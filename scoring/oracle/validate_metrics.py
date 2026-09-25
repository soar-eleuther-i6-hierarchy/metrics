"""Detector checks on ground-truth inputs: finiteness, non-degeneracy, and the s_res probe vs its cosine form.

Activations come from the true coefficients `A` or from `oracle_encode`; no trained SAE is involved.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from scoring.core.detectors import (DetectorInputs, compute_all, s_res_cosine,
                                    s_res_probe)
from scoring.core.registry import CONSTANTS as _REGISTRY_CONSTANTS
from toygen import labels

if TYPE_CHECKING:
    from scoring.core.world import WorldBundle

_TINY = 1e-12

# Ridge penalty for the oracle encoder's magnitudes; fixed, not tuned to any SAE.
RIDGE_LAMBDA = 1e-4

# With fewer defined pairs than this, `s_res_calibration` reports NaN.
_MIN_CALIB_PAIRS = 100


class OracleEncodeInfeasible(ValueError):
    """The oracle encoder cannot reach the requested L0 for this draw.

    Subclasses ValueError; catch it by name so a compute_all config error is not swallowed."""


def _gate_support(h: torch.Tensor, g: torch.Tensor, realized_l0: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Tied unit-g JumpReLU gate at mean L0 == realized_l0; returns (support [n, F], g_unit [F, D]).

    The threshold is set over positive projections only, so it stays > 0 on dense checkpoints."""
    target = float(realized_l0)
    if not (target > 0) or not math.isfinite(target):
        raise OracleEncodeInfeasible(
            f"oracle_encode: realized_l0 must be a positive finite number, got {target}")
    g_unit = g.double() / g.double().norm(dim=1, keepdim=True).clamp_min(_TINY)
    proj = h.double() @ g_unit.transpose(0, 1)             # [n, F]
    n, F = proj.shape
    pos = proj[proj > 0]                                   # [P]
    k = int(round(target * n))                             # total firing entries wanted
    P = int(pos.numel())
    if k <= 0:
        raise OracleEncodeInfeasible(
            f"oracle_encode: mean L0={target:.3g} rounds to k=0 firing entries over n={n} tokens, "
            f"i.e. a (near-)dead SAE the gate cannot represent. Failing loud rather than requesting a "
            f"0-th quantile.")
    if k >= P:
        raise OracleEncodeInfeasible(
            f"oracle_encode: mean L0={target:.2f} needs {k} firing entries but only {P} projections "
            f"are positive (~{P / max(n, 1):.1f}/row), which a non-negative JumpReLU gate cannot reach "
            f"(the tied-unit-g encoder fires at most ~F/2 latents/row). The checkpoint is denser than "
            f"this oracle encoder can mirror; refusing to silently plateau the ceiling.")
    theta = torch.kthvalue(pos, P - k + 1).values          # k-th largest positive projection
    return proj > theta, g_unit


def oracle_encode(h: torch.Tensor, g: torch.Tensor, realized_l0: float,
                  lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """Oracle activations: a tied unit-g gate picks the support, per-token ridge on raw `g` sets magnitudes.

    Pass the trained SAE's realized L0 on real h, not its nominal k. Raises `OracleEncodeInfeasible`."""
    support, _ = _gate_support(h, g, realized_l0)
    gd = g.double()
    n, F = support.shape
    acts = torch.zeros(n, F, dtype=torch.float64, device=gd.device)
    hd = h.double()
    for t in range(n):
        S = support[t].nonzero(as_tuple=True)[0]
        if S.numel() == 0:
            continue
        Gs = gd[S]                                         # [k, D] raw decoder rows
        gram = Gs @ Gs.transpose(0, 1)                     # [k, k]
        gram = gram + lam * torch.eye(S.numel(), dtype=torch.float64, device=gd.device)
        acts[t, S] = torch.linalg.solve(gram, Gs @ hd[t])  # [k]
    return acts


def reconstruction_fvu(h: torch.Tensor, acts: torch.Tensor, g: torch.Tensor) -> float:
    """`||h - acts @ g||^2 / ||h||^2` with the raw decoder `g`; unlike `core.recovery`, no mean-centering."""
    h_hat = acts.double() @ g.double()
    err = h.double() - h_hat
    return float((err ** 2).sum() / (h.double() ** 2).sum().clamp_min(_TINY))


def _constant_scored_detectors(detectors: dict[str, torch.Tensor],
                               pairs: list[tuple[int, int]], tol: float = 1e-9) -> list[str]:
    """Sorted names of detectors with fewer than 2 finite values over `pairs` or a finite range below `tol`."""
    degen: list[str] = []
    for det, mat in detectors.items():
        vals = torch.tensor([float(mat[p, c]) for (p, c) in pairs], dtype=torch.float64)
        vals = vals[torch.isfinite(vals)]                  # drop NaN and +/-inf
        if vals.numel() < 2 or float(vals.max() - vals.min()) < tol:
            degen.append(det)
    return sorted(degen)


# --- validation checks ---
def pure_inputs(bundle: "WorldBundle", feats: list[int], acts: torch.Tensor) -> DetectorInputs:
    """DetectorInputs on the recovered features `feats`, from the given activations and the true decoders `g`."""
    idx = torch.tensor(feats, dtype=torch.long)
    g_sel = bundle.g[idx]
    W_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
    return DetectorInputs(
        acts_rec=acts, W_unit=W_unit, W_raw=g_sel, h=bundle.h,
        b_dec=torch.zeros(bundle.g.shape[1], dtype=bundle.g.dtype),
        tokens=bundle.tokens, vocab=bundle.cfg.vocab,
    )


def _detector_validity(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                       pair_classes: torch.Tensor, tol: float) -> dict[str, dict]:
    """Per detector over `pairs`: finite count, finite (>= 2), degenerate, and finite fraction per true class.

    The per-class fraction tells designed abstention (NaN on a class it cannot measure) from breakage."""
    degen = set(_constant_scored_detectors(detectors, pairs, tol))
    pa = torch.tensor([a for (a, _b) in pairs], dtype=torch.long)
    pb = torch.tensor([b for (_a, b) in pairs], dtype=torch.long)
    class_masks = {name: pair_classes == labels._index(name) for name in labels.LABELS}
    out: dict[str, dict] = {}
    for det, mat in detectors.items():
        finite = torch.isfinite(mat[pa, pb]) if pa.numel() else torch.zeros(0, dtype=torch.bool)
        n_finite = int(finite.sum())
        by_class = {name: (float(finite[m].double().mean()) if int(m.sum()) else float("nan"))
                    for name, m in class_masks.items()}
        out[det] = {"n_finite": n_finite, "finite": n_finite >= 2, "degenerate": det in degen,
                    "finite_by_class": by_class}
    return out


def machinery_report(bundle: "WorldBundle", feats: list[int], pairs: list[tuple[int, int]],
                     realized_l0: float, constants: dict | None = None,
                     tol: float = 1e-9, lam: float = RIDGE_LAMBDA) -> dict:
    """Per-detector validity under `true_A` and `alpha_encoder` activations, plus the encoder's FVU.

    A detector passes if finite and non-degenerate in at least one regime; some are degenerate in one by design.
    """
    constants = _REGISTRY_CONSTANTS if constants is None else constants
    idx = torch.tensor(feats, dtype=torch.long)
    alpha_full = oracle_encode(bundle.h, bundle.g, realized_l0, lam=lam)   # full dictionary
    alpha_acts = alpha_full[:, idx]
    trueA_acts = bundle.A[:, idx]
    regimes = {
        "true_A": compute_all(pure_inputs(bundle, feats, trueA_acts), constants),
        "alpha_encoder": compute_all(pure_inputs(bundle, feats, alpha_acts), constants),
    }
    # truth label per pair; the only place truth enters
    pair_classes = torch.tensor(
        [int(bundle.pair_labels[feats[a], feats[b]]) for (a, b) in pairs], dtype=torch.long)
    per_regime = {name: _detector_validity(dets, pairs, pair_classes, tol)
                  for name, dets in regimes.items()}
    detectors = list(regimes["true_A"].keys())
    passed = {
        det: any(per_regime[r][det]["finite"] and not per_regime[r][det]["degenerate"]
                 for r in per_regime)
        for det in detectors
    }
    return {
        "per_regime": per_regime,
        "passed": passed,
        "all_passed": all(passed.values()),
        "failed": sorted(d for d, ok in passed.items() if not ok),
        # whole encoder output against all of g; the recovered slice would inflate FVU
        "alpha_encoder_fvu": reconstruction_fvu(bundle.h, alpha_full, bundle.g),
    }


def _corr(x: torch.Tensor, y: torch.Tensor, kind: str) -> float:
    if x.numel() < 2:
        return float("nan")                                # undefined, not a real zero correlation
    if kind == "spearman":
        x = x.argsort().argsort().double()
        y = y.argsort().argsort().double()
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    return float((x @ y) / (x.norm() * y.norm() + _TINY))


def s_res_calibration(bundle: "WorldBundle", feats: list[int], constants: dict | None = None,
                      device: str | None = None) -> dict:
    """Pearson, Spearman and mean |diff| between probe s_res and cosine s_res, on true firing and true `g`.

    Probe permutations are drawn on CPU, so CPU and GPU runs agree to optimizer tolerance, not bit for bit."""
    constants = _REGISTRY_CONSTANTS if constants is None else constants
    idx = torch.tensor(feats, dtype=torch.long)
    g_sel = bundle.g[idx]
    g_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
    A_rec = bundle.A[:, idx]
    h = bundle.h
    if device is not None:
        g_unit, A_rec, h = g_unit.to(device), A_rec.to(device), h.to(device)
    cosine_g = s_res_cosine(g_unit)
    probe_true_g = s_res_probe(A_rec, h, g_unit, constants, label_acts=A_rec)
    R = cosine_g.shape[0]
    offdiag = ~torch.eye(R, dtype=torch.bool, device=cosine_g.device)
    cv, pv = cosine_g[offdiag], probe_true_g[offdiag]
    ok = torch.isfinite(cv) & torch.isfinite(pv)
    cv, pv = cv[ok], pv[ok]
    enough = cv.numel() >= _MIN_CALIB_PAIRS
    return {
        "n_pairs": int(cv.numel()),
        "pearson": _corr(cv, pv, "pearson") if enough else float("nan"),
        "spearman": _corr(cv, pv, "spearman") if enough else float("nan"),
        "mean_abs_diff": float((cv - pv).abs().mean()) if enough else float("nan"),
    }
