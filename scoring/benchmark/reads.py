"""The oracle and trained reads, reduced to one `Read` shape.

The oracle read scores true coefficients against true directions; the trained read matches a
checkpoint's latents to true features and scores the recovered universe on a held-out draw.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from scoring.benchmark.registry import GATES, METRICS, probe_fit_sample_seed
from scoring.core.detectors import (compute_bundle, fit_probe_directions,
                                    s_res_cosine, s_res_from_directions)
from scoring.core.grid import held_out_sample_seed, pair_frame, reduce_to_recovered
from scoring.core.recovery import activation_corr, match_features, per_class_recovery
from scoring.core.registry import CONSTANTS, DETECTORS
from scoring.core.world import regenerate_world, signed_normalized_decoder
from scoring.oracle.validate_metrics import pure_inputs
from toygen import labels
from toygen import spec
from toygen.world import resolve_config

_NAN = float("nan")
_TINY = 1e-12


@dataclass(frozen=True)
class Read:
    """One (toy, seed, read) scored universe. `pairs` are positions into `feats`, the true ids.

    `vals` holds one float64 score per METRICS name; `gate_vals` one float64 tristate
    {1.0, 0.0, NaN} per GATES name, kept apart so a gate is never read as a continuous score.
    """

    toy: str
    seed: int
    read: str                       # "oracle" | "trained"
    n_tokens: int
    s_res_mode: str                 # "probe" | "absent"
    F: int
    feats: list[int]
    pairs: list[tuple[int, int]]
    y: torch.Tensor
    vals: dict[str, torch.Tensor]
    gate_vals: dict[str, torch.Tensor]
    pair_labels: torch.Tensor
    W_unit: torch.Tensor
    recovered: torch.Tensor         # [F] bool, the scored universe
    detector_matrices: dict[str, torch.Tensor]
    extra: dict

    @property
    def n_recovered(self) -> int:
        return len(self.feats)


# --- shared machinery ---
def wide_matrix(outdegree: torch.Tensor) -> torch.Tensor:
    """`min(outdegree[p,c], outdegree[c,p])`, NaN unless both orderings are finite (PRECOMMIT s6).

    Masked explicitly: `torch.minimum(inf, 5)` is 5, and `fmin` would do the same with a NaN.
    """
    ok = torch.isfinite(outdegree) & torch.isfinite(outdegree.transpose(0, 1))
    w = torch.minimum(outdegree, outdegree.transpose(0, 1))
    return torch.where(ok, w, torch.full_like(w, _NAN))


def _scored(mat: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    return mat[pa, pb].double()


def add_derived(vals: dict[str, torch.Tensor], outdegree_matrix: torch.Tensor | None,
                pairs: list[tuple[int, int]] | None) -> dict[str, torch.Tensor]:
    """Add `abs_asymmetry_R` and, when the outdegree matrix is given, `wide`."""
    out = dict(vals)
    if "asymmetry_R" in out:
        out["abs_asymmetry_R"] = out["asymmetry_R"].abs()
    if outdegree_matrix is not None and pairs is not None:
        out["wide"] = _scored(wide_matrix(outdegree_matrix), pairs)
    return out


def class_totals(pair_labels: torch.Tensor) -> dict[str, int]:
    """Generated ordered pairs per class over the full feature set: the `N_total` denominator.

    From the answer key, not the scored pairs, which drop every unrecovered target.
    """
    F = int(pair_labels.shape[0])
    eye = torch.eye(F, dtype=torch.bool)
    return {name: int(((pair_labels == labels._index(name)) & ~eye).sum())
            for name in labels.LABELS}


def class_recovered(pair_labels: torch.Tensor, in_universe: torch.Tensor) -> dict[str, int]:
    """Ordered pairs per class with BOTH endpoints in the scored universe (`N_recovered`)."""
    return {name: n for name, (n, _tot) in
            per_class_recovery(in_universe, pair_labels, list(labels.LABELS)).items()}


def assemble_metrics(dets: dict[str, torch.Tensor], W_unit: torch.Tensor,
                     probe: torch.Tensor | None, pairs: list[tuple[int, int]]
                     ) -> dict[str, torch.Tensor]:
    """Every detector except `s_res`, plus `G` (cosine), `S_res` (probe) and the derived metrics.

    Both reads go through here, so neither can fill `G` from the probe or `S_res` from the cosine.
    """
    vals = {d: _scored(dets[d], pairs) for d in DETECTORS if d != "s_res"}
    vals["G"] = _scored(s_res_cosine(W_unit), pairs)
    vals["S_res"] = (_scored(probe, pairs) if probe is not None
                     else torch.full((len(pairs),), _NAN, dtype=torch.float64))
    vals = add_derived(vals, dets["outdegree"], pairs)
    missing = [m for m in METRICS if m not in vals]
    if missing:
        raise RuntimeError(f"the read did not produce {missing}; a registered expression "
                           f"would go untestable for a harness reason, not a metric one")
    return vals


def assemble_gates(gate_mats: dict[str, torch.Tensor],
                   pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Every registered gate on the ordered-pair frame; raises if the read lacks one.

    A missing gate would otherwise pass for a rule that is untestable on the data.
    """
    missing = [g for g in GATES if g not in gate_mats]
    if missing:
        raise RuntimeError(f"the read did not produce {missing}; a registered expression "
                           f"would go untestable for a harness reason, not a metric one")
    return {g: _scored(gate_mats[g], pairs) for g in GATES}


# --- the oracle read ---
def oracle_read(toy: str, seed: int, n_tokens: int, with_probe: bool = True,
                probe_fit_seed: int | None = None) -> Read:
    """Score the true coefficients against the true directions on the held-out draw of `seed`.

    `with_probe=False` records `s_res_mode="absent"`, so probe rules read INVALID MEASUREMENT.
    """
    # `cfg.seed` sets the geometry, `sample_seed` the draw. `resolve_config` leaves the seed at 0
    # and rejects `seed=`, so use `spec.replace`; passing only `sample_seed` scores seed-0 geometry.
    cfg = spec.replace(resolve_config(toy), seed=int(seed))
    rc = dataclasses.asdict(cfg)
    score_seed = held_out_sample_seed(int(seed))
    bundle = regenerate_world(rc, sample_seed=score_seed, n_tokens=n_tokens)
    F = int(bundle.g.shape[0])
    feats = list(range(F))
    idx = torch.tensor(feats, dtype=torch.long)
    inp = pure_inputs(bundle, feats, bundle.A[:, idx])

    # The probe is fitted on its own draw and frozen before the detectors, because
    # `compute_bundle` builds `gate_sres_rank` from the directions (PRECOMMIT s6).
    fit_seed = probe_fit_sample_seed(int(seed)) if probe_fit_seed is None else int(probe_fit_seed)
    probe, P, avail = None, None, None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens)
        fi = pure_inputs(fw, feats, fw.A[:, idx])
        # true firing as the fitting label here; the trained read uses the SAE's own
        P, avail = fit_probe_directions(fi.h, fi.acts_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, inp.W_unit)

    # cosine mode: the internal `s_res` key holds G here and is dropped by `assemble_metrics`.
    bnd = compute_bundle(inp, CONSTANTS, s_res_mode="cosine",
                         probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]

    pairs, y = pair_frame(feats, bundle.pair_labels)
    return Read(
        toy=toy, seed=seed, read="oracle", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=F, feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, inp.W_unit, probe, pairs),
        gate_vals=assemble_gates(bnd["gates"], pairs),
        pair_labels=bundle.pair_labels, W_unit=inp.W_unit,
        recovered=torch.ones(F, dtype=torch.bool),
        detector_matrices=dets,
        extra={"support": bnd["support"],
               "true_l0": float(bundle.A.gt(0).double().sum(dim=1).mean()),
               "resolved_config": rc,
               # the draw actually scored, not just the experiment seed
               "scoring_sample_seed": score_seed,
               "matching_sample_seed": None,    # the oracle read has no matcher
               "probe_fit_sample_seed": (fit_seed if with_probe else None),
               "probe_fit_labels": ("true firing (A)" if with_probe else None)},
    )


# --- the trained read ---
def trained_read(ckpt: str, n_tokens: int, with_probe: bool = True,
                 probe_fit_seed: int | None = None) -> Read:
    """Load a checkpoint and score its recovered universe on the held-out draw.

    Matches on the training-seed draw, as the saved run did, so `harness_gate` can compare them.
    """
    from scoring.trained.loaders import load_sae

    L = load_sae(ckpt)
    W, rc = L.W_dec, L.meta["resolved_config"]
    train_seed = int(L.meta["train_seed"])

    # A checkpoint whose config seed disagrees with `train_seed` was trained on another world,
    # and no downstream check would see it.
    cfg_seed = rc.get("seed")
    if cfg_seed is not None and int(cfg_seed) != train_seed:
        raise RuntimeError(
            f"checkpoint {ckpt} has resolved_config.seed={cfg_seed} but train_seed={train_seed}. "
            f"Geometry comes from cfg.seed and the draw from sample_seed, so these disagreeing "
            f"means the trained world is not the world this checkpoint is labelled with.")

    inw = regenerate_world(rc, sample_seed=train_seed, n_tokens=n_tokens)
    ai = L.encode(inw.h)
    oi = signed_normalized_decoder(W, ai, inw.h)
    res = match_features(activation_corr(inw.A, ai), inw.g, oi, rho=CONSTANTS["rho_star"])

    ho = regenerate_world(rc, sample_seed=held_out_sample_seed(train_seed), n_tokens=n_tokens)
    ah = L.encode(ho.h)
    oh = signed_normalized_decoder(W, ah, ho.h)
    feats, di, _index_map = reduce_to_recovered(
        ah, oh, W, res.match, res.recovered,
        h=ho.h, b_dec=L.b_dec, tokens=ho.tokens, vocab=ho.cfg.vocab)
    # Same fitting draw as the oracle read, but the SAE's own activations as labels
    # (`probe_self_W`), so the self-label circularity remains. Fitted before the detectors.
    fit_seed = (probe_fit_sample_seed(train_seed) if probe_fit_seed is None
                else int(probe_fit_seed))
    probe, P, avail = None, None, None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens)
        af = L.encode(fw.h)
        # same recovered positions as the scoring frame, so columns line up with `di.W_unit`
        of = signed_normalized_decoder(W, af, fw.h)
        _f2, dfit, _im2 = reduce_to_recovered(
            af, of, W, res.match, res.recovered,
            h=fw.h, b_dec=L.b_dec, tokens=fw.tokens, vocab=fw.cfg.vocab)
        P, avail = fit_probe_directions(dfit.h, dfit.acts_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, di.W_unit)

    bnd = compute_bundle(di, CONSTANTS, s_res_mode="cosine",
                         probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]

    pairs, y = pair_frame(feats, ho.pair_labels)

    # `reduce_to_recovered` keeps recovered AND matched features; not `res.recovered` alone.
    in_universe = torch.zeros(int(res.match.shape[0]), dtype=torch.bool)
    in_universe[torch.tensor(feats, dtype=torch.long)] = True

    # True-direction cosine on the same endpoints, a diagnostic control: it separates training
    # moving the geometry from recovery dropping the hard endpoints.
    g_unit = (ho.g / ho.g.norm(dim=1, keepdim=True).clamp_min(_TINY)).double()
    g_matched = s_res_cosine(g_unit[torch.tensor(feats, dtype=torch.long)])

    return Read(
        toy=str(rc.get("name", "unknown")), seed=train_seed, read="trained", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=int(ho.g.shape[0]), feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, di.W_unit, probe, pairs),
        gate_vals=assemble_gates(bnd["gates"], pairs),
        pair_labels=ho.pair_labels, W_unit=di.W_unit,
        recovered=in_universe,
        detector_matrices=dets,
        extra={"support": bnd["support"], "ckpt": str(ckpt), "G_g_matched": _scored(g_matched, pairs),
               "match": res.match, "n_recovered": len(feats),
               "effective_alpha": effective_alpha(ho.tree),
               "alpha_designed_in_meta": rc.get("alpha"),
               "resolved_config": rc,
               # the same keys the oracle read records, cross-checked by `verify_manifest`
               "scoring_sample_seed": held_out_sample_seed(train_seed),
               "matching_sample_seed": train_seed,
               "probe_fit_sample_seed": (fit_seed if with_probe else None),
               "probe_fit_labels": ("the SAE's own activations" if with_probe else None),
               "sae_meta": {
                   k: L.meta.get(k) for k in ("train_seed", "variant", "k", "d_sae")}},
    )


def effective_alpha(tree) -> dict:
    """Summary of the alpha actually planted on containment edges.

    The config's nominal `alpha` reads 0.48 even on `only_firing`, where every edge is zeroed.
    """
    a = [float(al) for c in range(tree.F) for (_p, _pe, al) in tree.parents.get(c, [])]
    if not a:
        return {"n_edges": 0, "median": _NAN, "min": _NAN, "max": _NAN, "frac_zero": _NAN}
    t = torch.tensor(a, dtype=torch.float64)
    return {"n_edges": int(t.numel()), "median": float(t.median()),
            "min": float(t.min()), "max": float(t.max()),
            "frac_zero": float((t == 0).double().mean())}
