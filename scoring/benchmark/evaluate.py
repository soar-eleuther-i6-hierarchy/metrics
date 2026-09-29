"""Per-metric diagnostics for a read; grading is re-exported from `metrics.rules.grading`."""

from __future__ import annotations

import torch

from metrics.rules.grading import (  # noqa: F401
    BAR_CONFOUND_LEAK,
    VERDICT_SCOPE,
    VERDICTS,
    class_counts,
    grade_rules,
    leak_exceedances,
    leakage,
    leakage_over_recovered,
    rule_overlap,
    target_rollup,
    under_supported_confounds,
    unmeasurable_confounds,
    verdict,
)
from scoring.benchmark.registry import CONSTANT_TOL, NULL_CLASS
from toygen import labels


def _finite(v: torch.Tensor) -> torch.Tensor:
    return v[torch.isfinite(v)]


def _is_constant(v: torch.Tensor, tol: float) -> bool | None:
    """`None` below two finite values: one point cannot show constancy, zero is no measurement."""
    f = _finite(v)
    if f.numel() < 2:
        return None
    return bool(float(f.max() - f.min()) <= tol)


def constant_flags(vals: torch.Tensor, y: torch.Tensor, eval_null_idx: list[int],
                   tol: float = CONSTANT_TOL, null_class: str = NULL_CLASS,
                   target: tuple[str, ...] | None = None) -> dict:
    """`constant_null`, `constant_target` and `constant_overall`, with tolerance and finite counts.

    A point-mass null against a point-mass target separates perfectly without any real detection
    (PRECOMMIT.md s6), so these flags travel with the verdict.
    """
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    # The label fallback is a larger population (it admits pairs whose flip is `reversed`), so
    # it is labelled rather than silent.
    if idx.numel():
        null_vals, null_pop = vals[idx], "the supplied null population"
    else:
        null_vals, null_pop = (vals[y == labels._index(null_class)],
                               f"every pair labelled {null_class} (no null index set supplied)")
    by_class = {name: _is_constant(vals[y == labels._index(name)], tol)
                for name in labels.LABELS}
    n_by_class = {name: int(torch.isfinite(vals[y == labels._index(name)]).sum())
                  for name in labels.LABELS}
    # A multi-class target is constant only if its classes are constant together, so pool them.
    constant_target = None
    if target:
        sel = torch.zeros(vals.numel(), dtype=torch.bool)
        for name in target:
            sel |= (y == labels._index(name))
        constant_target = _is_constant(vals[sel], tol)
    return {
        "tol": tol,
        "null_population": null_pop,
        "n_finite_overall": int(torch.isfinite(vals).sum()),
        "n_finite_null": int(torch.isfinite(null_vals).sum()),
        "n_finite_by_class": n_by_class,
        "constant_overall": _is_constant(vals, tol),
        "constant_null": _is_constant(null_vals, tol),
        "constant_target": constant_target,
        "constant_by_class": by_class,
    }


# Values set by one endpoint (per-parent rows, per-child `recon_child_gain`) or by the two
# endpoints (`wide`, `gate_high_outdegree`), not by the pair. No pair-level holdout exists for
# them, and the pair support mask must not NaN them (`detectors.MASKED_DETECTORS`).
ENDPOINT_BROADCAST: tuple[str, ...] = ("outdegree", "joint_child_J", "joint_child_mass",
                                       "sibling_redundancy", "joint_child_supp",
                                       "sibling_redundancy_pc", "recon_child_gain",
                                       "wide", "gate_high_outdegree")

# Metrics and gates equal on (p, c) and (c, p): both orderings are one decision, so
# `n_effective_null` halves the null denominator for them.
# `S_res` is not here: it is the directional probe on both reads; `G` is the cosine one.
# `gate_strictly_contains` is not here either: it is antisymmetric.
SYMMETRIC_METRICS: tuple[str, ...] = ("pmi", "G", "wide", "abs_asymmetry_R",
                                      "gate_mutually_contains", "gate_high_outdegree", "gate_support")


def metric_diagnostics(vals: dict[str, torch.Tensor], y: torch.Tensor,
                       eval_null_idx: list[int], tol: float = CONSTANT_TOL,
                       targets: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Per-metric and per-gate distributions and constant flags. `metric_class` and
    `n_effective_null` mark rates resting on fewer independent decisions than the denominator."""
    out: dict[str, dict] = {}
    n_ev = len(eval_null_idx)
    for name, v in vals.items():
        fin = _finite(v)
        broadcast = name in ENDPOINT_BROADCAST
        symmetric = name in SYMMETRIC_METRICS
        out[name] = {
            "n": int(v.numel()), "n_finite": int(fin.numel()),
            "median": float(fin.median()) if fin.numel() else float("nan"),
            "metric_class": ("endpoint-broadcast: the value is a function of one endpoint (or "
                             "of the unordered pair), so no pair-level holdout exists for it"
                             if broadcast else "pair-level"),
            "symmetric": symmetric,
            "n_effective_null": (n_ev // 2) if symmetric else n_ev,
            "flags": constant_flags(v, y, eval_null_idx, tol,
                                    target=(targets or {}).get(name)),
        }
    return out
