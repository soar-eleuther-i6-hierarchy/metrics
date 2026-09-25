"""Per-metric diagnostics for a read. Grading itself lives in `metrics.rules.grading`."""

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
    """`None` below two finite values: one point cannot establish constancy, and zero points is
    an absent measurement rather than a perfectly stable one."""
    f = _finite(v)
    if f.numel() < 2:
        return None
    return bool(float(f.max() - f.min()) <= tol)


def constant_flags(vals: torch.Tensor, y: torch.Tensor, eval_null_idx: list[int],
                   tol: float = CONSTANT_TOL, null_class: str = NULL_CLASS,
                   target: tuple[str, ...] | None = None) -> dict:
    """`constant_null` / `constant_target` / `constant_overall` with the tolerance and the
    finite counts beside them (PRECOMMIT s6).

    A point-mass null and a different point-mass target separate perfectly, which is a
    structurally simple separation rather than a strong detector. The oracle `superparent`
    result is exactly that shape -- null `wide` at 0 against a constant target at -122 -- so
    this flag has to travel with the verdict or the number reads as a clean success.
    """
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    # The null population is the index set `run_read` supplies: the pairs whose BOTH orderings
    # carry the null label. That is strictly fewer than `y == null_class`, which also admits a
    # pair whose flip is `reversed` -- an ancestry pair seen backwards. Falling back to the
    # label when the index set is empty therefore uses a DIFFERENT population, so the fallback
    # is LABELLED rather than silent.
    if idx.numel():
        null_vals, null_pop = vals[idx], "the supplied null population"
    else:
        null_vals, null_pop = (vals[y == labels._index(null_class)],
                               f"every pair labelled {null_class} (no null index set supplied)")
    by_class = {name: _is_constant(vals[y == labels._index(name)], tol)
                for name in labels.LABELS}
    n_by_class = {name: int(torch.isfinite(vals[y == labels._index(name)]).sum())
                  for name in labels.LABELS}
    # `constant_target` is named in PRECOMMIT s6, so it is emitted under that name rather than
    # left to be read out of `constant_by_class` by someone who already knows the target. A
    # multi-class target (the containment baseline's union) is constant only if the classes
    # TOGETHER are, so the values are pooled rather than checked one class at a time.
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


# Metrics whose value is a function of ONE endpoint rather than of the pair, plus the two that
# are a function of the unordered pair's two endpoints. Three shapes, all in the same class:
#
#   per-PARENT, broadcast across a ROW (`detectors._broadcast_parent`): outdegree,
#       joint_child_J, joint_child_mass, sibling_redundancy, joint_child_supp,
#       sibling_redundancy_pc
#   per-CHILD, broadcast down a COLUMN (`detectors._broadcast_child`): recon_child_gain
#   both endpoints: wide, gate_high_outdegree
#
# No pair-level split holds any of these out: on the seed-0 worlds 100% of evaluation-half
# pairs shared both endpoints with some calibration-half pair, back when there were halves.
# The property is about the METRIC, not the split, so it outlived it. Recorded per metric so a rate on
# one of them is never read as held out. A real holdout for this class needs a FEATURE-level
# split, which is a different procedure and was never built.
#
# The same property is why the scorability mask is SELECTIVE (`detectors.MASKED_DETECTORS`):
# NaN-ing `outdegree[p, c]` because p and c rarely co-fire deletes a number that was never
# about that pair, and it propagates through `reads.wide_matrix` into `wide`.
ENDPOINT_BROADCAST: tuple[str, ...] = ("outdegree", "joint_child_J", "joint_child_mass",
                                       "sibling_redundancy", "joint_child_supp",
                                       "sibling_redundancy_pc", "recon_child_gain",
                                       "wide", "gate_high_outdegree")

# Detectors symmetric in (parent, child). Both orderings of a pair take the SAME value, so the
# two decisions are one decision and the reported denominator is twice the number of
# independent ones. This was compounded by both orderings sharing a calibration half by design;
# the halves are gone and the doubling is not, because it comes from the metric.
# Metrics that take the SAME value on (p, c) and (c, p), so both orderings of a pair are one
# decision, not two -- which is what `n_effective_null` halves the denominator for.
#
# `S_res` is deliberately NOT here. The repo's doctrine is at scoring/oracle/score_dump.py's
# `symmetric_detectors`: `s_res` is symmetric only in COSINE mode, where it is the Gram matrix
# `W @ W.T`. In this package `S_res` is the PROBE on both reads (reads.py:145) -- a directional
# margin -- and `G` is the cosine one. Listing `S_res` here halved its effective null denominator
# on a metric that never had the symmetry that justifies halving.
#
# The GATES are in this list too, and three of them belong: `gate_mutually_contains` is `ge & ge.T`,
# `gate_high_outdegree` is `flag[p] | flag[c]` and `gate_support` is built from a symmetric
# co-firing count, so each takes the same value on (p, c) and (c, p). Leaving them out was the
# live case, not a hypothetical: `rule_topical` and `rule_superparent` read exactly those, so the
# two rules whose null denominators are doubled were the two reported as undoubled.
# `gate_strictly_contains` is deliberately absent -- it is antisymmetric by construction, which is the
# opposite property.
SYMMETRIC_METRICS: tuple[str, ...] = ("pmi", "G", "wide", "abs_asymmetry_R",
                                      "gate_mutually_contains", "gate_high_outdegree", "gate_support")


def metric_diagnostics(vals: dict[str, torch.Tensor], y: torch.Tensor,
                       eval_null_idx: list[int], tol: float = CONSTANT_TOL,
                       targets: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Per-metric and per-gate distributions, constant-distribution flags, and how independent
    the null decisions behind a reported rate actually are.

    The thresholds, the calibration support and the boundary ties are gone with the quantile
    rule: there is no fitted boundary left for a value to tie against, and `gates` compares at
    0.5 against a tristate that never takes that value. What remains is the part that was never
    about the split -- `metric_class` and `n_effective_null` say when a reported rate rests on
    fewer independent decisions than its denominator suggests, which is a property of the
    metric and is still true.
    """
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
            # Both orderings of a pair take the same value for a symmetric metric, so the
            # independent-decision count is half the ordered-pair denominator.
            "n_effective_null": (n_ev // 2) if symmetric else n_ev,
            "flags": constant_flags(v, y, eval_null_idx, tol,
                                    target=(targets or {}).get(name)),
        }
    return out
