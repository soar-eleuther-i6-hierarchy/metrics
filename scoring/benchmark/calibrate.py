"""The calibration/evaluation null split and the null quantiles.

`PRECOMMIT.md` s6.4: assign unrelated UNORDERED pairs to calibration/evaluation groups with a
frozen seed, keeping both orderings together, and **assign using true feature IDs before
intersecting with recovered endpoints so a pair retains its group across reads**.

The pilot evaluator did the assignment on RECOVERED POSITIONS. Positions are read-specific --
the oracle read scores all F features and the trained read scores whatever survived matching --
so the same true pair sat at different indices in the two reads and could land in different
halves. That makes the "held-out" evaluation null overlap the calibration null it was supposed
to be disjoint from, and the reported FPR is then not the quantity the protocol names.

The fix is to build the assignment from the generator's `[F, F]` answer key alone. It is then a
pure function of `(pair_labels, seed)`: identical for every read of that (toy, seed), and
computed before recovery is known.

Thresholds come from the CALIBRATION half only, and from FINITE values only. Support below
`MIN_CAL_SUPPORT` yields `(nan, nan)`, which makes the dependent rules untestable rather than
letting a p99 be borrowed from a handful of points.

WHAT THIS SPLIT DOES AND DOES NOT BUY, stated here because it is easy to overread.
It holds out PAIRS, not FEATURES. Four of the ten detectors are per-parent vectors broadcast
across a row (`outdegree`, `joint_child_J`, `joint_child_mass`, `sibling_redundancy`), so such
a pair's score is a function of ONE endpoint, and `wide` is a function of the unordered pair's
two endpoints. Measured on the seed-0 worlds, 100% of evaluation-half pairs share both endpoints
with some calibration-half pair, and on `only_superparent` the `wide` calibration and evaluation
halves are one identical value. For those metrics the split holds out essentially nothing and
the resulting rate is NOT a held-out FPR in any useful sense. It is a real holdout only for
pair-level metrics such as `pmi`, whose calibration and evaluation quantiles do differ.
PRECOMMIT s6 discloses the dependence in general; this is its specific measured form, and
`evaluate.metric_diagnostics` records which kind each metric is.
"""

from __future__ import annotations

import math

import torch

from scoring.benchmark.registry import CAL_SPLIT_SEED, MIN_CAL_SUPPORT, NULL_CLASS, Q_HI, Q_LO
from toygen import labels

_NAN = float("nan")
Key = tuple[int, int]


# --------------------------------------------------------------------------
# the split, over TRUE feature ids
# --------------------------------------------------------------------------
def null_unordered_keys(pair_labels: torch.Tensor,
                        null_class: str = NULL_CLASS) -> list[Key]:
    """Sorted `(a, b)` with `a < b` where BOTH orderings carry the null label.

    Requiring both orderings makes the key unambiguous: a pair whose flip is `reversed` is an
    ancestry pair seen the wrong way round, not a null one, and it must not enter the null
    population under either ordering.
    """
    idx = labels._index(null_class)
    m = pair_labels == idx
    both = m & m.transpose(0, 1)
    F = int(pair_labels.shape[0])
    iu = torch.triu_indices(F, F, offset=1)
    sel = both[iu[0], iu[1]]
    return [(int(a), int(b)) for a, b, keep in zip(iu[0].tolist(), iu[1].tolist(),
                                                   sel.tolist()) if keep]


def null_split_assignment(pair_labels: torch.Tensor, seed: int = CAL_SPLIT_SEED,
                          null_class: str = NULL_CLASS) -> dict[Key, bool]:
    """`{(a, b) with a < b: True if calibration}` over the FULL true-feature null.

    Deterministic in `(pair_labels, seed)` and independent of any read's recovered set, which
    is what makes a pair's group stable across the oracle and trained reads of the same world.
    Both orderings share the key, so they share the half.
    """
    keys = null_unordered_keys(pair_labels, null_class)
    perm = torch.randperm(len(keys), generator=torch.Generator().manual_seed(int(seed)))
    cal = {keys[i] for i in perm[: len(keys) // 2].tolist()}
    return {k: (k in cal) for k in keys}


def split_masks(pairs: list[tuple[int, int]], feats: list[int],
                assignment: dict[Key, bool]
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map an ordered-pair frame onto the assignment. Returns `(cal, eval, unassigned)`.

    `pairs` are POSITIONS into `feats`; the lookup key is built from the TRUE feature ids
    `feats[a], feats[b]`. `unassigned` is every pair that is not in the null population
    (targets, confounds, reversed) -- reported rather than silently folded into a half.
    """
    n = len(pairs)
    cal = torch.zeros(n, dtype=torch.bool)
    ev = torch.zeros(n, dtype=torch.bool)
    unk = torch.zeros(n, dtype=torch.bool)
    for i, (a, b) in enumerate(pairs):
        fa, fb = int(feats[a]), int(feats[b])
        key = (fa, fb) if fa < fb else (fb, fa)
        if key not in assignment:
            unk[i] = True
        elif assignment[key]:
            cal[i] = True
        else:
            ev[i] = True
    return cal, ev, unk


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
def null_quantiles(vals: torch.Tensor, q_lo: float = Q_LO, q_hi: float = Q_HI,
                   min_support: int = MIN_CAL_SUPPORT) -> tuple[float, float]:
    """`(Q01, Q99)` of the FINITE values, linear interpolation (PRECOMMIT s6).

    Returns `(nan, nan)` below `min_support`. An inadequately supported null makes the rule
    untestable; it must not be silently replaced by a number borrowed from too few points.
    Non-finite entries do not count toward the support floor.
    """
    fin = vals[torch.isfinite(vals)].double()
    if fin.numel() < min_support:
        return (_NAN, _NAN)
    return (float(torch.quantile(fin, q_lo, interpolation="linear")),
            float(torch.quantile(fin, q_hi, interpolation="linear")))


def fit_thresholds(vals: dict[str, torch.Tensor], cal_mask: torch.Tensor,
                   min_support: int = MIN_CAL_SUPPORT,
                   q_lo: float = Q_LO, q_hi: float = Q_HI
                   ) -> dict[str, tuple[float, float]]:
    """`(Q01, Q99)` per metric from the CALIBRATION half of this world's null.

    One threshold per metric per context, reused across every expression in that context. It is
    never chosen from the evaluated pair's true label (PRECOMMIT s6), and never refitted on the
    evaluation half -- that difference is what separates a held-out FPR from a fitted-on one.
    """
    return {name: null_quantiles(v[cal_mask], q_lo, q_hi, min_support)
            for name, v in vals.items()}


def boundary_ties(vals: torch.Tensor, th: tuple[float, float]) -> dict[str, int | None]:
    """How many finite values sit EXACTLY on each threshold (PRECOMMIT s7 box 5).

    The predicates fix strict-vs-inclusive precisely because ties decide cases: `HIGH` is
    `> Q99` and `NOT-HIGH` is `<= Q99`, so a pair exactly at Q99 goes one way and not the other.
    Counting the ties makes a rule whose recall hangs on exact equality visible instead of
    inferred. The comparison is exact, not tolerant, so the count describes the same comparison
    the decision actually used. `None` where the threshold is unusable - no boundary is not the
    same as no boundary cases.
    """
    fin = vals[torch.isfinite(vals)]

    def _n(q: float) -> int | None:
        return int((fin == q).sum()) if math.isfinite(q) else None

    return {"n_at_q01": _n(th[0]), "n_at_q99": _n(th[1])}


def calibration_support(vals: dict[str, torch.Tensor],
                        cal_mask: torch.Tensor) -> dict[str, int]:
    """Finite calibration-score count per metric, reported next to each threshold so an
    unusable threshold can be told apart from a merely unlucky one."""
    return {name: int(torch.isfinite(v[cal_mask]).sum()) for name, v in vals.items()}
