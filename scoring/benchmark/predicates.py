"""The two gate predicates, and conjunction over them.

Every predicate returns `(mask, scorable)`:

  `scorable`  the gate has a finite value for this pair, i.e. the rule could be evaluated.
  `mask`      the predicate is TRUE, and the pair is scorable. `mask` is always a subset of
              `scorable`, so `N_pass <= N_scorable` is an invariant, not a hope.

WHAT WAS DELETED AND WHY. Eight quantile predicates lived here -- HIGH, NOT-HIGH, LOW, NOT-LOW,
IN-BAND, SYM, FREQ-LOCAL, SURVIVES -- comparing each metric against a percentile of the
unrelated-pair null. That rule cannot be carried to a real SAE. It accepts 1% of whatever
population it is pointed at, which is a few hundred pairs on a 24-feature toy and millions on a
16k-latent dictionary, so it stays computable and stops meaning anything. It also made the
false-positive bar unfailable: the threshold was fitted on one half of the null and measured on
the other, and two exchangeable halves put ~1% over a q99 cut by arithmetic, with conjunctions
pushing that lower still. Only recall could ever fail. The replacement compares against FIXED
constants instead, in `scoring/core/gates.py`.

Both defects the pilot evaluator had still live here, and both are still guarded.

1. NOT-HIGH was implemented as `~HIGH`. In IEEE arithmetic `NaN > q` is False, so `~(NaN > q)`
   is TRUE: a pair with a MISSING score satisfied the negative clause. `FAILS` is the
   descendant of that clause and is written LITERALLY below, never as `~PASSES`. A gate is
   tristate, so NaN must satisfy NEITHER predicate -- the whole reason gates carry a NaN state
   is to say "no evidence", and `~PASSES` would convert every one of those into a confident
   rejection.

2. Scorability was taken from ONE metric for every expression. A conjunction is scorable only
   where every clause it contains is, which is why `evaluate` intersects the per-clause masks
   rather than taking a single finiteness vector from the caller.

A predicate may only be applied to a REGISTERED GATE. `PASSES(coverage_R)` is shape-legal and
would silently test whether a coverage exceeds 0.5 -- a plausible number answering a question
nobody asked -- so `evaluate` refuses any clause naming something outside `GATE_NAMES`.
"""

from __future__ import annotations

import torch

from scoring.core.gates import GATE_NAMES

# A gate is {1.0, 0.0, NaN}. The comparison point sits between the two defined values rather
# than at either of them, so neither `>=` nor `>` can be got wrong at a boundary that no gate
# ever lands on.
GATE_TRUE = 0.5


def _finite(v: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(v)


# --------------------------------------------------------------------------
# the predicates
# --------------------------------------------------------------------------
def passes(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate holds. NaN (never measurable) is not a pass."""
    s = _finite(v)
    return (s & (v > GATE_TRUE), s)


def fails(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate does not hold, written literally. NOT `~passes`: that form passes on NaN.

    `passes` and `fails` are complements only on the FINITE values. Over the whole tensor they
    are not, and the gap is exactly the population a rule has no evidence about.
    """
    s = _finite(v)
    return (s & (v <= GATE_TRUE), s)


PREDICATES: dict[str, dict] = {
    "PASSES": {"fn": passes},
    "FAILS": {"fn": fails},
}


# --------------------------------------------------------------------------
# conjunction
# --------------------------------------------------------------------------
def evaluate(clauses, vals: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Evaluate a conjunction of `(PREDICATE, gate)` clauses.

    Returns `(mask, scorable, per_clause)`. `scorable` is the intersection of the clauses'
    scorable masks -- CLAUSE-SPECIFIC, which is the point: a pair with a finite
    `gate_parent_of` but no probe is not scorable for a rule that reads `gate_sres_rank`, and
    must not be counted as a rejection.

    An unknown predicate, an unknown name, or a name that is not a registered gate raises
    rather than being skipped: a typo in the frozen registry would otherwise loosen a rule
    after the freeze with no visible change.
    """
    if not clauses:
        raise ValueError("an expression needs at least one clause")
    n = len(next(iter(vals.values())))
    mask = torch.ones(n, dtype=torch.bool)
    scorable = torch.ones(n, dtype=torch.bool)
    per: dict[str, dict] = {}
    for pred_name, gate in clauses:
        if pred_name not in PREDICATES:
            raise KeyError(f"unknown predicate {pred_name!r} in a registered expression")
        if gate not in GATE_NAMES:
            raise KeyError(
                f"{pred_name}({gate}) names {gate!r}, which is not a registered gate. A "
                f"predicate applied to a METRIC compares a score against {GATE_TRUE}, which is "
                f"a number but not a decision.")
        if gate not in vals:
            raise KeyError(f"the read did not produce gate {gate!r}")
        m, s = PREDICATES[pred_name]["fn"](vals[gate])
        key = f"{pred_name}({gate})"
        per[key] = {"n_scorable": int(s.sum()), "n_pass": int(m.sum())}
        mask = mask & m
        scorable = scorable & s
    # mask is already a subset of every clause's scorable mask, but state the invariant.
    mask = mask & scorable
    return mask, scorable, per

# `wide_matrix` deliberately lives in `reads.py` and only there. It was briefly duplicated
# here, which is how a NaN-propagation rule ends up correct in one copy and wrong in the other.
