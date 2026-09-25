"""
Rules: named conjunctions of gates, and the evaluator that applies them.

A rule is `{target, clauses, text}`. `target` is the pair class it is meant to find, and each
clause is `(PREDICATE, gate)`. Rules are named after the class they target, never after a toy:
a toy can hold several classes and then needs several rules.

Rule names carry no version. `RULESET_VERSION` does instead: bump it whenever a rule, a gate
or a value in `constants.py` changes, and stamp it on every result so two results decided by
different rules are never pooled. History of each rule is in PRECOMMIT.md s4.
"""

from __future__ import annotations

import torch

from .gates import GATE_NAMES

RULESET_VERSION = 1

# --------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------
# A gate is {1.0, 0.0, NaN}. The comparison point sits between the two defined values, so
# `>=` versus `>` cannot be got wrong at a boundary no gate lands on.
GATE_TRUE = 0.5


def _finite(v: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(v)


def passes(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(mask, scorable)`: the gate holds. NaN (never measurable) is not a pass."""
    s = _finite(v)
    return (s & (v > GATE_TRUE), s)


def fails(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate does not hold, written literally. NOT `~passes`: that form passes on NaN.

    `passes` and `fails` are complements only on the finite values; the gap is exactly the
    population a rule has no evidence about.
    """
    s = _finite(v)
    return (s & (v <= GATE_TRUE), s)


PREDICATES: dict[str, dict] = {
    "PASSES": {"fn": passes},
    "FAILS": {"fn": fails},
}


def evaluate(clauses, vals: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Evaluate a conjunction of `(PREDICATE, gate)` clauses over gate tensors of any shape.

    Returns `(mask, scorable, per_clause)`. `scorable` is the INTERSECTION of the clauses'
    scorable masks: a pair with no probe is not scorable for a rule that reads
    `gate_sres_rank`, and must not be counted as a rejection.

    An unknown predicate, an unknown name, or a name that is not a registered gate raises
    rather than being skipped: a typo would otherwise loosen a rule with no visible change.
    """
    if not clauses:
        raise ValueError("an expression needs at least one clause")
    shape = next(iter(vals.values())).shape
    mask = torch.ones(shape, dtype=torch.bool)
    scorable = torch.ones(shape, dtype=torch.bool)
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


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------
RULES: dict[str, dict] = {
    "rule_is_a": {
        # containment, then reconstruction mass, then Tree SAE's refinement rank
        "target": ("is_a",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("PASSES", "gate_recon"),
                    ("PASSES", "gate_sres_rank")),
        "text": "PASSES(strictly_contains) AND PASSES(recon) AND PASSES(sres_rank)",
    },
    "rule_firing_only": {
        # the same containment and reconstruction mass, but the parent decoder does NOT rank
        # against the child's concept: co-firing without refinement
        "target": ("firing_only",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("PASSES", "gate_recon"),
                    ("FAILS", "gate_sres_rank")),
        "text": "PASSES(strictly_contains) AND PASSES(recon) AND FAILS(sres_rank)",
    },
    "rule_superparent": {
        # out-degree alone, as `metrics.outdegree.find_superparents` flags it
        "target": ("superparent",),
        "clauses": (("PASSES", "gate_high_outdegree"),),
        "text": "PASSES(high_outdegree)",
    },
    "rule_frequency": {
        # containment that does not survive removing the frequent tokens
        "target": ("frequency",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("FAILS", "gate_freq_survives")),
        "text": "PASSES(strictly_contains) AND FAILS(freq_survives)",
    },
    "rule_topical": {
        # containment that DOES survive removing the frequent tokens: the complement of
        # rule_frequency within strict containment. No gate models topical co-occurrence
        # directly, so this is the most the gates can say about it.
        "target": ("topical",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("PASSES", "gate_freq_survives")),
        "text": "PASSES(strictly_contains) AND PASSES(freq_survives)",
    },
    "rule_containment": {
        # baseline: any direct containment, is_a and firing_only together
        "target": ("is_a", "firing_only"),
        "clauses": (("PASSES", "gate_strictly_contains"),),
        "text": "PASSES(strictly_contains)",
    },
    "rule_is_a_no_recon": {
        # rule_is_a without the reconstruction clause
        "target": ("is_a",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("PASSES", "gate_sres_rank")),
        "text": "PASSES(strictly_contains) AND PASSES(sres_rank)",
    },
    "rule_firing_only_no_recon": {
        # rule_firing_only without the reconstruction clause
        "target": ("firing_only",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("FAILS", "gate_sres_rank")),
        "text": "PASSES(strictly_contains) AND FAILS(sres_rank)",
    },
}

for _name, _spec in RULES.items():
    for _pred, _gate in _spec["clauses"]:
        assert _pred in PREDICATES and _gate in GATE_NAMES, f"{_name}: {_pred}({_gate})"
