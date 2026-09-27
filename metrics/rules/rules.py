"""
Rules: named conjunctions of gates, and the evaluator that applies them.

A rule is `{target, clauses, text}`: the pair class it targets and its `(PREDICATE, gate)`
clauses. Every result carries `RULESET_VERSION`; bump it when a rule, a gate or a constant
changes. PRECOMMIT.md s4.
"""

from __future__ import annotations

import torch

from .gates import GATE_NAMES

RULESET_VERSION = 2

# --- predicates ---
# Midway between 0.0 and 1.0, so `>` versus `>=` cannot matter.
GATE_TRUE = 0.5


def _finite(v: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(v)


def passes(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(mask, scorable)`: the gate holds. NaN is not a pass."""
    s = _finite(v)
    return (s & (v > GATE_TRUE), s)


def fails(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(mask, scorable)`: the gate does not hold. Not `~passes`, which passes on NaN: NaN must
    fail both."""
    s = _finite(v)
    return (s & (v <= GATE_TRUE), s)


PREDICATES: dict[str, dict] = {
    "PASSES": {"fn": passes},
    "FAILS": {"fn": fails},
}


def evaluate(clauses, vals: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """`(mask, scorable, per_clause)` for a conjunction of `(PREDICATE, gate)` clauses.

    `scorable` is the intersection over clauses, so a pair with no probe is unscorable, not
    rejected, under a rule reading `gate_sres_rank`. Unknown names raise: a typo would loosen
    the rule silently.
    """
    if not clauses:
        raise ValueError("an expression needs at least one clause")
    first = next(iter(vals.values()))
    mask = torch.ones(first.shape, dtype=torch.bool, device=first.device)
    scorable = torch.ones(first.shape, dtype=torch.bool, device=first.device)
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
    # already a subset of scorable; kept as an explicit invariant
    mask = mask & scorable
    return mask, scorable, per


# --- the rules ---
RULES: dict[str, dict] = {
    "rule_hierarchy": {
        # containment, co-firing above independence, coverage that holds without frequent
        # tokens, then Tree SAE's refinement rank
        "target": ("hierarchy_overlap", "hierarchy_orthogonal"),
        "clauses": (("PASSES", "gate_strictly_contains"), ("PASSES", "gate_pmi_positive"),
                    ("PASSES", "gate_freq_survives"), ("PASSES", "gate_sres_rank")),
        "text": ("PASSES(strictly_contains) AND PASSES(pmi_positive) AND PASSES(freq_survives) "
                 "AND PASSES(sres_rank)"),
    },
    "rule_dense_endpoint": {
        # out-degree alone, as `metrics.outdegree.find_superparents` flags it
        "target": ("dense_lookalike",),
        "clauses": (("PASSES", "gate_high_outdegree"),),
        "text": "PASSES(high_outdegree)",
    },
    "rule_frequency_driven": {
        # containment that does not survive removing the frequent tokens
        "target": ("frequency_lookalike",),
        "clauses": (("PASSES", "gate_strictly_contains"), ("FAILS", "gate_freq_survives")),
        "text": "PASSES(strictly_contains) AND FAILS(freq_survives)",
    },
    "rule_containment": {
        # baseline: any direct containment
        "target": ("hierarchy_overlap", "hierarchy_orthogonal"),
        "clauses": (("PASSES", "gate_strictly_contains"),),
        "text": "PASSES(strictly_contains)",
    },
}

for _name, _spec in RULES.items():
    for _pred, _gate in _spec["clauses"]:
        assert _pred in PREDICATES and _gate in GATE_NAMES, f"{_name}: {_pred}({_gate})"
