"""
The shared decision layer: gates, rules and constant sets.

`metrics/` computes statistics; this package decides on them. Both pipelines import it, so a
rule means the same thing everywhere; each pipeline picks a named constant set and stamps it,
with `RULESET_VERSION`, on every result.

Not listed in `metrics.__all__`, which is the set of metric functions the Tier-1 calibration
must exercise.
"""

from .constants import CONSTANT_SETS, GEMMA_MATRYOSHKA, SYNTHETIC_TOYS, GateConstants
from .gates import (
    GATE_NAMES,
    coverage_gates,
    either_endpoint,
    freq_survives_gate,
    high_outdegree,
    nan_self_pairs,
    recon_gate,
    squash,
    sres_rank_gate,
    support_gate,
    supported,
    tristate,
)
from .rules import (
    GATE_TRUE,
    PREDICATES,
    RULES,
    RULESET_VERSION,
    evaluate,
    fails,
    passes,
)

__all__ = [
    "CONSTANT_SETS",
    "GATE_NAMES",
    "GATE_TRUE",
    "GEMMA_MATRYOSHKA",
    "PREDICATES",
    "RULES",
    "RULESET_VERSION",
    "SYNTHETIC_TOYS",
    "GateConstants",
    "coverage_gates",
    "either_endpoint",
    "evaluate",
    "fails",
    "freq_survives_gate",
    "high_outdegree",
    "nan_self_pairs",
    "passes",
    "recon_gate",
    "squash",
    "sres_rank_gate",
    "support_gate",
    "supported",
    "tristate",
]
