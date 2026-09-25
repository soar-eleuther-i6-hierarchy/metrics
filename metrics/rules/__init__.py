"""
The shared decision layer: gates, rules, grading and constant sets.

`score_pairs` turns one frame's statistics into gates and rule decisions; `grade_rules` grades
those decisions against labelled pairs. Kept out of `metrics.__all__`, which lists only the
metric functions the Tier-1 calibration must exercise.
"""

from .classes import LABELS, NULL_CLASS
from .constants import (
    CONSTANT_SETS,
    GEMMA_MATRYOSHKA,
    METRIC_SETTINGS,
    SYNTHETIC_TOYS,
    GateConstants,
    MetricSettings,
)
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
from .grading import (
    BAR_CONFOUND_LEAK,
    BAR_EVAL_NULL_FPR,
    BAR_RECALL,
    DESIGNATED,
    MIN_SCORABLE_SUPPORT,
    PROBE_RULES,
    REPORT_SCHEMA,
    VERDICTS,
    grade_rules,
)
from .pairs import PairStats, score_pairs
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
    "BAR_CONFOUND_LEAK",
    "BAR_EVAL_NULL_FPR",
    "BAR_RECALL",
    "CONSTANT_SETS",
    "DESIGNATED",
    "GATE_NAMES",
    "GATE_TRUE",
    "GEMMA_MATRYOSHKA",
    "LABELS",
    "METRIC_SETTINGS",
    "MIN_SCORABLE_SUPPORT",
    "NULL_CLASS",
    "PREDICATES",
    "PROBE_RULES",
    "REPORT_SCHEMA",
    "RULES",
    "RULESET_VERSION",
    "SYNTHETIC_TOYS",
    "VERDICTS",
    "GateConstants",
    "MetricSettings",
    "PairStats",
    "coverage_gates",
    "either_endpoint",
    "evaluate",
    "fails",
    "freq_survives_gate",
    "grade_rules",
    "high_outdegree",
    "nan_self_pairs",
    "passes",
    "recon_gate",
    "score_pairs",
    "squash",
    "sres_rank_gate",
    "support_gate",
    "supported",
    "tristate",
]
