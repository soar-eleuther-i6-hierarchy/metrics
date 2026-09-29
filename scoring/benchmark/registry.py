"""The registered metric columns: the detectors, the decoder metrics and the derived metrics.

Gates, rules, bars and the report schema live in `metrics.rules`.
"""

from __future__ import annotations

from metrics.rules import GATE_NAMES
from scoring.core.registry import DETECTORS

# Read off the decoder rows rather than the firing: `G` is the decoder cosine, a trial metric
# defined in `scoring/core/detectors.py`; `S_res` is Tree SAE's probe score, from `metrics/sres.py`.
DECODER_METRICS: tuple[str, ...] = ("G", "S_res")
# Computed from other metrics: `wide` is min(outdegree[p,c], outdegree[c,p]), via
# `metrics.outdegree.either_endpoint_outdegree`, and `abs_asymmetry_R` is |asymmetry_R|.
DERIVED_METRICS: tuple[str, ...] = ("wide", "abs_asymmetry_R")
METRICS: tuple[str, ...] = DETECTORS + DECODER_METRICS + DERIVED_METRICS

# A name that is both a metric and a gate would let a clause read a score as a decision.
assert not set(METRICS) & set(GATE_NAMES), (
    f"a name is registered as both a metric and a gate: {sorted(set(METRICS) & set(GATE_NAMES))}")
