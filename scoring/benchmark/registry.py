"""The benchmark's registered metrics, gates, rules and toys, plus the probe-fitting seed.

Rules, bars and the report schema live in `metrics.rules` and are re-exported here.
"""

from __future__ import annotations

from metrics.rules import RULES
from metrics.rules.grading import (  # noqa: F401
    BAR_CONFOUND_LEAK,
    BAR_EVAL_NULL_FPR,
    BAR_RECALL,
    DESIGNATED,
    MIN_SCORABLE_SUPPORT,
    NULL_CLASS,
    PROBE_RULES,
    REPORT_SCHEMA,
)
from scoring.core.gates import GATE_NAMES
from scoring.core.registry import DETECTORS

# --- constants ---
# The three draws of one experiment seed, kept apart so no two share sampling noise
# (PRECOMMIT.md s6 step 2):
#   seed          matching (trained read only)
#   seed + 10000  scoring (scoring.core.grid.held_out_sample_seed; both reads)
#   seed + 20000  probe fitting (both reads)
PROBE_FIT_SEED_OFFSET = 20_000


def probe_fit_sample_seed(seed: int) -> int:
    """The probe-fitting draw for an experiment seed: `seed + 20000`."""
    return int(seed) + PROBE_FIT_SEED_OFFSET
CONSTANT_TOL = 1e-9          # finite-range tolerance for the constant-distribution flags

# --- metrics ---
# Benchmark-side names: `G` is decoder cosine (true g on the oracle read, learned W_dec on the
# trained read), `S_res` the Tree SAE probe, `wide` min(outdegree[p,c], outdegree[c,p]).
DERIVED_METRICS: tuple[str, ...] = ("G", "S_res", "wide", "abs_asymmetry_R")
METRICS: tuple[str, ...] = tuple(d for d in DETECTORS if d != "s_res") + DERIVED_METRICS

# `s_res` is left out: the internal key is cosine in one mode and the probe in the other, and
# PRECOMMIT.md s5 keeps them separately named as `G` and `S_res`.

# --- gates ---
# A name in both METRICS and GATES would let a clause read a score as a decision.
GATES: tuple[str, ...] = GATE_NAMES
assert not set(METRICS) & set(GATES), (
    f"a name is registered as both a metric and a gate: {sorted(set(METRICS) & set(GATES))}")

# --- rules ---
# The shared rules; PRECOMMIT.md s4 maps the old rule names to these.
EXPRESSIONS: dict[str, dict] = RULES

# Rules that read the probe: INVALID MEASUREMENT on a `--no-probe` run, not rejections.
PROBE_EXPRESSIONS: tuple[str, ...] = PROBE_RULES

TOYS: tuple[str, ...] = ("only_isa", "only_firing", "only_superparent",
                         "only_frequency", "only_topical")
