"""The FROZEN registry: the eight expressions and the constants they are calibrated with.

Every value here is a transcription of `PRECOMMIT.md`, not a choice made in code. Changing one
after the freeze changes what the benchmark measured, so `tests_local/test_benchmark_evaluate.py`
pins the whole table against hardcoded literals transcribed from the document. That catches an
edit to THIS file; it does not catch an edit to `PRECOMMIT.md`, which is why the manifest also
records a hash of the document alongside the registry as evaluated.

An expression is an ordered tuple of `(PREDICATE, metric)` clauses, conjoined. Keeping the
clauses separate rather than baking them into one boolean is what lets `predicates.evaluate`
build a CLAUSE-SPECIFIC scorable mask: a pair is scorable for an expression only when every
score and threshold that expression actually reads is available. The pilot evaluator gated
every rule on one metric's finiteness, which is why unscorable pairs were reported as
rejections (`PRECOMMIT.md` s7).

METRICS AND GATES ARE DIFFERENT THINGS AND ARE KEPT APART. A metric is a continuous score; a
gate is a fixed-threshold DECISION about a pair, tristate over {1.0, 0.0, NaN}. Expressions read
GATES only -- `predicates.evaluate` refuses a clause naming a metric, because applying `PASSES`
to a coverage would compare it against 0.5 and produce a plausible number answering no question.
Metrics are still computed, persisted at full precision and reported in `metric_diagnostics`;
they are simply no longer what a rule decides on.

Metric names are the benchmark's own, not `compute_all`'s internal keys:
  `G`      decoder cosine -- `G_g` on the oracle read (true `g`), `G_W` on the trained read
           (learned `W_dec`). PRECOMMIT s5: cosine-mode `s_res` is exported as G and must NOT
           populate the probe comparator rows.
  `S_res`  the Tree-SAE linear probe (`probe_true_g` oracle, `probe_self_W` trained).
  `wide`   `min(outdegree[p,c], outdegree[c,p])`, the either-endpoint transformation.
  `abs_asymmetry_R`  `|asymmetry_R|`, registered separately. It was added so SYM's threshold
           could be fitted on the absolute distribution; SYM is gone, and it is kept as a
           reported metric because the artifacts carry it.
The rest are the registry names from `scoring.core.registry.DETECTORS`.
"""

from __future__ import annotations

from metrics.rules import RULES
from scoring.core.gates import GATE_NAMES
from scoring.core.registry import DETECTORS

# --------------------------------------------------------------------------
# frozen constants (PRECOMMIT.md s6, s8)
# --------------------------------------------------------------------------
# Version of the REPORTING CONTRACT: the names and arithmetic of the rate keys written into
# `expressions.json`. Bump it whenever a key's arithmetic changes under an unchanged name, and
# whenever a rate key is added, renamed or removed.
#
#   1  the pilot / SEED0-PREFREEZE contract. `recall_given_recovery` = N_pass / N_scorable, one
#      `fpr` key on the null rows, `leakage` reading `recall_given_recovery`.
#   2  B2.3. `recall_given_recovery` = N_pass / N_recovered (the bar), with the scorable rate
#      split out as `pass_rate_given_scorable`; `fpr` split into `fpr_given_scorable` (the bar)
#      and `fpr_over_half`; `leakage` explicitly on the scorable rate with
#      `leakage_over_recovered` beside it; the support floor extended to the evaluation-null and
#      confound rows; established failures ordered ahead of unmeasurable evidence in `verdict`.
#   3  the fixed-gate contract. Every rule decides on a GATE against a fixed constant rather
#      than a null quantile, so the null is no longer halved and `fpr_given_scorable` is
#      measured over the WHOLE null instead of an evaluation half; `fpr_over_half` is gone.
#      `pmi`, `coverage_R` and `asymmetry_R` additionally carry the scorability mask. The key
#      names did not change and the arithmetic under them did, which is what this number exists
#      to make visible.
#   4  the shared-rules contract (2026-09-23). Every rule and three gates were RENAMED
#      (PRECOMMIT.md s4 name map) when they moved to `metrics.rules`; the arithmetic is
#      unchanged. A schema-3 artifact keys its rules as `overlap_v6` etc., and a rollup reading
#      it under the new names would find nothing and report every rule UNTESTABLE instead of
#      failing, so schema-3 trees are refused rather than read.
#
# This exists because `recall_given_recovery` KEPT ITS NAME and CHANGED ITS ARITHMETIC, and a
# schema-1 artifact is on disk. Pooling one of those under the new names is a silent denominator
# mix -- the exact class of error this package exists to prevent -- so the cross-world rollup and
# `verify_manifest` REQUIRE this value rather than defaulting it.
REPORT_SCHEMA = 4

# `Q_LO`/`Q_HI`, `TAU_SURV`, `MIN_CAL_SUPPORT` and `CAL_SPLIT_SEED` are GONE. Nothing is fitted
# any more: every rule compares against a constant in `scoring.core.registry.CONSTANTS`, so
# there is no calibration half, no quantile and no fitted boundary. `TAU_SURV = 0.25` in
# particular is replaced by `gates.squash(CONSTANTS["freq_survival_min_raw"])` = 0.333, which is
# `config.FREQ_SURVIVAL_MIN` put through the transform the detector reports its ratio under.
#
# The null is no longer halved either. The split existed so a threshold fitted on one half
# could be measured on the other; with nothing fitted the calibration half has no job, and the
# false-positive rate is measured on the WHOLE null. Do not re-introduce a split "for holdout" --
# there is nothing left to leak.

# Floor on FINITE SCORABLE TARGET pairs before a recall may carry a verdict. Without one,
# MET CRITERIA is reachable off a single pair -- an adversarial review built a superparent
# target with N_scorable=1, N_pass=1 and got recall 1.000 and MET CRITERIA. PRECOMMIT s8
# lists 'minimum calibration and scorable support' as an OPEN setting, so this is a PROPOSAL
# awaiting approval. It matches the repo's existing N<10 reporting censor
# (scoring/oracle/score_dump.py:_MIN_TABLE_N) and changes no seed-0 verdict: the smallest
# scorable target there is 54.
MIN_SCORABLE_SUPPORT = 10

# The three draws of one experiment seed, kept distinct so no two share sampling noise:
#   seed          the MATCHING draw (trained read only: the Hungarian matcher runs here)
#   seed + 10000  the SCORING draw  (scoring.core.grid.held_out_sample_seed; both reads)
#   seed + 20000  the PROBE FITTING draw (below; both reads)
# Mirrors `held_out_sample_seed`'s form deliberately. PRECOMMIT s6 step 2 specifies the fitting
# separation; see scoring/core/detectors.py:s_res_probe for what it does and does not remove.
PROBE_FIT_SEED_OFFSET = 20_000


def probe_fit_sample_seed(seed: int) -> int:
    """The probe FITTING draw for an experiment seed. Distinct from the matching and scoring
    draws at every seed, which `test_benchmark_probe_draw.py` asserts rather than assumes."""
    return int(seed) + PROBE_FIT_SEED_OFFSET
CONSTANT_TOL = 1e-9          # finite-range tolerance for the constant-distribution flags
NULL_CLASS = "unrelated"

# Operating criteria (PRECOMMIT.md s8). Proposed bars, not mathematical guarantees; they are
# not loosened because a rule fails.
BAR_RECALL = 0.80            # recall given recovery, >=
BAR_EVAL_NULL_FPR = 0.01     # null FPR in each tested world, <=  (whole null as of schema 3)
BAR_CONFOUND_LEAK = 0.05     # each named complete-expression confound rate, <=

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
DERIVED_METRICS: tuple[str, ...] = ("G", "S_res", "wide", "abs_asymmetry_R")
# the nine registry detectors that are NOT s_res, plus the four benchmark-side metrics
METRICS: tuple[str, ...] = tuple(d for d in DETECTORS if d != "s_res") + DERIVED_METRICS

# `s_res` is deliberately absent: the internal key means cosine in one mode and the probe in
# the other, and PRECOMMIT s5 requires those to stay separately named (`G` vs `S_res`).

# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
# The fixed-threshold decisions the expressions are built from. Sourced from
# `scoring.core.gates` rather than re-listed, so a gate cannot exist in one place and not the
# other. Disjointness from METRICS is asserted rather than assumed: a name in both would let a
# clause read a score where it meant to read a decision, and every value involved is a float.
GATES: tuple[str, ...] = GATE_NAMES
assert not set(METRICS) & set(GATES), (
    f"a name is registered as both a metric and a gate: {sorted(set(METRICS) & set(GATES))}")

# --------------------------------------------------------------------------
# the eight expressions, rebuilt on gates
# --------------------------------------------------------------------------
# WHAT CHANGED AND WHY, because the targets are unchanged and the clauses are not.
#
# `C` is gone. It was `HIGH(coverage_R) AND HIGH(asymmetry_R) AND HIGH(pmi)`, and the first two
# of those are exactly what `metrics.in_block.directed_coverage` decides at a fixed tau:
# containment one way (`R >= tau`) and NOT the other (`R_reverse < tau`), on a supported pair.
# That is `gate_strictly_contains`, one clause in place of two.
#
# PMI HAS NO GATE, so the PMI clause has no successor and is simply gone from every rule.
# `metrics/` computes PMI (`independence_null.py`) but thresholds it nowhere, and `config.py`
# carries no PMI constant. It is still computed, persisted and reported as a metric; it no
# longer decides anything. The clause rejected frequent-parent base-rate coverage, and what
# now does that work is the support guard plus the fixed tau -- less of it, and the leak rates
# say how much less.
#
# THE GEOMETRY CHANNEL IS NOW THE PROBE ONLY. The quantile rules had two: `G` (decoder cosine)
# and `S_res` (the Tree-SAE probe). No fixed cosine threshold exists anywhere -- not in
# `config.py`, not in `metrics/`, and `scoring/trained/absorption.py`'s `null_cos_threshold` is
# itself a permutation quantile -- so the G clauses have no fixed-threshold form. All four
# geometry rules therefore read `gate_sres_rank`, Tree SAE's top-k rank rule, which carries no
# numeric threshold at all. The consequence is that four rules now need the probe instead of
# two; `PROBE_EXPRESSIONS` records it, so a `--no-probe` run marks them INVALID MEASUREMENT
# rather than reading an absent probe as a rejection.
#
# The rules themselves live in `metrics.rules.RULES`, shared with the Gemma pipeline, and are
# named after the class they target with no version suffix. Until 2026-09-23 they were defined
# here as overlap_v6, orthogonal_v6, superparent_v5, frequency_v6, topical_v6,
# containment_baseline, probe_overlap_v3 and probe_orthogonal_v3; PRECOMMIT.md s4 maps the old
# names to the new. A change to a rule now bumps `metrics.rules.RULESET_VERSION` instead.
EXPRESSIONS: dict[str, dict] = RULES

# The five DESIGNATED rules: one per planted property. `rule_containment` is excluded (it
# is a sub-expression of both G rules) and so are the two historical probe comparators (they
# overlap the G rules by construction). Only these five are counted for the multiple-rule /
# no-rule outcomes PRECOMMIT s6 requires -- including the others would manufacture ambiguity.
DESIGNATED: tuple[str, ...] = ("rule_is_a", "rule_firing_only", "rule_superparent",
                               "rule_frequency", "rule_topical")

# Expressions that read the probe -- FOUR of them since the gate rebuild, not two, because the
# geometry channel is now `gate_sres_rank` alone (see the note above EXPRESSIONS). A run
# without the probe marks these INVALID MEASUREMENT rather than letting an all-NaN column read
# as a rejection, and the artifact records `s_res_mode="absent"`. The run is still written --
# it is a labelled partial run, not a refused one. With four of the five DESIGNATED rules
# reading the probe, a `--no-probe` run now measures very little.
PROBE_EXPRESSIONS: tuple[str, ...] = ("rule_is_a", "rule_firing_only",
                                      "rule_is_a_no_recon", "rule_firing_only_no_recon")

TOYS: tuple[str, ...] = ("only_isa", "only_firing", "only_superparent",
                         "only_frequency", "only_topical")
