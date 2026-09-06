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

Metric names are the benchmark's own, not `compute_all`'s internal keys:
  `G`      decoder cosine -- `G_g` on the oracle read (true `g`), `G_W` on the trained read
           (learned `W_dec`). PRECOMMIT s5: cosine-mode `s_res` is exported as G and must NOT
           populate the probe comparator rows.
  `S_res`  the Tree-SAE linear probe (`probe_true_g` oracle, `probe_self_W` trained).
  `wide`   `min(outdegree[p,c], outdegree[c,p])`, the either-endpoint transformation.
  `abs_asymmetry_R`  `|asymmetry_R|`, registered separately so SYM's threshold is fitted on the
           absolute distribution rather than the signed one.
The remaining nine are the registry names from `scoring.core.registry.DETECTORS`.
"""

from __future__ import annotations

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
#
# This exists because `recall_given_recovery` KEPT ITS NAME and CHANGED ITS ARITHMETIC, and a
# schema-1 artifact is on disk. Pooling one of those under the new names is a silent denominator
# mix -- the exact class of error this package exists to prevent -- so the cross-world rollup and
# `verify_manifest` REQUIRE this value rather than defaulting it.
REPORT_SCHEMA = 2

Q_LO, Q_HI = 0.01, 0.99      # null quantiles, linear interpolation
TAU_SURV = 0.25              # token_freq_survival boundary; equality assigned to SURVIVES
MIN_CAL_SUPPORT = 200        # floor on FINITE calibration scores; below it, no threshold
# Floor on FINITE SCORABLE TARGET pairs before a recall may carry a verdict. Without one,
# MET CRITERIA is reachable off a single pair -- an adversarial review built a superparent
# target with N_scorable=1, N_pass=1 and got recall 1.000 and MET CRITERIA. PRECOMMIT s8
# lists 'minimum calibration and scorable support' as an OPEN setting, so this is a PROPOSAL
# awaiting approval. It matches the repo's existing N<10 reporting censor
# (scoring/oracle/score_dump.py:_MIN_TABLE_N) and changes no seed-0 verdict: the smallest
# scorable target there is 54.
MIN_SCORABLE_SUPPORT = 10
CAL_SPLIT_SEED = 0           # frozen partition seed for the calibration/evaluation null split

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
BAR_EVAL_NULL_FPR = 0.01     # evaluation-half null FPR in each tested world, <=
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
# the eight frozen expressions (PRECOMMIT.md s4)
# --------------------------------------------------------------------------
C: tuple[tuple[str, str], ...] = (
    ("HIGH", "coverage_R"), ("HIGH", "asymmetry_R"), ("HIGH", "pmi"),
)

EXPRESSIONS: dict[str, dict] = {
    "overlap_v6": {
        "target": ("is_a",),
        "clauses": C + (("HIGH", "G"),),
        "text": "C AND HIGH(G)",
    },
    "orthogonal_v6": {
        "target": ("firing_only",),
        "clauses": C + (("IN-BAND", "G"),),
        "text": "C AND IN-BAND(G)",
    },
    "superparent_v5": {
        "target": ("superparent",),
        "clauses": (("LOW", "wide"), ("NOT-HIGH", "pmi")),
        "text": "LOW(wide) AND NOT-HIGH(pmi)",
    },
    "frequency_v6": {
        "target": ("frequency",),
        "clauses": (("HIGH", "pmi"), ("FREQ-LOCAL", "token_freq_survival")),
        "text": "HIGH(pmi) AND FREQ-LOCAL(S)",
    },
    "topical_v6": {
        "target": ("topical",),
        "clauses": (("HIGH", "pmi"), ("SURVIVES", "token_freq_survival"),
                    ("SYM", "asymmetry_R")),
        "text": "HIGH(pmi) AND SURVIVES(S) AND SYM(asymmetry_R)",
    },
    "containment_baseline": {
        # Generated direct containment = is_a UNION firing_only. The two primary code labels
        # stay separate in every table; only this baseline's target row combines them.
        "target": ("is_a", "firing_only"),
        "clauses": C,
        "text": "C",
    },
    "probe_overlap_v3": {
        # NOT C: the historical probe comparator has no PMI clause. PRECOMMIT.md s4.
        "target": ("is_a",),
        "clauses": (("HIGH", "coverage_R"), ("HIGH", "asymmetry_R"), ("HIGH", "S_res")),
        "text": "HIGH(coverage_R) AND HIGH(asymmetry_R) AND HIGH(S_res)",
    },
    "probe_orthogonal_v3": {
        "target": ("firing_only",),
        "clauses": C + (("NOT-HIGH", "S_res"),),
        "text": "C AND NOT-HIGH(S_res)",
    },
}

# The five DESIGNATED rules: one per planted property. `containment_baseline` is excluded (it
# is a sub-expression of both G rules) and so are the two historical probe comparators (they
# overlap the G rules by construction). Only these five are counted for the multiple-rule /
# no-rule outcomes PRECOMMIT s6 requires -- including the others would manufacture ambiguity.
DESIGNATED: tuple[str, ...] = ("overlap_v6", "orthogonal_v6", "superparent_v5",
                               "frequency_v6", "topical_v6")

# Expressions that read the probe. A run without the probe marks these INVALID MEASUREMENT
# rather than letting an all-NaN column read as a rejection, and the artifact records
# `s_res_mode="absent"`. The run is still written -- it is a labelled partial run, not a
# refused one.
PROBE_EXPRESSIONS: tuple[str, ...] = ("probe_overlap_v3", "probe_orthogonal_v3")

TOYS: tuple[str, ...] = ("only_isa", "only_firing", "only_superparent",
                         "only_frequency", "only_topical")
