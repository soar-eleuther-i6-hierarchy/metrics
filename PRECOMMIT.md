# Precommit: fixed-gate rules on the toy benchmark

Status: DRAFT, not frozen.
Every rule decides on gates compared against fixed constants, defined once in `metrics/rules/` and shared with the Gemma pipeline, so the same rule means the same thing on a toy and on Gemma.
Section 3 keeps only the measured results that constrain how the rules may be read; the earlier pilot record, measured under the null-quantile rules, is in git history.

## 1. Status and remaining steps

The rules, gate constants and pass criteria in Sections 4, 8 and 9 were fixed before the three-seed damage matrix, and the matrix met every declared criterion (Section 3).
Trained-SAE reads of the toys are paused; the steps and settings below that mention an SAE, recovery or matching apply when they resume.

Remaining, in order:

1. Approve the registry in Section 4 and fill the open settings in Section 8.
   No rule needs to pass to be included.
2. Freeze this document, the registry, the evaluator revision and the seed plan together.
3. Run one independent confirmation seed, report every registered outcome, then the remaining fresh seeds.
4. Evaluate the complex toy and then Gemma under the scope in Section 10.

A valid negative result does not block freezing; an invalid measurement or an unresolved setting does.
Do not search for a subtype separator in this study.

## 2. Goal and target meanings

Establish which existing metrics and fixed Boolean expressions respond to which planted properties, where they fail, and how that behavior changes across seeds, SAE training, and more complex worlds.
Deliver a metric-by-property characterization and a complete-expression recall/leakage table with support, provenance, and uncertainty limits.
The benchmark succeeds when those results are trustworthy, not when every expression retrieves its target.

Keep three notions separate:

| Relationship | Meaning | Example |
| --- | --- | --- |
| Semantic is-a | Category inclusion by meaning | Every dog is an animal |
| Activation containment | Child firing implies parent firing on the observed distribution | All dog images in a dataset happen to have red backgrounds |
| Decoder overlap | Component directions have positive cosine | Two decoder vectors share a direction |

Containment can arise without semantic is-a, and a genuine hierarchy can use an orthogonal child-specific component.
Neither containment nor positive decoder overlap establishes semantic hierarchy alone.

The generator's labels and what each targets:

| Code label | Benchmark target |
| --- | --- |
| `is_a` | Generated direct containment edge with planted overlap, effective alpha > 0 |
| `firing_only` | Generated direct containment edge with orthogonal directions, effective alpha = 0 |
| `superparent` | Either endpoint is a declared high-base-rate feature; firing probability is 0.85, not 1 |
| `frequency` | Both features share the planted frequent-token set |
| `topical` | Both features share the planted document topic |

Do not merge is_a and firing_only into a semantic "true hierarchy" label; their union is the target of the containment baseline only.
They are mutual hard negatives for the planted-overlap distinction, and the generator supplies structural labels, not a validated semantic taxonomy.

Stop the single-property stage when every registered metric/expression has a valid result or explicit untestable status for every required seed/class and result-changing audit issues are resolved.
Do not redesign S_res, change the generator, or search for a better classifier merely because the current expressions fail.

## 3. Measured constraints

Unless stated otherwise these are at the oracle ceiling: a perfect dictionary, true activations, no damage, full-size toys, 50k tokens.
A rule that does not fire there cannot fire anywhere.

### Ceiling recall, seed 0, after the Section 4 changes

| Rule | Recall | Reading |
| --- | --- | --- |
| rule_is_a | 1.00 | all three clauses pass |
| rule_firing_only | 0.00 in only_firing | the geometry gate passes every firing_only pair there |
| rule_superparent | 1.00 | null rate 0.00 |
| rule_frequency | 0.25 | 100% of the ordered container-to-member pairs |
| rule_topical | 0.33 | 100% of the ordered register-to-member pairs |
| rule_containment | 1.00 | |

### The geometry channel does not separate is_a from firing_only

`gate_sres_rank` passes 100% of is_a and 100% of firing_only pairs at k = 5 and at k = 2, the strictest non-trivial value; k = 2 only tightened the null from 1.3% to at most 0.1%.
The probe is fitted on the child's firing and the parent co-fires on all of it, so the parent's decoder is the child probe's rank-2 correlate whether or not the two directions are orthogonal.
No constant inside the current rules separates the classes.
The S_res values overlap (is_a p1 0.328 is below firing_only p99 0.372), and the same class reads 0.315 in only_firing against 0.042 in only_superparent, an eightfold shift driven by the parent's firing density rather than by geometry.
The gate responds to density as well: at k = 5 it passed 100% of firing_only edges in only_firing but 58% in only_superparent, where the parent is dense.
The decoder cosine `G` separates the classes cleanly and identically in every world (median 0.480 for is_a, 0.000 for firing_only); it is reported and read by no rule (Section 4).

### Frequency and topical are capped at 0.25 and 0.33

Both classes are labelled on both orderings of a pair while `gate_strictly_contains` is directional.
The rules reach 100% of the ordered pairs the toys plant, which are 25% of the frequency class and 33% of the topical class.
The caps are accepted and the populations stay symmetric (decided 2026-09-19); Section 8 records that the recall bar does not apply to those two rows.

### The damage matrix, seeds 1 to 3, measured 2026-09-20

333 runs, 111 dial points on each seed at 50k tokens, under the criteria declared in Section 9 before any of these numbers existed.
All three seeds met every declared criterion, 15 of 15 each.
Seed 0 is a regression check and not evidence, because it selected `sres_rank_top_k` and the `rule_topical` clauses.

Replication is near-exact: every rule reproduces to three decimals on all three seeds, except `rule_firing_only` on `only_superparent` at 0.833, 0.917 and 0.750 (10, 11 and 9 of 12 pairs).
Every null false-positive rate is 0.0000, which is flattering rather than informative: unrelated pairs sit at coverage 0.11 to 0.18 against a cut of 0.5.
The informative negatives are the other structured classes, and against those the set largely fails:

| Rule | On target | Worst off-target | On which class |
| --- | --- | --- | --- |
| `rule_frequency` | 0.250 | **0.000** | nothing; clean |
| `rule_firing_only` | 0.833 | 0.468 | `superparent` |
| `rule_is_a` | 1.000 | **1.000** | `firing_only` |
| `rule_superparent` | 1.000 | **1.000** | `firing_only` and `reversed` |
| `rule_topical` | 0.333 | **1.000** | `is_a` and `firing_only` |

- `rule_topical` contains nothing topic-specific, so every true containment edge passes it; its recall is not a measurement of topical detection.
- `rule_superparent` passes `reversed` at 1.000, because its gate reads an endpoint's out-degree, which is symmetric in the pair.
- `rule_frequency` is the only rule with clean specificity.
- Each discriminator partitions containment exhaustively, so no rule in either pair can abstain; that is the direct cause of the leakage.

What the rules actually read:

- The encoder costs nothing: the largest recall shift between the oracle column and the undamaged NNLS column is 0.0000 everywhere.
  Of the eight registered metrics, six are bit-identical between the two (`coverage_R`, `asymmetry_R`, `pmi`, `token_freq_survival`, `S_res`, `G`), because they depend only on the firing pattern and the decoder directions, which NNLS on the planted support leaves unchanged.
  Only `recon_2a` and `recon_child_gain` move, and `gate_recon` shows zero flips.
  The rule set is therefore almost entirely a firing-pattern test, which will not transfer to a trained SAE, where the encoder sets the support.
- NNLS reconstructs better than the true coefficients in every toy and seed, so reconstruction quality is not evidence of correct recovery.
- One clause carries every damage result.
  On `only_isa` under absorption, at all twelve dial points, `PASSES(strictly_contains)` alone, `strictly_contains AND sres_rank` and `strictly_contains AND recon AND sres_rank` return the same number.
  Across the whole matrix `gate_recon` changes an answer in 25 of about 950 comparisons, all in one cell (`only_superparent`, composition, `union` readout), where its own question is ill-posed because the readout merged two features into a shared latent.
  `gate_mutually_contains` is read by no rule.
- Absorption and splitting act only through coverage, and the rules switch rather than degrade.
  Coverage falls to about 1 minus eta and crosses the cut between eta 0.3 and 0.6.
  A feature read on one of k shards has coverage exactly 1/k, so the rule dies above k = 2.
  The `union` readout restores it, which is an oracle repair: the readout is handed the true shard grouping.
- `gate_sres_rank` is unmoved by every damage (1.000 at all twelve absorption points in only_firing, 0.056 in only_superparent).
  `G` tracks the geometry half of absorption instead: 0.480, 0.640, 0.745 and 0.814 at beta 0, 0.25, 0.5 and 0.75, blind to eta and to k.

Caveats: only_frequency and only_topical reach only 24 pooled corrupted pairs, and their transition cells are the least stable (`rule_frequency` at beta 0.75, eta 0 reads 0.12, 0.38 and 0.75 across seeds).
`transitive` and `sibling` are not planted in any pure toy.
The matrix is indexed by (toy, class): absorption damages only only_superparent's firing_only edges, and splitting as configured only its superparent pairs.
Leakage under damage is not measured; the damaged cells report target pass rates only.

## 4. Registry: the committed expressions

Every rule decides on gates, never on a metric.
The rules are defined once in `metrics/rules/rules.py` and graded by `metrics/rules/grading.py`; both are shared with the Gemma pipeline, and `scoring/` imports them.
They replaced null-quantile predicates, which could not travel to a real SAE: a q99 cut accepts 1% of whatever population it is pointed at, so the false-positive bar could not fail.

| ID | Target | Exact expression |
| --- | --- | --- |
| rule_is_a | is_a | `PASSES(strictly_contains) AND PASSES(recon) AND PASSES(sres_rank)` |
| rule_firing_only | firing_only | `PASSES(strictly_contains) AND PASSES(recon) AND FAILS(sres_rank)` |
| rule_superparent | superparent | `PASSES(high_outdegree)` |
| rule_frequency | frequency | `PASSES(strictly_contains) AND FAILS(freq_survives)` |
| rule_topical | topical | `PASSES(strictly_contains) AND PASSES(freq_survives)` |
| rule_containment | Generated direct containment: is_a union firing_only | `PASSES(strictly_contains)` |
| rule_is_a_no_recon | is_a | `PASSES(strictly_contains) AND PASSES(sres_rank)` |
| rule_firing_only_no_recon | firing_only | `PASSES(strictly_contains) AND FAILS(sres_rank)` |

The first five are designated; the containment baseline and the two no-recon comparators are reported beside them.
`FAILS` is written literally, never as `NOT PASSES`: a gate that was never measurable satisfies neither predicate.
The four rules reading `gate_sres_rank` are INVALID MEASUREMENT on a run without the probe, not rejections.

Deliberately absent:

- `G` is in no rule (decided 2026-09-19), so the rule set matches the one the team uses; it is computed and reported as a diagnostic in every artifact.
  Consequence, registered rather than hidden: the geometry rules rest on `gate_sres_rank` and cannot separate is_a from firing_only (Section 3).
  A fixed-cut G gate is future work outside this freeze.
- PMI decides nothing: no fixed constant for it exists in `config.py` or `metrics/`.
  The support guard and the fixed tau do less of its work, and the leak rates say how much less.

Names carry no version.
A change to any rule, gate or constant bumps `metrics.rules.RULESET_VERSION` (now 1), which every artifact records beside `gate_constant_set`.
Rules and gates were renamed on 2026-09-23 without changing any decision, and the stored gate-era results (`MATRIX`, `MATRIX-P3`) were renamed with them; the old names are in git history and in `outputs_archive/gate_era_original_names.tar` on soar-gpu.
Schema-2 results predate the gate rules and keep their own names.
`gate_contains`, one-way containment and the Gemma pipeline's cross-block edge, was added at the same time; no rule reads it yet.

### Constant and clause decisions, 2026-09-19

Eyeballed from the oracle ceiling, not swept; they changed a constant and a clause, never a metric definition.

1. `sres_rank_top_k` is 2, the strictest non-trivial value: the child's own decoder is essentially always rank 1, so k = 5 admitted any parent in ranks 2 to 5.
   It did not buy the separation (Section 3) but is strictly tighter on the null at identical recall.
   `config.SRES_RANK_TOP_K` and Tree SAE use 5; this is the recorded departure.
2. `rule_topical` reads `PASSES(strictly_contains) AND PASSES(freq_survives)`.
   The earlier `gate_mutually_contains` form read 0 by construction: the toy plants a register at 0.9 and members at 0.32, and any tau below 0.32 would also make every is_a pair mutual.
3. `recon_rel_gain_min` stays at 0.01 and is marked inert: at the ceiling it passes about 100% of targets and nulls alike.
   Its job is rejecting pairs where an endpoint carries no reconstruction mass, a damaged or trained-read condition; if it stays inert under damage, drop the clause rather than tune it.
4. Unchanged because the ceiling says they work: `edge_tau` 0.5, `superparent_outdeg_frac` 0.30, `freq_survival_min` 0.5 (on the raw ratio), `min_fire_count` 20, `min_joint` 30.

Known failures may replicate; there is no requirement to find a winning expression for every property.
Do not add new metrics, probe objectives or learned decision trees to rescue a failing rule.

## 5. Instrument definitions

### Metrics

Sixteen registered per-ordered-pair scalars: twelve detectors, two decoder metrics (`S_res` and `G`), and two derived ones.
Every formula except `G` is defined once, in `metrics/`, the formula library the Gemma pipeline also uses.
`scoring/core/detectors.py` computes the counts those functions take from token activations and places the results in the square frame: NaN self-pairs, the support masks and the frozen signs.
Undefined cells are NaN: scoring passes `undefined=NaN` where the Gemma pipeline's default is 0.
`G`, the decoder cosine, is a trial metric defined in `scoring/core/detectors.py`: it moves to `metrics/` if it is kept, and is deleted otherwise.
`S_res` is always the Tree SAE probe and `G` always the decoder cosine; they are separately named and never interchangeable.

| Metric | Definition | Formula in `metrics/` | Read by |
| --- | --- | --- | --- |
| `coverage_R` | R(p,c) = P(parent fires given child fires) | `coverage.coverage_legs` | gate_strictly_contains, gate_mutually_contains |
| `asymmetry_R` | R(p,c) - R(c,p) | `coverage.coverage_asymmetry` | gate_strictly_contains, gate_mutually_contains |
| `recon_2a` | relative reconstruction gain from the parent on the child's tokens | `reconstruction.edge_reconstruction_condition` | gate_recon |
| `recon_child_gain` | the same quantity for the child on its own tokens | `reconstruction.edge_reconstruction_condition` | gate_recon |
| `S_res` | Tree SAE probe score, min over endpoints of decoder-probe alignment | `sres.sres_scores` | gate_sres_rank |
| `outdegree` | kept-children count per parent, sign-flipped | `outdegree.kept_outdegree` | gate_high_outdegree |
| `wide` | the larger out-degree of the two endpoints, sign-flipped | `outdegree.either_endpoint_outdegree` | gate_high_outdegree |
| `token_freq_survival` | r/(1+r) with r the restricted-corpus coverage ratio | `token_control.frequency_controlled_coverage`, unclamped and floored on total firing | gate_freq_survives |
| `pmi` | log(P(p,c) / (P(p) P(c))), unsmoothed | `independence_null.independence_scores` | none |
| `joint_child_J` | min(1, sum of forward coverage over kept children) | `coverage.joint_child_coverage_upper` | none |
| `joint_child_supp` | share of the parent's firing tokens where a kept child fires | `joint_child.r_supp` | none |
| `joint_child_mass` | the same share of the parent's activation energy | `joint_child.r_mass` | none |
| `sibling_redundancy` | mean pairwise Jaccard of the kept children | `sibling_redundancy.sibling_redundancy` | none |
| `sibling_redundancy_pc` | the same Jaccard within the parent's firing tokens | `sibling_redundancy.parent_conditioned_redundancy` | none |
| `abs_asymmetry_R` | the absolute value of `asymmetry_R` | `coverage.coverage_asymmetry` | none |
| `G` | cos(d_p, d_c) between unit decoders; trial metric | `scoring/core/detectors.py` | none |

An artifact whose `detector_constants` still lists `pmi_laplace` carries the earlier smoothed `pmi`, log((cofire+1)N / ((fire_p+1)(fire_c+1))).
On the pairs the support mask keeps it sat at most 0.03 nats above the current value (100 re-scored points, 2026-09-25); on rare pairs that never co-fire it is large and positive, which the mask hides.
No rule reads it.

Eight of the sixteen decide nothing; they are persisted at full precision and reported in `metric_diagnostics`, so a reader can tell a diagnostic from a number a verdict rests on.

### Gates

Eight fixed-threshold decisions, defined once in `metrics/rules/gates.py`.
Each is tristate: 1.0 the rule holds, 0.0 it does not, NaN it was never measurable.
The constants are eyeballed, as Chanin's absorption paper sets its cutoffs and says so, and form the named set `SYNTHETIC_TOYS` in `metrics/rules/constants.py`.
The Gemma pipeline uses `GEMMA_MATRYOSHKA` from the same file, which differs in `fire_threshold` (1e-3) and `sres_rank_top_k` (5).
The defence of an arbitrary constant is that one fixed rule applies identically to every arm of a comparison.

| Gate | Rule | Constants | Defined where |
| --- | --- | --- | --- |
| `gate_support` | both endpoints fire at least `min_fire_count` times and they co-fire at least `min_joint` times | 20, 30 | every pair |
| `gate_contains` | `R(p,c) >= edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_strictly_contains` | `R(p,c) >= edge_tau` AND `R(c,p) < edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_mutually_contains` | `R(p,c) >= edge_tau` AND `R(c,p) >= edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_recon` | `recon_2a >= recon_rel_gain_min` AND `recon_child_gain >= recon_rel_gain_min` | 0.01 | both gains finite |
| `gate_sres_rank` | the parent's decoder AND the child's own rank inside the child probe's top k correlations | `sres_rank_top_k` | children whose probe trained |
| `gate_high_outdegree` | either endpoint has out-degree at least `superparent_outdeg_frac * (R - 1)` | 0.30 | endpoints firing at least `min_fire_count` times |
| `gate_freq_survives` | `token_freq_survival >= squash(freq_survival_min)`, equality surviving | 0.5 raw, 0.333 squashed | pairs with a defined survival ratio |

`squash(x) = x / (1 + x)` converts a threshold stated on the raw ratio to the scale the detector reports; comparing the raw 0.5 against the reported value would demand a raw ratio of 1.0.
`min_joint` is also the co-firing floor of the edge set and of `token_freq_survival`, as in the Gemma pipeline.
`gate_sres_rank` scores its competitor pool over the scored frame, not the whole dictionary; on a trained read with recovery attrition that bar is weaker than the paper's, and `n_recovered_features` is the number to watch.

For probe comparisons, use the same probe formula on true firing with true unit g and on learned firing with learned unit W, fitted on the recorded fitting draw and frozen before scoring.
A probe trained on child firing can exploit parent co-firing even with orthogonal decoders, which is the measured reason `gate_sres_rank` passes firing_only pairs (Section 3).
The paper's top-five probe ranking is our rule rather than a contrast, so the remaining differences from Tree SAE are the competitor pool and the all-pairs population.
Source: [Tree SAE Section 3](https://arxiv.org/html/2605.07922v2#S3), [Section 5.2](https://arxiv.org/html/2605.07922v2#S5.SS2), [Appendix H](https://arxiv.org/html/2605.07922v2#A8).
Exact containment implies oracle R = 1 when the child fires; it does not imply a recall guarantee for a rule after training.

## 6. Populations and required reports

Nothing is fitted: every rule compares a gate against a constant from Section 5, the same in every world, arm and read.
The false-positive rate is therefore measured on the whole unrelated null, with no calibration half, and the bar can fail.

Procedure:

1. Fit the SAE and determine recovery without evaluation observations.
2. Keep the three draws of an experiment seed apart: matching on `seed`, scoring on `seed + 10000`, probe fitting on `seed + 20000`.
3. Use held-out observations for firing statistics and token-frequency buckets.
4. Freeze the probe directions before the detectors and the gates, because `gate_sres_rank` reads them.

Primary evaluation uses all ordered pairs on the stated endpoint universe, excluding self-pairs, with no coverage shortlist.
For `frequency` and `topical` the population is the symmetric class, capped as in Section 3.

For each target, read and seed, report:

- N_total: generated ordered target pairs in the stated population.
- N_recovered: target pairs with both endpoints recovered.
- N_scorable: recovered target pairs where every gate the rule reads is measurable.
- N_pass: complete-expression positives among those pairs.

Missing or unrecovered targets count as end-to-end misses, and zero denominators are untestable.
Recall rides on N_recovered, and the null FPR and each confound's leakage ride on N_scorable (Section 8 says why).
Report every rule's null FPR and leakage in every world, plus multiple-rule and no-rule outcomes; never select a winning class or expression using the answer key.

The scorability guard is a reported population, not a silent filter: the fraction `gate_support` excludes is reported by cause, because "the rule rejected these pairs" and "the rule could not see them" are different results.
Survival is `S = r/(1+r)` with `r = R_rest/R_all`, where the restricted corpus drops the high-frequency bucket; unchanged coverage maps to 0.5.
`wide(p,c) = min(outdegree[p,c], outdegree[c,p])` requires both values finite; `wide` and `gate_high_outdegree` are endpoint-broadcast, so a per-pair scorability mask must never be applied to them.
Record `constant_null`, `constant_target` and `constant_overall` with their tolerance and finite counts: a point-mass null against a point-mass target separates perfectly without any real detection.

## 7. Reproducibility requirements

- Pairs are keyed on true feature ids, not recovered positions, so a true pair is in the null for every read of a (toy, seed).
- Scores, gates, and each rule's pass and scorable masks are persisted at full precision, so a borderline decision is reproducible; a rounded value near a constant would move a decision silently.
- `FAILS` is literal, a conjunction is scorable only where every gate it reads is measurable, and a rule may not read a metric: `metrics.rules.evaluate` refuses any clause naming something outside the registered gates.
- Every artifact records its resolved config, code revision, all seeds and draws, recovery mappings, the gate constants and the scoring precision.

Checklist before freezing; an item closes only when the listed verification exists:

- [x] Literal negative predicates, clause-specific scorable masks, no rule reading a metric, both directions finite before `wide`.
- [x] Re-run the oracle ceiling after the Section 4 decisions (2026-09-20; it reproduced the Section 3 table exactly).
- [ ] Persist enough exact data to reproduce every expression: original-precision gate matrices and required metric scores, or sufficient scorable and pass masks, alongside the constants.
- [ ] Verify instrument names and pair identity: probe `S_res` distinct from cosine `G`, matching feature ids and both pair orderings, the matched-oracle control retained.
- [ ] Add distribution diagnostics without changing verdicts: constant-null, constant-target and constant-overall flags, support, and dictionary or candidate-pool size beside the four outcome labels.
- [ ] Finalize output provenance and verify the complete tables: every rule on every world, including the null and the named confounds.

Passing the test suites is not certification of the benchmark while this list is open.

## 8. Settings to approve and freeze

The bars, the support floor, the designated rules and the grading arithmetic live in `metrics/rules/grading.py`, shared with the Gemma pipeline.
Complete this manifest before any fresh-seed result is inspected:

| Setting | Current proposal / action needed |
| --- | --- |
| Candidate registry | Five designated expressions, the containment baseline and two no-recon comparators (Section 4); approve exact inclusion |
| Metric definitions | Pin the implementation revision, score modes, signs, normalization, the undefined-cell convention and support gates. The probe and token-bucket settings are `METRIC_SETTINGS`, stamped as `metric_settings` |
| Gate constants | `edge_tau` 0.5, `min_fire_count` 20, `min_joint` 30, `recon_rel_gain_min` 0.01, `superparent_outdeg_frac` 0.30, `freq_survival_min` 0.5 (squashed 0.333), `sres_rank_top_k` 2: the set `SYNTHETIC_TOYS`, stamped whole as `gate_constants` in every artifact |
| Operational recall bar | Recall-given-recovery >= 0.8, except `rule_frequency` and `rule_topical`: their caps (Section 3) mean those rows report the curve and the cap, and no MET or DID NOT MEET verdict on recall is read from them. The verdict function still prints one, so the table must carry this note |
| Null-FPR bar | Null FPR <= 0.01 in each tested world, over the whole unrelated class |
| Confound-leakage bar | Each named complete-expression confound rate <= 0.05 |
| Support requirements | `MIN_SCORABLE_SUPPORT = 10`, applying to the target row, the null row and every confound row |
| Rate denominators | Recall on `N_recovered`; null FPR and confound leakage on `N_scorable` |
| Cross-world combination | Null FPR: the worst world, with the pooled rate beside it. Leakage: pooled by counts, with the worst world beside it |
| Reporting contract version | `REPORT_SCHEMA = 4` (history below). Stamped into every artifact; `write_artifacts` refuses any other value |
| Recovery/end-to-end criterion | Report both; specify an additional bar only if making an operational-recovery success claim |
| Seeds and draws | Record the exact three unused world/training seed ids, the draw derivations (Section 6) and sample sizes. The draw offsets are in `scoring/config.py`, stamped as `benchmark_settings`, and the census thresholds as `pathology_settings` in `census.json` |
| SAE setup | Pin per-toy variant, sparsity, dictionary/prefix sizes, training configuration and checkpoint provenance |
| Baselines | Endpoint firing-count comparators at the same nominal null budget, reporting achieved FPR, recall and leakage on the same population; matching a baseline is a valid result, and a superiority claim needs measured added discrimination |
| Uncertainty | Report each seed separately and between-seed variation; approve any confidence-bound method before using it |

### Report schema versions

`REPORT_SCHEMA` versions the keys `grade_rules` writes and the arithmetic under them; it is bumped when a key's arithmetic changes under an unchanged name, or a rate key is added, renamed or removed.

1. The pilot contract: `recall_given_recovery` was N_pass / N_scorable, with one `fpr` key on the null rows.
2. `recall_given_recovery` moved to N_pass / N_recovered, with the scorable rate split out as `pass_rate_given_scorable`; `fpr` split into `fpr_given_scorable` and `fpr_over_half`; `leakage` read the scorable rate, with `leakage_over_recovered` beside it; the support floor extended to the null and confound rows; `verdict` decides established failures before unmeasurable evidence.
3. The fixed-gate contract: the null is no longer halved, so `fpr_given_scorable` is measured over the whole null and `fpr_over_half` is gone.
4. The shared-rules contract: every rule and three gates were renamed (Section 4); the arithmetic is unchanged.

### Why the settings are shaped this way

Every bar takes the denominator that is harder to clear.
A target pair the rule could not score counts as a miss, while an unscorable null or confound pair does not dilute a rate; symmetric denominators would let a rule clear the leakage budget by leaving confound pairs unmeasurable.
Both denominators are reported under names that say which is which.

The support floor guards the negative evidence too: an adversarial review reached MET CRITERIA off a one-pair null and a one-pair confound row.
An under-supported negative row blocks a pass without rescuing a failure.

The null-FPR bar holds "in each tested world", so the worst world faces it; a pooled rate lets four quiet worlds absorb one loud one.
Leakage has no per-world wording and is pooled by counts.

The bars are proposed operating criteria, not mathematical guarantees; do not loosen them because a metric fails.
Shared-feature pairs are not independent replications, and p10-p90 ranges are not confidence intervals.

Use MET CRITERIA, DID NOT MEET CRITERIA, UNTESTABLE and INVALID MEASUREMENT as distinct labels; all three bars are inclusive.
A missing class has zero end-to-end recall and an untestable conditional detector, not perfect specificity evidence.
A valid negative score is not INVALID MEASUREMENT.
High base rate is the superparent target, so its firing-count baseline is substantive rather than something every detector must beat.

## 9. Freeze, replicate, and report

- [ ] Close Section 7's correctness/reporting items and approve Section 8's manifest.
- [ ] Mark a specific document/registry/evaluator/configuration version FROZEN and preserve it.
- [ ] Run the three fresh seeds with no candidate reselection, predicate flips, threshold search, or seed replacement.
- [ ] Produce the full metric-by-property and expression-by-world/class tables, including recovery, scorable support, evaluation FPR, and uncertainty limits.
- [ ] Report whether each pilot success or failure replicated; state when the signal is within uncertainty rather than calling it an improvement.
- [ ] Finish the single-property stage, including properties for which no designated expression met criteria.

### Pass criteria for the damage-matrix run, declared 2026-09-20 before any seed 1-3 result

These are written down before the run so they cannot be shaped by what comes back.
A criterion that fails stops the run and gets debugged; it is not relaxed to fit the output.

Instrument checks, which must all hold or the read is not trustworthy:

- Each designated rule reproduces its seed-0 oracle ceiling recall within 0.05 absolute, on the toy that plants its target class.
- The worst null false-positive rate across every rule and every toy is at or below 0.01.
- No expression row reads INVALID, and every arm row carries its counts for all four populations (corrupted, touched, intact, null).
- Absorption at `beta = 0, eta = 0` reproduces that toy's undamaged cell exactly. These are the same dictionary, so any difference is a bug in the damage path, not a result.
- Composition at `pi = 0` under the `union` readout reproduces that toy's undamaged cell exactly, per the invariant recorded in `synthdict/PARAMETERS.md`.
- Every damage cell has a non-empty intact arm. Coverage is 0.5 precisely to guarantee this; a cell that arrives without one is reported as uncontrolled rather than compared.

Registered expectations, declared so that confirming them is not mistaken for discovery and contradicting them is not quietly absorbed:

- `rule_firing_only` reads 0 recall and `rule_is_a` leaks fully onto `firing_only`, on every toy and every read. The geometry channel does not separate the two classes at any `k`; this is settled and is not reopened by this run.
- `rule_frequency` and `rule_topical` cap at 0.25 and 0.33 because their classes are labelled on both orderings while the rules are directional. Those are ceilings, not failures, and the 0.80 bar does not apply to those rows.
- Under absorption on `only_superparent` the `superparent` class has no corrupted pairs by construction, because absorbed edges are `firing_only`. Its signal is in the touched arm.
- Hedging is not run on `only_isa`. Every parent there has exactly one child, so hedging deletes the parent's only `is_a` pair and `gamma_rel` cannot move recall by any amount. Measured seed 0: corrupted and touched arms are 0 at every coverage. The column runs on `only_superparent` instead, where the same coverage gives 1686 corrupted and 3144 intact pairs.

Reporting discipline:

- Counts are printed beside every rate. `only_frequency` and `only_topical` reach only about 8 corrupted pairs, so their rates carry wide intervals.
- No difference is called real unless its interval excludes the comparison. A gap inside the interval is reported as within noise, in those words.
- Seed 0 is a regression check against the recorded ceiling and is never quoted as evidence for a constant, because it selected `sres_rank_top_k = 2` and the `rule_topical` clauses.

Expected findings are hypotheses, not requirements imposed on the outputs.
If a previously failing rule succeeds, report that result with the same checks; do not engineer either success or failure.
After freezing, an implementation bug requires a versioned correction and rerunning affected results with the same seed plan.
A new formula, orientation, expression, or threshold-selection procedure belongs to a new development version with fresh validation, not a hidden repair inside this benchmark.

## 10. Later stages and parked questions

Complex toy: keep the rule and gate identities fixed, and specify additional classes such as siblings and transitive pairs, nuisance settings and any overlapping-label scoring before viewing that stage.
Do not assume pure-toy success must transfer or pure-toy failure must persist; label any expression change as a separate development study.

Gemma: define an independent annotation target and a justified null protocol before reporting semantic precision or recall.
The toy's answer key is unavailable there, and random feature pairs are not automatically known unrelated examples.
Until then, report applicability and candidate relations rather than semantic correctness, and do not equate recovery of planted geometric labels with validation of semantic is-a.

Parked outside this study:

- Finding a new separator for the two containment subtypes.
- Cross-only S_res, nuisance-conditioned probes, redesigned metrics, and learned decision trees.
- Causal attribution of decoder changes to a particular training objective without a suitable comparison.
- Additional diagnostics that do not affect validity or interpretation of the reported benchmark.
