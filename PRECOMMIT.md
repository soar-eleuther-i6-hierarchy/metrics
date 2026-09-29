# Precommit: fixed-gate rules on the toy benchmark

Status: DRAFT, not frozen.
Every rule decides on gates compared against fixed constants, defined once in `metrics/rules/` and shared with the Gemma pipeline, so the same rule means the same thing on a toy and on Gemma.
Section 3 keeps only the measured results that constrain how the rules may be read; the earlier pilot record, measured under the null-quantile rules, is in git history.

## 1. Status and remaining steps

Ruleset 2 (Sections 2 and 4) replaces the rules the first damage matrix ran under; its pass criteria (Section 9) are declared before its own run.
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

The question is how properties that look like hierarchy mask it, and whether the rules can tell the two apart.
The generator's labels, for an ordered pair (a, b) with a the candidate parent and b the candidate child:

| Code label | Pair |
| --- | --- |
| `hierarchy_overlap` | a is b's direct parent, and b's direction overlaps a's (alpha > 0) |
| `hierarchy_orthogonal` | a is b's direct parent, with orthogonal directions (alpha = 0) |
| `dense_lookalike` | a is a dense feature (firing on 85% of tokens), b a sparse feature outside a's lineage |
| `dense_other` | every other ordering of a dense pair: sparse to dense, dense to dense |
| `frequency_lookalike` | a fires on 90% of the tokens with one frequent token id, b on 32% of them |
| `frequency_other` | every other pair sharing a planted frequent-token set |
| `topical_lookalike` | a fires on 90% of one topic's tokens, b on 32% of them |
| `topical_other` | every other pair sharing a planted document topic |

The two hierarchy classes are one target.
Park et al. (Theorem 8a), Costa et al. (Definition 2.3) and Chanin et al.'s toys model a child's own direction as orthogonal to its parent, so geometry is not what makes an edge hierarchical; the sub-label only lets each geometry be reported separately.
The two geometries are planted in two worlds, `hierarchy_overlap` and `hierarchy_orthogonal`, that fire identically and differ only in the child's direction.
A look-alike is the ordering of a spurious pair whose firing looks like a parent-to-child edge: P(a | b) is high and P(b | a) is low.
It is the only ordering in which the property can pass for hierarchy, so it is the spurious target; the other orderings are scored negatives, kept out of the unrelated null.
The generator supplies structural labels, not a validated semantic taxonomy.

Stop the single-property stage when every registered metric/expression has a valid result or explicit untestable status for every required seed/class and result-changing audit issues are resolved.
Do not redesign S_res, change the generator, or search for a better classifier merely because the current expressions fail.

## 3. Measured constraints

Unless stated otherwise these are at the oracle ceiling: a perfect dictionary, true activations, no damage, full-size toys, 50k tokens.
A rule that does not fire there cannot fire anywhere.

Ruleset 1's results (the damage matrix `MATRIX`, seeds 1 to 3, and its seed-0 ceiling `MATRIX-P3`, report schema 4) are kept here only where they constrain ruleset 2; the full record is in git history.
Ruleset 2's measurements are below; they come from the second damage matrix, `MATRIX2`.

### What ruleset 2 measured

Seeds 1 to 3 pooled, 50k tokens, counts in pairs.
With a perfect dictionary:

- `rule_hierarchy` found all 360 edges on each hierarchy toy and passed no unrelated pair on any toy.
- On `dense` it found 14 of the 36 dense-parent edges (5, 5 and 4 per seed), where `rule_containment` found all 36: the rank gate rejects them.
- The rank gate and `PMI > 0` reject dense look-alikes: `rule_hierarchy` passed 47 of 7,092 (0.7%), where the rank gate alone passes 95 and `rule_containment` all of them.
- The token-frequency check blocks all 45 frequency look-alikes; `rule_hierarchy` passed all 48 topical look-alikes, as `rule_containment` does, so no check stops shared topics.
- `rule_frequency_driven` passed every frequency look-alike and nothing else; `rule_dense_endpoint` passed every pair with a dense endpoint, the 36 dense-parent edges included.

Under damage, on the damaged pairs; the undamaged pairs of the same runs stay at their no-damage rates:

- Absorption removes detection through the parent going silent on the child's tokens, not through the child's decoder taking on the parent's direction.
  With the parent silent on 60% of those tokens `rule_hierarchy` found none of the absorbed edges at any mixing; with no silencing it found all of them at every mixing.
  The two geometries differ only at mixing 0.75 with 30% silencing: 62% of edges found with overlapping directions, 76% with orthogonal ones.
- Splitting a parent and reading only its strongest piece drops `rule_hierarchy` to 52% at k = 2 and to 0% at k = 3 or more, in both geometries; recombining the pieces keeps it at 100%.
- Dense features that are split, or composed with a share of 0.5 or more, hide every dense look-alike from `rule_dense_endpoint` when each piece is read alone; recombining restores it.
- Composition does not change `rule_hierarchy` on the hierarchy toys.
- Hedging leaves `rule_dense_endpoint` at 100%, and its effect on `rule_hierarchy`'s dense look-alikes (0.5% to 0.1%) is within noise.
- The dense-parent edges under damage are too few (9 to 18 pairs) for any change to exceed noise.
- On seeds 2 and 3 the frequency toy's 12 absorption runs keep no undamaged container-to-member pair, so they have no within-run comparison (Section 9); `rule_frequency_driven`'s dose trend there still runs from 100% at zero dose to 0%.

### What ruleset 1 established

- Across seeds 1 to 3, `gate_recon` changed a rule's answer in 8 of 1,026 point-by-class cells for the is_a rule and 17 of 1,026 for the firing_only rule (5,685 pairs), all in one setting (`only_superparent`, composition, `union` readout), where the readout merged two features into one latent and its question is ill-posed.
  No rule reads it (Section 4).
- `gate_sres_rank` does not separate the two geometries at k = 5 or at k = 2: the probe is fitted on the child's firing and the parent co-fires on all of it, so the parent's decoder ranks near the top whether or not the directions overlap.
- `gate_sres_rank` responds to the parent's firing density: on the seed-0 ceiling at k = 5 it passed 100% of orthogonal edges when the parent fired on 18% of tokens, and 58% (7 of 12) when it fired on 85%.
- `gate_freq_survives` separates the frequency pairs only because every planted token id is in the high-frequency bucket: survival was exactly 0 on all 180 frequency pairs and 0.500 on every tree edge, against a cut of 0.333.

## 4. Registry: the committed expressions

Every rule decides on gates, never on a metric.
The rules are defined once in `metrics/rules/rules.py` and graded by `metrics/rules/grading.py`; both are shared with the Gemma pipeline, and `scoring/` imports them.
They replaced null-quantile predicates, which could not travel to a real SAE: a q99 cut accepts 1% of whatever population it is pointed at, so the false-positive bar could not fail.

| ID | Target | Exact expression |
| --- | --- | --- |
| rule_hierarchy | hierarchy_overlap, hierarchy_orthogonal | `PASSES(strictly_contains) AND PASSES(pmi_positive) AND PASSES(freq_survives) AND PASSES(sres_rank)` |
| rule_dense_endpoint | dense_lookalike | `PASSES(high_outdegree)` |
| rule_frequency_driven | frequency_lookalike | `PASSES(strictly_contains) AND FAILS(freq_survives)` |
| rule_containment | hierarchy_overlap, hierarchy_orthogonal (baseline) | `PASSES(strictly_contains)` |

The first three are designated; the containment baseline is reported beside them.
Every rule is graded on every class, so a look-alike class that `rule_hierarchy` passes shows up as its leak: that leak is the measure of how each property masks hierarchy.
No rule targets `topical_lookalike`, because no gate reads a topic; the topical toy is scored as a negative only.
`FAILS` is written literally, never as `NOT PASSES`: a gate that was never measurable satisfies neither predicate.
`rule_hierarchy`, the one rule reading `gate_sres_rank`, is INVALID MEASUREMENT on a run without the probe, not a rejection.
`PMI > 0` is the independence check of the metrics paper, and only `rule_hierarchy` reads it: on seed 0 it halves the dense look-alikes that pass containment (2,364 to 1,195) and keeps all 12 dense-parent edges, and on the other toys containment already implies it.
`rule_hierarchy` also requires the paper's token-frequency check, `R_low+mid / R_all >= 0.5`: on seed 0 it removes all 15 frequency look-alikes and no true edge, a success the frequency toy builds in by putting every planted token in the frequent group.

Deliberately absent:

- `G` is in no rule (decided 2026-09-19), so the rule set matches the one the team uses; it is computed and reported as a diagnostic in every artifact.
  No rule separates the two hierarchy geometries; they are one target, and `G` reports the geometry beside it.
- `gate_recon` is computed and reported, and read by no rule (Section 3).
- A spurious-coverage rule, `PASSES(strictly_contains) AND FAILS(sres_rank)`, is left out: within containment, wherever the rank gate is measurable, it passes exactly the pairs `rule_hierarchy` rejects, so it measures nothing new.

Names carry no version.
A change to any rule, gate or constant bumps `metrics.rules.RULESET_VERSION` (now 2), which every artifact records beside `gate_constant_set`.
Ruleset 2 (2026-09-26) merged `is_a` and `firing_only` into the hierarchy target, split each spurious class by ordering, dropped the recon clause and `rule_topical`, and returned `sres_rank_top_k` to 5.
It also added `PMI > 0` and the token-frequency check to `rule_hierarchy` the same day; the version stayed 2 and `MATRIX2` was rerun under it.
Its toys are `hierarchy_overlap` and `hierarchy_orthogonal`, the ruleset-1 `only_isa` and `only_firing` worlds under new names, and `dense`, `frequency` and `topical`.
Those three were rebuilt on 2026-09-27 from the ruleset-1 `only_superparent`, `only_frequency` and `only_topical` worlds with more copies of the same construction, because they held too few pairs: each dense parent has two children that never fire together (24 dense-parent edges), the frequency toy has 24 token groups of one frequent token id each (ids 1 to 24), and the topical toy 24 topics (48 look-alikes each).
Rules and gates were renamed on 2026-09-23 without changing any decision, and the stored gate-era results (`MATRIX`, `MATRIX-P3`) were renamed with them; the old names are in git history and in `outputs_archive/gate_era_original_names.tar` on soar-gpu.
Schema-2 results predate the gate rules and keep their own names.
`gate_contains`, one-way containment and the Gemma pipeline's cross-block edge, was added at the same time; no rule reads it yet.

### Constant and clause decisions

They change a constant or a clause, never a metric definition.

1. `sres_rank_top_k` is 5, Tree SAE's operational rule, in both constant sets.
   Ruleset 1 used 2 to separate is_a from firing_only, which it did not do (Section 3); with the two merged the reason is gone.
2. `recon_rel_gain_min` stays at 0.01 for the reported `gate_recon`; no rule reads it.
3. `gate_strictly_contains` is kept rather than Tree SAE's one-way coverage: the two differ only on pairs that cover each other, and in these toys all of those are dense-to-dense pairs.
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
| `token_freq_survival` | r/(1+r) with r the restricted-corpus coverage ratio | `token_control.frequency_controlled_coverage`, unclamped; a child with no rare-token firing scores 0, one with 1 to 4 is unmeasurable | gate_freq_survives |
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

Ten of the sixteen decide nothing, the two reconstruction metrics included, since no rule reads `gate_recon`; they are persisted at full precision and reported in `metric_diagnostics`, so a reader can tell a diagnostic from a number a verdict rests on.

### Gates

Nine fixed-threshold decisions, defined once in `metrics/rules/gates.py`.
Each is tristate: 1.0 the rule holds, 0.0 it does not, NaN it was never measurable.
The constants are eyeballed, as Chanin's absorption paper sets its cutoffs and says so, and form the named set `SYNTHETIC_TOYS` in `metrics/rules/constants.py`.
The Gemma pipeline uses `GEMMA_MATRYOSHKA` from the same file, which differs in `fire_threshold` (1e-3) and `sres_rank_pool` (the whole dictionary).
The defence of an arbitrary constant is that one fixed rule applies identically to every arm of a comparison.

| Gate | Rule | Constants | Defined where |
| --- | --- | --- | --- |
| `gate_support` | both endpoints fire at least `min_fire_count` times and they co-fire at least `min_joint` times | 20, 30 | every pair |
| `gate_contains` | `R(p,c) >= edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_strictly_contains` | `R(p,c) >= edge_tau` AND `R(c,p) < edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_mutually_contains` | `R(p,c) >= edge_tau` AND `R(c,p) >= edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_recon` | `recon_2a >= recon_rel_gain_min` AND `recon_child_gain >= recon_rel_gain_min` | 0.01 | both gains finite |
| `gate_sres_rank` | the parent's decoder AND the child's own rank inside the child probe's top k correlations | `sres_rank_top_k` | children whose probe trained |
| `gate_high_outdegree` | either endpoint has out-degree at least `superparent_outdeg_frac * (R - 1)` | 0.30 | every pair, unless both endpoints fire under `min_fire_count` times |
| `gate_freq_survives` | `token_freq_survival >= squash(freq_survival_min)`, equality surviving | 0.5 raw, 0.333 squashed | pairs with a defined survival ratio |
| `gate_pmi_positive` | `PMI(p,c) = log(cofire * N / (fire_p * fire_c)) > 0`, so exact independence fails | 0 | supported pairs |

`squash(x) = x / (1 + x)` converts a threshold stated on the raw ratio to the scale the detector reports; comparing the raw 0.5 against the reported value would demand a raw ratio of 1.0.
`min_joint` is also the co-firing floor of the edge set and of `token_freq_survival`, as in the Gemma pipeline.
`gate_sres_rank` scores its competitor pool over the scored frame, not the whole dictionary; on a trained read with recovery attrition that bar is weaker than the paper's, and `n_recovered_features` is the number to watch.

For probe comparisons, use the same probe formula on true firing with true unit g and on learned firing with learned unit W, fitted on the recorded fitting draw and frozen before scoring.
A probe trained on child firing can exploit parent co-firing even with orthogonal decoders, which is the measured reason `gate_sres_rank` passes orthogonal edges (Section 3).
The paper's top-five probe ranking is our rule rather than a contrast, so the remaining differences from Tree SAE are the competitor pool, the all-pairs population and strict rather than one-way containment (Section 4).
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
The spurious targets are the look-alike orderings (Section 2); the other orderings of each family are reported as their own classes and graded as confounds.

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
The high-frequency bucket is the most common half of the scored tokens, so the check sees only causes inside it: a pair tied to token ids outside that half keeps its coverage when the bucket is dropped and is not flagged.
The half is cut from the sampled counts, so ids near its edge move in and out between draws; the frequency toy plants ids 1 to 24, which fell outside the half on 1 of 300 random 50k-token draws (ids up to 30: 240 of 300).
`wide(p,c) = min(outdegree[p,c], outdegree[c,p])` requires both values finite, while `gate_high_outdegree` is undefined only when both endpoints are; both are endpoint-broadcast, so a per-pair scorability mask must never be applied to them.
Record `constant_null`, `constant_target` and `constant_overall` with their tolerance and finite counts: a point-mass null against a point-mass target separates perfectly without any real detection.

## 7. Reproducibility requirements

- Pairs are keyed on true feature ids, not recovered positions, so a true pair is in the null for every read of a (toy, seed).
- Scores, gates, and each rule's pass and scorable masks are persisted at full precision, so a borderline decision is reproducible; a rounded value near a constant would move a decision silently.
- `FAILS` is literal, a conjunction is scorable only where every gate it reads is measurable, and a rule may not read a metric: `metrics.rules.evaluate` refuses any clause naming something outside the registered gates.
- Every artifact records its resolved config, code revision, all seeds and draws, recovery mappings, the gate constants and the scoring precision.

Checklist before freezing; an item closes only when the listed verification exists:

- [x] Literal negative predicates, clause-specific scorable masks, no rule reading a metric, both directions finite before `wide`.
- [ ] Re-run the oracle ceiling under ruleset 2: the seed-0 checks in Section 9 (ruleset 1's re-run on 2026-09-20 reproduced its own table exactly).
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
| Candidate registry | Three designated rules and the containment baseline (Section 4); approve exact inclusion |
| Metric definitions | Pin the implementation revision, score modes, signs, normalization, the undefined-cell convention and support gates. The probe and token-bucket settings are `METRIC_SETTINGS`, stamped as `metric_settings` |
| Gate constants | `edge_tau` 0.5, `min_fire_count` 20, `min_joint` 30, `recon_rel_gain_min` 0.01, `superparent_outdeg_frac` 0.30, `freq_survival_min` 0.5 (squashed 0.333), `sres_rank_top_k` 5: the set `SYNTHETIC_TOYS`, stamped whole as `gate_constants` in every artifact |
| Operational recall bar | Recall-given-recovery >= 0.8 |
| Null-FPR bar | Null FPR <= 0.01 in each tested world, over the whole unrelated class |
| Confound-leakage bar | Each named complete-expression confound rate <= 0.05 |
| Support requirements | `MIN_SCORABLE_SUPPORT = 10`, applying to the target row, the null row and every confound row |
| Rate denominators | Recall on `N_recovered`; null FPR and confound leakage on `N_scorable` |
| Cross-world combination | Null FPR: the worst world, with the pooled rate beside it. Leakage: pooled by counts, with the worst world beside it |
| Reporting contract version | `REPORT_SCHEMA = 5` (history below). Stamped into every artifact; `write_artifacts` refuses any other value |
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
5. Ruleset 2's classes: `is_a` and `firing_only` became `hierarchy_overlap` and `hierarchy_orthogonal`, and each spurious class split into `_lookalike` and `_other`, so a stored class code means a different class; `synthdict/export.py` refuses any other schema.

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
High base rate is the dense target, so its firing-count baseline is substantive rather than something every detector must beat.

## 9. Freeze, replicate, and report

- [ ] Close Section 7's correctness/reporting items and approve Section 8's manifest.
- [ ] Mark a specific document/registry/evaluator/configuration version FROZEN and preserve it.
- [ ] Run the three fresh seeds with no candidate reselection, predicate flips, threshold search, or seed replacement.
- [ ] Produce the full metric-by-property and expression-by-world/class tables, including recovery, scorable support, evaluation FPR, and uncertainty limits.
- [ ] Report whether each pilot success or failure replicated; state when the signal is within uncertainty rather than calling it an improvement.
- [ ] Finish the single-property stage, including properties for which no designated expression met criteria.

### Pass criteria for the ruleset-2 damage-matrix run (`MATRIX2`), declared 2026-09-26 before any result

These are written down before the run so they cannot be shaped by what comes back.
A criterion that fails stops the run and gets debugged; it is not relaxed to fit the output.
The run is 129 dial points per seed at 50k tokens: seed 0 at the oracle and undamaged points first, then seeds 1 to 3.
The `dense`, `frequency` and `topical` toys were rebuilt after the first run (Section 4); the lines below that name their sizes were updated on 2026-09-27, before those three toys were rerun.

Instrument checks, which must all hold or the read is not trustworthy:

- Every declared point is written: 10 on seed 0, 129 on each of seeds 1 to 3.
- On seed 0 each hierarchy toy's arrays equal the ruleset-1 seed-0 arrays of the same world in `MATRIX-P3` (`only_isa`, `only_firing`) bit for bit, at the oracle and undamaged reads, on every gate except `gate_sres_rank`, which k moves, and `gate_contains` and `gate_pmi_positive`, which that run did not store.
  The rebuilt `dense`, `frequency` and `topical` have no earlier twin.
- Each designated rule's oracle recall is 1.00 within 0.05 on the toys built for its target: `rule_hierarchy` on `hierarchy_overlap` and on `hierarchy_orthogonal`, `rule_dense_endpoint` on `dense`, `rule_frequency_driven` on `frequency`.
  This follows from the ruleset-1 gate arrays (Section 3), so a miss means the labels or the rule wiring are wrong.
- The worst null false-positive rate across every rule and every toy is at or below 0.01.
- No expression row reads INVALID.
- Absorption at `beta = 0, eta = 0` reproduces that toy's undamaged cell exactly.
  These are the same dictionary, so any difference is a bug in the damage path, not a result.
- Composition at `pi = 0` under the `union` readout reproduces that toy's undamaged cell exactly, per the invariant recorded in `synthdict/PARAMETERS.md`.
- Every damage cell has a non-empty intact arm.
  Coverage is 0.5 precisely to guarantee this; a cell that arrives without one is reported as uncontrolled rather than compared.

Registered expectations, declared so that confirming them is not mistaken for discovery and contradicting them is not quietly absorbed:

- `rule_hierarchy` passes every edge on both hierarchy toys.
- It passes about half of the dense parents' edges on `dense`, now 24 per seed (on the earlier one-child toy: 7 of 12 on seed 0 and 5 of 12 on seed 1 under ruleset-1 ranks at k = 5, and 14 of 36 over seeds 1 to 3 in the first `MATRIX2` run).
- It leaks onto `topical_lookalike`: the pair from the feature firing on 90% of a topic's tokens to one firing on 32% passes containment, and the probe ranks the 90% feature high by the same co-firing; the token-frequency check blocks `frequency_lookalike`.
- `rule_dense_endpoint` passes every dense look-alike, and also `dense_other` and the dense parents' edges, because its gate reads either endpoint's out-degree.
- `rule_frequency_driven` passes every frequency look-alike.
- Every null rate is 0.
- Hedging is run on neither hierarchy toy: every parent there has exactly one child, so hedging deletes the parent's only hierarchy pair and `gamma_rel` cannot move recall.
  The column runs on `dense`.
- Under absorption on `dense` the corrupted pairs are the dense parents' edges (`hierarchy_orthogonal`); the dense look-alikes are touched, not corrupted.

Reporting discipline:

- Counts are printed beside every rate, for all four populations of an arm row (corrupted, touched, intact, null).
  `frequency` and `topical` reach about 24 corrupted pairs per seed, so their rates carry wide intervals.
- No difference is called real unless its interval excludes the comparison.
  A gap inside the interval is reported as within noise, in those words.
- Seed 0 is the instrument check above and is not quoted as evidence.
- Each target class is reported under `rule_hierarchy` (its recall on a hierarchy class, its leak on a look-alike) and under the class's own designated detector.
  A table cell's rate is the pass rate among that class's measurable pairs; the recall criterion above is checked separately.

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
