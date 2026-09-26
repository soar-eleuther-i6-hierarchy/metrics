# Precommit: existing metrics and fixed property expressions

Status: DRAFT, revised 2026-09-19 for the FIXED-GATE rules.
The null-quantile predicates are gone, `G` is dropped, and every rule now decides on a gate compared against a constant, so the same rule means the same thing on a toy and on Gemma.
Sections 4, 5 and 6 are rewritten for that contract; Sections 2, 3 and 11 keep the pilot record, which was measured under the quantile rules and is labelled as history.
Stop searching for a subtype separator in this study.
Keep the committed expressions, including valid failures, and test whether their behavior replicates.
Do not open the fresh seeds until the freeze checklist is complete.
Revised again 2026-09-20 to record the three-seed damage matrix in Section 3 and the pass criteria in Section 9.
The 2026-09-19 revision updated the document only; the 2026-09-20 one reports 333 runs and changes no rule.

## 1. Immediate next actions

Work in this order:

1. DONE 2026-09-19: `sres_rank_top_k` is 2 and `rule_topical` reads `PASSES(strictly_contains) AND PASSES(freq_survives)`, both measured at the ceiling before and after (Section 3).
2. DONE 2026-09-19: the 0.25 and 0.33 caps are accepted, the target populations stay symmetric, and `BAR_RECALL` does not apply to those two rows (Sections 4 and 8).
3. DONE 2026-09-19: G stays out of every rule and remains a reported diagnostic, so the geometry failure is a registered negative result rather than an open search (Sections 3 and 4).
4. DONE 2026-09-20: gate mutation anchors filled and killed, and the oracle ceiling re-run on all five toys, reproducing the Section 3 table exactly.
5. DONE 2026-09-20: the pass criteria were written into Section 9 before any seed 1-3 result existed, and the three-seed damage matrix was run and met all of them (Section 3).
6. Approve the registry in Section 4 and fill the remaining settings in Section 8.
No rule needs to pass to be included.
7. Freeze the document, registry, evaluator revision and seed plan together.
8. Run one independent confirmation seed, report every registered outcome, then the remaining fresh seeds.
9. Evaluate complex-toy transfer and then Gemma under the scope in Section 10.

The next implementation task is the two constant and clause decisions, NOT another metric-development experiment.
A valid negative result does not block freezing; an invalid measurement or an unresolved setting does.

## 2. Goal, target meanings, and stopping condition

Establish which existing metrics and fixed Boolean expressions respond to which planted properties, where they fail, and how that behavior changes across seeds, SAE training, and more complex worlds.
Deliver a metric-by-property characterization and a complete-expression recall/leakage table with support, provenance, and uncertainty limits.
The benchmark succeeds when those results are trustworthy, not when every expression retrieves its target.

Keep three notions separate:

| Relationship | Meaning | Example |
| --- | --- | --- |
| Semantic is-a | Category inclusion by meaning | Every dog is an animal |
| Activation containment | Child firing implies parent firing on the observed distribution | All dog images in a dataset happen to have red backgrounds |
| Decoder overlap | Component directions have positive cosine | Two decoder vectors share a direction |

Containment can arise without semantic is-a.
A genuine animal/dog hierarchy can also use an animal component and an orthogonal dog-specific component.
Neither containment nor positive decoder overlap establishes semantic hierarchy alone.

Retain the existing generator labels for compatibility:

| Code label | Benchmark target |
| --- | --- |
| `is_a` | Generated direct containment edge with planted overlap, effective alpha > 0 |
| `firing_only` | Generated direct containment edge with orthogonal directions, effective alpha = 0 |
| `superparent` | Either endpoint is a declared high-base-rate feature; firing probability is 0.85, not 1 |
| `frequency` | Both features share the planted frequent-token set |
| `topical` | Both features share the planted document topic |

Do not merge is_a and firing_only into a semantic "true hierarchy" label.
Their union is the target for a separate generated-direct-containment baseline; primary class labels do not change.
They are mutual hard negatives for the planted-overlap distinction, not automatically semantic negatives of each other.
The abstract generator supplies structural/geometric labels rather than an independently validated semantic taxonomy.

Stop the single-property stage when every registered metric/expression has a valid result or explicit untestable status for every required seed/class and result-changing audit issues are resolved.
Do not redesign S_res, flip G predicates, change the generator, or search for a better classifier merely because the current expressions fail.

## 3. Pilot conclusions to carry forward

The subsections down to "What the pilot currently says" were measured under the NULL-QUANTILE rules and are kept as history.
They are not restatements of how the current rules behave; the gate-era measurement is the ceiling check at the end of this section.

### G_W has now been measured

Sources: [G_W findings](outputs_local/gw_check/FINDINGS.md), the five `outputs_local/gw_check/<toy>.json` files, and `_raw/<toy>_gw.npz` in that directory.
The server study used seed-0 Matryoshka checkpoints, 200k scoring tokens, and a held-out scoring draw.
G_W is cosine between learned parent/child decoders; G_g_matched is true-direction cosine restricted to the same recovered endpoints WITHIN each world.
There are 112 recovered is_a target pairs and 120 recovered firing_only target pairs, not one shared 112-pair universe across both worlds.

| Read | is_a median [p10, p90] | firing_only median [p10, p90] |
| --- | --- | --- |
| G_g_matched | 0.480 [0.480, 0.480], n=112 | approximately 0 [0, 0], n=120 |
| G_W | 0.04495 [0.01485, 0.07296], n=112 | 0.06069 [0.02462, 0.36525], n=120 |

These are within-seed distributions, not confidence intervals.
On the matched endpoints, the selected learned decoders do not preserve the planted overlap sufficiently for the current HIGH/IN-BAND rules to retrieve the subtypes.
The matched-oracle control rules out dropped endpoints as the explanation for this within-recovered comparison.
Recovery loss remains a separate end-to-end limitation, and the matching procedure remains part of the operational definition of a learned counterpart.

Supported conclusion: the registered G-based expressions fail on this pilot.
Unsupported conclusions: all subtype information is destroyed, no geometric classifier can work, semantic hierarchy is lost, or the multi-level loss alone caused the change.
Tree SAE Appendix H provides a plausible interpretation of near-orthogonal decoders, not a causal experiment on these checkpoints.
Do not call the oracle a guaranteed numerical ceiling, or use the observed HIGH-rate reversal as a stable ordering before fresh-seed replication.

### Complete expressions: target and confound counts

Define `C = HIGH(coverage_R) AND HIGH(asymmetry_R) AND HIGH(pmi)`.
These counts are from the saved G_W JSONs, using each world's own calibration thresholds:

| Expression | is_a target pairs | firing_only target pairs | Important additional leakage |
| --- | --- | --- | --- |
| C | 112/112 | 117/120 | 62/240 frequency and 18/56 topical |
| C AND HIGH(G_W) | 2/112 | 22/120 | 7/56 topical |
| C AND IN-BAND(G_W) | 110/112 | 95/120 | 62/240 frequency and 11/56 topical |

C accepts zero reversed targets in both containment worlds.
It is sensitive to generated containment and rejects reversed/unrelated pairs well in those worlds, but its frequency leakage is 25.8% and topical leakage is 32.1%.
Register C as a containment-oriented baseline with known confounds, NOT a successful selective containment detector across all five worlds.
The two G expressions fail their intended subtype classification; do not relabel their false positives as true hierarchy.

The frequency expression retrieves 236/240 trained targets and accepts zero observed non-frequency targets; trained superparent targets are absent and untestable.
The topical expression retrieves 18/56 targets and accepts 4/120 firing_only pairs and 4/120 reversed pairs in that world.
The trained superparent target is absent after recovery: 0/726 generated target pairs are available for conditional scoring.

### Expression-level FPR correction is not yet reflected in the saved reports

The original G_W JSONs provide `unrelated_eval` for individual geometry readings but count complete expressions over the FULL unrelated population.
Consequently, the original write-up's expression FPRs include calibration pairs and must not be quoted as evaluation-half FPRs.
The source now has `expression_counts` to emit an evaluation-null entry, but existing JSONs/write-up still need corrected generation and review.

A previous review reapplied saved evaluation masks to aligned legacy score arrays, with the reported numerical gate error used to bound float32-rounding ambiguity.
That check yielded the following target-world evaluation-null counts; they are cross-checks to reproduce in the corrected evaluator output, not newly regenerated reports:

| Expression / target world | Evaluation-null positives / population |
| --- | --- |
| is_a / only_isa | 0/26684 |
| firing_only / only_firing | 11/28560 |
| frequency / only_frequency | 56/9060 = approximately 0.00618 |
| topical / only_topical | 22/10268 = approximately 0.00214 |
| superparent / only_superparent | 0/7140; target class remains untestable |

Do not substitute full-null rates such as frequency 98/18120 for these evaluation rates.
Report null FPR for every rule in EVERY evaluated world, not only its target world.
For example, the same review found topical-expression null counts 318/28560 in only_firing and 80/7140 in only_superparent, both above a literal 0.01 bar.
This does not change topical's already failed recall verdict, but it belongs in the complete benchmark table.
When a rounded cache is too close to a threshold to recover an exact decision, regenerate from the original-precision scores instead of silently choosing a side.

### What the pilot currently says

| Candidate | Provisional interpretation, pending corrected reporting |
| --- | --- |
| is_a G-rule | DID NOT MEET CRITERIA: low recall and confound leakage |
| firing_only G-rule | DID NOT MEET CRITERIA: low recall and substantial is_a/frequency/topical leakage |
| Frequency rule | Promising pilot success on measured targets/confounds; correct evaluation FPR before finalizing the table |
| Topical rule | DID NOT MEET CRITERIA: low recall, plus some off-target-world null-FPR violations |
| Superparent rule | UNTESTABLE on trained targets; retain the oracle result separately |
| C baseline | DID NOT MEET selective-containment criteria because of frequency/topical leakage |

These are outcomes for fixed expressions and stated populations, not impossibility results about the underlying properties.
Single-seed counts supply no across-seed uncertainty estimate.

### Oracle-ceiling check of the gate rebuild, measured 2026-09-19

Every rule was run on a PERFECT dictionary: the oracle read, true activations, no damage, full-size toys, 50k tokens, seed 0.
This is the ceiling case, so a rule that does not fire here cannot fire anywhere.

| Toy | Rule | Recall at the ceiling | Cause |
| --- | --- | --- | --- |
| only_isa | rule_is_a | 1.00 | all three clauses pass |
| only_firing | rule_firing_only | 0.00 | `gate_sres_rank` passes 100% of firing_only pairs, so `FAILS` never holds |
| only_superparent | rule_superparent | 1.00 | null rate 0.00 |
| only_frequency | rule_frequency | 0.25 | `gate_strictly_contains` is directional; the class is labelled both ways |
| only_topical | rule_topical | 0.00 | `gate_mutually_contains` passes 0% of topical pairs |

Per-gate pass rates behind those numbers, target class against the unrelated null:

| Gate | is_a | firing_only | superparent | frequency | topical | null |
| --- | --- | --- | --- | --- | --- | --- |
| gate_strictly_contains | 1.00 | 1.00 | 0.47 | 0.25 | 0.33 | 0.00 |
| gate_mutually_contains | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gate_recon | 1.00 | 1.00 | - | 1.00 | 1.00 | 0.98 to 1.00 |
| gate_sres_rank | 1.00 | 1.00 in only_firing, 0.58 on only_superparent's dense edges | 0.017 | 0.62 | 0.69 | 0.013 to 0.03 |
| gate_high_outdegree | 0.00 | 0.00 in only_firing, 1.00 on only_superparent's dense edges | 1.00 | 0.00 | 0.00 | 0.00 |
| gate_freq_survives | 1.00 | 1.00 | - | 0.00 | 1.00 | 0.75 to 1.00 |

Three findings carry into the proposals in Section 4:

1. The geometry channel does not separate the two primary classes.
`gate_sres_rank` passes on 100% of is_a AND 100% of firing_only pairs at F = 240, so `rule_firing_only` reads 0 and `rule_is_a` fires on every firing_only pair in only_firing, a 100% cross-toy leak.
The cause is the mechanism Section 5 names: the child's probe is fitted on the child's own firing and the parent co-fires on every one of those tokens.
2. Dropping the PMI clause costs specificity against dense features, not against independent ones.
`gate_strictly_contains` passes 0% of unrelated pairs everywhere, but 47% of superparent pairs, so the containment baseline reads base-rate coverage as containment.
`gate_sres_rank` absorbs it here (1.7% on those pairs), which means the dense-feature defence now rests on the probe rather than on an association test.
3. `gate_recon` is inert at the ceiling, passing targets and nulls alike.
4. The same gate reads differently in different worlds: `gate_sres_rank` passes 100% of firing_only edges in only_firing but 58% of them in only_superparent, where the parent is dense.
Those are the same class with the same geometry, so the gate is responding to density rather than to what it claims to measure.

### The same ceiling after the two changes, measured 2026-09-19

`sres_rank_top_k` is now 2 and `rule_topical` reads `PASSES(strictly_contains) AND PASSES(freq_survives)`.
Same protocol: oracle read, true activations, no damage, full-size toys, 50k tokens, seed 0.

| Rule | Before | After | Reading |
| --- | --- | --- | --- |
| rule_is_a | 1.00 | 1.00 | unchanged |
| rule_firing_only | 0.00 | 0.00 | NOT fixed; see below |
| rule_superparent | 1.00 | 1.00 | unchanged |
| rule_frequency | 0.25 | 0.25 | its ceiling on the symmetric class; 100% of the ordered container-to-member pairs |
| rule_topical | 0.00 | 0.33 | fixed; 100% of the ordered register-to-member pairs |
| rule_containment | 1.00 | 1.00 | unchanged |

`gate_sres_rank` at k = 2, measured per class:

| Toy | Target pass | Null pass |
| --- | --- | --- |
| only_isa (is_a) | 1.00 | 0.000 |
| only_firing (firing_only) | 1.00 | 0.001 |

So k = 2 tightened the null from 1.3% to at most 0.1% and left both target rates at 100%.
The parent's decoder is the child probe's rank-2 correlate whether or not the two directions are orthogonal, because the probe is fitted on the child's firing and the parent co-fires on all of it.
The geometry channel therefore does not separate is_a from firing_only at ANY k, and no constant inside this rule will change that.
This is a registered negative result, not an invitation to search for a separator: `rule_firing_only` and `rule_firing_only_no_recon` are expected to read 0 recall, and `rule_is_a` is expected to leak onto firing_only, on every read.
Anyone wanting that separation back needs a geometry measurement the benchmark does not currently have.

### Is it a threshold issue? No, measured both ways 2026-09-19

Both forms the geometry metric can take were checked at the ceiling, same protocol.

| Form | Setting | is_a | firing_only | null |
| --- | --- | --- | --- | --- |
| rank rule | k = 5 | 1.00 | 1.00 | 0.013 |
| rank rule | k = 2, the strictest possible | 1.00 | 1.00 | 0.001 |
| S_res value | median | 0.380 | 0.315 | 0.000 |
| S_res value | p1 / p99 | 0.328 / 0.453 | 0.266 / 0.372 | +/- 0.13 |
| G, decoder cosine | median | 0.480 | 0.000 | +/- 0.12 |

k = 1 is impossible: the rule requires the child's own decoder and the parent's.
A value cut near 0.35 would split the two distributions in these two toys, but they overlap (is_a p1 = 0.328 is below firing_only p99 = 0.372), and the same class reads 0.315 in only_firing against 0.042 in only_superparent, an eightfold shift driven by the parent's firing density rather than by geometry.
So no constant inside the current rules separates the classes, and a value threshold would not transfer between worlds even where it works.

The decoder cosine does separate them, cleanly and identically in every world, and it is bounded and dictionary-size independent, so a fixed cut would carry to Gemma.
It is a reported diagnostic here and is read by no rule, which is a deliberate decision recorded in Section 4.

### The damage matrix, three seeds, measured 2026-09-20

333 runs: 111 dial points on each of seeds 1, 2 and 3, at 50k tokens, written under the criteria declared in Section 9 before any of these numbers existed.
Seed 0 was run separately as a regression check against the ceiling recorded above and is not quoted as evidence, because it selected `sres_rank_top_k` and the `rule_topical` clauses.
All three seeds met every declared criterion, 15 of 15 each.

Replication is near-exact.
Every rule reproduces to three decimals on all three seeds, with one exception: `rule_firing_only` on `only_superparent` reads 0.833, 0.917 and 0.750, which is 10, 11 and 9 of 12 pairs.
Every null false-positive rate is 0.0000, on every rule, in every toy, on every seed.

### Recall was the wrong instrument: the set does not discriminate

The null rate of 0.0000 is flattering rather than informative.
`unrelated` pairs sit at coverage 0.11 to 0.18 against a cut of 0.5, so they are trivially separable.
The informative negatives are the other STRUCTURED classes, and measured against those the set largely fails.

Pass rate on each rule's own target, beside its worst pass rate on a class it does not target, at the oracle ceiling with no damage:

| Rule | On target | Worst off-target | On which class |
| --- | --- | --- | --- |
| `rule_frequency` | 0.250 | **0.000** | nothing; clean |
| `rule_firing_only` | 0.833 | 0.468 | `superparent` |
| `rule_is_a` | 1.000 | **1.000** | `firing_only` |
| `rule_superparent` | 1.000 | **1.000** | `firing_only` and `reversed` |
| `rule_topical` | 0.333 | **1.000** | `is_a` and `firing_only` |

Three specific readings.

`rule_topical` fires more often on classes that are not topical than on the class that is.
It reads `PASSES(strictly_contains) AND PASSES(freq_survives)`, which contains nothing topic-specific, so every true containment edge passes it.
The change recorded above took it from a structural zero to a structural false positive, and the cross-class table is what should have been run at the time.

`rule_superparent` passes `reversed` at 1.000.
A rule that accepts the flipped ordering of a pair is not testing a directional relation.
Its single gate reads an ENDPOINT's out-degree, which is symmetric in the pair by construction.

`rule_frequency` is the only rule in the set with clean specificity: 0.000 on every non-target class in every toy.
Its 0.250 is the directional ceiling, not a failure.

Two structural notes.
Each discriminator partitions containment exhaustively: every containment pair either passes `sres_rank` or fails it, and either passes `freq_survives` or fails it, so no rule in either pair can abstain.
That is the direct cause of the leakage above.
And 25% of `frequency` pairs and 33% of `topical` pairs lie inside containment, which is exactly the recall those two rules achieve; the caps are containment fractions, not rule failures.

### The encoder costs nothing, because the rules barely read it

Oracle and undamaged differ only in how activations are obtained on the same perfect dictionary.
The largest recall shift caused by the encoder alone is **0.0000**, on every rule, in every toy, on all three seeds.

The mechanism is not encoder accuracy.
Of the eight registered metrics, six are bit-identical between the two columns (`coverage_R`, `asymmetry_R`, `pmi`, `token_freq_survival`, `S_res`, `G`) because they are functions of the firing pattern and the decoder directions, both of which are fixed.
Only `recon_2a` and `recon_child_gain` move, and `gate_recon` shows zero flips.
NNLS is solved ON the planted support and `zeroed_rate` is exactly 0, so the realized support equals the planted support.

The correct statement is therefore that the rule set is almost entirely a FIRING-PATTERN test, so an encoder that only errs in magnitudes is invisible to it.
This will not transfer to a trained SAE, where the encoder determines the support and every firing-derived quantity moves.

NNLS also reconstructs BETTER than the true coefficients in every toy and seed (for example 0.0069 against 0.0089 on `only_isa`), because it minimizes squared error on the support while the true coefficients are the generative values.
Reconstruction quality is therefore not evidence of correct recovery.

### One clause carries every damage result

On `only_isa` under absorption, at all twelve dial points, `PASSES(strictly_contains)` alone, `strictly_contains AND sres_rank`, and `strictly_contains AND recon AND sres_rank` return the SAME number.
`gate_recon` and `gate_sres_rank` never change a verdict.
`gate_recon` changes an answer in 25 of about 950 comparisons across the whole matrix, all of them in one cell (`only_superparent`, composition, `union` readout), where its own question is ill-posed because the readout has deliberately merged two features into a shared latent.
`gate_mutually_contains` is read by no rule at all.

Absorption acts only through coverage.
`eta` removes the parent on a share of co-firing tokens, so coverage falls to about 1 minus eta and crosses the cut between eta 0.3 and 0.6; `beta` removes no firing but rewrites the child's row, so the solver attributes parent mass to the child and suppresses the parent's own coefficient, which surfaces as a firing effect only at beta 0.75.
Ten of twelve cells read exactly 1.00 or exactly 0.00: the rules do not degrade, they switch at the threshold.

Splitting acts only through coverage too, and its effect is arithmetic.
Reading a feature on one of k shards gives coverage of exactly 1/k, measured at 0.50, 0.33, 0.25 and 0.17 for k of 2, 3, 4 and 6.
The rule dies when 1/k falls below the cut, so at k above 2.
The single intermediate value in the sweep, 0.522 at k = 2, is a coin flip because 1/2 equals the cut exactly.
The `union` readout restores coverage to its undamaged value at every k and the rules read 1.000 throughout, so splitting SCATTERS information rather than destroying it.
That is an oracle repair, not a capability: the readout is handed the true shard grouping, which a real pipeline would have to discover.

### The geometry channel does not respond to damage either

`gate_sres_rank` is unmoved by every damage applied.
Under absorption it reads 1.000 at all twelve dial points in `only_firing` and 0.056 at all twelve in `only_superparent` — pinned at opposite ends of its range, responding to neither.
Under splitting it is likewise flat.
Absorption at beta 0.75 rewrites the child's decoder row to carry three quarters of the parent direction, and the only gate that claims to read geometry does not move.

The decoder cosine does respond, and separates the two damages:

| | coverage | G |
| --- | --- | --- |
| splitting, k = 2, 3, 4, 6 | falls as 1/k | 0.480, 0.480, 0.480, 0.480 |
| absorption, beta = 0, 0.25, 0.5, 0.75 | unchanged | 0.480, 0.640, 0.745, 0.814 |
| absorption, eta = 0, 0.3, 0.6 | falls | unchanged at every beta |

G is a clean monotone readout of the geometry half of absorption, blind to eta and to k, and the rule set reads only the other sensor.
This is the second independent argument for a fixed-cut G gate; the first is that G is the only measurement separating `is_a` from `firing_only`.
No rule change is made here, per Section 9.

### Caveats on the above

`only_frequency` and `only_topical` reach only 24 pooled corrupted pairs, and their non-trivial cells are the least stable: `rule_frequency` at beta 0.75, eta 0 reads 0.12, 0.38 and 0.75 across the three seeds.
The cells reading exactly 0.00 or 1.00 are unanimous across seeds; the transition cells, which are the interesting ones, are the ones measured worst.

`transitive` and `sibling` are not planted in any pure toy, so no rule is exercised against them here.

`rule_topical` recall is reported above for completeness but should not be read as a measurement of topical detection, since the rule has no topic-specific clause.

The matrix is indexed by (toy, CLASS), not by toy: `only_superparent` carries two planted classes, and absorption damages only its `firing_only` edges while splitting as configured damages only its `superparent` pairs.

Leakage under damage is NOT measured: the damaged cells report target pass rates only.

## 4. Registry: the committed expressions

Every rule decides on GATES, never on a metric.
A gate is a fixed-threshold decision, tristate over pass / fail / not measurable, defined in `metrics/rules/gates.py` and listed in Section 5.
The rules are defined once, in `metrics/rules/rules.py`, and graded by `metrics/rules/grading.py`; both are shared with the Gemma pipeline, and `scoring/` imports them.
This replaced the null-quantile predicates, which could not travel to a real SAE: a q99 cut accepts 1% of whatever population it is pointed at, and the false-positive bar it produced could not fail.

Recommended freeze set: five designated expressions, the containment baseline, and two historical probe comparators.

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

`FAILS` is written literally, never as `NOT PASSES`: a gate that was never measurable must satisfy neither predicate.
Keep primary code labels separate when reporting the baseline's combined target.
Four of the eight rules read the probe (`gate_sres_rank`), so a no-probe run marks those INVALID MEASUREMENT rather than reading an absent probe as a rejection.

What was deleted, and what carries its work:

- `C` was `HIGH(coverage_R) AND HIGH(asymmetry_R) AND HIGH(pmi)`. Its first two clauses are `gate_strictly_contains`: containment one way and not the other, at a fixed tau, on a supported pair.
- The PMI clause has no successor. PMI is computed and reported; no fixed constant for it exists in `config.py` or `metrics/`, so it decides nothing. The support guard plus the fixed tau do less of its work, and the leak rates say how much less.
- G is dropped from every RULE at the team's decision, and is DELIBERATELY NOT REINSTATED (decided 2026-09-19), so the benchmark's rule set matches the set the team is using. All four geometry rules therefore rest on `gate_sres_rank`, Tree SAE's top-k rank rule, which carries no numeric threshold.
  G is still COMPUTED and reported as a diagnostic column in every artifact (`reads.py` writes it from the decoder cosine). That is what makes the Section 3 contrast measurable: the rules cannot separate is_a from firing_only, and the column shows a measurement that can. Keeping the column costs one cosine per pair and removes nothing from the team's set.
  The consequence is registered rather than hidden: `rule_firing_only` and `rule_firing_only_no_recon` are expected to read 0 recall on every read, and `rule_is_a` is expected to leak onto firing_only. A proposal to add the cosine back as a fixed-cut gate is future work, outside this freeze.
- Names carry no version since 2026-09-23. A change to any rule, gate or constant bumps `metrics.rules.RULESET_VERSION` (now 1), which every artifact written since then records beside `gate_constant_set`.

Rules and gates were renamed on 2026-09-23 without changing any decision, and the stored gate-era results (`MATRIX`, `MATRIX-P3`) were renamed with them; the old names are in git history and in `outputs_archive/gate_era_original_names.tar` on soar-gpu.
Schema-2 results predate the gate rules and keep their own names.
`gate_contains`, one-way containment and the Gemma pipeline's cross-block edge, was added at the same time; no rule reads it yet.

### Changes applied 2026-09-19, with the measurement behind each

Eyeballed from the oracle-ceiling read in Section 3, not swept.
They changed a constant and a clause, never a metric definition.
Items 1 and 2 are IN THE CODE; item 3 is a decision to leave something alone; item 4 lists what was not touched.

1. **`sres_rank_top_k` 5 to 2. APPLIED, and it did not buy the separation.**
The gate requires the child's own decoder AND the parent's to rank inside the child probe's top k.
The probe is fitted on the child's firing, so the child's own decoder is essentially always rank 1, which means k = 5 admits any parent in ranks 2 to 5.
At the ceiling that passed 100% of is_a AND 100% of firing_only pairs, so the rule separated nothing.
k = 2 is the strictest non-trivial form: the parent must be the child's single strongest competitor after itself.
Measured after the change: both classes still pass 100%, and the null fell from 1.3% to at most 0.1%.
The constant is kept at 2 because it is strictly tighter on the null at identical recall, and `config.SRES_RANK_TOP_K` stays 5, so the departure is recorded here and in `scoring/core/registry.py`.
The separation failure is now a registered negative result (Section 3), not an open tuning question.

2. **`rule_topical` is now `PASSES(strictly_contains) AND PASSES(freq_survives)`. APPLIED; recall went 0.00 to 0.33.**
`gate_mutually_contains` requires coverage at or above tau in both directions; the toy plants a register at 0.9 and members at 0.32, so no topical pair is ever a duplicate and the rule reads 0 by construction.
No tau fixes this: any cut below 0.32 also makes every is_a pair a duplicate.
Measured at the ceiling, the ordered register-to-member pairs are exactly the 33% of the topical class that passes `strictly_contains`, they pass `freq_survives` 100%, and the frequency class fails it 100%.
So the two confounds separate cleanly on the survival gate alone, and each reaches its ceiling on the direction the toy actually plants.
`gate_mutually_contains` is now read by no rule; it stays registered and reported.

DECIDED 2026-09-19: the 0.25 and 0.33 caps are ACCEPTED and the target populations stay symmetric.
A directional rule on a symmetrically labelled class cannot exceed those numbers, and the alternative costs either directional labels in `toygen/labels.py`, which moves the answer key and `evaluator_sha256`, or a reporting-layer population built from the tree.
The cap changes no comparison: every arm and every dial point carries it, so the dose-response, the corrupted-versus-intact gap and the leak rates are unaffected.
What it does break is the absolute recall bar, so Section 8 records that `BAR_RECALL` does not apply to those two rows; they are reported as a curve and a cap, never as MET or DID NOT MEET on recall.

3. **`recon_rel_gain_min` stays at 0.01, marked inert.**
At the ceiling it passes about 100% of targets AND about 100% of nulls, so it neither helps nor hurts there.
It is kept because its job is to reject pairs where an endpoint carries no reconstruction mass, which is a damaged-dictionary and trained-read condition, not a ceiling one.
If the one-seed run shows it still inert under damage, drop the clause rather than tune it.

4. **Unchanged, because the ceiling says they work:** `edge_tau` 0.5 (null pass rate 0.0 in every toy), `superparent_outdeg_frac` 0.30 (recall 1.00, null 0.00), `freq_survival_min_raw` 0.5 (frequency 0% pass, topical 100% pass), `min_fire_count` 20 and `support_min_joint` 30.

Development history remains disclosed:

- The earliest oracle S_res read was decoder cosine; it is not interchangeable with the probe.
- The probe's p99-based overlap-specific predicates failed on seed 0.
- G was introduced as a separately named existing-formula measurement, not a repaired S_res, and is now dropped entirely.
- Frequency dropped the symmetry clause after the bounded cached comparison: 236/240 trained targets without it versus 120/240 with it.
- Topical kept symmetry under the quantile rules; under the gates the symmetric form cannot fire at all, which is proposal 2 above.

Known failures may replicate; there is no requirement to find a winning expression for every property.
Do not add new metrics, probe objectives or learned decision trees to rescue a failing rule.

## 5. Instrument definitions

### Metrics

Sixteen registered per-ordered-pair scalars, from `scoring/core/detectors.py` and `scoring/core/registry.py`.
Thirteen are detector keys; `G`, `S_res`, `wide` and `abs_asymmetry_R` are the benchmark's own derived names.
Pin the implementation revision, signs, normalization and orientation, smoothing, support gates, recovery settings, and probe settings.
Do not mix similarly named implementations from `metrics/` with these scores.

| Metric | Definition | Read by |
| --- | --- | --- |
| `coverage_R` | R(p,c) = P(parent fires given child fires) | gate_strictly_contains, gate_mutually_contains |
| `asymmetry_R` | R(p,c) - R(c,p) | gate_strictly_contains, gate_mutually_contains |
| `recon_2a` | relative reconstruction gain from the parent on the child's tokens | gate_recon |
| `recon_child_gain` | the same quantity for the child on its own tokens | gate_recon |
| `S_res` | Tree SAE probe score, min over endpoints of decoder-probe alignment | gate_sres_rank |
| `outdegree` | kept-children count per parent, sign-flipped | gate_high_outdegree |
| `wide` | min of the two orientations of outdegree, the either-endpoint form | gate_high_outdegree |
| `token_freq_survival` | r/(1+r) with r the restricted-corpus coverage ratio | gate_freq_survives |
| `pmi`, `joint_child_J`, `joint_child_supp`, `joint_child_mass`, `sibling_redundancy`, `sibling_redundancy_pc`, `G`, `abs_asymmetry_R` | as defined in the registry | NO GATE: reported as diagnostics only |

Eight of the sixteen decide nothing.
They are persisted at full precision and reported in `metric_diagnostics`, and a reader must be able to tell a reported diagnostic from a number a verdict rests on.

### Gates

Eight fixed-threshold decisions, defined once in `metrics/rules/gates.py` and checked against the functions in `metrics/` they share a rule with.
Each is tristate: 1.0 the rule holds, 0.0 it does not, NaN it was never measurable.
The constants are eyeballed values, not derived ones, exactly as Chanin's absorption paper sets its cutoffs and says so.
They are the named set `SYNTHETIC_TOYS` in `metrics/rules/constants.py`; the Gemma pipeline uses `GEMMA_MATRYOSHKA` from the same file, which differs in `fire_threshold` (1e-3) and `sres_rank_top_k` (5).
Relocating the arbitrariness somewhere visible does not remove it; the defence is that one fixed rule applies identically to every arm of a comparison.

| Gate | Rule | Constants | Defined where |
| --- | --- | --- | --- |
| `gate_support` | both endpoints fire at least `min_fire_count` times and they co-fire at least `support_min_joint` times | 20, 30 | every pair |
| `gate_contains` | `R(p,c) >= edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_strictly_contains` | `R(p,c) >= edge_tau` AND `R(c,p) < edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_mutually_contains` | `R(p,c) >= edge_tau` AND `R(c,p) >= edge_tau`, on a supported pair | 0.5 | supported pairs |
| `gate_recon` | `recon_2a >= recon_rel_gain_min` AND `recon_child_gain >= recon_rel_gain_min` | 0.01 | both gains finite |
| `gate_sres_rank` | the parent's decoder AND the child's own rank inside the child probe's top k correlations | `sres_rank_top_k` | children whose probe trained |
| `gate_high_outdegree` | either endpoint has out-degree at least `superparent_outdeg_frac * (R - 1)` | 0.30 | endpoints firing at least `min_fire_count` times |
| `gate_freq_survives` | `token_freq_survival >= squash(freq_survival_min_raw)`, equality surviving | 0.5 raw, 0.333 squashed | pairs with a defined survival ratio |

`squash(x) = x / (1 + x)` converts a threshold stated on the raw ratio to the scale the detector reports.
Comparing the raw 0.5 against the reported value would demand a raw ratio of 1.0, twice as strict as intended.

`gate_sres_rank` scores its competitor pool over the SCORED frame, not the whole dictionary.
On the oracle and synthetic reads and on a latent-frame Gemma read the two coincide; on a trained read with recovery attrition the bar is weaker than the paper's, and `n_recovered_features` is the number to watch.

### Theoretical applicability

For probe comparisons, use the same probe formula on true firing with true unit g and on learned firing with learned unit W.
Use an explicitly recorded probe-fitting draw and reuse the fitted directions for scoring.
A probe trained on child firing can exploit parent co-firing even with orthogonal decoders, so perfect latent recovery does not guarantee recovery of an isolated generative direction.
This is not a hypothesis here: it is the measured reason `gate_sres_rank` passes on firing_only pairs at the ceiling (Section 3).

Source: [Tree SAE Section 3](https://arxiv.org/html/2605.07922v2#S3), [Section 5.2](https://arxiv.org/html/2605.07922v2#S5.SS2), [Appendix H](https://arxiv.org/html/2605.07922v2#A8).
S_res measures alignment of both decoders with a third child-concept direction.
The paper's top-five probe ranking is now our rule rather than a contrast, so the remaining difference from the paper is the competitor pool and the all-pairs population.
Exact containment implies oracle R = 1 when the child fires; it does not imply a numerical recall guarantee for a rule after training.
A valid poor score is a benchmark result, not an obligation to improve the metric.

## 6. Populations and required reports

Nothing is fitted, so there is no calibration procedure to freeze.
Every rule compares a gate against a constant from Section 5, and the same constant applies to every world, every arm and every read.
Three consequences, all deliberate:

- The unrelated null is no longer halved. The split existed so a threshold fitted on one half could be measured on the other; with nothing fitted there is nothing to leak, and the false-positive rate is measured on the WHOLE null. Do not reintroduce a split for holdout.
- A threshold is no longer context-specific. The same numbers apply on a 24-feature toy and on a 16k-latent SAE, which is the property this rewrite exists to buy.
- The bar can now fail. Under the quantile rule a q99 cut put about 1% of an exchangeable null over the line by arithmetic, so only recall could ever fail a criterion.

Procedure:

1. Fit the SAE and determine recovery without evaluation observations.
2. For probe measurements, separate the recorded fitting draw from scoring. The fitting draw is `seed + 20000`, scoring is `seed + 10000`, matching is `seed`.
3. Use held-out observations for firing statistics and token-frequency buckets.
4. Freeze the probe directions before the detectors and the gates: `gate_sres_rank` reads them, so a refit between the two would score one rule against directions no other rule saw.

Primary evaluation uses all ordered pairs on the stated endpoint universe, excluding self-pairs, with no coverage shortlist.
This exposes both candidate-selection recall loss and leakage into competing properties.
Keep any paper-style shortlist evaluation separate, with its selection recall and conditional acceptance reported.

For `frequency` and `topical`, state which population the recall is over.
The class label is assigned to both orderings of a pair, while `gate_strictly_contains` is directional, so a directional rule on the symmetric population is capped at 25% and 33% by construction.
Section 4 proposal 2 scores those two targets on their ordered subset and reports the symmetric count beside it.

For each target, read and seed, report:

- N_total: generated ordered target pairs in the stated population.
- N_recovered: target pairs with both endpoints recovered.
- N_scorable: recovered target pairs where every gate the rule reads is measurable.
- N_pass: complete-expression positives among those pairs.

Compute recovery, scorable fraction, recall-given-recovery, and end-to-end recall from these counts.
Missing or unrecovered targets count as end-to-end misses; zero denominators are untestable.
Report the null FPR over the whole unrelated class, each confound's leakage, and each row's recovery and scorability, plus multiple-rule and no-rule outcomes.
Precision requires a stated population and prevalence.
Never select a winning class or expression using the answer key.

The scorability guard is a reported population, not a silent filter.
`gate_support` excludes a pair when an endpoint fires fewer than 20 times or the two co-fire fewer than 30 times, and the excluded fraction is reported by cause, because "the rule rejected these pairs" and "the rule could not see them" are different results.

Survival is `S = r/(1+r)` with `r = R_rest/R_all`; unchanged coverage maps to 0.5.
The restricted corpus excludes the high-frequency bucket and retains the mid and rare buckets.
Zero restricted child support with adequate total support maps to zero by convention, not to an estimated conditional probability.
The boundary is `squash(0.5) = 0.333`, with equality assigned to surviving, the same side the deleted `SURVIVES` predicate used.

`wide(p,c) = min(outdegree[p,c], outdegree[c,p])` on the sign-flipped scores, requiring both values finite.
`wide` and `gate_high_outdegree` are endpoint-broadcast quantities: they are functions of the two endpoints, not of the pair, so they have no pair-level holdout and a per-pair scorability mask must never be applied to them.
Save pair IDs and recovered-feature mappings; do not reconstruct trained reverse-pair identity from unannotated class arrays.

Alongside per-metric distributions, record `constant_null`, `constant_target` and `constant_overall`, with the numerical tolerance and finite counts used.
A point-mass null and a different point-mass target can separate perfectly; that is a structurally simple separation, not evidence a detector discriminates.
The oracle superparent result is exactly this shape and may not transfer unchanged.

## 7. Evaluator work required before freezing

These are implementation and reporting tasks, not changes to the tested metrics.
Keep this checklist open until the listed verification exists.

Closed by the gate rewrite:

- [x] **Literal negative predicates.** `FAILS` is written literally against the finite values, so a gate that was never measurable satisfies neither predicate. `NOT-HIGH` as `~HIGH` is gone with the quantile rules.
- [x] **Clause-specific scorable masks.** A conjunction is scorable only where every gate it reads is measurable; `evaluate` intersects the per-clause masks instead of taking one finiteness vector from the caller.
- [x] **A rule may not read a metric.** `predicates.evaluate` refuses any clause naming something outside the registered gates, so `PASSES(coverage_R)` cannot silently compare a coverage against 0.5.
- [x] **Evaluation-null reporting.** The question is gone with the calibration half: the FPR is measured over the whole unrelated class and `REPORT_SCHEMA = 3` marks artifacts written under that contract.
- [x] **Both directions finite before `wide`.** Retained from the previous revision.

Still open:

- [ ] **Re-run the oracle ceiling after the Section 4 decisions.**
Every designated rule must fire on its own class on a perfect dictionary, with the null rate at zero.
Record the table in Section 3 again rather than replacing it, so the before and after are both visible.
- [ ] **Persist enough exact data to reproduce every expression.**
Save original-precision gate matrices and required metric scores, or sufficient scorable and pass masks, alongside the constants.
A rounded legacy value near a constant would move a decision silently.
- [ ] **Verify instrument names and pair identity.**
Keep probe `S_res` distinct from cosine-mode `G`, verify matching feature IDs and both pair orderings, and retain the matched-oracle control.
Use stable true-feature pair IDs across reads, not recovered-position indices that change after attrition.
- [ ] **Add distribution diagnostics without changing verdicts.**
Record constant-null, constant-target and constant-overall flags, support, tied boundaries, and dictionary or candidate-pool size alongside the four outcome labels.
- [ ] **Finalize output provenance and verify the complete tables.**
Export resolved configs, code revision, all random seeds, recovery mappings, the gate constants, and scoring precision.
Check every rule on every world, including the null and the named confounds.

Verification status at this revision:

- `tests_local` and `synthdict/tests` pass together on the server: 845 passed, 5 skipped (2026-09-19).
- The oracle-ceiling numbers in Section 3 were measured on this code, on all five toys at full size.
- The mutation harness has not been re-run against the gate rewrite; the synthdict half is at 135 of 135 anchors killed, and the scoring half needs anchors for the new gates.
- Passing these suites is not certification of the benchmark while the checklist above remains open.

## 8. Settings to approve and freeze

The bars, the support floor, the designated rules and the grading arithmetic live in `metrics/rules/grading.py`, shared with the Gemma pipeline.

Complete this manifest before any fresh-seed result is inspected:

| Setting | Current proposal / action needed |
| --- | --- |
| Candidate registry | Five designated expressions, C baseline, and two historical probe comparators in Section 4; approve exact inclusion |
| Metric definitions | Pin canonical implementation revision, score modes, signs, normalization, smoothing, and support gates |
| Gate constants | `edge_tau` 0.5, `min_fire_count` 20, `support_min_joint` 30, `recon_rel_gain_min` 0.01, `superparent_outdeg_frac` 0.30, `freq_survival_min_raw` 0.5 (squashed 0.333), `sres_rank_top_k` 2, applied 2026-09-19 (config.py keeps 5; Section 4). Stamped in every artifact's `gate_constants` |
| Nothing fitted | No quantiles, no calibration half, no per-read threshold block. The null FPR is measured over the whole unrelated class |
| Operational recall bar | Recall-given-recovery >=0.8, EXCEPT `rule_frequency` and `rule_topical`, whose symmetric class labels cap them at 0.25 and 0.33 (Section 4). Those two rows report the curve and the cap; the bar does not apply and no MET or DID NOT MEET verdict on recall is read from them. The verdict function still prints one, because the bar lives in code; the table must carry this note beside it |
| Null-FPR bar | Null FPR <=0.01 in each tested world, measured over the whole unrelated class |
| Confound-leakage bar | Each named complete-expression confound rate <=0.05 |
| Support requirements | `MIN_SCORABLE_SUPPORT = 10`, applying to the TARGET row, the null row, and EVERY confound row. The calibration-support floor is gone with the calibration half |
| Rate denominators | Recall on `N_recovered`; null FPR and confound leakage on `N_scorable`. Deliberately asymmetric - see below |
| Cross-world FPR combination | WORST world, not pooled. The pooled rate is reported beside it as a labelled diagnostic |
| Cross-world leakage combination | POOLED by counts, with the worst world reported beside it |
| Probe fitting draw | Fitted on `seed + 20000`, distinct from the matching draw (`seed`) and the scoring draw (`seed + 10000`) |
| Reporting contract version | `REPORT_SCHEMA = 4` (history below). Stamped into every artifact and required by the manifest check and the cross-world rollup |
| Recovery/end-to-end criterion | Report both; specify an additional bar only if making an operational-recovery success claim |
| Seeds and draws | Record exact three unused world/training seed IDs, fitting/scoring/split seed derivations, and sample sizes |
| SAE setup | Pin per-toy variant, sparsity, dictionary/prefix sizes, training configuration, and checkpoint provenance |
| Baselines | Fix endpoint firing-count comparators and their calibration; do not compare their AUROC directly with expression recall |
| Uncertainty | Report each seed separately and between-seed variation; approve any confidence-bound decision method before using it |

### Report schema versions

`REPORT_SCHEMA` versions the keys `grade_rules` writes and the arithmetic under them.
It is bumped whenever a key's arithmetic changes under an unchanged name, or a rate key is added, renamed or removed.

1. The pilot contract. `recall_given_recovery` was N_pass / N_scorable, with one `fpr` key on the null rows.
2. `recall_given_recovery` moved to N_pass / N_recovered, with the scorable rate split out as `pass_rate_given_scorable`; `fpr` split into `fpr_given_scorable` and `fpr_over_half`; `leakage` read the scorable rate, with `leakage_over_recovered` beside it; the support floor extended to the null and confound rows; `verdict` decides established failures before unmeasurable evidence.
3. The fixed-gate contract. Rules decide on gates against fixed constants, so the null is no longer halved and `fpr_given_scorable` is measured over the whole null; `fpr_over_half` is gone.
4. The shared-rules contract. Every rule and three gates were renamed (Section 4); the arithmetic is unchanged.

### Rate denominators are deliberately asymmetric

Every bar takes the denominator that is HARDER to clear.
Recall rides on `N_recovered`, so a target pair the rule could not score counts as a miss: a rule that cannot measure a pair has not found it.
An FPR or a leak rides on `N_scorable`, so an unscorable null or confound pair does not dilute the rate.
Making the two symmetric would let a rule clear the leakage budget by rendering confound pairs unmeasurable, which is the rule being rewarded for the pairs it failed to evaluate.
Both denominators are reported for all three quantities under names that say which is which, so a reader can always see the one the bar did not use.

### The support floor guards the negative evidence too

A floor on the target alone is not enough.
An adversarial review reached MET CRITERIA off a one-pair null and a one-pair confound row, both of which cleared their bars arithmetically without establishing anything.
The floor therefore applies to the null row and to every confound row as well as to the target.
An under-supported negative row BLOCKS a pass without RESCUING a failure, and that asymmetry is deliberate: a rule whose own recall failed on well-supported evidence still failed, but a rule cannot be certified as passing over evidence nobody could measure.
Measured across all 80 read-expression-class rows on seed 0, zero rows have `0 < N_scorable < 10`, so this floor changes no seed-0 verdict; it is a guard for the fresh seeds and the complex toy.

### The cross-world FPR is the worst world, not the pooled rate

Section 8 words the null-FPR bar as holding "in each tested world", so the quantity that faces it is the worst one.
A pooled rate lets four quiet worlds absorb one loud one.
This is not hypothetical: measured on the saved seed-0 arrays, `rule_topical` flips at the 0.01 bar on BOTH reads depending on which rule is used.
Leakage is pooled by counts instead, because the confound bar has no per-world wording.
The anti-pooling argument applies to leakage in exactly one place: `reversed` is the only class the five toys generate in more than one world, so it is the only row where pooled and worst-world can differ at all.
Both are reported either way.

The numerical bars are proposed operating criteria, not mathematical guarantees or conditions for publishing a result.
Do not loosen them because a metric fails.
Shared-feature/topic pairs are not independent replications; p10-p90 score ranges are not confidence intervals.
If formal confidence-bound decisions are used, predeclare a dependence-aware method and an inconclusive category.

Use MET CRITERIA, DID NOT MEET CRITERIA, UNTESTABLE, and INVALID MEASUREMENT as distinct labels.
Attach theoretical applicability and distribution diagnostics separately.
A missing trained superparent class has zero end-to-end recall and an untestable conditional detector, not perfect specificity evidence.
A valid negative score is not INVALID MEASUREMENT.

For endpoint firing-count baselines, use calibration-only thresholds at the same nominal null budget and report achieved FPR, recall, and leakage on the same population.
Matching a baseline is a valid result; a superiority claim requires measured added discrimination and uncertainty.
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

### Complex toy

Keep the metric and expression identities fixed and apply only the registered calibration adaptation.
Before viewing that stage, specify additional classes such as siblings/transitive pairs, nuisance settings, and any overlapping-label scoring.
Evaluate whether the pure-toy null separation, C's sensitivity/confounds, and the G-rule failures change when mechanisms coexist.
Do not assume pure-toy success must transfer or pure-toy failure must persist.
Label any expression changes as a separate development study rather than transfer of the original rule.

### Gemma

Define an independent annotation/ground-truth target and a justified null protocol before reporting semantic precision/recall.
The toy's answer key is unavailable, and random feature pairs are not automatically known unrelated examples.
Until those choices are fixed, report applicability and candidate relations rather than semantic correctness.
Do not equate recovery of planted geometric labels with validation of semantic is-a.

### Parked outside this study

- Finding a new separator for the two containment subtypes.
- Cross-only S_res, nuisance-conditioned probes, redesigned metrics, and learned decision trees.
- Causal attribution of decoder changes to a particular training objective without a suitable comparison.
- Additional diagnostics that do not affect validity or interpretation of the reported benchmark.

The min-component diagnostic remains optional if needed to explain an existing result.
Algebra already gives `min(cross,self)>t` only when both terms exceed t, so dropping the self term cannot remove those false positives at an unchanged threshold.
No further rule search is needed to close this pilot.

## 11. Earlier evidence and provenance retained

The old `alpha_designed=0.48` fields are nominal config values, not effective geometry for every toy.
The G_W run records effective alpha=0.48 on 120 only_isa edges, alpha=0 on 120 only_firing edges, and no generated containment edges for the three confound-only worlds.
Keep old artifacts with an explicit correction rather than claiming their numerical scores were computed with the wrong geometry.

The old only_isa cache's median `cos(W_child,g_parent)=-0.215538` and `cos(W_child,g_child)=0.693478` are different measurements from G_W.
The inference G_W ~= -0.216 was unsupported without learned-parent alignment; direct G_W measurement supersedes it.
The separate only_firing absorption census reports 11 absorbed edges even though the is_a-filtered legacy `absorb__*` arrays are empty.
Neither an empty export nor a changed score alone establishes absence or presence of absorption.

Sources for the earlier probe check are [probe_check.json](outputs_local/probe_check/probe_check.json) and its [tables](outputs_local/probe_check/probe_check.md).
The historical firing_only NOT-HIGH probe clause passed 0/120 oracle and 13/120 trained targets; the topical IN-BAND probe clause passed 19/56 oracle and 32/56 trained targets.
Those counts bound complete-expression recall on the same populations/thresholds and explain why the failed comparators remain in the record.
The trained probe pilot calibrated on its whole saved null because pair identity was absent; those null rates were not held-out validation.
The staged per-toy probe JSONs were previously checked against their merged file and matched.

This document supersedes overstatements in the earlier findings without overwriting the source artifacts.
It records the G_W negative result, C's actual confounds, and the finite remaining work needed to freeze an accurate benchmark.
