# Round-2 design notes (decisions taken in discussion; NOT implemented)

Captured 2026-09-12 for reference when round 2 is built.
Nothing here has been coded, pre-registered, or run.
A `SYNTH_PRECOMMIT` round-2 section must be written and approved before any of it executes.

## Decision 1 - the synthetic path is MATCHER-FREE

> **IMPLEMENTED 2026-09-12.** `synthdict/planted.py` holds `PlantedMap`; `match_mode` is now
> `readout`; the matcher imports are gone from `read.py` and `census.py`, enforced by a
> structural guard with a positive control (`test_planted.py`). Two accessors, deliberately
> distinct: `columns()` is POSITION-indexed (the scored frame) and `feature_lookup()` is
> FEATURE-indexed (what `classify_dictionary` needs) - conflating them is invisible under an
> identity map and wrong the moment a feature is missing, so a test drives the gapped case.
> A shared latent is refused at construction (that is a merge, and it needs its own readout).
> Matcher grading against the planted key remains UNBUILT and is still the recorded seam.

Matching solves an INVERSE problem: someone else built the dictionary and you must infer which latent corresponds to which feature.
Synthesis has no inverse problem - we build the dictionary, so the correspondence is an INPUT, not an inference.

So round 2 drops `match_features` / `activation_corr` / `reduce_to_recovered` from the synthetic path entirely and replaces them with a planted lookup written at construction time.

What that changes, concretely:

- **Planted lookup.** The corruption record carries `feature -> [latent ids]` for every feature, populated when the damage is applied. For 1-1 damages (absorption, hedging, noise, missing-latent) this is the identity map; splitting and merging are the only many-to-one / one-to-many cases.
- **Declared readout policy.** A lookup entry with several shards does not tell the scorer which single row/column to consume, and the metrics need exactly one. So the readout is DECLARED per cell, stored beside the shards (e.g. `{feature 7: shards [7a,7b,7c,7d], readout: "union"}`), and the scorer only does a dictionary lookup. This is a measurement decision made at construction time, NOT an inference - which is what makes it matcher-free.
  For split cells report BOTH policies: `strongest_shard` (what a 1-1 pipeline effectively sees) and `union` (what the dictionary actually contains). The gap between them IS the measurement of what splitting costs the metrics, with no matcher noise in it.
- **Recovery becomes declarative.** Today "recovered" means "matched with corr >= rho_star". On the synthetic path every planted feature is present by construction, so the synthetic read reports NO recovery loss. That is correct: recovery loss is a property of training, and there is no training here.

**Recorded scope reduction (deliberate, not an oversight):** round 1's P5 comparison ("identity == hungarian at all 46 dial points") came from running both modes. It disappears from the synthetic side under this design. The matcher's real weakness - a split feature reading as "missing" - is better measured where it actually bites, on the TRAINED side, where it already was (superparent 0/726).

**Division of labour this settles:**

| path | correspondence | measures |
| --- | --- | --- |
| synthetic | planted lookup + declared readout | the metric's transfer function, zero pipeline confound |
| trained toy | matcher (no alternative: the dictionary is not ours) | metric + pipeline, and the matcher as its own object of study |
| Gemma | none exists | pairwise over latents; only null-calibrated statements survive |

## Decision 2 - what a cell buys, and the claim it supports

"Survives" always has a hidden "up to how much damage". Three claim sizes:

- **A (binary):** "E survives damage at the level real training produces." 2 points (zero + trained anchor). Expires if a future SAE is more damaged than the toy anchor.
- **B (tolerance):** "E works up to severity s*, fails beyond." Needs a sweep. This is the claim that TRAVELS - measure a new SAE's damage level, compare to s*. A falls out of B for free by marking the anchor on the curve.
- **C (no-signal):** "this instrument carries no signal for this pathology at ANY severity." Only a sweep can license the words "at any severity"; round 1's S_res result is this shape.

You do not choose B or C in advance - one sweep per cell, and the curve's shape decides which claim it supports.

**Rule adopted:** SWEEP (zero + ~4 levels) wherever an expression's own TARGET is under attack; two anchored points where the question is genuinely binary (does damage X make unrelated rules false-fire). Mark the trained-measured level on every curve - that line converts a curve into a gate verdict.

## Decision 3 - the cell matrix

**TEST** = new cell to run · **DONE** = verdict extractable from round-1 artifacts, no new run · **skip** = named covering cell · **-** = impossible to construct.

| | absorption | hedging | splitting | merging | missing latent | noise |
| --- | --- | --- | --- | --- | --- | --- |
| only_isa | DONE | DONE | **TEST** (sweep) | skip (= absorption's eta=1 end) | skip (superparent covers) | **TEST** (2 pts) |
| only_firing | DONE | DONE | skip (only_isa covers) | - | skip | **TEST** (2 pts) |
| only_superparent | - | - | **TEST** (sweep, headline) | - | **TEST** (2 pts) | skip |
| only_frequency | - | - | **TEST** (2 pts) | skip (sibling covers) | skip | skip |
| only_topical | - | - | skip (frequency covers) | skip | skip | skip |
| sibling world (NEW config) | skip | skip | skip | **TEST** (sweep) | skip | skip |

Why the dashes are facts, not choices: absorption needs a parent-child edge (superparent/frequency/topical worlds have none); merging in only_firing is degenerate (a latent firing "when parent AND child fire" fires exactly when the child does, so it IS the child latent).

Why the headline is superparent x splitting: it reproduces a MEASURED trained outcome (always-on features split across 7-18 weak latents), turning a one-off trained mystery into a curve. Its neighbour, superparent x missing-latent, plants the competing explanation (mass into the decoder bias) so the two have distinguishable fingerprints.

## Decision 4 - anchors come from the trained side

Every non-arbitrary damage level is a trained measurement, never a choice:

- splitting: 8 shards (superparent also 16) - measured range was 7-18.
- absorption/hedging: severity 0.48 on 11/120 edges - already covered by round 1's bridge point, so no new runs.
- missing latent: the two competing explanations of the trained superparent.
- merging: saturating (no trained measurement exists to anchor to).
- noise: sigma matched to the measured trained decoder spread (G_W).

## Code changes this implies (for whoever builds it)

- `corruptions.py`: each corruption returns a planted `feature -> [latent ids]` lookup; `CORRUPTIONS` registry gains `split`, `merge`, `missing`, `noise`; splitting/merging change the latent count, so `W_raw` is no longer `[F, D]`.
- `read.py`: `match_mode` is replaced by `readout` (`identity` | `strongest_shard` | `union`); the `hungarian` branch and its three imports leave the synthetic path; `recovered` is all-True by construction; `synth_encode` produces `[n, L]` with L != F.
- `run_synth.py` / `report.py`: stamp `readout` and the lookup hash in meta; group by it as `acts_mode` is grouped today.
- `census.py`: the census locates latents through `match[c]` / `match[p]`, so it needs the planted lookup instead - and note the known limit already recorded, that absorption-of-a-split-child is invisible to a per-latent cosine test (carry spread over shards, each below eps).
- Tests: the passthrough anchor stays the load-bearing gate; add lookup-integrity anchors (a shard set that does not cover its feature's firing must fail loudly) and readout-policy anchors (strongest vs union must differ where they should).

## Open, needing a decision before round 2 runs

- Write the `SYNTH_PRECOMMIT` round-2 section: per-cell bars, the readout policy per cell, the verdict vocabulary (SURVIVES / FAILS / INVALID), and the downstream rule (what a FAILS verdict obliges in the PCFG / Gemma stages).
- Seeds: 2-3, only on the numbers that get quoted.
- The sibling world config (branching 2-3, depth 1) exists only as a plan; it also revives the `sibling` and `transitive` pair classes, which have no registered expression yet.

## Evidence the synthetic construction itself is sound (round 1, measured)

Recorded because these are what license using synthetic dictionaries at all - separate from any finding ABOUT the metrics.
Every item is a measurement from the round-1 artifacts or the committed test suite, not an assertion.

**The undamaged synthetic read IS the benchmark pipeline.**
The passthrough anchor (no corruption, true-A activations) reproduces the real `scoring/benchmark/reads.py::oracle_read` BIT-FOR-BIT across all 13 metrics, same pairs, same labels, same decoder rows.
So anything the corrupted reads show is attributable to the planted damage, not to a re-implementation drifting from the pipeline it claims to test.
It doubles as the proof that `signed_normalized_decoder` is a no-op on planted dictionaries.

**The activation model is honest at zero damage.**
Ridge magnitudes on the planted support land at the noise floor (FVU within 0.02 of true-A) with a support-flip rate under 0.005 (exactly 0 in clean mode).
So the encoder substitute is not quietly changing the firing structure before the metrics see it.

**The dials do what they say, checked against closed form.**
`G` on corrupted edges equals the planted severity exactly (identity mode); realized severity matches beta/sqrt(1+beta^2) on orthogonal edges to 1e-6; `coverage_R` on corrupted edges equals (1-eta) exactly in clean mode (1.0, 0.8, 0.6, 0.4, 0.2, 0.0 along the diagonal).
These are manipulation checks, not results - and they are the reason a flat or non-monotone metric response can be read as a property of the METRIC rather than a broken generator.

**The corrupted/intact split is real.**
Intact edges hold G ~ 0 while corrupted edges move with the dial, in the same world, same run, same thresholds.
Corrupted + intact partition the recovered target class exactly in every artifact row.

**An independent instrument agrees with the planting.**
The census (`classify_dictionary`, written for trained checkpoints, not for this study) counts exactly the planted set - zero false positives outside it across all 92 artifact rows, and exactly 11/11 at the bridge point.
Cross-check: severity = sin(theta_hat) to 2.8e-16, two quantities computed by different code paths.

**The synthetic read reproduces a pre-existing, independently measured defect.**
`topical_v6` false-fires at ~0.010 on the evaluation null at EVERY dial including zero corruption, in both toys - the same off-world FPR violation the benchmark had already recorded on trained reads (PRECOMMIT s3).
The synthetic pipeline was not tuned to produce this; reproducing a known artifact it did not aim at is external evidence the read is faithful.

**The matcher adds nothing when the map is 1-1 - the empirical basis for Decision 1.**
Identity and Hungarian columns are byte-identical at all 46 dial points, and recovery is 240/240 at every setting including maximal damage.
That is what makes dropping the matcher from the synthetic path a scope reduction rather than a loss of information.

**Verification discipline behind all of the above.**
51 tests and 33/33 mutation anchors killed on the committed state; three independent reviews (adversarial correctness, instrument validity, test quality) closed; the frozen `tests_local` suite unaffected and `evaluator_sha256` unchanged, so nothing in the benchmark moved.
The review process also caught the one construct defect BEFORE any interpretation (beta leaking into the firing channel via the unconstrained ridge), which is what produced the two-activation-model design - evidence the loop works, not just the code.

**What none of this establishes:** that the planted damages resemble what training produces (only the trained side can say), or that the five planted properties cover the reasons real features co-occur.
