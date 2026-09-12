# Round-2 design notes (decisions taken in discussion; NOT implemented)

Captured 2026-09-12 for reference when round 2 is built.
Nothing here has been coded, pre-registered, or run.
A `SYNTH_PRECOMMIT` round-2 section must be written and approved before any of it executes.

## Decision 1 - the synthetic path is MATCHER-FREE

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
