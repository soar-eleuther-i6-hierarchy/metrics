# `metrics_v2/` — the same metrics, arranged by the question they answer

`metrics/` grew one file per metric in the order they were written, with numbered names
(1a, 1c, 2a, 2b, 9). This package holds the same functions under two questions:

| Package | Question | Files |
| ------- | -------- | ----- |
| `structure/` | is this candidate edge a real parent → child? | `containment`, `independence`, `frequency`, `contribution`, `geometry` |
| `pathology/` | if not, which pathology produced it? | `splitting`, `absorption`, `composition`, `parent_coverage`, `degree` |

Decisions (gates, rules, grading, constant sets) stay in [`metrics/rules/`](../metrics/rules/),
which both pipelines import; `metrics_v2.rules` is that package, not a copy.

**Phase 1 (this state).** Every function is the *identical object* as its `metrics.*` original.
No formula moved, no default changed, so every published number reproduces from either package.
`tests/test_metrics_v2.py` asserts identity (`is`) for every row of the table below.

## Old name → new name

| `metrics/` | `metrics_v2/` | Why the name |
| ---------- | ------------- | ------------ |
| `coverage_legs` | `structure.containment_ratios` | returns R = P(p\|c) and F = P(c\|p) |
| `keep_edges` | `structure.candidate_edges` | it defines the candidate set |
| `coverage_asymmetry` | `structure.containment_asymmetry` | R − F |
| `directed_coverage` | `structure.in_block_containment` | containment inside one block |
| `independence_scores` | `structure.pmi_scores` | PMI and Dev |
| `frequency_buckets`, `local_frequency_buckets` | same names | |
| `frequency_controlled_coverage` | `structure.frequency_survival` | the ratio it returns |
| `edge_reconstruction_condition` | `structure.contribution_gains` | a contribution filter, not S_res |
| `per_token_ablation_gain` | same name | |
| `train_probe` | `structure.child_probe` | fitted on the child's firing |
| `sres_rank_check`, `sres_scores` | `structure.refinement_rank`, `structure.refinement_scores` | Tree SAE's refinement test |
| `negative_parent_composition` | `structure.probe_negative_parent_share` | "composition" here never meant the pathology |
| `sibling_redundancy` | `pathology.sibling_overlap` | mean pairwise Jaccard |
| `parent_conditioned_redundancy` | `pathology.sibling_overlap_within_parent` | restricted to the parent's tokens |
| `duplicate_pairs` | `pathology.coextensive_pairs` | R ≥ τ both ways |
| `share_energy` | `pathology.child_energy_share` | |
| `r_supp`, `r_mass` | `pathology.parent_support_covered`, `pathology.parent_energy_covered` | how much of the parent its children cover |
| `joint_child_coverage_exact` | `pathology.parent_support_covered_exact` | same ratio from a union count; drift guard |
| `degree_stats` | `pathology.degree_summary` | |
| `kept_outdegree`, `either_endpoint_outdegree` | `pathology.parent_outdegree`, `pathology.pair_max_outdegree` | |
| `find_superparents` | `pathology.dense_parents` | flagged on out-degree alone |

Dropped, not renamed: `joint_child_coverage_upper` (min(1, ΣF) double-counts co-firing children
and saturates near 1; the exact form replaces it).

## New, and not calibrated

Three functions exist only here. No pipeline calls them and no toy has graded them yet.

| Function | Reads | Signature it reports |
| -------- | ----- | -------------------- |
| `structure.decoder_cosine` | two decoder matrices | cosine between every parent and child row |
| `pathology.absorption_signature` | R and that cosine | aligned decoders with failed containment: cos ≥ cos_min and R < τ. Absorption as injected in the paper (hole η, carry β) moves exactly these two numbers in opposite directions; a healthy edge is R high, cos low |
| `pathology.composition_signature` | kept edges and the parent block's co-firing | a child with two kept parents that are neither nested nor co-extensive. This is the *conjunction* reading of composition (a latent on the intersection of two surviving parents); the *merge* reading (two concepts become one latent) reduces through coverage to a small absorption hole and has no signature of its own |
| `pathology.direction_duplicates` | one block's decoder rows | pairs of latents whose directions coincide (|cosine| ≥ cut, sign and scale ignored): the sharded form of splitting, which firing overlap cannot see. On the synthetic shard toy a 0.9 cut recovers every planted pair with no false positive; on a trained SAE the cut needs its own null |

Both are reports, never cuts. Absorption is outside the paper's precision goal (an absorbed edge
never enters the candidate set), so these stay out of every rule until a toy grades them.

## Two constructions of "splitting", and which detector sees which

The repository plants splitting in two different ways, and the supporting metrics do not see
both:

| Construction | Where | Firing overlap (`pathology.sibling_overlap*`) | Decoder cosine (`structure.decoder_cosine`) |
| --- | --- | --- | --- |
| **copies**: latents that fire on the *same* tokens | `validation/synthetic_toy_world.py` (Tier-1 split parent) | 1.00 vs 0.00 on a genuine parent | not needed |
| **shards**: latents that share one *direction* and *partition* the tokens | `synthdict.corruptions.split` (the synthetic-dictionary toys) | 0.00 by construction: blind | 1.00 vs at most 0.12 between unsplit parents |

`validation/split_detectors.py` runs the shard construction on the `only_isa` toy and reports
both signals per `k` (`outputs/metrics_v2/split_detectors.md`). The consequence for the paper:
"sibling redundancy detects splitting" is true for duplicated latents under one parent and
false for a feature sharded into disjoint pieces; the second is what decoder cosine reads. Both
detectors belong in `pathology/splitting.py` eventually; the cosine form is not yet a rule.

## What this does not settle

Four decisions carried over from the review of the shared-rules PR, to be answered here rather
than by changing a default in `metrics/`:

1. survival floor: since the shared-rules PR's review fix, `frequency_controlled_coverage` takes
   `no_rare_firing_scores_zero`; the Gemma pipeline leaves it off (a child that never fires on
   rare tokens is untestable), the synthetic benchmark turns it on (it scores 0 and fails).
   One flag, two pipelines, both recorded;
2. a block pair with no edges: `degree_summary` returns 0.0 for poly-parenting, top-1 share and
   Gini, the values of a healthy spread; NaN or skipping would keep it out of averages;
3. a width-independent dense-parent cut: 30% of the child block is 115 children on gemma's
   B0→B1 and 3,932 on T-SAE's, which is unreachable, so T-SAE's zero superparents is a
   block-width effect;
4. decoder cosine as a rule input: one working document proposes it for is-a, the precommit file
   records it as a negative result kept out of every rule.

## Other architectures

Nothing here knows about Matryoshka. Block boundaries come from the caller (the T-SAE adapter
passes `[3276, 13108]` from the checkpoint), and every function takes tensors. The one caution:
a temporal block is built to co-fire across neighbouring tokens, so mutual containment inside it
is a property of the architecture before it is read as splitting.

## Phases

1. **Now**: re-exports under the new names, identity-tested. `metrics/` untouched.
2. Move the code here, leave `metrics/` as a thin facade importing from here, update
   `run_metrics.py`, `run_token_metrics.py`, `in_block_edges.py`, `validation/` and `reporting/`.
   `scoring/` needs nothing: it imports only `metrics.rules` and `metrics.sres.train_probe`.
3. Calibrate the three new functions on the absorption and composition toys, then decide whether
   either enters a rule.

Keep the table above forever. A previous rename dropped its old-to-new table and made every later
diff harder to read.
