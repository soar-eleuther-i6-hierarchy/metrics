<style>
.x0nav{position:sticky;top:0;z-index:999;background:#fff;border-bottom:1px solid #E3DAFB;
font:500 13px/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 14px;}
.x0nav .row{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding:9px 18px;}
.x0nav .row+.row{border-top:1px solid #F1ECFD;}
.x0nav a{text-decoration:none;color:#5A6B7B;}
.x0nav a:hover{color:#7C22CE;}
.x0nav .brand{font-weight:700;color:#7C22CE;letter-spacing:.2px;}
.x0nav .on{color:#7C22CE;font-weight:700;}
.x0nav .lbl{color:#9AA7B3;font-size:11px;text-transform:uppercase;letter-spacing:.7px;}
.x0nav .pill{border:1px solid #E3DAFB;border-radius:7px;padding:5px 10px;background:#F6F3FE;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{width:1px;height:17px;background:#E3DAFB;}
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;margin-left:auto;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
.x0nav details.dens{flex:1 1 100%;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;display:inline-block;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
.x0nav .drow{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding-top:9px;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="" href="../outputs/">Results</a><a class="" href="../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# `metrics/` — the metric library

One file per metric. Every metric is a **pure function over cached statistics** (co-firing counts,
per-edge reconstruction sums, token-bucket counts, decoder vectors) with no model and no IO, so the
identical code runs on the real gemma-2-2b caches, on the synthetic toy in [`validation/`](../validation/), and
on the trained toy SAE.

They are **competing measurements of the same edge**. The candidate edge *set* comes from Metric 1;
Metrics 2–7 grade those edges. A "real" edge has to survive all of them.

| File | Metric | Stage | What it adds |
| ---- | ------ | ----- | ------------ |
| [`coverage.py`](coverage.py) | 1a/1b — reverse + forward coverage, `keep_edges` | 02 | defines the candidate edge set |
| [`joint_child.py`](joint_child.py) | 1c — exact joint-child coverage + energy shares | 02 | do the children actually account for the parent |
| [`reconstruction.py`](reconstruction.py) | 2a — ablation contribution filter | 02 | both endpoints carry reconstruction mass |
| [`sres.py`](sres.py) | 2b — probe-based `S_res`, rank-scored | 03 | the parent decoder points at the child concept |
| [`sibling_redundancy.py`](sibling_redundancy.py) | 3 — child diversity | 02 / 03 | feature splitting in disguise |
| [`outdegree.py`](outdegree.py) | 4 — degree distribution, Gini, superparents | 02 | superparents and poly-parenting |
| [`token_control.py`](token_control.py) | 5 — frequency-bucketed coverage | 02 | does the edge hold on rare tokens |
| [`independence_null.py`](independence_null.py) | 6 — PMI / Dev vs the independence null | 02 | is the co-firing above chance |

All thresholds live in [`config.py`](../config.py). A feature "fires" on a token when its activation
exceeds `FIRE_THRESHOLD = 1e-3` (post-JumpReLU); every matrix below is built on that.

**Undefined cells.**
With default arguments an empty denominator (a feature that never fires, a zero energy or error sum) is clamped, and `parent_conditioned_redundancy` returns 0 for fewer than two children.
The keyword-only `undefined=` returns that value in those cells instead.
On the ratios (`coverage_legs`, `r_supp`, `r_mass`, `edge_reconstruction_condition`) it also drops the clamp elsewhere, so a tiny denominator keeps its true ratio; on `parent_conditioned_redundancy` it changes only the no-children and dead-parent returns, and on `kept_outdegree` only a parent that never fires.

**The synthetic benchmark calls these same functions.**
`scoring/core/detectors.py` passes `undefined=NaN`, so an unmeasurable cell is never read as a zero.
It also calls `coverage_asymmetry`, `kept_outdegree`, and `frequency_controlled_coverage` with `clamp_max=None, floor="total"`; `scoring/benchmark/reads.py` calls `either_endpoint_outdegree`.
No default changed.
The new functions stay out of `metrics.__all__`, which lists only what the Tier-1 calibration covers.

---

## 1. Coverage — defines the candidate edge set

From `cofire[p,c]` (tokens where both fire):

```
R[p,c] = cofire[p,c] / fire_count[c]     reverse: is the child contained in the parent?
F[p,c] = cofire[p,c] / fire_count[p]     forward: how much of the parent this one child explains
```

Keep the edge when `R[p,c] ≥ EDGE_TAU = 0.50`, both endpoints fire at least `MIN_FIRE_COUNT = 20`
times, **and** the pair co-fires at least `MIN_JOINT = 30` times. (`F` is computed but is not part of
the accept rule.) The joint-support guard matters: a child firing 20 times inside a near-always-on
parent reaches `R = 1.0` on base rate alone. Excluded edges are **reported, not silently dropped**.

**Blind spot.** Coverage only sees co-*occurrence*. A child that happens to fire inside a very
frequent parent gets a high `R` with no semantic or reconstructive relationship — which is the entire
reason the other metrics exist. It is also blind to **absorption** in the opposite direction: if the
child absorbed a case from the parent, the parent goes silent exactly where the child fires, `R`
collapses, and the pair never enters the candidate set at all.

### 1c. Joint-child coverage and energy shares

`min(1, Σ F)` double-counts co-firing children and saturates near 1 for any parent with many
children (the 0.92–0.97 numbers the older reports showed regardless of structure). The exact forms:

```
R_supp(p) = |{x : f_p>0 and ≥1 child fires}| / |{x : f_p>0}|          support version
R_mass(p) = Σ f_p² · 1[≥1 child fires] / Σ f_p²                      energy version
Share_energy(c,p) = Σ f_p² · 1[f_c>0] / Σ f_p²                       per-child energy share
```

The **superparent signature** is `R ≈ 1` for many children while `R_mass ≈ 0`: the parent's mass
lives where no child is. One child with `Share_energy ≈ 1` is a rename/duplicate candidate. Shares
do not sum to 1 — siblings co-fire.

## 2a. Reconstruction ablation — the contribution filter

Ablating feature `f` on a token changes the reconstruction only by its own contribution, so the
error delta has a closed form:

```
g_f = 2·a_f·⟨d_f, x − x̂⟩ + a_f²‖d_f‖²
parent_gain[p,c] = Σ g_p / Σ err        child_gain[c] = Σ g_c / Σ err      (over the child's tokens)
```

Pass when **both** gains ≥ `RECON_REL_GAIN_MIN = 0.01` — ablating the parent must make the child's
reconstruction at least 1% worse.

**Honest-name note.** This is *not* Tree SAE's `S_res`; it is a contribution filter. It tests neither
refinement nor semantic coherence: two strong but unrelated co-firing features pass both sides, and
its pass rate scales with block energy, mechanically suppressing deep blocks. Kept as a cheap
baseline; the real thing is 2b.

## 2b. `S_res` — probe-based, rank-scored

```
S_res(p,c) = min( (d_c*)ᵀ d_c , (d_c*)ᵀ d_p )
```

with `d_c*` a linear-probe direction for the **child concept** trained on the residual stream. The
parent decoder must point toward the child concept (refinement); the child decoder must too (sanity).

Scored by Tree SAE's operational **rank rule** — both decoders in the top `SRES_RANK_TOP_K = 5` probe
correlations over all 32768 features — never a threshold. Healthy pairs have `d_p ⟂ d_c`, which caps
`min(·,·)` at `1/√2 ≈ 0.707`, so any τ above that rejects every healthy pair by construction.
`sres_rank_check` scores one pair; `sres_scores` returns the value for every pair at once, and is
what `scoring/` reports as its `S_res` column.

**Circularity caveat.** The probe target `1[f_c > 0]` is a **self-label**: a corrupted (absorbed or
split) latent yields a corrupted probe that then validates the corruption. Report these as
"self-labeled `S_res`" — a self-consistency check, not ground truth. The negative-class parent
composition is reported per probe, because a parent-rich negative class suppresses the shared parent
component of a discriminative probe.

## 3. Sibling redundancy — the feature-splitting detector

If children are a real refinement they partition the parent's firing set; if they are one feature
split into near-copies they co-fire massively.

```
J(ci,cj) = cofire(ci,cj) / (fire(ci) + fire(cj) − cofire(ci,cj))
redundancy(p) = mean over sibling pairs        flag when ≥ SIBLING_REDUNDANCY_FLAG = 0.50
```

The property under test is disjointness **within the parent's firing set**. The global Jaccard scores
co-firing anywhere, which in an unconstrained architecture is partly irrelevant;
`parent_conditioned_redundancy` (the second pass, needs per-token masks) is the corrected form, and the
global form is kept alongside it for auditability. Within-block co-firing is `C×C`, so B4's 24576²
does not fit — `SIBLING_BLOCKS = [1, 2, 3]`, and sibling redundancy is unavailable for a B4 child.

## 4. Out-degree — the superparent / poly-parenting detector

With `fire_rate[p] = fire_count[p] / total_tokens`, flag a parent when
`outdeg[p] / n_children ≥ SUPERPARENT_OUTDEG_FRAC = 0.30`.

The flag is on **out-degree alone**. The older AND-gate (`fan-out ≥ 30%` **and**
`fires ≥ 10%`) leaked in both directions — at layer 24 feature 14 clears firing by 4× (41.9%) but
fans out to only 21.9%, so it slipped through. Dropping the firing conjunct raised superparent recall
without dropping precision; `fire_frac` is still reported as an attribute and the old gate survives
as `strict` for comparison. Also reported: `poly_frac` (share of parented children with ≥2 parents),
top-1 edge share, and the out-degree Gini (0 = evenly spread, →1 = concentrated).

## 5. Token-frequency control

Token ids are bucketed by their share of corpus **mass**, not rank: bucket 0 = the ids covering the
top `FREQ_HIGH_MASS = 50%` of tokens, bucket 1 = the next 40%, bucket 2 = the long tail.

```
survival(p,c) = R over buckets 1+2  ÷  R over all buckets      pass when ≥ FREQ_SURVIVAL_MIN = 0.50
```

`≈1` the edge holds on rare tokens too (genuine); `≈0` it exists only on frequent tokens (frequency
capture). Edges whose child barely fires outside bucket 0 are marked **untestable** (`NaN`), not
failed.

## 6. Independence null — PMI and Dev

Raw reverse coverage rewards frequent parents: a parent firing on ~all tokens has `R ≈ 1` against
every child by base rate alone. Score the surprise over the independent-firing null instead:

```
PMI(p,c) = log[ P(p ∧ c) / (P(p)·P(c)) ] = log[ n_joint · N / (n_p · n_c) ]
Dev(p,c) = R(p,c) − ρ_p          (ρ_p = parent firing rate; sign-equivalent, in coverage units)
```

`PMI ≈ 0` for a base-rate-only "parent" even when `R ≈ 1`. Pairs below the support guard are `NaN`
(not `−inf`, which would poison means and sorts) and are counted.

**Scope.** This controls the frequency confound only. **Topical co-occurrence passes it** — `enzyme`
and `CT scan` are not independent, they share the latent `biology`, so `PMI > 0`. See the open gap in
the [root README](../README.md#what-each-metric-catches).

## The verdict: when is an edge a real parent → child?

```
R ≥ 0.5  and  fire_p ≥ 20  and  fire_c ≥ 20  and  cofire ≥ 30     (coverage)
and  parent_gain ≥ 0.01  and  child_gain ≥ 0.01                   (reconstruction)
and  survival ≥ 0.5                                               (frequency control)
and  parent not flagged as a superparent
```

The three rejection categories in the reports map exactly onto these: **superparent** /
**freq-driven** (`survival < 0.5`) / **no-recon** (`parent_gain < 0.01`). The second pass adds the `S_res`
rank verdict on the surviving shortlist.

## Adding a metric

1. **Cache its raw sums in the collect step.** `collect_statistics.py` streams the corpus **once**;
   you cannot compute a new signal in `run_metrics.py` if the collect step did not accumulate what it
   needs. Per-token detail (probes, parent-conditioned masks) goes through the token cache and the
   second pass (`run_token_metrics.py`) instead.
2. **Write it as a pure function here**, over tensors only — no model, no IO — so the toy calibration
   can run it.
3. **Export it** from [`__init__.py`](__init__.py).
4. **Calibrate it** in [`validation/`](../validation/): it must recover the genuine edges of the synthetic tree
   and reject the pathology it claims to catch.
5. **Add its row to the properties matrix** in the [root README](../README.md). A metric earns its
   place by adding a column no existing metric covers; if its row duplicates another's, it is
   redundant and should be dropped.

Feature indices are **global** (0–32767). `config.block_of()` and `sae_utils.block_slice()` convert
to and from block-local indices, and `analyse_pair` mixes both — watch which space a variable is in
(`parent_local` vs `parent_global`).

## `rules/` - shared gates, rules, grading and constant sets

The metrics above compute statistics. [`rules/`](rules/) decides on them, once, for every pipeline:

| File | Holds |
| ---- | ----- |
| [`rules/gates.py`](rules/gates.py) | eight fixed-threshold gates on a `[P, C]` candidate frame, each `1.0` / `0.0` / `NaN` (not measurable) |
| [`rules/rules.py`](rules/rules.py) | the named rules (conjunctions of gates), the `PASSES` / `FAILS` evaluator, and `RULESET_VERSION` |
| [`rules/pairs.py`](rules/pairs.py) | `PairStats` and `score_pairs`: every gate and rule decision for one frame, in one call |
| [`rules/grading.py`](rules/grading.py) | `grade_rules`: counts, rates and verdicts against labelled pairs, with the bars and the support floor |
| [`rules/classes.py`](rules/classes.py) | the nine pair classes a rule can target, in their stored index order |
| [`rules/constants.py`](rules/constants.py) | named constant sets: `GEMMA_MATRYOSHKA` (read by `config.py`) and `SYNTHETIC_TOYS` (read by `scoring/`) |

The gates take plain tensors, so a block pair, a within-block frame or a square toy frame all work.
The edge gates, in terms of reverse coverage `R(p,c) = P(p fires | c fires)`:

| Gate | Test | Same as |
| ---- | ---- | ------- |
| `gate_contains` | `R(p,c) >= tau` | the cross-block edge, `keep_edges` |
| `gate_strictly_contains` | `R(p,c) >= tau` and `R(c,p) < tau` | in-block `parent_of` |
| `gate_mutually_contains` | both directions `>= tau` | in-block `duplicate` |

Rules are named after the pair class they target (`rule_is_a`, `rule_firing_only`, `rule_superparent`,
`rule_frequency`, `rule_topical`, ...), never after a toy. Names carry no version: change a rule, a gate
or a constant and bump `RULESET_VERSION`, and stamp it with the constant set's `name` on every result.
The rules are not in `metrics.__all__`, which lists the metric functions the Tier-1 calibration must call.

Scoring one block pair, with the statistics the pipeline already computes:

```python
from metrics.rules import GEMMA_MATRYOSHKA as C, PairStats, high_outdegree, score_pairs

out = score_pairs(PairStats(
    cofire=cofire, fire_p=fire_p, fire_c=fire_c, R=R, R_rev=F,
    parent_gain=parent_gain, child_gain=child_gain, survival=survival, survival_scale="raw",
    high_outdeg_p=high_outdegree(edges, fire_p, n_children, C.superparent_outdeg_frac, C.min_fire_count),
    high_outdeg_c=high_outdeg_of_children,       # from the next block pair down
    probe_corr=corr, parent_ids=parent_ids, child_ids=child_ids, probe_available=trained,
), C)
out["gates"]["gate_contains"]     # [P, C] tristate
out["rules"]["rule_is_a"]         # (pass mask, scorable mask)
```

Grading needs a pair label for every scored pair, as an index into `LABELS`:
`grade_rules(gates, y, n_total, null_idx)` takes the gates flattened to one vector per gate, the labels `y`,
the generated pair count per class, and the indices of the null pairs.
