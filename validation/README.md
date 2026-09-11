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

# `validation/` — metric calibration

This is **calibration**, not a unit-test suite: how a metric gets *scored* rather than eyeballed.
(The directory was called `tests/`, which promised coverage of `metrics/` and delivered a toy-world
generator. [`tests/`](../tests/) now holds real unit tests.)

Tiers 1–2 live here. They have a ground-truth tree and run offline.

**Tier 3 is the odd one that does live here** — [`qualitative_check.py`](qualitative_check.py)
needs the real `exp0_stats.pt`, needs the network for labels, is judged by human reading rather
than against a known answer, and writes a published artifact into `RUN_DIR`.

**The PCFG SAE is a control, not a tier.** It used to be Tier 3, and it never met the bar the column
headings promise: a rung licenses the one above it by scoring the battery against a known answer, and
this run has none. Its grammar *is* known, and the PCFG repo ships
`pcfg_bridge.grammar.vocab.role_of(token_id)` for building a latent→symbol mapping from it, but
nothing consumes that yet, so it reports the same battery outputs as Tier 3 rather than a recovery
score. What it does answer is a different question — what corpus complexity does to the same battery,
between a hand-built world and natural language — which is why it is a control. It has no calibration
script here: the statistics come from `adapters/from_pcfg.py` in the umbrella repo. A
`calibrate_on_pcfg.py` beside the other two is what closing the gap would look like, and would make
it a rung.

The numbering has moved twice, which is worth stating because a reader of an older page has it
differently: it was once 3 for the qualitative pass and 4 for PCFG, then 3 for PCFG and 4 for
qualitative, and it is now three tiers with the qualitative pass at 3 and PCFG off the ladder.

Full results and the tier table: [outputs/README.md](../outputs/README.md#how-the-metrics-are-validated-three-tiers)

| File | Tier | Runs on | Scored against | What it does |
| ---- | ---- | ------- | -------------- | ------------ |
| [`synthetic_toy_world.py`](synthetic_toy_world.py) | 1 | — | — | builds the synthetic world: a known 5-parent tree plus **six** injected structures (superparent, feature-split parent, frequency-coincidence edge, an absorbed child, a shared-topic pair, and a within-block containment + duplicate pair), reduced to the statistics the metrics read **and** to the per-token view the probes need |
| [`calibrate_on_synthetic_toy.py`](calibrate_on_synthetic_toy.py) | 1 | hand-built statistics + per-token residuals | **known tree** | runs every metric on that world and scores it on the job it claims — **14/14 pass**, covering **21/21 metric functions**. One seed: `calibrate(seed=0)`, called once. `build_world` takes a seed so the run can be varied, but nothing loops over one, and this page claimed a 0–7 sweep that has never existed |
| [`calibrate_on_trained_toy.py`](calibrate_on_trained_toy.py) | 2 | a Matryoshka SAE *actually trained* on that tree | **known tree** | matches learned latents back to true features, scores edge recovery — **precision 1.00, recall 1.00** (9/9 edges, 0 false positives, all 20 features learned) — and runs the probe functions on the **learned** latents: `S_res` accepts **9/9** testable true edges against a chance rate of 0.25 |
| [`qualitative_check.py`](qualitative_check.py) | 3 | the released `gemma-2-2b` SAE | **nothing — no ground truth** | contrasts survivor vs rejected edges and reads both endpoint labels against Neuronpedia. Also pipeline stage 02b |
| `adapters/from_pcfg.py` *(umbrella repo, not this one)* | control | a PCFG-trained Matryoshka SAE (zipf 1.5, 1792 latents, 8 blocks) | **nothing yet** — the grammar is known but no latent→symbol mapping exists | same battery as Tier 3, on a source that has a base model; layer 01: 327 candidates, 100% recon, 0/327 S_res; layer 03: 781 candidates, 95% recon, 4/772 S_res |

**Why Tier 3 is not named `calibrate_*`.** The first two score metrics against an answer we know.
Tier 3 has no such answer: it is judged by reading labels that are themselves model-generated
(Neuronpedia's autointerp). Calling it a calibration would claim a ground truth that does not exist,
so the verb differs on purpose. The PCFG control is not a `calibrate_*` either, for a different
reason — it *has* an answer available and does not use it yet.

```bash
python3 validation/calibrate_on_synthetic_toy.py               # Tier 1
PYTHONPATH=src python3 validation/calibrate_on_trained_toy.py  # Tier 2, needs outputs/toy_trained/
python3 -m validation.qualitative_check                        # Tier 3, needs exp0_stats.pt + labels
```

## The notebooks

[`notebooks/`](notebooks/) holds the two runnable walk-throughs. Neither writes into `outputs/`.

| Notebook | Tier | What it adds over the script beside it |
| -------- | ---- | ------------------------------------- |
| [`calibrate_on_synthetic_toy.ipynb`](notebooks/calibrate_on_synthetic_toy.ipynb) | 1 | the world and the scorecard figure by figure — what is in the world, which 55 candidates coverage proposes out of it, what each gate removes, the tree before and after, and the two negative controls — by calling `build_world`, `_run_metrics` and `_score` rather than reimplementing them. Plus a **seed sweep**: 14/14 rows over seeds 0–7, which is the sweep this page once claimed existed |
| [`train_and_calibrate_on_toy.ipynb`](notebooks/train_and_calibrate_on_toy.ipynb) | 2 | trains the toy checkpoint itself — Matryoshka **and** a vanilla SAE differing in two config entries — then scores feature recovery and edge recovery on both. The vanilla contrast is the part `calibrate_on_trained_toy.py` does not have; that script grades one existing checkpoint |

The Tier-2 notebook lived in `sae-training/scripts/` until 20 August. It trains against the
`sae-training` checkout (found beside `metrics/`, or via `EXP0_SAE_TRAINING` — the same variable
`calibrate_on_trained_toy.py` reads) and writes its checkpoints there, **not** into
`outputs/toy_trained/`, so re-running it cannot overwrite the checkpoint the paper's Tier-2 numbers
are read from.

**The two tiers do not use the same toy.** This page said they did — "both tiers use the same toy,
Bussmann's tree" — and that is wrong, as its own table two rows above shows. Tier 1's world is built
by [`synthetic_toy_world.py`](synthetic_toy_world.py): a 5-parent tree over **42 features**
(`P = 10` parents, `C = 32` children) with six pathologies injected on purpose, and it never reads
`tree.json`. Tier 2's world is Bussmann's tree from `sae-training/configs/tree.json`: **20
features, 9 edges, no injected pathologies**.

So the step from Tier 1 to Tier 2 changes the world as well as where the statistics come from, and
does not isolate "the SAE had to learn it" as a single moving part. What it does isolate cleanly is
**blame** — a missed edge counts against a metric only if the SAE learned both endpoints. On the
current checkpoint there is nothing to attribute: it learned all 20 features and the battery returned
all 9 edges. That rule mattered on the checkpoint graded until 19 August, which learned 17 of 20 and
whose three misses were exactly the three features it never learned.

Tier 2 needs a checkpoint in `outputs/toy_trained/`. The reference one is trained by
[`notebooks/train_and_calibrate_on_toy.ipynb`](notebooks/train_and_calibrate_on_toy.ipynb) in this
directory, which writes `cfg.json` + `sae_weights.safetensors` there directly;
`sae-training/scripts/train_toy.py` in the team repo trains the same world with that repo's own
trainer. Either way the ground-truth tree is read from `sae-training/configs/tree.json`, expected
**beside** `metrics/` (`../sae-training/`); set `EXP0_SAE_TRAINING` if your clone lives elsewhere.

Both tiers write into [`outputs/`](../outputs/) (`synthetic_toy_calibration.json`,
`trained_toy_calibration.json`) and have a dashboard: `python3 -m reporting.visualize --calibration`
and `python3 -m reporting.visualize --trained-calibration`.

## What Tier 2 found in the SAE itself — and what the current checkpoint does not

**On the checkpoint graded until 19 August**, the sibling metric caught a defect nobody injected.
The tree declares all three parents `mutually_exclusive_children`, and in 200,000 draws true features
5 and 7 co-fired **zero** times, while the latents that recovered them co-fired **27,592** times — the
one matched to feature 5 never firing alone. That SAE had conflated two concepts the grammar keeps
apart, `parent_conditioned_redundancy` reported **0.958** for that parent against 0.000 for the other
two, and edge recovery still called both of its edges recovered: the sibling metric was adding a
column coverage and reconstruction do not have, exactly as the properties matrix claims.

**The checkpoint graded now has no such defect.** `parent_conditioned_redundancy` reports **0.000
for all three parents** and the recovered latents co-fire zero times, matching the ground truth. The
metric is doing the same thing in both cases — reporting what is there — so the demonstration above
is kept as what it was: evidence that this metric can find a defect Tier 1 structurally cannot,
because there every pathology is one we put in. It is not a claim about the current SAE.

## What Tier 1 did not cover until 7 August

Tier 1's own page used to say the four per-token functions (`train_probe`, `sres_rank_check`,
`negative_parent_composition`, `parent_conditioned_redundancy`) were *calibrated in Tier 2*. They
were not. Tier 2 imports `coverage_legs`, `keep_edges`, `edge_reconstruction_condition`,
`frequency_controlled_coverage` and `frequency_buckets` — nothing else. So **metric 2b, the strict
test that rejects most of what survives on gemma, had no ground-truth calibration anywhere**, and
neither did the within-block metric. The claim read as verified because it named a tier rather than
a file.

Both are covered now. `build_world` additionally returns `resid`, `fired` and `W_dec` — the
per-token view the probes read — so the four functions run against a known tree instead of against
nothing.

**One thing the toy cannot show.** The rank rule passes an unrelated parent whenever chance puts it
in the top *k* of *D*, so its null rate is `k/D`: 11.9% here, 0.28% on PCFG's 1792 latents, 0.015%
on gemma's 32768. The row asserts that genuine edges all pass and that the superparent passes *at
chance*, because asserting zero would assert something the rule does not claim. The dependence is
worth carrying into any cross-source comparison of S_res pass rates.

## Why both

Tier 1 is certain but artificial — it proves the arithmetic is right, and nothing about whether an
SAE would ever learn such a structure. Tier 2 closes exactly that gap: the toy passes through a real
training run first, so only what the SAE actually learned reaches the metrics. That is also what lets
it attribute a miss: a missed edge counts against a metric only if the SAE learned both endpoints.
The current checkpoint learned all 20 and missed nothing, so the rule has nothing to do; it did the
work on the checkpoint graded until 19 August, which learned 17 and missed exactly the three edges
whose child was among the missing.

## The lateral control

The three tiers are a ladder along one axis: ground truth traded against realism, all answering
*are the metrics trustworthy?* A control answers a different question — one that has to be closed
before the gemma result means what we say it means.

| File | Question | Result |
| ---- | -------- | ------ |
| [`block_tree_alignment.py`](block_tree_alignment.py) | does the **Matryoshka nesting itself** put a parent in an earlier block than its children? | **6/6 testable edges respected**; mean block 1.7 for parents, 4.5 for children — **on the 9 August checkpoint; not re-run since, see below** |

```bash
python3 -m validation.block_tree_alignment                       # needs outputs/toy_trained/
python3 -m reporting.visualize --trained-calibration             # renders it onto the Tier-2 page
```

It writes `outputs/block_tree_alignment.json`, which the Tier-2 dashboard picks up if it is there
and skips if it is not — the two are separate questions and separate scripts, sharing a page.

**The result above is stale, and the script no longer runs.** `block_tree_alignment.json` is dated
9 August against a `trained_toy_calibration.json` of 19 August: it describes the checkpoint that
learned 17 of 20 features, where the three untestable edges were exactly the three whose child was
never learned. Re-running it on the current checkpoint raises `KeyError: 'latent_sizes'` — the
`cfg.json` the notebook now writes carries `notebook_config` (`n_prefixes`, `n_latents`) instead.

That is not only a key rename. The 9 August file assumed ten equal blocks,
`[2, 4, … , 20]`, and upstream's `sample_prefixes` **draws prefix lengths at random every step**
(Pareto-biased toward short ones); it has no fixed block boundaries at all. What makes latent order
meaningful is `permute_latents` pushing high-contribution latents toward the front, not a partition.
So the control needs its discretisation stated and defended before it is re-run, rather than an
even split restored to make the script pass.

The page also states what the 6/6 costs. Three true features were recovered by more than one latent,
and the check takes the **earliest** block for each — the reading most favourable to the
architecture. A later choice could turn a respected edge into a violation. The narrowest respected
edge clears by a single block.

**The objection it closes.** The tiers establish that the metrics are sound; gemma then says the
hierarchy fails. A reader is entitled to reply: *maybe Matryoshka simply cannot produce a coherent
hierarchy, so you have measured a broken architecture rather than discovered anything.* This is the
control for that. On a clean toy with a known tree the nesting produces the **right** ordering, so
the architecture is not structurally incapable — and the production failure is about what the data
distribution does to it, which is exactly what Exp 2 sweeps.

**Why it is answerable here and nowhere else.** Tier 2 indexes by ground truth rather than by block
on purpose: mixing the two would confound "is the metric right?" with "did Matryoshka order the
features right?", and that separation is what let it report the SAE's recall as the SAE's ceiling rather
than the metrics'. So the second question stayed open. The toy can answer it because it has ten
Matryoshka blocks *and* a known tree. On gemma it cannot be asked at all — the correct ordering is
unknown, so a violation is indistinguishable from a concept we misread.

On that 9 August checkpoint the 3 untestable edges were the same 3 children the SAE never learned,
the same ceiling its recall reported. Feature splitting did show up: 3 true features recovered by two
latents each.

## The threshold sweep — what `threshold_sweep.py` does and what it found

Two review comments, both on 5 September, asked the same thing about different constants. On
`EDGE_TAU` and `MIN_JOINT`: "why these particular numbers? What would happen if we selected
tau = 0.9?" On `FREQ_SURVIVAL_MIN`: "Again we need to either have ablations or a good reasoning
for the any default number."

`threshold_sweep.py` answers both by varying one threshold at a time and holding the rest at
their defaults. That is the ablation proper: it attributes a change to a single constant. It
runs in two modes, because the two settings answer different questions.

    python3 validation/threshold_sweep.py                     # trained toy: ground truth
    python3 validation/threshold_sweep.py --stats ../data/fmt # 12 PCFG runs: no ground truth

On the toy the score is precision and recall against the known tree. On PCFG there is no tree,
so the score is the accepted edge count together with the share of accepted edges whose PMI is
below 0.5. PMI near zero means two features co-fire at about the rate their individual firing
rates already predict, so that share estimates how much of the accepted set is frequency rather
than structure.

**The script asserts before it sweeps.** Its default point must reproduce
`outputs/trained_toy_calibration.json` on precision, recall and the error counts, or it exits.
Without that check this file would become a second source of truth for the same checkpoint,
reporting its own numbers whenever the gating here drifted from `run_metrics.analyse_pair`.

### What it found, and it is not a justification of the defaults

**The toy cannot rank thresholds at all.** Precision 1.00 and recall 1.00 at every value of
every threshold, except `EDGE_TAU` at or below 0.1. Any claim that these constants were
calibrated on the toy is unsupported: the toy accepts all of them equally.

**Only one of the five acceptance thresholds does any work.** On PCFG, block pair B0->B1, 12
runs, one knob at a time:

| threshold | range tested | accepted edges | share at chance |
| --- | --- | --- | --- |
| `EDGE_TAU` | 0.1 to 0.95 | 2027 down to 28 | 0.71 down to 0.15 |
| `MIN_FIRE_COUNT` | 5 to 200 | 135 to 132 | 0.31 to 0.34 |
| `MIN_JOINT` | 0 to 100 | 134 to 132 | 0.33 to 0.34 |
| `RECON_REL_GAIN_MIN` | 0 to 0.1 | 136 to 44 | 0.33 to 0.17 |
| `FREQ_SURVIVAL_MIN` | 0 to 0.75 | 151 to 131 | 0.33 to 0.33 |

`MIN_FIRE_COUNT` moves the accepted set by 2 percent across its whole range and `MIN_JOINT` by
1 percent. `RECON_REL_GAIN_MIN` does nothing near its default and only begins to act at 0.05,
five times the value in use.

**At the current `EDGE_TAU`, about a third of accepted edges sit at chance.** 0.5 gives
134 +/- 151 edges at a chance share of 0.33 +/- 0.24. One step to 0.6 gives 84 +/- 130 at
0.12 +/- 0.20. Past 0.6 nothing improves: the chance share stays between 0.09 and 0.17 through
0.95 while the edge count falls by two thirds. The direct answer to the review question is that
0.9 is not better than 0.6, only smaller.

**`FREQ_SURVIVAL_MIN` does not reduce the chance share.** Measured before and after its own
gate, at the default tau: 0.33 before, 0.36 after. It removes edges, but not the ones co-firing
at chance. We report that instead of a justification, because the evidence does not support one.

### Three limits on reading those numbers

The PMI score is not independent of the `FREQ_SURVIVAL_MIN` gate. The paper states this about
itself, that the gate and the independence null both shift weight away from frequent tokens, and
that it does not correct for the overlap. The before and after numbers therefore bound the
gate's effect rather than measure it.

The spread across grammar configurations is larger than the mean. At the default, 134 +/- 151.
Any single PCFG number is misleading, and the appendix should show the spread rather than the
average.

This covers one block pair. Deeper pairs accept too few edges to compare: B1->B2 accepts
3 +/- 5 at the default. It also covers the toy and PCFG only. Gemma needs the GPU node, and
would make a third source rather than a second.

### What is still open

No selection rule has been proposed. Showing that 0.6 dominates 0.5 on this evidence is not the
same as a stated criterion applied consistently to every threshold and every source. Six further
constants were not swept, because they gate other metrics rather than edge acceptance:
`FIRE_THRESHOLD`, `FREQ_HIGH_MASS`, `SRES_RANK_TOP_K`, `SUPERPARENT_OUTDEG_FRAC`,
`SIBLING_REDUNDANCY_FLAG` and `SHARE_ENERGY_SPLIT`.

Raw numbers: `outputs/threshold_sweep.json` for the toy, `outputs/threshold_sweep_pcfg.json` for
PCFG. The claim, with its limits, is `findings/I6-F006` in the umbrella repository.

## What is *not* here

[`../tests/`](../tests/) holds unit tests of the pipeline code — they measure nothing about
hierarchy and score no metric, they guard claims the code makes about itself.
`tests/test_collect_generic.py` is the first: it asserts that `collect_statistics.collect()` still
runs on a source that is not gemma, a property one leftover `config` global in the accumulation loop
would silently break.

The naming follows that split: everything here is `calibrate_*` or a named control, and a `test_`
prefix means a unit test in `../tests/`. Tier 1 used to be `test_metric_calibration.py` — a
calibration wearing a unit-test name, which also invited pytest to collect a file that is not a
pytest test. It is now `calibrate_on_synthetic_toy.py`, parallel to `calibrate_on_trained_toy.py`, so the
ladder is legible from the filenames alone. The names say `synthetic` and `trained` because that
is what differs about the *statistics* — they are not two conditions on one toy, as this page
once claimed; the two worlds differ as well (see above).

## Adding a calibration for a new metric

A new metric earns its place by catching a property no existing metric catches
(see the matrix in the [root README](../README.md#what-each-metric-catches)). So the calibration has
two halves: inject that property into `synthetic_toy_world.py`, then assert the new metric flags it **and**
that it spares the genuine tree edges. A metric that only fires on the pathology is a detector; one
that also kills healthy edges is a filter with a false-positive problem.

## `audit_comparability.py` — what may be compared with what

Added 2026-09-11, after a Temporal SAE result was compared against a Matryoshka baseline
graded before BOS positions were excluded from the co-firing counts. Both files loaded
without error and the ratio they produced was wrong by more than a factor of two.

The script reads the settings each result file records, never the prose written about it,
and reports four things:

1. Reports whose `config` omits `bos_excluded` or `min_joint`, which means they predate
   those guards. The test is on absence, because a key that did not exist cannot carry a
   value.
2. Files claiming one corpus but disagreeing on token count.
3. Probe results (`second_pass.json`) whose shortlist is larger than the candidate set in
   the report beside them. Those files carry no config of their own, so this is the only
   way to tell whether the pair came from one run.
4. The source files each finding names, resolved, with a verdict for each.

Exit status is non-zero when anything is flagged.

```bash
python3 -m validation.audit_comparability
python3 -m validation.audit_comparability --json /tmp/audit.json
```

It does not check block width, seed or architecture. Those are still matched by hand, and
the project's design notes say why that matters.
