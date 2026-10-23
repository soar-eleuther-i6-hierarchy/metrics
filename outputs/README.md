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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../">SOAR I-6 · metrics</a><a class="on" href="../outputs/">Results</a><a class="" href="../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# `outputs/` — results

Everything the pipeline produces, grouped by SOURCE: `gemma-2-2b/layer_NN/`, `pcfg-matryoshka/layer_01/`. Layer-independent artifacts (the toy calibrations) sit directly here.

The layers moved out of `outputs/layer_NN/` on 7 August, when a second source was published beside
them and the old layout implied "layer 6" was a global fact rather than a fact about one model.
Old URLs 404 — see *Moved pages* below.

`assets/plotly.min.js` is the one shared plotly bundle every dashboard links to. Inlining it in each
page instead cost 4.6 MB per file and added ~70 MB of blobs to git on every regeneration; the pages
are ~20 KB now and still open offline. `reporting/visualize.py` writes it automatically if missing.

Link to the `.html` form, not `.md`: GitHub Pages serves `.md` as raw markdown text.

## Across all layers

- [**Synthetic toy calibration scorecard**](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/synthetic_toy_calibration.html) — Tier 1, synthetic ground truth (14/14 rows).
- [**Trained-toy calibration**](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/trained_toy_calibration.html) — Tier 2, edge recovery on a Matryoshka SAE trained on the known tree (precision 1.00, recall 1.00).

## Other sources

- [**`pcfg-matryoshka/`**](pcfg-matryoshka/) — Exp 2, the same battery on a Matryoshka SAE trained on a PCFG corpus
  (`zipf_exponent` 1.5): 1792 latents in 8 blocks over a 4-layer toy transformer, not gemma's 32768
  in 5. Produced by `adapters/from_pcfg.py` in the umbrella repo; the metric code is untouched,
  which is the point of it being here rather than in a repo of its own.

  Same shape as gemma: a source directory of layers, with its own entry in the nav and its own layer
  row. Two pills, **1 and 3** — the base transformer has four layers in total (0–3), so gemma's
  6/12/18/24 do not exist for this model and never will; of those four, an SAE has been trained on
  two. Its Page row is five wide rather than seven: the two qualitative pages read Neuronpedia
  labels, which exist for gemma's dictionary and no other.

### Publishing a source that is not gemma

Stages 02–04 read the block structure, the model and the dictionary out of the stats file, so an SAE
of a different shape needs an **adapter and no metric code** — `adapters/from_pcfg.py` in the
umbrella repo, which also writes the `token_cache/` stage 03 reads and the `w_dec.pt` it scores
`S_res` with.

- `EXP0_RUN=pcfg-matryoshka/layer_01` names the output directory and publishes at
  [`outputs/pcfg-matryoshka/layer_01/`](pcfg-matryoshka/layer_01/README.md). Any depth works — every
  page derives its distance to the site root and its link to the shared plotly bundle from its own
  path — but the **last component** decides how the page reads: a directory named `layer_NN` gets the
  layer nav, anything else gets the site-wide one.
- Give the directory an index with `python3 -m reporting.layer_index --run`, and the source its own
  with `--source <name>`, **or those URLs 404 on GitHub Pages**.
- Register the source in `config.SOURCES` (label, layers, which page kinds it actually has) and it
  appears in the nav with its own layer row. Each source is one entry in `NAV_GLOBAL` — a place you
  go, not a row that follows you: the layer pills belong to gemma and appear only inside
  `outputs/gemma-2-2b/`, because offering to "switch layer" from a PCFG dashboard means switching to
  a different model entirely.
- Adding to `NAV_GLOBAL` makes every already-generated bar wrong, and regenerating a dashboard needs
  that layer's ~700 MB cache. `python3 -m reporting.refresh_nav` re-renders the bar in place instead,
  deriving each page's identity from its path; `--check` reports without writing. It changes
  navigation and nothing else, so it is not a way to freshen stale numbers.

## Moved pages

The five layer directories were at `outputs/layer_NN/` until 7 August, so **every URL of that shape
is now a 404** — 25 pages plus 5 directory indexes. The replacement is the same path with the source
in it: `outputs/layer_06/metrics_dashboard.html` → [`outputs/gemma-2-2b/layer_06/metrics_dashboard.html`](gemma-2-2b/layer_06/metrics_dashboard.html).

Redirect stubs at the old paths were written and then removed: they duplicated all five layer
directories in the tree for 120 KB of files holding no results, which reads as five more sets of
results to anyone browsing the repo. If old links turn out to be in circulation, the generator is
in this commit's parent — `reporting/moved_pages.py`, `git show 0139852`.

## Withdrawn pages

`kill_rates.html` and `cross_depth_comparison.html` are no longer here. Both were hand-built with no
generator, and both were written against caches that counted the BOS token. BOS is an attention sink,
so every feature fires on it: with 400 documents every pair in the dictionary collected 400 joint
firings and sailed past the `MIN_JOINT = 30` support guard. Excluding it inverted the very numbers
those two pages existed to display — deep-pair reconstruction, the frequency-driven share, the death
rate. They were kept in `outputs_archive/` with a banner until 23 Oct 2026, when they were removed in
the repo cleanup. They are still in git history and on the `backup` branch.

They were withdrawn rather than fixed on purpose. Editing the numbers by hand would leave two pages
that no rerun can reproduce and no rerun can invalidate — which is how they went stale in the first
place. If the cross-depth view is wanted back, it should come back as a generator under `reporting/`.

## The second pass has run on four runs, not on all of them

Stage 03 (`run_token_metrics.py` — S_res, parent-conditioned sibling redundancy, the kept-children
union) has produced a `second_pass.json` for **`gemma-2-2b/layer_01`, `gemma-2-2b/layer_06`,
`pcfg-matryoshka/layer_01` and `pcfg-matryoshka/layer_03`**. Stages 01, 01b, 01c, 02 and 02b have
run on all eight runs (six gemma layers, two PCFG layers). Gemma layers 3, 12, 18 and 24 have no
second pass, so their `S_res` columns and the strict stages of their dashboards are empty by
absence, not by result — do not read them as zero.

This heading used to say "layer 6 only", which was true when it was written and stopped being true
as soon as the PCFG runs and gemma layer 1 were published; the S_res figures quoted elsewhere on
this page for PCFG could only have come from a stage-03 output.

It is committed even though it is a generated artifact, because it cannot be regenerated from this
clone: stage 03 reads `token_cache/` and `exp0_stats.pt`, both far too large for git and both absent
here. It was produced on the compute node and pulled down with the rest of the v2 results.

The file is not a duplicate of the `second_pass` key inside `metrics_report.json`. `run_token_metrics`
deliberately strips the per-edge `edges` list when merging into the report, so the report carries the
summary (`n_pass`, `n_edges_scored`) and this file carries the rows — parent, child, both probe ranks,
both correlations, the verdict. `reporting/make_report_figures.py` and `reporting/visualize.py` both
read those rows.

That distinction matters beyond bookkeeping: `run_metrics.py` labels its own sibling-redundancy figure
`global_jaccard_confounded` and defers the verdict to this pass. The number on the dashboards is not
the answer; the answer is here.

## Per layer

Every layer has the same seven pages: four interactive dashboards, then the three rendered text
reports behind them. Each layer also has its own landing page — `gemma-2-2b/layer_NN/`, written by
`python3 -m reporting.layer_index` — which is where the nav bar's layer buttons go, and the source
itself has one at [`gemma-2-2b/`](gemma-2-2b/) from `--source`.

| Layer | Metrics dashboard | Superparent fan-out | Qualitative dashboard | metrics report | qualitative report |
| ----- | ----------------- | ------------------- | --------------------- | -------------- | ------------------ |
| **L1**  | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_01/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_01/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_01/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_01/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_01/qualitative_check.html) |
| **L3**  | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_03/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_03/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_03/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_03/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_03/qualitative_check.html) |
| **L6**  | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_06/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_06/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_06/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_06/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_06/qualitative_check.html) |
| **L12** | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_12/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_12/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_12/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_12/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_12/qualitative_check.html) |
| **L18** | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_18/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_18/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_18/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_18/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_18/qualitative_check.html) |
| **L24** | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_24/metrics_dashboard.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_24/superparent_sankey.html) | [open](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_24/qualitative_dashboard.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_24/metrics_report.html) | [report](https://soar-eleuther-i6-hierarchy.github.io/metrics/outputs/gemma-2-2b/layer_24/qualitative_check.html) |

## What is in a `layer_NN/` directory

| Artifact | Written by | What it holds |
| -------- | ---------- | ------------- |
| `exp0_stats.pt` | `collect_statistics.py` | ~700 MB of cached statistics: co-firing counts, per-bucket co-firing, per-edge reconstruction sums, within-block sibling co-firing, energy. **Not in git** — see below. |
| `token_cache/` | `collect_statistics.py` | fp16 residuals + sparse latents, so the second pass can train probes without re-running the model. Not in git. |
| `feature_labels.json` | `fetch_labels.py` | all 32768 autointerp descriptions from Neuronpedia's S3 export (~99.9% coverage; ~26 features fall back to `feature <idx>`). |
| `metrics_report.{json,md}` | `run_metrics.py` | per-block-pair summaries and the top edges, annotated with labels. |
| `second_pass.json` | `run_token_metrics.py` | `S_res` verdicts per edge, parent-conditioned sibling redundancy, exact kept-children union. |
| `in_block_edges.{json,md}` | `in_block_edges.py` | same-level directed edges and co-extensive duplicates per block. |
| `qualitative_check.{json,md}` | `validation/qualitative_check.py` | survivor vs rejected edges with both endpoint labels, for human reading. |
| `metrics_dashboard.html`, `superparent_sankey.html`, `qualitative_dashboard.html`, `in_block_dashboard.html` | `reporting/visualize.py` (`--qualitative`, `--in-block`) | the interactive pages linked above. |
| `paper_figuers/*.png` | `reporting/make_report_figures.py` | the static proof-figures for the write-up, written once into `outputs/paper_figuers/` (not per run) and tracked in git. |
| `npedia_labels_cache.json` | `validation/qualitative_check.py` | per-feature Neuronpedia API fallback for the handful missing from the bulk export. |

`python3 -m utils.organize_outputs` sorts a run directory into `dashboards/` and `reports/` for browsing;
it leaves the data files the scripts read exactly where they expect them, and is idempotent.

## The big caches are not in git

`outputs/**/*.pt` is gitignored. The ~700 MB-per-layer caches live on the Hub at
[`soar-eleuther-i6-hierarchy/experiment_0-stats`](https://huggingface.co/datasets/soar-eleuther-i6-hierarchy/experiment_0-stats):

```bash
hf download soar-eleuther-i6-hierarchy/experiment_0-stats --repo-type dataset --local-dir outputs/
```

They are loaded with `weights_only=False`. The HTML dashboards **are** tracked and deliberately not
in LFS (`.gitattributes` sets `-filter -diff -merge` on `outputs/**/*.html`), because GitHub Pages
does not resolve LFS objects and would serve the pointer stub instead of the page.

To keep an experimental run away from the published directory, redirect it:
`EXP0_OUT=outputs_local python3 run_metrics.py`.

A run always writes to the same path — `outputs/layer_NN/` — because the site links to it by
name, and timestamping that directory would 404 every page. So that a rerun does not simply
erase the previous numbers, `collect_statistics.py` copies the current artifacts to
`outputs_local/archive/layer_NN__<date>T<time>/` before it starts. The copy skips `*.pt`,
`token_cache/` and `figures/`: the caches are on the Hub and rebuildable, and copying them per
run would fill the disk. Archives are gitignored — they are history, not results.

## How the metrics are validated: three tiers

The same metrics are checked at three tiers of increasing realism. Each tier gives up one guarantee
and gains one dose of reality; a metric we trust has to hold across all three. ("Tier", not "layer",
to avoid confusion with the model's residual-stream layers.)

The PCFG SAE **was** the third rung and is not one any more. A tier earns its place by scoring the
battery against a known answer; that run has none, so it moved to
[Other sources](#other-sources) as a control on corpus complexity — see below.

| Tier | What it is | Ground truth? | What it proves |
| ---- | ---------- | ------------- | -------------- |
| **1. Synthetic toy** | [`validation/synthetic_toy_world.py`](../validation/synthetic_toy_world.py): a known 5-parent tree plus six injected structures, reduced to the statistics the metrics read *and* to the per-token residuals the probes need | yes, by construction | the maths is right — **14/14 scorecard rows**, covering 21/21 metric functions; the last two rows are negative controls that pass when nothing catches them |
| **2. Trained toy** | [`validation/calibrate_on_trained_toy.py`](../validation/calibrate_on_trained_toy.py): a Matryoshka SAE actually trained on Bussmann's tree, metrics run on the *learned* features | yes, the tree is known | the metrics survive a real training run — **precision 1.00, recall 1.00** (9/9 edges, 0 false positives, all 20 features learned) — and the probe functions run here too: `S_res` accepts **9/9** testable true edges against a chance rate of 0.25. On the checkpoint graded until 19 August, parent-conditioned redundancy caught a conflation the SAE itself introduced (0.958 against 0.000); on this one it reports 0.000 for all three parents, because there is none |
| **3. Released SAE** | [`validation/qualitative_check.py`](../validation/qualitative_check.py) on `gemma-2-2b / NN-res-matryoshka-dc`, read against Neuronpedia labels | no, human judgement stands in | the metrics mean something on a checkpoint we did not train |

Tier 1 is certain but artificial; Tier 3 is realistic but has no ground truth and is a published
checkpoint we did not train — which is what *released* names, since the rungs below it are real SAEs
too. Tier 2 is the only rung with both a trained SAE and a known answer.

**Tier 2 does not isolate the variable it is named for.** It runs on a clean 20-feature tree while
Tier 1 grades a larger pathology-injected world, so the toy changes along with the statistics. What
it does isolate cleanly is **blame**: a missed edge counts against a metric only if the SAE learned
both endpoints.

### The PCFG SAE: a control, not a rung

It has a base model between the concepts and the SAE, which neither toy has, and it runs the same
battery unchanged — but it cannot license anything above it, because it has no recovery score. Its
ground truth is available and unconsumed: the PCFG repo's
`pcfg_bridge.grammar.vocab.role_of(token_id)` returns the grammar role of any token id
(subject/verb/object/connector/eos/section/paragraph/document) and its `analysis/README.md` names it
for exactly this purpose, but no latent→symbol mapping is built yet, so the run reports the same
battery outputs as Tier 3 rather than a recovery score.

It also changes the grammar, the base model, the corpus and the dictionary size all at once against
the toys, so it *bounds* base-model dependence rather than isolating it — isolating it would mean
training a transformer on the toy's own tree, which nothing here does. And both published layers are
a **single grammar configuration** (zipf 1.5, EOS the only delimiter): one point of the three-axis
sweep Exp 2 specifies, not a sweep. Its numbers are in [Other sources](#other-sources) —
layer 01: 327 candidates, 100% recon, 0/327 S_res; layer 03: 781 candidates, 95% recon, 4/772 S_res.

**What the tiers do and do not cover.** Tier 1 scores **14/14 rows**, covering
**21/21 metric functions**. Until 7 August this paragraph said `S_res` was "calibrated in Tier 2,
not here". It was not: Tier 2 imports `coverage_legs`, `keep_edges`,
`edge_reconstruction_condition`, `frequency_controlled_coverage` and `frequency_buckets`, and
nothing else — so the strict test, the one that decides which edges survive on gemma, was graded
against no known answer at all, and neither was in-block directed coverage.
[`validation/synthetic_toy_world.py`](../validation/synthetic_toy_world.py) now also returns the per-token view
(`resid`, `fired`, `W_dec`) those functions read, and carries three further structures: an absorbed
child, a shared-topic pair and a within-block containment plus duplicate pair.

Absorption and topical co-occurrence are still **not caught** — that has not changed and cannot be
fixed by another threshold. What changed is that they are now *scored*, as negative controls that
pass when the battery does nothing: the absorbed edge has `R = 0.00` and never enters the candidate
set, and the shared-topic non-edge clears coverage, reconstruction, the frequency control and PMI.
A limitation that is measured regresses visibly; one that is only written down does not.

One thing the toy cannot show: the rank rule passes an unrelated parent whenever chance puts it in
the top *k* of *D*, so its null rate is `k/D` — 11.9% here, 0.28% on PCFG's 1792 latents, 0.015% on
gemma's 32768. An `S_res` pass rate is only comparable between dictionaries of similar size.

**Tier 1 detail.** The per-metric scorecard is not copied here any more. It was, and it drifted: it still read 9 rows and "28 non-genuine" after the toy had grown to 14 rows and 35. The live table is [`synthetic_toy_calibration.md`](synthetic_toy_calibration.md), written by the calibration itself on every run, so it cannot say something the run did not.

**Tier 2 detail** — on the trained toy the metrics keep **every** edge whose two endpoints the SAE
actually recovered, with **no false positives**. The current checkpoint recovered all 20 features, so
all 9 edges were testable and all 9 came back. The rule that a miss counts against a metric only when
the SAE learned both endpoints did its work on the checkpoint graded until 19 August: that one
recovered 17 of 20, and its three missed edges were exactly the three whose child was among the
missing — a limit of the trained SAE, not of the metrics. The loop:

1. **Rebuild the toy** — Bussmann's compositional tree from the team's
   [`sae-training`](https://github.com/soar-eleuther-i6-hierarchy/sae-training) repo
   (`configs/tree.json`): 3 parents, each with 3 mutually-exclusive children, plus rare features. A
   child only fires when its parent fires — that is the ground-truth hierarchy.
2. **Train a Matryoshka SAE on it** (batch-topk, `k=2`); the checkpoint lands in `outputs/toy_trained/`.
   The SAE sees only activations, never the tree.
3. **Match learned latents to true features** — for each latent, the true feature its decoder points
   at most (cosine ≥ 0.4). This is how we know which latent *is* parent 0.
4. **Run the metrics on the learned latents**, same functions and thresholds as the real pipeline.
5. **Score against the known tree** — precision and recall over true parent→child edges. An edge kept
   on a semantically wrong pair shows up immediately as a false positive.

Steps 3–5 are what separates "the metric failed" from "the SAE failed": a miss only counts against a
metric if the SAE learned both endpoints, which is why per-feature recovery is reported too.

```bash
python3 validation/calibrate_on_synthetic_toy.py                    # Tier 1
python3 -m reporting.visualize --calibration                # Tier 1 dashboard

PYTHONPATH=src python3 validation/calibrate_on_trained_toy.py    # Tier 2 (needs outputs/toy_trained/)
python3 -m reporting.visualize --trained-calibration        # Tier 2 dashboard

python3 -m validation.qualitative_check                     # Tier 3
python3 -m reporting.visualize --qualitative                # Tier 3 dashboard
```

## The finding these outputs support

**This section stated the pre-BOS numbers until 7 August.** It said B1→B2 at layer 6 held 271k
candidates of which ~0% improved reconstruction and 99.4% were frequency-driven; that pair holds
**280** candidates, **35.7%** of them pass reconstruction and **27.8%** are frequency-driven. It also
said semantic agreement "collapses by L24", which was withdrawn with the rest of the depth claim.

What the regenerated outputs support:

- **The cheap filters barely bite; the refinement test does.** At layer 6, B0→B1 proposes 2,428
  candidates and 85.9% of them improve reconstruction — but only **10 of 1,700 (0.6%)** pass probe
  `S_res`. Stage 03 has run on four of the eight runs, and gemma layers 3, 12, 18 and 24 are not
  among them, so that ratio has one gemma layer behind it.
- **The graph is not a tree.** Multi-parenting in B0→B1 is 99 / 100 / 100 / 89 / 100% across L3–L24.
  The one claim that did not move, because it is a ratio over children that already have a parent.
- **About half the survivors read as genuine refinement**, and the commonest way the rest are wrong
  is a semantic parent with a function-word or formatting child — "formal legal terminology" → the
  word "the". That is topical co-occurrence: it clears coverage, reconstruction, the frequency
  control *and* PMI, and nothing in the battery detects it.

Figures for the write-up are in [`paper_figuers/`](paper_figuers/), rebuilt with
`python3 -m reporting.make_report_figures`; each one derives every number in its title from the JSON
it plots, so a caption cannot outlive its data.
