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

# outputs_archive/

Superseded results, kept so a withdrawn number stays citable.

`outputs/layer_NN/` is a fixed path — the published site links to it by name, so timestamping it
would 404 every page. That means a rerun overwrites the previous run in place. `config.ARCHIVE_DIR`
takes a dated copy first, into this directory, before the overwrite happens.

This directory is **tracked in git, not ignored**. That is the point: when a number changes, the
version it replaced has to survive in the repository rather than in one person's working copy.
`outputs_local/` was the old home for this and is gitignored, so an archive there vanished on a fresh
clone — precisely the one case where you need it.

Nothing here is linked from the site nav. These pages are evidence, not reading material.

## What is in here

### `layer_NN__v1__2026-08-06T17-45/`

The five gemma-2-2b layers as they read before BOS was excluded from the co-firing statistics.
Distinguishable at a glance by the token count: **48,971** here, **48,571** in `outputs/`. The
difference is exactly 400 — one BOS position per document, 400 documents.

These are the only files kept here, because the paper still reads them:
`reporting/make_report_figures.py` builds `shared_input_moved_every_metric.png` from them, and
`reporting/make_report_tables.py` builds the BOS table.

## Removed on 23 Oct 2026

The repo cleanup removed everything else that used to sit here: the dated snapshots of 9 and 16
August (`layer_NN__2026-08-09T…`, `layer_NN__2026-08-16T…`) and the two withdrawn hand-built pages
(`kill_rates__v1__*.html`, `cross_depth_comparison__v1__*.html`). No code read them. They are still in
git history and on the `backup` branch.

## What was touched, and what was not

The numbers, tables and figures in every archived file are exactly as they were published. Only the
nav bar around them was edited, so that navigating inside the archive still works.

Two file kinds are deliberately *not* archived, via `config.ARCHIVE_SKIP`. `exp0_stats.pt`,
`token_cache/` and `figures/` are too large — the cache lives on the Hub. `feature_labels.json` and
`npedia_labels_cache.json` are not run output at all: they are Neuronpedia's labels for the layer's
SAE, identical in v1 and v2 because nothing we compute touches them. They were 9.8 MB of an 11 MB
archive, byte-identical to the tracked copies in `outputs/`. The archived dashboards inline their
labels at generation time, so they still read correctly without them.

## Why v1 was wrong

BOS is an attention sink: every feature fires there. With `PREPEND_BOS = True` and 400 documents,
every parent/child pair in the dictionary accumulated 400 joint firings — against a `MIN_JOINT = 30`
support guard. The guard exists to kill pairs whose co-firing is coincidence, and BOS handed every
pair enough co-firing to clear it. So the guard passed everything, and the metrics downstream graded
a candidate set that should never have existed.

The wider methodological point: five of the six metrics read the same co-firing matrix. They were
designed as independent detectors that would fail independently, and one contaminated token position
defeated them together. Agreement between them is weaker evidence than it looks.

## Reading an archived page

Do not compare a v1 page against a v2 page metric by metric and treat the delta as a finding. The two
were computed over different token sets. The delta is the contamination, not a result.
