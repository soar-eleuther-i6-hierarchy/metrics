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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/">3</a><a class="pill on" href="../../../outputs/gemma-2-2b/layer_06/">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../../outputs/gemma-2-2b/layer_06/metrics_dashboard.html">Dashboard</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/superparent_sankey.html">Superparents</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/in_block_dashboard.html">In-block</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/qualitative_dashboard.html">Qualitative</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/metrics_report.html">Metrics report</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/in_block_edges.html">In-block report</a><a class="" href="../../../outputs/gemma-2-2b/layer_06/qualitative_check.html">Qualitative report</a></div></nav>

# Edge activation magnitudes

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 745,305 | 2.19 | 5.93 | 24.75 |
| B1 | 564,121 | 1.91 | 3.32 | 12.41 |
| B2 | 735,460 | 1.86 | 2.88 | 9.58 |
| B3 | 1,394,296 | 1.85 | 2.77 | 6.92 |
| B4 | 928,341 | 1.84 | 2.69 | 6.86 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 2428 | 0.67 | 15% | 1.06 | 0% |
| 1->2 | 280 | 0.96 | 47% | 1.16 | 7% |
| 2->3 | 762 | 0.88 | 42% | 1.26 | 9% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 15 → 194 | 0.62 | 0.25 | feature 15 | feature 194 |
| 15 → 406 | 0.64 | 0.27 | feature 15 | feature 406 |
| 15 → 333 | 0.66 | 0.23 | feature 15 | feature 333 |
| 15 → 353 | 0.66 | 0.22 | feature 15 | feature 353 |
| 15 → 319 | 0.67 | 0.13 | feature 15 | feature 319 |
| 15 → 508 | 0.69 | 0.20 | feature 15 | feature 508 |
| 15 → 349 | 0.70 | 0.16 | feature 15 | feature 349 |
| 15 → 392 | 0.70 | 0.22 | feature 15 | feature 392 |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 320 → 1694 | 0.29 | 2.78 | feature 320 | feature 1694 |
| 405 → 1694 | 0.31 | 2.45 | feature 405 | feature 1694 |
| 457 → 1694 | 0.32 | 3.06 | feature 457 | feature 1694 |
| 446 → 1936 | 0.34 | 2.57 | feature 446 | feature 1936 |
| 417 → 1936 | 0.34 | 2.45 | feature 417 | feature 1936 |
| 247 → 1694 | 0.34 | 2.49 | feature 247 | feature 1694 |
| 311 → 1694 | 0.35 | 1.34 | feature 311 | feature 1694 |
| 187 → 1670 | 0.36 | 1.13 | feature 187 | feature 1670 |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1501 → 4217 | 0.13 | 0.92 | feature 1501 | feature 4217 |
| 1501 → 3014 | 0.14 | 0.45 | feature 1501 | feature 3014 |
| 1554 → 5936 | 0.16 | 0.81 | feature 1554 | feature 5936 |
| 1554 → 2624 | 0.18 | 0.72 | feature 1554 | feature 2624 |
| 1297 → 5963 | 0.22 | 12.80 | feature 1297 | feature 5963 |
| 1297 → 3772 | 0.23 | 3.97 | feature 1297 | feature 3772 |
| 1297 → 3513 | 0.23 | 1.79 | feature 1297 | feature 3513 |
| 1297 → 5838 | 0.24 | 0.89 | feature 1297 | feature 5838 |
