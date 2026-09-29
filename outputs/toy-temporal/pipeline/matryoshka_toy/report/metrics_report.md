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
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../../">SOAR I-6 · metrics</a><a class="" href="../../../outputs/">Results</a><a class="" href="../../../outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="../../../outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="../../../outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="on" href="../../../outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../../outputs/gemma-2-2b/layer_01/metrics_report.html">1</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_03/metrics_report.html">3</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_06/metrics_report.html">6</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_12/metrics_report.html">12</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_18/metrics_report.html">18</a><a class="pill" href="../../../outputs/gemma-2-2b/layer_24/metrics_report.html">24</a><span class="sep"></span><span class="lbl">Page</span></div></nav>

# Exp 0 - metrics report

**Layer 0**　·　gemma-2-2b / trained_toy　·　toy_trained　·　199,936 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

## Block pair 0->1  -  9 candidate edges

- **Out-degree**: 3 parents, 9 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 33.3% of edges, Gini 0.000, max out-degree 3.
- **Superparents** (out-degree flag): 3 (3 also pass the old fire-rate AND-gate) — e.g. feature 0 _conjunctions and relational words that suggest connections …_: 3 children, fires on 15.0% of tokens
- **Independence null**: mean edge PMI 1.90; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 9/9 edges pass (100.0%).
- **Frequency control**: mean survival 1.000 over 9 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.000 over 3 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.671, R_mass mean 0.739; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.601.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 0 -> 11 | 1.00 | 0.20 | 1.90 | 85.91/15.29 | Y | 1.00 | 0.00 | conjunctions and relational words that … | programmatic structures and error handl… |
| 0 -> 14 | 1.00 | 0.21 | 1.90 | 65.42/18.13 | Y | 1.00 | 0.00 | conjunctions and relational words that … | phrases indicating progression or conti… |
| 0 -> 18 | 1.00 | 0.20 | 1.90 | 29.05/9.25 | Y | 1.00 | 0.00 | conjunctions and relational words that … | references to legal context or terminol… |
| 1 -> 12 | 1.00 | 0.20 | 1.90 | 32.67/4.67 | Y | 1.00 | 0.00 | geographic locations and names associat… | mathematical expressions and notations |
| 1 -> 15 | 1.00 | 0.20 | 1.90 | 26.15/3.13 | Y | 1.00 | 0.00 | geographic locations and names associat… | references to specific entities or impo… |
| 1 -> 17 | 1.00 | 0.20 | 1.90 | 33.93/4.40 | Y | 1.00 | 0.00 | geographic locations and names associat… | the presence of various types of file e… |
| 2 -> 13 | 1.00 | 0.20 | 1.90 | 94.42/20.68 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | scientific terms and concepts related t… |
| 2 -> 16 | 1.00 | 0.20 | 1.90 | 84.50/15.00 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | terms related to errors or exceptions i… |
