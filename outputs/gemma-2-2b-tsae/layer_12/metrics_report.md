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
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="../../">SOAR I-6 · metrics</a><a class="" href="../../outputs/">Results</a><a class="" href="../../outputs/cross_depth_comparison.html">Cross-depth</a><a class="" href="../../outputs/kill_rates.html">Kill rates</a><a class="" href="../../outputs/toy_calibration.html">Toy calibration</a><a class="" href="../../outputs/trained_toy_calibration.html">Trained toy</a></div><div class="row"><span class="lbl">Layer</span><a class="pill" href="../../outputs/layer_03/metrics_report.html">3</a><a class="pill" href="../../outputs/layer_06/metrics_report.html">6</a><a class="pill on" href="../../outputs/layer_12/metrics_report.html">12</a><a class="pill" href="../../outputs/layer_18/metrics_report.html">18</a><a class="pill" href="../../outputs/layer_24/metrics_report.html">24</a><span class="sep"></span><span class="lbl">Page</span><a class="" href="../../outputs/layer_12/metrics_dashboard.html">Dashboard</a><a class="" href="../../outputs/layer_12/superparent_sankey.html">Superparents</a><a class="" href="../../outputs/layer_12/qualitative_dashboard.html">Qualitative</a><a class="on" href="../../outputs/layer_12/metrics_report.html">Metrics report</a><a class="" href="../../outputs/layer_12/qualitative_check.html">Qualitative report</a></div></nav>

# Exp 0 - metrics report

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  1621 candidate edges

- **Out-degree**: 494 parents, 978 children, 281 multi-parented (PolyFrac 28.7%); top-1 parent holds 3.6% of edges, Gini 0.928, max out-degree 59.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.15; 0 edges (0.0%) at chance level (PMI < 0.5). 2281 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1598/1621 edges pass (98.6%).
- **Frequency control**: mean survival 0.909 over 1298 testable edges; 107 (8.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.091 over 241 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 26 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.381.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1455 -> 9045 | 1.00 | 0.14 | 5.33 | 0.09/0.08 | Y | 1.00 | - | terms related to electronics, telecommu… | the words "clinical" and "preclinical",… |
| 2690 -> 5847 | 1.00 | 0.12 | 4.81 | 0.30/0.13 | Y | 1.00 | 0.20 | abbreviations, acronyms, and names that… | numerical references within square brac… |
| 2690 -> 6185 | 1.00 | 0.09 | 4.81 | 0.41/1.04 | Y | 1.00 | 0.20 | abbreviations, acronyms, and names that… | numbers, especially those representing … |
| 2690 -> 8353 | 1.00 | 0.15 | 4.81 | 2.00/96.88 | Y | - | 0.20 | abbreviations, acronyms, and names that… | the abbreviation "TLS" |
| 1138 -> 8353 | 1.00 | 0.15 | 4.81 | 1.99/96.88 | Y | - | 0.20 | URLs | the abbreviation "TLS" |
| 3002 -> 8353 | 1.00 | 0.15 | 4.81 | 1.95/96.88 | Y | - | 0.20 | references to legal case numbers | the abbreviation "TLS" |
| 2559 -> 6185 | 1.00 | 0.09 | 4.81 | 0.39/1.04 | Y | 1.00 | 0.20 | terms related to fine particulate matter | numbers, especially those representing … |
| 1491 -> 8353 | 1.00 | 0.15 | 4.81 | 1.95/96.88 | Y | - | 0.20 | references to figures in research papers | the abbreviation "TLS" |
