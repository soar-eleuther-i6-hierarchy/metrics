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

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,971 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  3260 candidate edges

- **Out-degree**: 80 parents, 384 children, 383 multi-parented (PolyFrac 99.7%); top-1 parent holds 11.8% of edges, Gini 0.732, max out-degree 384.
- **Superparents** (out-degree flag): 5 (5 also pass the old fire-rate AND-gate) — e.g. feature 44 _years represented as four-digit numbers, often with additio…_: 384 children, fires on 98.8% of tokens
- **Independence null**: mean edge PMI 1.26; 1227 edges (37.6%) at chance level (PMI < 0.5). 2 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 533/3260 edges pass (16.3%).
- **Frequency control**: mean survival 0.594 over 3209 testable edges; 1407 (43.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.273 over 70 parents; 4 over the 0.5 global threshold.
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.789.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 86 -> 162 | 1.00 | 0.02 | 1.08 | -0.00/0.00 | n | - | 0.22 | occurrences of the words "it", "is", an… | articles ("the", "a") and prepositions … |
| 91 -> 162 | 1.00 | 0.36 | 3.79 | -0.00/0.00 | n | - | 0.46 | terms related to medical and biological… | articles ("the", "a") and prepositions … |
| 123 -> 162 | 1.00 | 0.16 | 2.99 | -0.00/0.00 | n | - | 0.42 | numerical data and special characters o… | articles ("the", "a") and prepositions … |
| 44 -> 318 | 1.00 | 0.03 | 0.01 | 0.18/-0.02 | n | 1.00 | 0.03 | years represented as four-digit numbers… | physical units of measurement |
| 44 -> 283 | 1.00 | 0.00 | 0.01 | 1.69/0.05 | Y | 1.00 | 0.03 | years represented as four-digit numbers… | legal citations |
| 18 -> 162 | 1.00 | 0.20 | 3.18 | -0.01/0.00 | n | - | 0.39 | quantitative data and statistical resul… | articles ("the", "a") and prepositions … |
| 44 -> 272 | 1.00 | 0.02 | 0.01 | 0.18/-0.00 | n | 1.00 | 0.03 | years represented as four-digit numbers… | the word "with" near words like "the", … |
| 44 -> 162 | 1.00 | 0.01 | 0.01 | 0.18/0.00 | n | - | 0.03 | years represented as four-digit numbers… | articles ("the", "a") and prepositions … |

## Block pair 1->2  -  34061 candidate edges

- **Out-degree**: 181 parents, 607 children, 385 multi-parented (PolyFrac 63.4%); top-1 parent holds 0.9% of edges, Gini 0.665, max out-degree 303.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.05; 170 edges (0.5%) at chance level (PMI < 0.5). 81 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 132/34061 edges pass (0.4%).
- **Frequency control**: mean survival 0.067 over 33540 testable edges; 33090 (98.7%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.476 over 154 parents; 122 over the 0.5 global threshold.
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.766.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 436 -> 601 | 0.99 | 0.14 | 2.81 | -0.00/-0.00 | n | - | 0.52 | citations to legal authority in case la… | a mashup of topics, activating on menti… |
| 505 -> 601 | 0.99 | 0.06 | 1.98 | 0.00/-0.00 | n | - | 0.50 | mathematical symbols and notation | a mashup of topics, activating on menti… |
| 424 -> 1307 | 0.99 | 0.07 | 2.06 | -0.00/-0.01 | n | - | 0.48 | LaTeX code | the word "to" followed by a verb |
| 487 -> 1307 | 0.99 | 0.14 | 2.82 | -0.00/-0.01 | n | - | 0.50 | phrases starting with the word "all". | the word "to" followed by a verb |
| 292 -> 1307 | 0.99 | 0.22 | 3.27 | -0.00/-0.01 | n | - | 0.52 | the word "non" and phrases containing "… | the word "to" followed by a verb |
| 259 -> 1307 | 0.99 | 0.05 | 1.84 | -0.00/-0.01 | n | - | 0.48 | words and short phrases indicating chan… | the word "to" followed by a verb |
| 292 -> 601 | 0.99 | 0.22 | 3.27 | -0.00/-0.00 | n | - | 0.52 | the word "non" and phrases containing "… | a mashup of topics, activating on menti… |
| 506 -> 601 | 0.99 | 0.12 | 2.70 | -0.00/-0.00 | n | - | 0.51 | code and programming language-related c… | a mashup of topics, activating on menti… |

## Block pair 2->3  -  430547 candidate edges

- **Out-degree**: 578 parents, 1423 children, 1196 multi-parented (PolyFrac 84.0%); top-1 parent holds 0.3% of edges, Gini 0.711, max out-degree 1151.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.74; 0 edges (0.0%) at chance level (PMI < 0.5). 580 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 119/430547 edges pass (0.0%).
- **Frequency control**: mean survival 0.022 over 401144 testable edges; 400623 (99.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.459 over 517 parents; 47 over the 0.5 global threshold.
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.872.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1343 -> 6486 | 1.00 | 0.20 | 5.06 | -0.00/0.01 | n | - | 0.63 | the word "title" in various contexts, e… | the word "geometric", often found in ma… |
| 1134 -> 7820 | 1.00 | 0.30 | 3.61 | -0.00/-0.00 | n | - | 0.46 | acronyms, abbreviations, and symbols | code snippets and URLs |
| 1572 -> 5875 | 1.00 | 0.18 | 5.33 | 0.24/0.06 | Y | 1.00 | - | words that begin with the "ph" sound | phrases indicating something has been s… |
| 1938 -> 6486 | 1.00 | 0.08 | 4.13 | 0.02/0.01 | Y | - | 0.47 | numbered/lettered lists with parenthesi… | the word "geometric", often found in ma… |
| 1685 -> 7820 | 1.00 | 0.71 | 4.46 | 0.00/-0.00 | n | - | 0.47 | the word "format" followed by a colon | code snippets and URLs |
| 1233 -> 7820 | 1.00 | 0.19 | 3.12 | 0.00/-0.00 | n | - | 0.45 | mentions of physical cards, stamps, and… | code snippets and URLs |
| 956 -> 7820 | 1.00 | 0.55 | 4.21 | 0.00/-0.00 | n | - | 0.47 | code syntax including length, parenthes… | code snippets and URLs |
| 1348 -> 6486 | 1.00 | 0.10 | 4.39 | 0.00/0.01 | n | - | 0.47 | mathematical notation | the word "geometric", often found in ma… |
