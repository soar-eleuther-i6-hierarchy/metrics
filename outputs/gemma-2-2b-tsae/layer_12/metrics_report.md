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
