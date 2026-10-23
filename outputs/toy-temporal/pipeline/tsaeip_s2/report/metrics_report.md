# Exp 0 - metrics report

**Layer 0**　·　gemma-2-2b / trained_toy　·　toy_trained　·　20 latents in 2 blocks　·　199,936 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

## Block pair 0->1  -  10 candidate edges

- **Out-degree**: 3 parents, 8 children, 2 multi-parented (PolyFrac 25.0%); top-1 parent holds 40.0% of edges, Gini 0.067, max out-degree 4.
- **Superparents** (out-degree flag): 3 (3 also pass the old fire-rate AND-gate) — e.g. feature 2 _mathematical terminology and concepts related to gauge theo…_: 4 children, fires on 15.0% of tokens
- **Independence null**: mean edge PMI 1.64; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 10/10 edges pass (100.0%).
- **Frequency control**: mean survival 1.000 over 10 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.161 over 3 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.671, R_mass mean 0.684; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.773.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 0 -> 12 | 1.00 | 0.20 | 1.90 | 98.00/66.22 | Y | 1.00 | 0.00 | conjunctions and relational words that … | mathematical expressions and notations |
| 0 -> 14 | 1.00 | 0.20 | 1.90 | 99.38/68.59 | Y | 1.00 | 0.00 | conjunctions and relational words that … | phrases indicating progression or conti… |
| 0 -> 15 | 1.00 | 0.20 | 1.90 | 99.43/67.39 | Y | 1.00 | 0.00 | conjunctions and relational words that … | references to specific entities or impo… |
| 2 -> 18 | 1.00 | 0.20 | 1.90 | 97.82/69.10 | Y | 1.00 | 0.06 | mathematical terminology and concepts r… | references to legal context or terminol… |
| 2 -> 19 | 1.00 | 0.20 | 1.90 | 92.91/64.31 | Y | 1.00 | 0.06 | mathematical terminology and concepts r… | function definitions and calls in progr… |
| 3 -> 8 | 0.71 | 0.42 | 1.55 | 53.32/177.19 | Y | 1.00 | 0.42 | instances of the verb "to be" in variou… | articles and determiners in the text |
| 2 -> 13 | 0.58 | 0.23 | 1.35 | 42.88/275.59 | Y | 1.00 | 0.06 | mathematical terminology and concepts r… | scientific terms and concepts related t… |
| 2 -> 17 | 0.57 | 0.23 | 1.34 | 42.62/282.49 | Y | 1.00 | 0.06 | mathematical terminology and concepts r… | the presence of various types of file e… |
