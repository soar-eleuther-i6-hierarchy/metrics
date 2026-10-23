# Exp 0 - metrics report

**Layer 0**　·　gemma-2-2b / trained_toy　·　toy_trained　·　20 latents in 2 blocks　·　199,936 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

## Block pair 0->1  -  9 candidate edges

- **Out-degree**: 3 parents, 9 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 33.3% of edges, Gini 0.000, max out-degree 3.
- **Superparents** (out-degree flag): 3 (3 also pass the old fire-rate AND-gate) — e.g. feature 1 _geographic locations and names associated with specific pla…_: 3 children, fires on 14.9% of tokens
- **Independence null**: mean edge PMI 1.90; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 9/9 edges pass (100.0%).
- **Frequency control**: mean survival 1.000 over 9 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.000 over 3 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.671, R_mass mean 0.682; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.601.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1 -> 6 | 1.00 | 0.20 | 1.90 | 72.15/49.91 | Y | 1.00 | 0.00 | geographic locations and names associat… | references to numerical data or statist… |
| 1 -> 17 | 1.00 | 0.20 | 1.90 | 70.17/48.23 | Y | 1.00 | 0.00 | geographic locations and names associat… | the presence of various types of file e… |
| 1 -> 18 | 1.00 | 0.20 | 1.90 | 66.95/47.04 | Y | 1.00 | 0.00 | geographic locations and names associat… | references to legal context or terminol… |
| 2 -> 4 | 1.00 | 0.20 | 1.90 | 69.42/47.32 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | terms related to data management and da… |
| 2 -> 5 | 1.00 | 0.20 | 1.90 | 70.63/49.71 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | phrases and concepts related to finalit… |
| 2 -> 8 | 1.00 | 0.21 | 1.90 | 70.38/49.80 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | articles and determiners in the text |
| 3 -> 11 | 1.00 | 0.20 | 1.90 | 73.67/50.98 | Y | 1.00 | 0.00 | instances of the verb "to be" in variou… | programmatic structures and error handl… |
| 3 -> 12 | 1.00 | 0.20 | 1.90 | 69.99/47.84 | Y | 1.00 | 0.00 | instances of the verb "to be" in variou… | mathematical expressions and notations |
