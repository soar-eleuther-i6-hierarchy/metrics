# Exp 0 - metrics report

**Layer 0**　·　gemma-2-2b / trained_toy　·　toy_trained　·　20 latents in 2 blocks　·　199,936 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

## Block pair 0->1  -  9 candidate edges

- **Out-degree**: 3 parents, 9 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 33.3% of edges, Gini 0.000, max out-degree 3.
- **Superparents** (out-degree flag): 3 (3 also pass the old fire-rate AND-gate) — e.g. feature 0 _conjunctions and relational words that suggest connections …_: 3 children, fires on 14.9% of tokens
- **Independence null**: mean edge PMI 1.90; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 9/9 edges pass (100.0%).
- **Frequency control**: mean survival 1.000 over 9 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.000 over 3 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.671, R_mass mean 0.681; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.601.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 0 -> 9 | 1.00 | 0.20 | 1.90 | 68.91/48.42 | Y | 1.00 | 0.00 | conjunctions and relational words that … | specific dates and times mentioned in t… |
| 0 -> 16 | 1.00 | 0.20 | 1.90 | 72.00/51.09 | Y | 1.00 | 0.00 | conjunctions and relational words that … | terms related to errors or exceptions i… |
| 0 -> 17 | 1.00 | 0.20 | 1.90 | 66.63/47.04 | Y | 1.00 | 0.00 | conjunctions and relational words that … | the presence of various types of file e… |
| 1 -> 7 | 1.00 | 0.20 | 1.90 | 69.11/46.79 | Y | 1.00 | 0.00 | geographic locations and names associat… | terms related to health, particularly t… |
| 1 -> 10 | 1.00 | 0.20 | 1.90 | 69.21/46.88 | Y | 1.00 | 0.00 | geographic locations and names associat… | names of individuals in various contexts |
| 1 -> 18 | 1.00 | 0.21 | 1.90 | 74.01/52.72 | Y | 1.00 | 0.00 | geographic locations and names associat… | references to legal context or terminol… |
| 2 -> 5 | 1.00 | 0.20 | 1.90 | 67.47/46.39 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | phrases and concepts related to finalit… |
| 2 -> 6 | 1.00 | 0.20 | 1.90 | 73.16/51.03 | Y | 1.00 | 0.00 | mathematical terminology and concepts r… | references to numerical data or statist… |
