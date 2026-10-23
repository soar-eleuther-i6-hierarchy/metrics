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
