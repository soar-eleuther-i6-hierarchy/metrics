# Exp 0 - metrics report

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  1473 candidate edges

- **Out-degree**: 63 parents, 382 children, 381 multi-parented (PolyFrac 99.7%); top-1 parent holds 25.9% of edges, Gini 0.906, max out-degree 382.
- **Superparents** (out-degree flag): 2 (2 also pass the old fire-rate AND-gate) — e.g. feature 44 _technical terminology related to programming and coding con…_: 382 children, fires on 98.8% of tokens
- **Independence null**: mean edge PMI 0.37; 1118 edges (75.9%) at chance level (PMI < 0.5). 2 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 794/1473 edges pass (53.9%).
- **Frequency control**: mean survival 0.998 over 1473 testable edges; 26 (1.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.033 over 49 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.506.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 44 -> 318 | 1.00 | 0.03 | 0.01 | 1.66/0.03 | Y | 1.00 | 0.01 | technical terminology related to progra… | code-related constructs and syntax |
| 44 -> 203 | 1.00 | 0.00 | 0.01 | 1.01/-0.00 | n | 1.00 | 0.01 | technical terminology related to progra… | mathematical symbols and expressions |
| 44 -> 283 | 1.00 | 0.00 | 0.01 | 1.69/0.05 | Y | 1.00 | 0.01 | technical terminology related to progra… | terms related to cell biology, specific… |
| 44 -> 139 | 1.00 | 0.00 | 0.01 | 1.43/0.01 | Y | 1.00 | 0.01 | technical terminology related to progra… | mathematical expressions and calculatio… |
| 44 -> 272 | 1.00 | 0.01 | 0.01 | 1.44/0.01 | n | 1.00 | 0.01 | technical terminology related to progra… | expressions of encouragement and suppor… |
| 44 -> 161 | 1.00 | 0.03 | 0.01 | 1.74/-0.00 | n | 1.00 | 0.01 | technical terminology related to progra… | testimonies related to statements made … |
| 44 -> 452 | 1.00 | 0.03 | 0.01 | 2.05/0.00 | n | 1.00 | 0.01 | technical terminology related to progra… | scientific terms and results related to… |
| 44 -> 158 | 1.00 | 0.02 | 0.01 | 1.92/0.01 | Y | 1.00 | 0.01 | technical terminology related to progra… | terms related to cellular processes and… |

## Block pair 1->2  -  621 candidate edges

- **Out-degree**: 97 parents, 402 children, 88 multi-parented (PolyFrac 21.9%); top-1 parent holds 44.0% of edges, Gini 0.932, max out-degree 273.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 1.43; 260 edges (41.9%) at chance level (PMI < 0.5). 93 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 197/621 edges pass (31.7%).
- **Frequency control**: mean survival 0.751 over 620 testable edges; 150 (24.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.178 over 52 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 1 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.223.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 292 -> 831 | 0.95 | 0.05 | 3.46 | 0.03/-0.00 | n | 1.03 | 0.00 | terms related to account verification a… | references to outcomes and results rela… |
| 260 -> 768 | 0.94 | 0.03 | 3.94 | 0.52/0.38 | Y | - | 0.14 | elements related to programming imports… | code import statements from programming… |
| 383 -> 1503 | 0.93 | 0.01 | 0.82 | 0.04/0.16 | Y | 1.00 | 0.01 | terms related to mathematical proofs an… | formal language used to describe invent… |
| 375 -> 1634 | 0.86 | 0.08 | 1.36 | 0.03/0.00 | n | 0.96 | 0.04 | terms related to promotional offers and… | persuasive language related to service … |
| 383 -> 936 | 0.86 | 0.01 | 0.74 | 0.01/0.06 | Y | 0.97 | 0.01 | terms related to mathematical proofs an… | terms related to inventions and their d… |
| 450 -> 1786 | 0.86 | 0.00 | 1.05 | 0.01/0.35 | Y | 1.00 | 0.01 | terms related to statistical models and… | references to inventions and their desc… |
| 383 -> 1786 | 0.86 | 0.00 | 0.74 | 0.04/0.35 | Y | 1.02 | 0.01 | terms related to mathematical proofs an… | references to inventions and their desc… |
| 383 -> 1535 | 0.85 | 0.16 | 0.74 | 0.02/0.01 | Y | 0.98 | 0.01 | terms related to mathematical proofs an… | terms and concepts related to Scala pro… |

## Block pair 2->3  -  1747 candidate edges

- **Out-degree**: 247 parents, 337 children, 74 multi-parented (PolyFrac 22.0%); top-1 parent holds 2.2% of edges, Gini 0.934, max out-degree 38.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.21; 0 edges (0.0%) at chance level (PMI < 0.5). 618 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 156/1747 edges pass (8.9%).
- **Frequency control**: mean survival 0.244 over 1343 testable edges; 1043 (77.7%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.349 over 138 parents; 71 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 1 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.407.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1572 -> 5875 | 1.00 | 0.18 | 5.32 | 0.24/0.06 | Y | 1.00 | - | ellipses and incomplete thoughts | ellipses or dramatic pauses in text |
| 1938 -> 6486 | 1.00 | 0.16 | 4.83 | 0.02/0.01 | Y | - | 0.62 | comments or annotations within code | mathematical symbols and expressions, p… |
| 1343 -> 6486 | 1.00 | 0.20 | 5.05 | -0.00/0.01 | n | - | 0.62 | instances of the word "when" in various… | mathematical symbols and expressions, p… |
| 1348 -> 6486 | 1.00 | 0.30 | 5.46 | 0.00/0.01 | n | - | 0.62 | documents related to legal proceedings … | mathematical symbols and expressions, p… |
| 1734 -> 6486 | 0.98 | 0.15 | 4.76 | -0.00/0.01 | n | - | 0.62 | phrases regarding minimum requirements … | mathematical symbols and expressions, p… |
| 889 -> 6486 | 0.98 | 0.30 | 5.47 | 0.00/0.01 | n | - | 0.62 | negative numerical values or operations | mathematical symbols and expressions, p… |
| 1438 -> 6486 | 0.98 | 0.19 | 4.98 | 0.00/0.01 | n | - | 0.53 | time-related information, particularly … | mathematical symbols and expressions, p… |
| 1149 -> 6486 | 0.98 | 0.04 | 3.42 | 0.00/0.01 | n | - | 0.62 | structural and procedural elements in a… | mathematical symbols and expressions, p… |
