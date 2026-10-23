# Exp 0 - metrics report

**Layer 0**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L0　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  749 candidate edges

- **Out-degree**: 49 parents, 137 children, 137 multi-parented (PolyFrac 100.0%); top-1 parent holds 18.3% of edges, Gini 0.962, max out-degree 137.
- **Superparents** (out-degree flag): 5 (5 also pass the old fire-rate AND-gate) — e.g. feature 129 _feature 129_: 137 children, fires on 100.0% of tokens
- **Independence null**: mean edge PMI 0.40; 674 edges (90.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 724/749 edges pass (96.7%).
- **Frequency control**: mean survival 1.000 over 749 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.014 over 19 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.756, R_mass mean 0.780; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.187.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 46 -> 259 | 1.00 | 0.00 | 0.55 | 1.02/0.02 | Y | 1.00 | 0.03 | feature 46 | feature 259 |
| 129 -> 401 | 1.00 | 0.00 | 0.00 | 2.39/0.01 | n | 1.00 | 0.03 | feature 129 | feature 401 |
| 129 -> 368 | 1.00 | 0.02 | 0.00 | 2.20/0.49 | Y | 1.00 | 0.03 | feature 129 | feature 368 |
| 149 -> 401 | 1.00 | 0.00 | 2.39 | 0.34/0.01 | n | 1.00 | - | feature 149 | feature 401 |
| 129 -> 405 | 1.00 | 0.02 | 0.00 | 2.55/0.44 | Y | 1.00 | 0.03 | feature 129 | feature 405 |
| 19 -> 401 | 1.00 | 0.00 | 1.86 | 0.56/0.01 | n | 1.00 | - | feature 19 | feature 401 |
| 213 -> 259 | 1.00 | 0.00 | 0.28 | 0.40/0.02 | Y | 1.00 | 0.03 | feature 213 | feature 259 |
| 186 -> 401 | 1.00 | 0.00 | 2.22 | 0.58/0.01 | n | 1.00 | - | feature 186 | feature 401 |

## Block pair 1->2  -  8 candidate edges

- **Out-degree**: 8 parents, 3 children, 2 multi-parented (PolyFrac 66.7%); top-1 parent holds 12.5% of edges, Gini 0.964, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.28; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 8/8 edges pass (100.0%).
- **Frequency control**: mean survival 0.981 over 8 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.935, R_mass mean 0.937; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.022.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 399 -> 530 | 0.67 | 0.03 | 2.49 | 0.43/0.97 | Y | 0.93 | - | feature 399 | feature 530 |
| 427 -> 530 | 0.65 | 0.03 | 2.42 | 0.16/0.97 | Y | 0.93 | - | feature 427 | feature 530 |
| 373 -> 530 | 0.65 | 0.03 | 2.41 | 0.11/0.97 | Y | 0.94 | - | feature 373 | feature 530 |
| 332 -> 617 | 0.56 | 0.01 | 1.97 | 0.12/0.81 | Y | 1.02 | - | feature 332 | feature 617 |
| 438 -> 617 | 0.55 | 0.02 | 2.40 | 0.15/0.81 | Y | 1.00 | - | feature 438 | feature 617 |
| 353 -> 637 | 0.53 | 0.02 | 2.25 | 0.15/2.34 | Y | 1.00 | - | feature 353 | feature 637 |
| 261 -> 617 | 0.52 | 0.02 | 2.29 | 0.15/0.81 | Y | 0.99 | - | feature 261 | feature 617 |
| 394 -> 617 | 0.51 | 0.01 | 2.01 | 0.08/0.81 | Y | 1.02 | - | feature 394 | feature 617 |

## Block pair 2->3  -  2 candidate edges

- **Out-degree**: 2 parents, 2 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 50.0% of edges, Gini 0.991, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.08; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2/2 edges pass (100.0%).
- **Frequency control**: mean survival 1.015 over 2 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.752, R_mass mean 0.699; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.030.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 453 -> 798 | 0.84 | 0.03 | 3.49 | 0.22/0.75 | Y | 1.00 | - | feature 453 | feature 798 |
| 660 -> 802 | 0.56 | 0.03 | 2.67 | 0.28/1.39 | Y | 1.03 | - | feature 660 | feature 802 |

## Block pair 3->4  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.65; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 0.994 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.629, R_mass mean 0.733; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.095.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 819 -> 924 | 0.95 | 0.09 | 4.65 | 0.99/0.59 | Y | 0.99 | - | feature 819 | feature 924 |

## Block pair 4->5  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 5->6  -  2 candidate edges

- **Out-degree**: 2 parents, 2 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 50.0% of edges, Gini 0.991, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 5.08; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2/2 edges pass (100.0%).
- **Frequency control**: mean survival 0.964 over 2 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.555, R_mass mean 0.273; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.231.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1281 -> 1544 | 0.69 | 0.20 | 5.12 | 0.09/0.61 | Y | 1.00 | - | feature 1281 | feature 1544 |
| 1126 -> 1457 | 0.63 | 0.26 | 5.05 | 0.07/0.44 | Y | 0.93 | - | feature 1126 | feature 1457 |

## Block pair 6->7  -  6 candidate edges

- **Out-degree**: 6 parents, 5 children, 1 multi-parented (PolyFrac 20.0%); top-1 parent holds 16.7% of edges, Gini 0.973, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 5.53; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 6/6 edges pass (100.0%).
- **Frequency control**: mean survival 0.994 over 6 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.389, R_mass mean 0.179; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.275.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1349 -> 1663 | 0.94 | 0.24 | 5.51 | 0.14/0.54 | Y | 1.01 | - | feature 1349 | feature 1663 |
| 1441 -> 1663 | 0.92 | 0.28 | 5.65 | 0.12/0.54 | Y | 1.01 | - | feature 1441 | feature 1663 |
| 1386 -> 1669 | 0.88 | 0.27 | 5.63 | 0.06/0.45 | Y | 1.01 | - | feature 1386 | feature 1669 |
| 1450 -> 1640 | 0.85 | 0.18 | 5.49 | 0.05/0.20 | Y | 0.93 | - | feature 1450 | feature 1640 |
| 1383 -> 1715 | 0.67 | 0.31 | 5.62 | 0.06/0.26 | Y | 1.00 | - | feature 1383 | feature 1715 |
| 1442 -> 1729 | 0.52 | 0.37 | 5.30 | 0.07/0.25 | Y | 1.00 | - | feature 1442 | feature 1729 |
