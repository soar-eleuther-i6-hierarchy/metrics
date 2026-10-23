# Exp 0 - metrics report

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  647 candidate edges

- **Out-degree**: 67 parents, 145 children, 59 multi-parented (PolyFrac 40.7%); top-1 parent holds 22.4% of edges, Gini 0.906, max out-degree 145.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 94 _feature 94_: 145 children, fires on 98.6% of tokens
- **Independence null**: mean edge PMI 1.43; 147 edges (22.7%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 623/647 edges pass (96.3%).
- **Frequency control**: mean survival 0.907 over 625 testable edges; 67 (10.7%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.149 over 50 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.984, R_mass mean 0.982; 8 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.538.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 47 -> 355 | 1.00 | 0.02 | 2.58 | 3.25/0.04 | Y | - | 0.07 | feature 47 | feature 355 |
| 157 -> 355 | 1.00 | 0.02 | 2.44 | 0.54/0.04 | Y | - | 0.08 | feature 157 | feature 355 |
| 94 -> 266 | 1.00 | 0.01 | 0.01 | 0.31/0.44 | Y | - | 0.02 | feature 94 | feature 266 |
| 94 -> 355 | 1.00 | 0.00 | 0.01 | 0.38/0.04 | Y | - | 0.02 | feature 94 | feature 355 |
| 94 -> 292 | 1.00 | 0.00 | 0.01 | 0.03/0.06 | Y | 1.00 | 0.02 | feature 94 | feature 292 |
| 94 -> 415 | 1.00 | 0.01 | 0.01 | 0.32/0.32 | Y | - | 0.02 | feature 94 | feature 415 |
| 150 -> 355 | 1.00 | 0.06 | 3.45 | 2.62/0.04 | Y | - | 0.16 | feature 150 | feature 355 |
| 129 -> 355 | 1.00 | 0.01 | 1.71 | 4.28/0.04 | Y | - | 0.07 | feature 129 | feature 355 |

## Block pair 1->2  -  18 candidate edges

- **Out-degree**: 15 parents, 8 children, 2 multi-parented (PolyFrac 25.0%); top-1 parent holds 22.2% of edges, Gini 0.943, max out-degree 4.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.76; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 18/18 edges pass (100.0%).
- **Frequency control**: mean survival 1.017 over 18 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.109 over 1 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.975, R_mass mean 0.981; 5 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.091.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 273 -> 635 | 0.96 | 0.17 | 2.71 | 0.09/0.05 | Y | 1.02 | 0.11 | feature 273 | feature 635 |
| 273 -> 558 | 0.95 | 0.15 | 2.70 | 0.08/0.08 | Y | 1.00 | 0.11 | feature 273 | feature 558 |
| 273 -> 649 | 0.92 | 0.10 | 2.67 | 0.08/0.04 | Y | 1.02 | 0.11 | feature 273 | feature 649 |
| 273 -> 617 | 0.86 | 0.03 | 2.61 | 0.02/0.02 | Y | 1.00 | 0.11 | feature 273 | feature 617 |
| 389 -> 545 | 0.86 | 0.02 | 2.18 | 0.14/0.04 | Y | 0.99 | - | feature 389 | feature 545 |
| 321 -> 545 | 0.86 | 0.02 | 2.22 | 0.13/0.04 | Y | 0.99 | - | feature 321 | feature 545 |
| 286 -> 545 | 0.78 | 0.01 | 1.94 | 0.07/0.04 | Y | 1.00 | - | feature 286 | feature 545 |
| 317 -> 617 | 0.77 | 0.09 | 3.71 | 0.08/0.02 | Y | 0.98 | - | feature 317 | feature 617 |

## Block pair 2->3  -  8 candidate edges

- **Out-degree**: 5 parents, 8 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 50.0% of edges, Gini 0.984, max out-degree 4.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.61; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 6/8 edges pass (75.0%).
- **Frequency control**: mean survival 1.019 over 8 testable edges; 1 (12.5%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.015 over 1 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.799, R_mass mean 0.821; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.106.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 639 -> 842 | 0.99 | 0.04 | 3.91 | 3.15/0.57 | Y | 1.00 | - | feature 639 | feature 842 |
| 520 -> 767 | 0.84 | 0.09 | 2.66 | 0.13/0.10 | Y | 1.11 | 0.02 | feature 520 | feature 767 |
| 520 -> 860 | 0.72 | 0.05 | 2.51 | 0.16/0.01 | n | 1.13 | 0.02 | feature 520 | feature 860 |
| 573 -> 836 | 0.71 | 0.06 | 3.02 | 0.09/0.09 | Y | 1.06 | - | feature 573 | feature 836 |
| 520 -> 707 | 0.69 | 0.18 | 2.47 | 0.16/0.06 | Y | 1.25 | 0.02 | feature 520 | feature 707 |
| 630 -> 894 | 0.57 | 0.05 | 2.10 | 0.01/0.02 | n | 0.20 | - | feature 630 | feature 894 |
| 633 -> 839 | 0.55 | 0.05 | 2.03 | 0.05/0.16 | Y | 1.00 | - | feature 633 | feature 839 |
| 520 -> 888 | 0.53 | 0.00 | 2.19 | 0.10/0.01 | Y | 1.40 | 0.02 | feature 520 | feature 888 |

## Block pair 3->4  -  2 candidate edges

- **Out-degree**: 1 parents, 2 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 2.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.04; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2/2 edges pass (100.0%).
- **Frequency control**: mean survival 0.960 over 2 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.273 over 1 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.769, R_mass mean 0.889; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.610.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 707 -> 1022 | 0.93 | 0.17 | 4.09 | 0.09/0.02 | Y | 1.02 | 0.27 | feature 707 | feature 1022 |
| 707 -> 971 | 0.85 | 0.44 | 4.00 | 0.08/0.29 | Y | 0.90 | 0.27 | feature 707 | feature 971 |

## Block pair 4->5  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 5->6  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 6->7  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.
