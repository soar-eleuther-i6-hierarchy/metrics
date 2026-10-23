# Exp 0 - metrics report

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  71 candidate edges

- **Out-degree**: 23 parents, 24 children, 11 multi-parented (PolyFrac 45.8%); top-1 parent holds 31.0% of edges, Gini 0.946, max out-degree 22.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.12; 14 edges (19.7%) at chance level (PMI < 0.5). 3 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 64/71 edges pass (90.1%).
- **Frequency control**: mean survival 0.901 over 49 testable edges; 5 (10.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.280 over 14 parents; 2 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.966, R_mass mean 0.958; 4 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.280.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 94 -> 351 | 1.00 | 0.12 | 3.70 | -0.02/0.38 | n | - | 0.99 | feature 94 | feature 351 |
| 202 -> 351 | 1.00 | 0.02 | 2.08 | -0.01/0.38 | n | - | 0.99 | feature 202 | feature 351 |
| 103 -> 351 | 1.00 | 0.02 | 1.83 | -0.00/0.38 | n | - | 0.33 | feature 103 | feature 351 |
| 117 -> 351 | 1.00 | 0.16 | 3.96 | 0.26/0.38 | Y | - | 0.24 | feature 117 | feature 351 |
| 42 -> 351 | 1.00 | 0.23 | 4.34 | 0.20/0.38 | Y | - | 0.45 | feature 42 | feature 351 |
| 197 -> 351 | 1.00 | 0.22 | 4.29 | 3.11/0.38 | Y | - | 0.33 | feature 197 | feature 351 |
| 106 -> 351 | 1.00 | 0.10 | 3.49 | 1.45/0.38 | Y | - | 0.24 | feature 106 | feature 351 |
| 58 -> 351 | 1.00 | 0.01 | 0.93 | 3.85/0.38 | Y | - | 0.03 | feature 58 | feature 351 |

## Block pair 1->2  -  18 candidate edges

- **Out-degree**: 13 parents, 8 children, 2 multi-parented (PolyFrac 25.0%); top-1 parent holds 33.3% of edges, Gini 0.957, max out-degree 6.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.14; 0 edges (0.0%) at chance level (PMI < 0.5). 3 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 7/18 edges pass (38.9%).
- **Frequency control**: mean survival 1.002 over 17 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.086 over 1 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.962, R_mass mean 0.961; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.087.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 245 -> 518 | 1.00 | 0.00 | 2.54 | 0.05/-0.00 | n | 1.00 | - | feature 245 | feature 518 |
| 254 -> 518 | 1.00 | 0.00 | 2.56 | 0.04/-0.00 | n | 1.00 | - | feature 254 | feature 518 |
| 304 -> 518 | 1.00 | 0.00 | 2.22 | 0.08/-0.00 | n | 1.00 | - | feature 304 | feature 518 |
| 307 -> 518 | 1.00 | 0.00 | 2.47 | 0.06/-0.00 | n | 1.00 | - | feature 307 | feature 518 |
| 309 -> 518 | 1.00 | 0.00 | 2.64 | 0.06/-0.00 | n | 1.00 | - | feature 309 | feature 518 |
| 396 -> 518 | 1.00 | 0.00 | 2.49 | 0.04/-0.00 | n | 1.00 | - | feature 396 | feature 518 |
| 438 -> 518 | 1.00 | 0.00 | 2.56 | 0.04/-0.00 | n | 1.00 | - | feature 438 | feature 518 |
| 301 -> 632 | 1.00 | 0.36 | 4.07 | 0.54/0.60 | Y | - | 0.09 | feature 301 | feature 632 |

## Block pair 2->3  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 1 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 3->4  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

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
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 2 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 6->7  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.70; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/1 edges pass (0.0%).
- **Frequency control**: mean survival 1.000 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.635, R_mass mean 0.610; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.009.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1403 -> 1651 | 1.00 | 0.01 | 4.70 | -0.02/-0.00 | n | 1.00 | - | feature 1403 | feature 1651 |
