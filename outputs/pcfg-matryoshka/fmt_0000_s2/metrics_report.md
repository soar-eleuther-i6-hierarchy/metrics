# Exp 0 - metrics report

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　1,022,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  184 candidate edges

- **Out-degree**: 18 parents, 110 children, 49 multi-parented (PolyFrac 44.5%); top-1 parent holds 54.9% of edges, Gini 0.981, max out-degree 101.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 191 _feature 191_: 101 children, fires on 55.8% of tokens
- **Independence null**: mean edge PMI 0.58; 102 edges (55.4%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 184/184 edges pass (100.0%).
- **Frequency control**: mean survival 1.003 over 184 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.145 over 12 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.978, R_mass mean 0.967; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.302.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 99 -> 424 | 0.99 | 0.01 | 1.15 | 5.24/0.19 | Y | 1.00 | 0.08 | feature 99 | feature 424 |
| 156 -> 424 | 0.98 | 0.16 | 4.02 | 1.05/0.19 | Y | 1.00 | 0.15 | feature 156 | feature 424 |
| 156 -> 330 | 0.97 | 0.08 | 4.01 | 2.10/0.13 | Y | 1.00 | 0.15 | feature 156 | feature 330 |
| 29 -> 317 | 0.96 | 0.07 | 3.72 | 1.56/0.15 | Y | 1.00 | - | feature 29 | feature 317 |
| 140 -> 430 | 0.95 | 0.15 | 2.07 | 0.35/0.07 | Y | 1.00 | 0.09 | feature 140 | feature 430 |
| 126 -> 424 | 0.95 | 0.13 | 3.80 | 2.45/0.19 | Y | 1.00 | 0.31 | feature 126 | feature 424 |
| 140 -> 317 | 0.92 | 0.01 | 2.03 | 0.68/0.15 | Y | 1.01 | 0.09 | feature 140 | feature 317 |
| 181 -> 317 | 0.83 | 0.01 | 1.51 | 0.71/0.15 | Y | 1.01 | 0.00 | feature 181 | feature 317 |

## Block pair 1->2  -  3 candidate edges

- **Out-degree**: 3 parents, 2 children, 1 multi-parented (PolyFrac 50.0%); top-1 parent holds 33.3% of edges, Gini 0.987, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 1.96; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/3 edges pass (33.3%).
- **Frequency control**: mean survival 0.991 over 3 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.987, R_mass mean 0.988; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.002.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 229 -> 645 | 0.67 | 0.00 | 1.99 | 0.02/0.00 | n | 0.93 | - | feature 229 | feature 645 |
| 376 -> 645 | 0.66 | 0.00 | 1.78 | 0.01/0.00 | n | 1.02 | - | feature 376 | feature 645 |
| 299 -> 562 | 0.61 | 0.00 | 2.11 | 0.02/0.29 | Y | 1.02 | - | feature 299 | feature 562 |

## Block pair 2->3  -  2 candidate edges

- **Out-degree**: 2 parents, 2 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 50.0% of edges, Gini 0.991, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.30; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/2 edges pass (50.0%).
- **Frequency control**: mean survival 1.073 over 2 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.923, R_mass mean 0.929; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.002.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 615 -> 752 | 0.61 | 0.00 | 2.51 | 0.03/0.01 | n | 1.01 | - | feature 615 | feature 752 |
| 594 -> 888 | 0.54 | 0.00 | 2.08 | 0.02/0.02 | Y | 1.14 | - | feature 594 | feature 888 |

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

## Block pair 5->6  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.37; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/1 edges pass (0.0%).
- **Frequency control**: mean survival 0.808 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.767, R_mass mean 0.736; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.002.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1167 -> 1541 | 0.55 | 0.00 | 3.37 | 0.13/-0.00 | n | 0.81 | - | feature 1167 | feature 1541 |

## Block pair 6->7  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.10; 0 edges (0.0%) at chance level (PMI < 0.5). 1 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 0.901 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.846, R_mass mean 0.803; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.005.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1463 -> 1710 | 0.68 | 0.01 | 4.10 | 0.20/0.02 | Y | 0.90 | - | feature 1463 | feature 1710 |
