# Exp 0 - metrics report

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　1,022,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  30 candidate edges

- **Out-degree**: 8 parents, 20 children, 5 multi-parented (PolyFrac 25.0%); top-1 parent holds 56.7% of edges, Gini 0.985, max out-degree 17.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 0.82; 5 edges (16.7%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 26/30 edges pass (86.7%).
- **Frequency control**: mean survival 1.005 over 30 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.007 over 4 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.991, R_mass mean 0.975; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.214.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 185 -> 287 | 0.99 | 0.00 | 1.19 | 0.66/-0.03 | n | 1.00 | 0.02 | feature 185 | feature 287 |
| 125 -> 446 | 0.96 | 0.11 | 1.05 | 0.25/0.03 | Y | 1.00 | 0.00 | feature 125 | feature 446 |
| 193 -> 446 | 0.96 | 0.11 | 1.10 | 0.42/0.03 | Y | 1.00 | 0.00 | feature 193 | feature 446 |
| 185 -> 343 | 0.96 | 0.03 | 1.15 | 0.72/0.05 | Y | 1.01 | 0.02 | feature 185 | feature 343 |
| 141 -> 287 | 0.93 | 0.01 | 1.99 | 0.06/-0.03 | n | 1.01 | - | feature 141 | feature 287 |
| 185 -> 434 | 0.85 | 0.06 | 1.03 | 0.47/0.07 | Y | 1.00 | 0.02 | feature 185 | feature 434 |
| 125 -> 253 | 0.82 | 0.13 | 0.89 | 0.21/0.09 | Y | 1.01 | 0.00 | feature 125 | feature 253 |
| 108 -> 287 | 0.82 | 0.01 | 1.88 | 0.05/-0.03 | n | 1.02 | - | feature 108 | feature 287 |

## Block pair 1->2  -  6 candidate edges

- **Out-degree**: 6 parents, 2 children, 1 multi-parented (PolyFrac 50.0%); top-1 parent holds 16.7% of edges, Gini 0.973, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.54; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 5/6 edges pass (83.3%).
- **Frequency control**: mean survival 1.057 over 6 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.965, R_mass mean 0.957; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.002.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 445 -> 617 | 0.96 | 0.00 | 3.41 | 0.04/0.02 | Y | 1.02 | - | feature 445 | feature 617 |
| 404 -> 617 | 0.79 | 0.00 | 2.83 | 0.02/0.02 | Y | 1.08 | - | feature 404 | feature 617 |
| 394 -> 617 | 0.74 | 0.00 | 2.45 | 0.02/0.02 | Y | 1.08 | - | feature 394 | feature 617 |
| 359 -> 617 | 0.59 | 0.00 | 2.19 | 0.01/0.02 | Y | 1.08 | - | feature 359 | feature 617 |
| 444 -> 617 | 0.51 | 0.00 | 2.06 | 0.01/0.02 | Y | 1.10 | - | feature 444 | feature 617 |
| 290 -> 626 | 0.50 | 0.01 | 2.30 | 0.00/0.00 | n | 0.99 | - | feature 290 | feature 626 |

## Block pair 2->3  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 7 edges dropped by the joint-support guard (n_joint < 30).
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
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 2 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 5->6  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.19; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 0.935 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.510, R_mass mean 0.510; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.001.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1141 -> 1361 | 0.53 | 0.00 | 2.19 | 0.01/0.02 | Y | 0.94 | - | feature 1141 | feature 1361 |

## Block pair 6->7  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.
