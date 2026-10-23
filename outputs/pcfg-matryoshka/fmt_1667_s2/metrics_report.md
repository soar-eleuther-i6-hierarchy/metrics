# Exp 0 - metrics report

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　598,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  125 candidate edges

- **Out-degree**: 18 parents, 109 children, 5 multi-parented (PolyFrac 4.6%); top-1 parent holds 81.6% of edges, Gini 0.984, max out-degree 102.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 83 _feature 83_: 102 children, fires on 37.5% of tokens
- **Independence null**: mean edge PMI 0.63; 78 edges (62.4%) at chance level (PMI < 0.5). 24 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 116/125 edges pass (92.8%).
- **Frequency control**: mean survival 0.959 over 125 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.029 over 3 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.887, R_mass mean 0.895; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.141.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 198 -> 444 | 0.98 | 0.22 | 1.75 | 0.32/0.06 | Y | 1.01 | - | feature 198 | feature 444 |
| 212 -> 241 | 0.95 | 0.00 | 1.30 | 0.19/-0.00 | n | 0.97 | 0.02 | feature 212 | feature 241 |
| 83 -> 240 | 0.94 | 0.08 | 0.92 | 0.37/0.06 | Y | 0.82 | 0.06 | feature 83 | feature 240 |
| 189 -> 240 | 0.93 | 0.17 | 1.64 | 0.91/0.06 | Y | 0.54 | - | feature 189 | feature 240 |
| 212 -> 375 | 0.92 | 0.04 | 1.27 | 0.64/0.05 | Y | 1.01 | 0.02 | feature 212 | feature 375 |
| 81 -> 240 | 0.90 | 0.15 | 1.50 | 1.13/0.06 | Y | 0.80 | 0.01 | feature 81 | feature 240 |
| 212 -> 330 | 0.89 | 0.06 | 1.24 | 0.54/0.09 | Y | 1.00 | 0.02 | feature 212 | feature 330 |
| 212 -> 293 | 0.85 | 0.07 | 1.19 | 0.54/0.06 | Y | 1.00 | 0.02 | feature 212 | feature 293 |

## Block pair 1->2  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 2->3  -  5 candidate edges

- **Out-degree**: 1 parents, 5 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 5.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 1.37; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 5/5 edges pass (100.0%).
- **Frequency control**: mean survival 1.016 over 5 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.026 over 1 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.903, R_mass mean 0.902; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.308.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 590 -> 800 | 0.52 | 0.07 | 1.38 | 0.04/0.11 | Y | 1.02 | 0.03 | feature 590 | feature 800 |
| 590 -> 854 | 0.51 | 0.06 | 1.37 | 0.04/0.10 | Y | 1.01 | 0.03 | feature 590 | feature 854 |
| 590 -> 685 | 0.51 | 0.06 | 1.37 | 0.04/0.10 | Y | 0.99 | 0.03 | feature 590 | feature 685 |
| 590 -> 867 | 0.50 | 0.06 | 1.36 | 0.04/0.10 | Y | 1.01 | 0.03 | feature 590 | feature 867 |
| 590 -> 793 | 0.50 | 0.06 | 1.36 | 0.04/0.10 | Y | 1.05 | 0.03 | feature 590 | feature 793 |

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
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 1 edges dropped by the joint-support guard (n_joint < 30).
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
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 1 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.
