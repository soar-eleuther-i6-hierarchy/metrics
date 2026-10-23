# Exp 0 - metrics report

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  493 candidate edges

- **Out-degree**: 55 parents, 167 children, 100 multi-parented (PolyFrac 59.9%); top-1 parent holds 33.3% of edges, Gini 0.945, max out-degree 164.
- **Superparents** (out-degree flag): 2 (2 also pass the old fire-rate AND-gate) — e.g. feature 105 _feature 105_: 164 children, fires on 97.5% of tokens
- **Independence null**: mean edge PMI 0.65; 252 edges (51.1%) at chance level (PMI < 0.5). 3 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 485/493 edges pass (98.4%).
- **Frequency control**: mean survival 0.746 over 482 testable edges; 135 (28.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.119 over 35 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.993, R_mass mean 0.993; 9 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.372.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 105 -> 362 | 1.00 | 0.00 | 0.03 | 0.09/0.01 | Y | - | 0.02 | feature 105 | feature 362 |
| 105 -> 354 | 1.00 | 0.01 | 0.03 | 0.18/-0.01 | n | 1.00 | 0.02 | feature 105 | feature 354 |
| 105 -> 398 | 1.00 | 0.05 | 0.03 | 0.20/0.03 | Y | 1.00 | 0.02 | feature 105 | feature 398 |
| 105 -> 353 | 1.00 | 0.04 | 0.02 | 0.12/0.07 | Y | 1.00 | 0.02 | feature 105 | feature 353 |
| 105 -> 273 | 1.00 | 0.10 | 0.02 | 0.15/0.08 | Y | 1.00 | 0.02 | feature 105 | feature 273 |
| 105 -> 337 | 1.00 | 0.02 | 0.02 | 0.16/0.34 | Y | 1.00 | 0.02 | feature 105 | feature 337 |
| 105 -> 258 | 0.99 | 0.01 | 0.02 | 0.16/0.32 | Y | 1.00 | 0.02 | feature 105 | feature 258 |
| 105 -> 335 | 0.99 | 0.02 | 0.02 | 0.16/0.26 | Y | 1.00 | 0.02 | feature 105 | feature 335 |

## Block pair 1->2  -  21 candidate edges

- **Out-degree**: 21 parents, 1 children, 1 multi-parented (PolyFrac 100.0%); top-1 parent holds 4.8% of edges, Gini 0.906, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.85; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 14/21 edges pass (66.7%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.959, R_mass mean 0.960; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.060.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 345 -> 561 | 1.00 | 0.03 | 2.38 | 0.01/0.05 | Y | - | - | feature 345 | feature 561 |
| 369 -> 561 | 1.00 | 0.03 | 2.38 | 0.01/0.05 | n | - | - | feature 369 | feature 561 |
| 385 -> 561 | 1.00 | 0.05 | 2.81 | 0.00/0.05 | n | - | - | feature 385 | feature 561 |
| 235 -> 561 | 1.00 | 0.05 | 2.71 | 0.01/0.05 | Y | - | - | feature 235 | feature 561 |
| 244 -> 561 | 1.00 | 0.05 | 2.89 | 0.01/0.05 | Y | - | - | feature 244 | feature 561 |
| 245 -> 561 | 1.00 | 0.10 | 3.48 | 0.04/0.05 | Y | - | - | feature 245 | feature 561 |
| 253 -> 561 | 1.00 | 0.04 | 2.69 | 0.01/0.05 | n | - | - | feature 253 | feature 561 |
| 276 -> 561 | 1.00 | 0.05 | 2.81 | 0.03/0.05 | Y | - | - | feature 276 | feature 561 |

## Block pair 2->3  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.64; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 0.844 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.911, R_mass mean 0.929; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.138.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 468 -> 850 | 0.73 | 0.14 | 2.64 | 0.11/0.05 | Y | 0.84 | - | feature 468 | feature 850 |

## Block pair 3->4  -  3 candidate edges

- **Out-degree**: 3 parents, 2 children, 1 multi-parented (PolyFrac 50.0%); top-1 parent holds 33.3% of edges, Gini 0.987, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.47; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/3 edges pass (33.3%).
- **Frequency control**: mean survival 0.355 over 3 testable edges; 2 (66.7%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.804, R_mass mean 0.736; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.182.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 679 -> 982 | 0.81 | 0.22 | 3.38 | 0.01/0.02 | n | 0.04 | - | feature 679 | feature 982 |
| 765 -> 982 | 0.80 | 0.25 | 3.49 | 0.01/0.02 | n | 0.03 | - | feature 765 | feature 982 |
| 823 -> 1076 | 0.75 | 0.07 | 3.55 | 0.23/0.02 | Y | 1.00 | - | feature 823 | feature 1076 |

## Block pair 4->5  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 5->6  -  4 candidate edges

- **Out-degree**: 3 parents, 3 children, 1 multi-parented (PolyFrac 33.3%); top-1 parent holds 50.0% of edges, Gini 0.989, max out-degree 2.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.47; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2/4 edges pass (50.0%).
- **Frequency control**: mean survival 0.711 over 4 testable edges; 2 (50.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.052 over 1 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.813, R_mass mean 0.835; 1 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.294.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1159 -> 1525 | 0.70 | 0.05 | 1.91 | 0.03/0.01 | n | 1.10 | 0.05 | feature 1159 | feature 1525 |
| 1204 -> 1345 | 0.67 | 0.12 | 3.01 | 0.05/0.02 | Y | 0.47 | - | feature 1204 | feature 1345 |
| 1159 -> 1402 | 0.59 | 0.57 | 1.74 | 0.03/0.05 | Y | 1.27 | 0.05 | feature 1159 | feature 1402 |
| 1325 -> 1345 | 0.52 | 0.14 | 3.21 | 0.01/0.02 | n | 0.00 | - | feature 1325 | feature 1345 |

## Block pair 6->7  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.
