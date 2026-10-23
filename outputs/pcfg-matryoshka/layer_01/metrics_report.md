# Exp 0 - metrics report

**Layer 1**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L1　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  327 candidate edges

- **Out-degree**: 17 parents, 129 children, 128 multi-parented (PolyFrac 99.2%); top-1 parent holds 39.1% of edges, Gini 0.985, max out-degree 128.
- **Superparents** (out-degree flag): 2 (2 also pass the old fire-rate AND-gate) — e.g. feature 153 _feature 153_: 128 children, fires on 56.8% of tokens
- **Independence null**: mean edge PMI 0.30; 312 edges (95.4%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 327/327 edges pass (100.0%).
- **Frequency control**: mean survival 1.020 over 327 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.030 over 5 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.698, R_mass mean 0.695; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.236.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 18 -> 326 | 0.99 | 0.34 | 1.37 | 0.58/0.08 | Y | 1.00 | 0.03 | feature 18 | feature 326 |
| 109 -> 247 | 0.86 | 0.01 | 1.22 | 1.55/0.04 | Y | 1.03 | - | feature 109 | feature 247 |
| 31 -> 247 | 0.85 | 0.01 | 1.16 | 0.57/0.04 | Y | 1.03 | - | feature 31 | feature 247 |
| 34 -> 247 | 0.84 | 0.01 | 1.19 | 1.91/0.04 | Y | 1.02 | - | feature 34 | feature 247 |
| 98 -> 247 | 0.84 | 0.01 | 1.18 | 1.60/0.04 | Y | 1.02 | - | feature 98 | feature 247 |
| 142 -> 247 | 0.84 | 0.01 | 1.21 | 0.60/0.04 | Y | 1.02 | - | feature 142 | feature 247 |
| 201 -> 247 | 0.84 | 0.01 | 1.15 | 1.52/0.04 | Y | 1.02 | - | feature 201 | feature 247 |
| 120 -> 247 | 0.83 | 0.01 | 1.23 | 0.22/0.04 | Y | 1.02 | - | feature 120 | feature 247 |

## Block pair 1->2  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.45; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 1.007 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.908, R_mass mean 0.840; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.035.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 232 -> 669 | 0.56 | 0.04 | 2.45 | 0.09/0.33 | Y | 1.01 | - | feature 232 | feature 669 |

## Block pair 2->3  -  2 candidate edges

- **Out-degree**: 2 parents, 1 children, 1 multi-parented (PolyFrac 100.0%); top-1 parent holds 50.0% of edges, Gini 0.991, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.31; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2/2 edges pass (100.0%).
- **Frequency control**: mean survival 0.970 over 2 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.761, R_mass mean 0.738; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.049.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 541 -> 675 | 0.75 | 0.05 | 3.26 | 0.08/0.19 | Y | 1.00 | - | feature 541 | feature 675 |
| 568 -> 675 | 0.67 | 0.05 | 3.36 | 0.04/0.19 | Y | 0.94 | - | feature 568 | feature 675 |

## Block pair 3->4  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 4->5  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.94; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 0.997 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.641, R_mass mean 0.766; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.053.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 965 -> 1172 | 0.98 | 0.05 | 3.94 | 0.27/0.40 | Y | 1.00 | - | feature 965 | feature 1172 |

## Block pair 5->6  -  0 candidate edges

- **Out-degree**: 0 parents, 0 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 0.0% of edges, Gini 0.000, max out-degree 0.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI nan; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 0/0 edges pass (0.0%).
- **Frequency control**: mean survival nan over 0 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean nan, R_mass mean nan; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): n/a.

## Block pair 6->7  -  1 candidate edges

- **Out-degree**: 1 parents, 1 children, 0 multi-parented (PolyFrac 0.0%); top-1 parent holds 100.0% of edges, Gini 0.996, max out-degree 1.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.75; 0 edges (0.0%) at chance level (PMI < 0.5). 0 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1/1 edges pass (100.0%).
- **Frequency control**: mean survival 0.838 over 1 testable edges; 0 (0.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.638, R_mass mean 0.744; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.077.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1506 -> 1634 | 0.56 | 0.08 | 3.75 | 0.12/0.17 | Y | 0.84 | - | feature 1506 | feature 1634 |
