# Exp 0 - metrics report

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

## Block pair 0->1  -  1473 candidate edges

- **Out-degree**: 63 parents, 382 children, 381 multi-parented (PolyFrac 99.7%); top-1 parent holds 25.9% of edges, Gini 0.906, max out-degree 382.
- **Superparents** (out-degree flag): 2 (2 also pass the old fire-rate AND-gate) — e.g. feature 44 _feature 44_: 382 children, fires on 98.8% of tokens
- **Independence null**: mean edge PMI 0.37; 1118 edges (75.9%) at chance level (PMI < 0.5). 2 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 794/1473 edges pass (53.9%).
- **Frequency control**: mean survival 0.998 over 1473 testable edges; 28 (1.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.033 over 49 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.506.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 44 -> 318 | 1.00 | 0.03 | 0.01 | 1.66/0.03 | Y | 1.00 | 0.01 | feature 44 | feature 318 |
| 44 -> 203 | 1.00 | 0.00 | 0.01 | 1.01/-0.00 | n | 1.00 | 0.01 | feature 44 | feature 203 |
| 44 -> 139 | 1.00 | 0.00 | 0.01 | 1.43/0.01 | Y | 1.00 | 0.01 | feature 44 | feature 139 |
| 44 -> 283 | 1.00 | 0.00 | 0.01 | 1.69/0.05 | Y | 1.00 | 0.01 | feature 44 | feature 283 |
| 44 -> 272 | 1.00 | 0.01 | 0.01 | 1.44/0.01 | n | 1.00 | 0.01 | feature 44 | feature 272 |
| 44 -> 161 | 1.00 | 0.03 | 0.01 | 1.74/-0.00 | n | 1.00 | 0.01 | feature 44 | feature 161 |
| 44 -> 452 | 1.00 | 0.03 | 0.01 | 2.05/0.00 | n | 1.00 | 0.01 | feature 44 | feature 452 |
| 44 -> 158 | 1.00 | 0.02 | 0.01 | 1.92/0.01 | Y | 1.00 | 0.01 | feature 44 | feature 158 |

## Block pair 1->2  -  621 candidate edges

- **Out-degree**: 97 parents, 402 children, 88 multi-parented (PolyFrac 21.9%); top-1 parent holds 44.0% of edges, Gini 0.932, max out-degree 273.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 1.43; 260 edges (41.9%) at chance level (PMI < 0.5). 93 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 197/621 edges pass (31.7%).
- **Frequency control**: mean survival 0.752 over 620 testable edges; 148 (23.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.178 over 52 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 1 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.223.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 292 -> 831 | 0.95 | 0.05 | 3.46 | 0.03/-0.00 | n | 1.03 | 0.00 | feature 292 | feature 831 |
| 260 -> 768 | 0.94 | 0.03 | 3.94 | 0.52/0.38 | Y | - | 0.14 | feature 260 | feature 768 |
| 383 -> 1503 | 0.93 | 0.01 | 0.82 | 0.04/0.16 | Y | 0.99 | 0.01 | feature 383 | feature 1503 |
| 375 -> 1634 | 0.86 | 0.08 | 1.36 | 0.03/0.00 | n | 0.96 | 0.04 | feature 375 | feature 1634 |
| 383 -> 936 | 0.86 | 0.01 | 0.74 | 0.01/0.06 | Y | 0.97 | 0.01 | feature 383 | feature 936 |
| 383 -> 1786 | 0.86 | 0.00 | 0.74 | 0.04/0.35 | Y | 0.97 | 0.01 | feature 383 | feature 1786 |
| 450 -> 1786 | 0.86 | 0.00 | 1.05 | 0.01/0.35 | Y | 0.93 | 0.01 | feature 450 | feature 1786 |
| 383 -> 1535 | 0.85 | 0.16 | 0.74 | 0.02/0.01 | Y | 0.99 | 0.01 | feature 383 | feature 1535 |

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
| 1572 -> 5875 | 1.00 | 0.18 | 5.32 | 0.24/0.06 | Y | 1.00 | - | feature 1572 | feature 5875 |
| 1348 -> 6486 | 1.00 | 0.30 | 5.46 | 0.00/0.01 | n | - | 0.62 | feature 1348 | feature 6486 |
| 1343 -> 6486 | 1.00 | 0.20 | 5.05 | -0.00/0.01 | n | - | 0.62 | feature 1343 | feature 6486 |
| 1938 -> 6486 | 1.00 | 0.16 | 4.83 | 0.02/0.01 | Y | - | 0.62 | feature 1938 | feature 6486 |
| 1149 -> 6486 | 0.98 | 0.04 | 3.42 | 0.00/0.01 | n | - | 0.62 | feature 1149 | feature 6486 |
| 1916 -> 6486 | 0.98 | 0.04 | 3.48 | -0.00/0.01 | n | - | 0.31 | feature 1916 | feature 6486 |
| 1262 -> 6486 | 0.98 | 0.06 | 3.81 | -0.01/0.01 | n | - | 0.62 | feature 1262 | feature 6486 |
| 666 -> 6486 | 0.98 | 0.22 | 5.14 | -0.00/0.01 | n | - | 0.31 | feature 666 | feature 6486 |

## Block pair 3->4  -  3277 candidate edges

- **Out-degree**: 464 parents, 609 children, 167 multi-parented (PolyFrac 27.4%); top-1 parent holds 2.8% of edges, Gini 0.972, max out-degree 93.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 4.66; 0 edges (0.0%) at chance level (PMI < 0.5). 2160 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 170/3277 edges pass (5.2%).
- **Frequency control**: mean survival 0.269 over 2312 testable edges; 1707 (73.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 14 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.415.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 2209 -> 24367 | 1.00 | 0.70 | 6.32 | 0.00/-0.00 | n | - | - | feature 2209 | feature 24367 |
| 7009 -> 20071 | 1.00 | 0.80 | 6.46 | 0.00/-0.00 | n | - | - | feature 7009 | feature 20071 |
| 5591 -> 20071 | 1.00 | 0.07 | 4.07 | -0.00/-0.00 | n | - | - | feature 5591 | feature 20071 |
| 4052 -> 24367 | 1.00 | 0.36 | 5.67 | -0.00/-0.00 | n | - | - | feature 4052 | feature 24367 |
| 2728 -> 24367 | 1.00 | 0.17 | 4.92 | -0.00/-0.00 | n | - | - | feature 2728 | feature 24367 |
| 7009 -> 24367 | 1.00 | 0.80 | 6.46 | 0.00/-0.00 | n | - | - | feature 7009 | feature 24367 |
| 4052 -> 20071 | 1.00 | 0.36 | 5.67 | -0.00/-0.00 | n | - | - | feature 4052 | feature 20071 |
| 2728 -> 20071 | 1.00 | 0.17 | 4.92 | -0.00/-0.00 | n | - | - | feature 2728 | feature 20071 |
