# Exp 0 - metrics report

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  2428 candidate edges

- **Out-degree**: 58 parents, 382 children, 382 multi-parented (PolyFrac 100.0%); top-1 parent holds 15.7% of edges, Gini 0.919, max out-degree 382.
- **Superparents** (out-degree flag): 5 (5 also pass the old fire-rate AND-gate) — e.g. feature 15 _technical documentation-like language, including code snipp…_: 382 children, fires on 98.9% of tokens
- **Independence null**: mean edge PMI 0.22; 2087 edges (86.0%) at chance level (PMI < 0.5). 14 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 2086/2428 edges pass (85.9%).
- **Frequency control**: mean survival 1.031 over 2428 testable edges; 25 (1.0%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.054 over 42 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.559.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 15 -> 211 | 1.00 | 0.00 | 0.01 | 1.17/0.39 | Y | 1.00 | 0.01 | technical documentation-like language, … | Wikipedia categories |
| 15 -> 158 | 1.00 | 0.00 | 0.01 | 0.91/0.15 | Y | 1.00 | 0.01 | technical documentation-like language, … | code-like structures and spacing |
| 15 -> 348 | 1.00 | 0.01 | 0.01 | 1.36/0.00 | n | 1.00 | 0.01 | technical documentation-like language, … | references to scientific publications |
| 42 -> 235 | 1.00 | 0.00 | 0.05 | 0.06/0.48 | Y | 1.00 | 0.01 | code and code-like snippets, with a pre… | math-related symbols like multiplicatio… |
| 15 -> 486 | 1.00 | 0.01 | 0.01 | 1.94/-0.00 | n | 1.00 | 0.01 | technical documentation-like language, … | mathematical equations or references to… |
| 15 -> 133 | 1.00 | 0.01 | 0.01 | 1.08/0.10 | Y | 1.00 | 0.01 | technical documentation-like language, … | the word "name" and its variants in pro… |
| 15 -> 433 | 1.00 | 0.01 | 0.01 | 1.69/0.05 | Y | 1.00 | 0.01 | technical documentation-like language, … | occurrences of the word "then" and near… |
| 15 -> 177 | 1.00 | 0.01 | 0.01 | 1.52/0.06 | Y | 1.00 | 0.01 | technical documentation-like language, … | sentences containing the word "must" or… |

## Block pair 1->2  -  280 candidate edges

- **Out-degree**: 98 parents, 193 children, 36 multi-parented (PolyFrac 18.7%); top-1 parent holds 27.9% of edges, Gini 0.895, max out-degree 78.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 1.89; 75 edges (26.8%) at chance level (PMI < 0.5). 48 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 100/280 edges pass (35.7%).
- **Frequency control**: mean survival 0.764 over 277 testable edges; 77 (27.8%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.032 over 28 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 1 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.137.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 216 -> 1501 | 0.89 | 0.24 | 2.96 | 0.27/0.89 | Y | 1.07 | 0.10 | sentences starting with "the" followed … | numbers starting with 1 |
| 302 -> 1694 | 0.86 | 0.03 | 2.90 | 0.06/-0.01 | n | 0.42 | 0.02 | source code and/or documents with very … | references to US legal cases |
| 216 -> 1554 | 0.85 | 0.57 | 2.91 | 0.10/0.31 | Y | 1.07 | 0.10 | sentences starting with "the" followed … | the numeral 1 |
| 139 -> 1939 | 0.84 | 0.06 | 2.91 | 0.10/0.09 | Y | 0.79 | 0.02 | code containing `import` statements, `@… | terms in Android Java import statements |
| 216 -> 1694 | 0.82 | 0.03 | 2.88 | 0.03/-0.01 | n | 0.44 | 0.10 | sentences starting with "the" followed … | references to US legal cases |
| 311 -> 1694 | 0.81 | 0.04 | 3.28 | -0.00/-0.01 | n | 0.00 | 0.05 | something, but it's too difficult to de… | references to US legal cases |
| 139 -> 1694 | 0.81 | 0.03 | 2.88 | 0.00/-0.01 | n | 0.11 | 0.02 | code containing `import` statements, `@… | references to US legal cases |
| 287 -> 1694 | 0.80 | 0.06 | 3.56 | 0.01/-0.01 | n | 0.11 | 0.09 | math problems involving the letter 'q' … | references to US legal cases |

## Block pair 2->3  -  762 candidate edges

- **Out-degree**: 234 parents, 428 children, 59 multi-parented (PolyFrac 13.8%); top-1 parent holds 13.1% of edges, Gini 0.925, max out-degree 100.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 3.19; 0 edges (0.0%) at chance level (PMI < 0.5). 572 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 239/762 edges pass (31.4%).
- **Frequency control**: mean survival 0.642 over 668 testable edges; 277 (41.5%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.255 over 139 parents; 48 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 2 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.273.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1841 -> 3374 | 1.00 | 0.16 | 5.54 | 0.28/0.32 | Y | - | - | URLs and email addresses | periods appearing in a URL |
| 1554 -> 7399 | 0.99 | 0.14 | 3.46 | 1.03/0.06 | Y | 1.00 | 0.10 | the numeral 1 | scientific and technical terminology |
| 1778 -> 4624 | 0.97 | 0.05 | 3.49 | 0.01/0.01 | n | - | 0.52 | URLs, version numbers, paths and people | mentions of asynchronous operations in … |
| 1571 -> 4624 | 0.97 | 0.22 | 5.06 | 0.00/0.01 | n | - | 0.13 | answers to programming questions | mentions of asynchronous operations in … |
| 1524 -> 4624 | 0.97 | 0.23 | 5.11 | 0.01/0.01 | n | - | 0.58 | code comments | mentions of asynchronous operations in … |
| 767 -> 5432 | 0.97 | 0.06 | 4.45 | 0.02/0.00 | n | - | 0.16 | various kinds of reference markers | math question/answer section separators |
| 1223 -> 7901 | 0.96 | 0.15 | 4.11 | 0.22/0.49 | Y | 0.98 | 0.14 | age ranges using the word "year" | dates, specifically the year 1924, 1938… |
| 1257 -> 4624 | 0.96 | 0.08 | 4.00 | -0.00/0.01 | n | - | 0.52 | Microsoft code snippets including Windo… | mentions of asynchronous operations in … |
