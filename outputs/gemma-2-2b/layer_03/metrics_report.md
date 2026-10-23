# Exp 0 - metrics report

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  1971 candidate edges

- **Out-degree**: 60 parents, 381 children, 377 multi-parented (PolyFrac 99.0%); top-1 parent holds 19.3% of edges, Gini 0.917, max out-degree 381.
- **Superparents** (out-degree flag): 5 (5 also pass the old fire-rate AND-gate) — e.g. feature 70 _proper nouns that have mixed upper and lowercase letters or…_: 381 children, fires on 99.3% of tokens
- **Independence null**: mean edge PMI 0.34; 1361 edges (69.1%) at chance level (PMI < 0.5). 17 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1770/1971 edges pass (89.8%).
- **Frequency control**: mean survival 1.041 over 1971 testable edges; 18 (0.9%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.059 over 46 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 0.999, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.482.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 70 -> 476 | 1.00 | 0.01 | 0.01 | 1.32/0.17 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the word "most" followed by "of" |
| 70 -> 352 | 1.00 | 0.00 | 0.01 | 1.95/0.01 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | square brackets surrounding numbers and… |
| 70 -> 316 | 1.00 | 0.01 | 0.01 | 1.18/0.07 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | reflexive pronouns. |
| 70 -> 169 | 1.00 | 0.01 | 0.01 | 1.62/0.10 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the phrase "into the" |
| 70 -> 164 | 1.00 | 0.00 | 0.01 | 0.92/0.16 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | LaTeX mathmatical notation |
| 70 -> 392 | 1.00 | 0.01 | 0.01 | 0.88/0.21 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the word "so" in various contexts |
| 70 -> 148 | 1.00 | 0.01 | 0.01 | 0.88/0.00 | n | 1.00 | 0.01 | proper nouns that have mixed upper and … | trademarked product names |
| 70 -> 310 | 1.00 | 0.01 | 0.01 | 1.27/0.04 | Y | 1.00 | 0.01 | proper nouns that have mixed upper and … | the keyword "public" in code |

## Block pair 1->2  -  1387 candidate edges

- **Out-degree**: 122 parents, 856 children, 300 multi-parented (PolyFrac 35.0%); top-1 parent holds 57.5% of edges, Gini 0.949, max out-degree 798.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 448 _a grab bag of proper nouns including names, places, and cod…_: 798 children, fires on 48.0% of tokens
- **Independence null**: mean edge PMI 0.92; 793 edges (57.2%) at chance level (PMI < 0.5). 213 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 632/1387 edges pass (45.6%).
- **Frequency control**: mean survival 1.011 over 1313 testable edges; 110 (8.4%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.237 over 72 parents; 10 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 56 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.197.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 208 -> 1032 | 0.95 | 0.01 | 2.34 | 0.00/-0.00 | n | - | 0.47 | scientific and technical texts, especia… | the `aligned` environment tag in LaTeX. |
| 370 -> 1032 | 0.95 | 0.04 | 3.30 | 0.01/-0.00 | n | - | 0.10 | code, math equations, and youtube links | the `aligned` environment tag in LaTeX. |
| 154 -> 1032 | 0.95 | 0.02 | 2.55 | -0.00/-0.00 | n | - | 0.28 | words from scientific documents related… | the `aligned` environment tag in LaTeX. |
| 145 -> 1032 | 0.95 | 0.00 | 1.25 | -0.01/-0.00 | n | - | 0.05 | words, terms, and related symbols in so… | the `aligned` environment tag in LaTeX. |
| 448 -> 1032 | 0.95 | 0.00 | 0.69 | 0.06/-0.00 | n | - | 0.02 | a grab bag of proper nouns including na… | the `aligned` environment tag in LaTeX. |
| 199 -> 1032 | 0.95 | 0.01 | 1.53 | 0.01/-0.00 | n | - | 0.07 | technical or scientific terms, includin… | the `aligned` environment tag in LaTeX. |
| 474 -> 1032 | 0.95 | 0.13 | 4.55 | -0.00/-0.00 | n | - | 0.56 | square brackets, often used in mathemat… | the `aligned` environment tag in LaTeX. |
| 249 -> 1032 | 0.95 | 0.11 | 4.41 | -0.00/-0.00 | n | - | 0.56 | law references, including numbers, foot… | the `aligned` environment tag in LaTeX. |

## Block pair 2->3  -  4379 candidate edges

- **Out-degree**: 289 parents, 2810 children, 436 multi-parented (PolyFrac 15.5%); top-1 parent holds 64.1% of edges, Gini 0.972, max out-degree 2806.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 1457 _words used in official documents and scientific publications_: 2806 children, fires on 87.9% of tokens
- **Independence null**: mean edge PMI 1.25; 2812 edges (64.2%) at chance level (PMI < 0.5). 2089 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1786/4379 edges pass (40.8%).
- **Frequency control**: mean survival 0.803 over 3918 testable edges; 833 (21.3%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.213 over 163 parents; 25 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 5 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.351.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1457 -> 2589 | 1.00 | 0.00 | 0.13 | 0.03/0.21 | Y | 1.00 | 0.02 | words used in official documents and sc… | the word "of" usually preceeding a date… |
| 1606 -> 6997 | 1.00 | 0.15 | 4.76 | 0.00/-0.00 | n | - | 0.56 | the word "at", especially within code a… | the word "spatial" |
| 598 -> 6997 | 1.00 | 0.28 | 5.42 | -0.00/-0.00 | n | - | 0.56 | the words "every" and "everyone." | the word "spatial" |
| 2046 -> 6997 | 1.00 | 0.12 | 4.57 | -0.00/-0.00 | n | - | 0.53 | code snippets, particularly function de… | the word "spatial" |
| 1023 -> 6997 | 1.00 | 0.14 | 4.75 | -0.00/-0.00 | n | - | 0.56 | words or phrases directly related to le… | the word "spatial" |
| 1457 -> 6403 | 1.00 | 0.00 | 0.13 | 0.06/0.47 | Y | - | 0.02 | words used in official documents and sc… | dates |
| 655 -> 6997 | 1.00 | 0.79 | 6.45 | -0.00/-0.00 | n | - | 0.56 | mathematical or physics notation, speci… | the word "spatial" |
| 1998 -> 6997 | 1.00 | 0.12 | 4.58 | -0.00/-0.00 | n | - | 0.56 | words containing the "re" prefix follow… | the word "spatial" |
