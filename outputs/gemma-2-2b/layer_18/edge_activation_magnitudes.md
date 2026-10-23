# Edge activation magnitudes

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 596,173 | 6.36 | 16.30 | 74.19 |
| B1 | 606,369 | 5.68 | 9.62 | 28.81 |
| B2 | 851,607 | 5.55 | 8.47 | 26.33 |
| B3 | 1,384,694 | 5.54 | 8.16 | 20.50 |
| B4 | 935,193 | 5.53 | 8.07 | 19.94 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 1129 | 0.54 | 10% | 1.06 | 0% |
| 1->2 | 1748 | 1.00 | 50% | 0.99 | 3% |
| 2->3 | 4195 | 1.02 | 53% | 0.63 | 30% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 89 → 216 | 0.29 | 0.24 | technical or scientific language relate… | LaTeX math environments |
| 89 → 222 | 0.37 | 0.50 | technical or scientific language relate… | code-related terms, particularly those … |
| 89 → 201 | 0.48 | 0.25 | technical or scientific language relate… | code snippets with certain structural e… |
| 89 → 374 | 0.49 | 0.34 | technical or scientific language relate… | internet URLs beginning with "http" |
| 89 → 163 | 0.50 | 0.10 | technical or scientific language relate… | LaTeX equations and expressions. |
| 89 → 328 | 0.50 | 0.14 | technical or scientific language relate… | scientific references and quantitative … |
| 89 → 442 | 0.51 | 0.27 | technical or scientific language relate… | code indentation |
| 89 → 435 | 0.52 | 0.10 | technical or scientific language relate… | French characters with accents |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 420 → 592 | 0.07 | 1.19 | blank lines | references to figures |
| 420 → 1746 | 0.07 | 1.12 | blank lines | mathematical code, in particular the ke… |
| 420 → 964 | 0.07 | 1.71 | blank lines | legal citations |
| 317 → 592 | 0.18 | 1.07 | identifies months of the year, especial… | references to figures |
| 317 → 1746 | 0.18 | 1.01 | identifies months of the year, especial… | mathematical code, in particular the ke… |
| 317 → 964 | 0.18 | 1.54 | identifies months of the year, especial… | legal citations |
| 420 → 656 | 0.20 | 0.38 | blank lines | legal jargon |
| 374 → 592 | 0.21 | 1.12 | internet URLs beginning with "http" | references to figures |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1156 → 2287 | 0.23 | 1.83 | comments in code with a star preceeding… | references to religious texts |
| 1156 → 2858 | 0.23 | 1.76 | comments in code with a star preceeding… | equations with delta functions, especia… |
| 1156 → 3007 | 0.23 | 1.11 | comments in code with a star preceeding… | ranges, such as between two numbers or … |
| 1156 → 3349 | 0.23 | 1.61 | comments in code with a star preceeding… | uses common words when making logical a… |
| 1156 → 3413 | 0.23 | 1.30 | comments in code with a star preceeding… | the word "master" in various contexts |
| 1156 → 3732 | 0.23 | 2.91 | comments in code with a star preceeding… | seemingly random letters or numbers, of… |
| 1156 → 3952 | 0.23 | 1.14 | comments in code with a star preceeding… | grammatical errors and suggestions on f… |
| 1156 → 4235 | 0.23 | 1.23 | comments in code with a star preceeding… | legal jargon in court documents |
