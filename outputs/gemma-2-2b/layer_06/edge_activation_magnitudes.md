# Edge activation magnitudes

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 745,305 | 2.19 | 5.93 | 24.75 |
| B1 | 564,121 | 1.91 | 3.32 | 12.41 |
| B2 | 735,460 | 1.86 | 2.88 | 9.58 |
| B3 | 1,394,296 | 1.85 | 2.77 | 6.92 |
| B4 | 928,341 | 1.84 | 2.69 | 6.86 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 2428 | 0.67 | 15% | 1.06 | 0% |
| 1->2 | 280 | 0.96 | 47% | 1.16 | 7% |
| 2->3 | 762 | 0.88 | 42% | 1.26 | 9% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 15 → 194 | 0.62 | 0.25 | feature 15 | feature 194 |
| 15 → 406 | 0.64 | 0.27 | feature 15 | feature 406 |
| 15 → 333 | 0.66 | 0.23 | feature 15 | feature 333 |
| 15 → 353 | 0.66 | 0.22 | feature 15 | feature 353 |
| 15 → 319 | 0.67 | 0.13 | feature 15 | feature 319 |
| 15 → 508 | 0.69 | 0.20 | feature 15 | feature 508 |
| 15 → 349 | 0.70 | 0.16 | feature 15 | feature 349 |
| 15 → 392 | 0.70 | 0.22 | feature 15 | feature 392 |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 320 → 1694 | 0.29 | 2.78 | feature 320 | feature 1694 |
| 405 → 1694 | 0.31 | 2.45 | feature 405 | feature 1694 |
| 457 → 1694 | 0.32 | 3.06 | feature 457 | feature 1694 |
| 446 → 1936 | 0.34 | 2.57 | feature 446 | feature 1936 |
| 417 → 1936 | 0.34 | 2.45 | feature 417 | feature 1936 |
| 247 → 1694 | 0.34 | 2.49 | feature 247 | feature 1694 |
| 311 → 1694 | 0.35 | 1.34 | feature 311 | feature 1694 |
| 187 → 1670 | 0.36 | 1.13 | feature 187 | feature 1670 |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1501 → 4217 | 0.13 | 0.92 | feature 1501 | feature 4217 |
| 1501 → 3014 | 0.14 | 0.45 | feature 1501 | feature 3014 |
| 1554 → 5936 | 0.16 | 0.81 | feature 1554 | feature 5936 |
| 1554 → 2624 | 0.18 | 0.72 | feature 1554 | feature 2624 |
| 1297 → 5963 | 0.22 | 12.80 | feature 1297 | feature 5963 |
| 1297 → 3772 | 0.23 | 3.97 | feature 1297 | feature 3772 |
| 1297 → 3513 | 0.23 | 1.79 | feature 1297 | feature 3513 |
| 1297 → 5838 | 0.24 | 0.89 | feature 1297 | feature 5838 |
