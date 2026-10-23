# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **815** directed edges, **21** duplicate pairs, 614 survive PMI>0; PolyFrac 68%, Gini 0.895.
- **S_res: 19/614** edges are genuine refinements (0 children untestable).
- In-block superparents: 2 (e.g. F105 _feature 105_: 193 children, fires 98%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 105 → 90 | 1.00 | 0.02 | feature 105 | feature 90 |
| 105 → 4 | 1.00 | 0.02 | feature 105 | feature 4 |
| 172 → 45 | 1.00 | 1.12 | feature 172 | feature 45 |
| 105 → 131 | 1.00 | 0.02 | feature 105 | feature 131 |
| 185 → 8 | 1.00 | 2.40 | feature 185 | feature 8 |
| 105 → 186 | 1.00 | 0.02 | feature 105 | feature 186 |
| 202 → 88 | 1.00 | 0.52 | feature 202 | feature 88 |
| 80 → 88 | 1.00 | 1.51 | feature 80 | feature 88 |

_Duplicate pairs (rename/split candidates):_ 4≈51 (feature 4); 4≈90 (feature 4); 8≈88 (feature 8); 8≈134 (feature 8); 8≈139 (feature 8); 14≈25 (feature 14)

## Block B1  (224 features)
- **12** directed edges, **0** duplicate pairs, 12 survive PMI>0; PolyFrac 100%, Gini 0.968.
- **S_res: 0/12** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 317 → 362 | 0.90 | 2.52 | feature 317 | feature 362 |
| 357 → 362 | 0.77 | 2.90 | feature 357 | feature 362 |
| 438 → 373 | 0.74 | 1.94 | feature 438 | feature 373 |
| 438 → 411 | 0.73 | 1.93 | feature 438 | feature 411 |
| 438 → 374 | 0.71 | 1.90 | feature 438 | feature 374 |
| 436 → 362 | 0.65 | 1.60 | feature 436 | feature 362 |
| 345 → 362 | 0.62 | 1.89 | feature 345 | feature 362 |
| 341 → 373 | 0.57 | 2.09 | feature 341 | feature 373 |

## Block B2  (224 features)
- **7** directed edges, **0** duplicate pairs, 7 survive PMI>0; PolyFrac 100%, Gini 0.969.
- **S_res: 0/7** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 463 → 561 | 1.00 | 3.20 | feature 463 | feature 561 |
| 491 → 561 | 1.00 | 3.49 | feature 491 | feature 561 |
| 528 → 561 | 1.00 | 3.28 | feature 528 | feature 561 |
| 574 → 561 | 1.00 | 4.08 | feature 574 | feature 561 |
| 590 → 561 | 1.00 | 3.29 | feature 590 | feature 561 |
| 619 → 561 | 1.00 | 3.32 | feature 619 | feature 561 |
| 668 → 561 | 1.00 | 3.35 | feature 668 | feature 561 |

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1101 → 945 | 0.64 | 3.62 | feature 1101 | feature 945 |

## Block B5  (224 features)
- **5** directed edges, **0** duplicate pairs, 5 survive PMI>0; PolyFrac 25%, Gini 0.985.
- **S_res: 2/5** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1159 → 1143 | 0.90 | 2.17 | feature 1159 | feature 1143 |
| 1141 → 1330 | 0.81 | 3.49 | feature 1141 | feature 1330 |
| 1159 → 1204 | 0.73 | 1.96 | feature 1159 | feature 1204 |
| 1185 → 1165 | 0.71 | 3.66 | feature 1185 | feature 1165 |
| 1204 → 1143 | 0.56 | 2.84 | feature 1204 | feature 1143 |

## Block B6  (224 features)
- **3** directed edges, **0** duplicate pairs, 3 survive PMI>0; PolyFrac 50%, Gini 0.993.
- **S_res: 0/3** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1402 → 1525 | 0.73 | 2.00 | feature 1402 | feature 1525 |
| 1402 → 1345 | 0.70 | 1.95 | feature 1402 | feature 1345 |
| 1519 → 1345 | 0.52 | 3.65 | feature 1519 | feature 1345 |

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
