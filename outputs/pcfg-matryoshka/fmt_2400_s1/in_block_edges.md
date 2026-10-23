# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **78** directed edges, **12** duplicate pairs, 78 survive PMI>0; PolyFrac 49%, Gini 0.953.
- **S_res: 3/75** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 58 → 190 | 1.00 | 0.93 | feature 58 | feature 190 |
| 58 → 169 | 1.00 | 0.93 | feature 58 | feature 169 |
| 58 → 47 | 1.00 | 0.93 | feature 58 | feature 47 |
| 58 → 197 | 1.00 | 0.93 | feature 58 | feature 197 |
| 58 → 182 | 1.00 | 0.93 | feature 58 | feature 182 |
| 58 → 106 | 0.99 | 0.92 | feature 58 | feature 106 |
| 149 → 6 | 0.99 | 2.91 | feature 149 | feature 6 |
| 58 → 55 | 0.99 | 0.92 | feature 58 | feature 55 |

_Duplicate pairs (rename/split candidates):_ 2≈142 (feature 2); 35≈51 (feature 35); 35≈143 (feature 35); 35≈148 (feature 35); 35≈180 (feature 35); 47≈182 (feature 47)

## Block B1  (224 features)
- **5** directed edges, **1** duplicate pairs, 5 survive PMI>0; PolyFrac 0%, Gini 0.994.
- **S_res: 0/5** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 301 → 351 | 1.00 | 4.07 | feature 301 | feature 351 |
| 301 → 299 | 1.00 | 4.07 | feature 301 | feature 299 |
| 353 → 432 | 1.00 | 2.59 | feature 353 | feature 432 |
| 301 → 408 | 0.99 | 4.06 | feature 301 | feature 408 |
| 301 → 398 | 0.93 | 3.99 | feature 301 | feature 398 |

_Duplicate pairs (rename/split candidates):_ 351≈408 (feature 351)

## Block B2  (224 features)
- **6** directed edges, **1** duplicate pairs, 6 survive PMI>0; PolyFrac 25%, Gini 0.981.
- **S_res: 1/3** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 645 → 518 | 1.00 | 2.87 | feature 645 | feature 518 |
| 649 → 518 | 1.00 | 2.62 | feature 649 | feature 518 |
| 584 → 632 | 1.00 | 4.27 | feature 584 | feature 632 |
| 584 → 514 | 0.99 | 4.26 | feature 584 | feature 514 |
| 600 → 518 | 0.97 | 2.53 | feature 600 | feature 518 |
| 639 → 454 | 0.51 | 1.79 | feature 639 | feature 454 |

_Duplicate pairs (rename/split candidates):_ 557≈643 (feature 557)

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **0** directed edges, **1** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

_Duplicate pairs (rename/split candidates):_ 1252≈1312 (feature 1252)

## Block B6  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/0** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1465 → 1475 | 0.87 | 3.14 | feature 1465 | feature 1475 |

## Block B7  (224 features)
- **2** directed edges, **0** duplicate pairs, 2 survive PMI>0; PolyFrac 100%, Gini 0.991.
- **S_res: 0/2** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1729 → 1651 | 1.00 | 5.37 | feature 1729 | feature 1651 |
| 1632 → 1651 | 0.75 | 3.21 | feature 1632 | feature 1651 |
