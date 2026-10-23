# In-block (same-level) directed edges

**Layer 1**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L1　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **550** directed edges, **78** duplicate pairs, 550 survive PMI>0; PolyFrac 99%, Gini 0.956.
- **S_res: 15/550** edges are genuine refinements (0 children untestable).
- In-block superparents: 3 (e.g. F153 _feature 153_: 134 children, fires 57%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 201 → 8 | 1.00 | 1.33 | feature 201 | feature 8 |
| 120 → 8 | 1.00 | 1.41 | feature 120 | feature 8 |
| 13 → 8 | 1.00 | 1.44 | feature 13 | feature 8 |
| 109 → 8 | 0.99 | 1.36 | feature 109 | feature 8 |
| 7 → 8 | 0.99 | 2.91 | feature 7 | feature 8 |
| 31 → 127 | 0.99 | 1.31 | feature 31 | feature 127 |
| 34 → 127 | 0.99 | 1.35 | feature 34 | feature 127 |
| 13 → 127 | 0.99 | 1.43 | feature 13 | feature 127 |

_Duplicate pairs (rename/split candidates):_ 7≈101 (feature 7); 13≈31 (feature 13); 13≈34 (feature 13); 13≈46 (feature 13); 13≈52 (feature 13); 13≈97 (feature 13)

## Block B1  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 513 → 453 | 0.67 | 2.40 | feature 513 | feature 453 |

## Block B3  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 708 → 675 | 0.59 | 3.77 | feature 708 | feature 675 |

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1407 → 1551 | 0.52 | 4.22 | feature 1407 | feature 1551 |

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
