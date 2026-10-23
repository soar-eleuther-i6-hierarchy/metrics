# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　646,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **99** directed edges, **32** duplicate pairs, 99 survive PMI>0; PolyFrac 82%, Gini 0.959.
- **S_res: 3/99** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 68 → 70 | 1.00 | 1.47 | feature 68 | feature 70 |
| 11 → 70 | 1.00 | 1.46 | feature 11 | feature 70 |
| 11 → 177 | 1.00 | 1.45 | feature 11 | feature 177 |
| 68 → 177 | 0.99 | 1.47 | feature 68 | feature 177 |
| 169 → 138 | 0.99 | 1.43 | feature 169 | feature 138 |
| 11 → 112 | 0.99 | 1.45 | feature 11 | feature 112 |
| 68 → 112 | 0.99 | 1.46 | feature 68 | feature 112 |
| 11 → 5 | 0.99 | 1.45 | feature 11 | feature 5 |

_Duplicate pairs (rename/split candidates):_ 2≈7 (feature 2); 2≈15 (feature 2); 2≈18 (feature 2); 2≈85 (feature 2); 2≈88 (feature 2); 2≈151 (feature 2)

## Block B1  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 381 → 288 | 0.52 | 1.86 | feature 381 | feature 288 |

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 569 → 522 | 0.66 | 2.14 | feature 569 | feature 522 |

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1286 → 1292 | 0.83 | 4.19 | feature 1286 | feature 1292 |

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
