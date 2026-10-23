# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　598,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **148** directed edges, **5** duplicate pairs, 148 survive PMI>0; PolyFrac 18%, Gini 0.967.
- **S_res: 3/139** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F13 _feature 13_: 82 children, fires 39%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 135 → 57 | 1.00 | 1.37 | feature 135 | feature 57 |
| 68 → 222 | 1.00 | 1.50 | feature 68 | feature 222 |
| 29 → 176 | 1.00 | 1.45 | feature 29 | feature 176 |
| 29 → 164 | 0.99 | 1.45 | feature 29 | feature 164 |
| 80 → 222 | 0.99 | 1.41 | feature 80 | feature 222 |
| 135 → 12 | 0.99 | 1.36 | feature 135 | feature 12 |
| 176 → 164 | 0.99 | 2.43 | feature 176 | feature 164 |
| 13 → 220 | 0.98 | 0.93 | feature 13 | feature 220 |

_Duplicate pairs (rename/split candidates):_ 13≈50 (feature 13); 29≈68 (feature 29); 29≈80 (feature 29); 50≈220 (feature 50); 68≈80 (feature 68)

## Block B1  (224 features)
- **2** directed edges, **0** duplicate pairs, 2 survive PMI>0; PolyFrac 100%, Gini 0.991.
- **S_res: 1/2** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 422 → 342 | 0.63 | 1.81 | feature 422 | feature 342 |
| 306 → 342 | 0.56 | 1.81 | feature 306 | feature 342 |

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 509 → 600 | 0.68 | 2.32 | feature 509 | feature 600 |

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
