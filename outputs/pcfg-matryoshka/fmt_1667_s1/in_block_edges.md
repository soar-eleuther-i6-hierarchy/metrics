# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　598,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **73** directed edges, **17** duplicate pairs, 73 survive PMI>0; PolyFrac 82%, Gini 0.950.
- **S_res: 2/73** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 85 → 134 | 1.00 | 4.05 | feature 85 | feature 134 |
| 30 → 153 | 1.00 | 1.38 | feature 30 | feature 153 |
| 22 → 153 | 1.00 | 1.37 | feature 22 | feature 153 |
| 25 → 153 | 1.00 | 1.37 | feature 25 | feature 153 |
| 42 → 134 | 1.00 | 4.63 | feature 42 | feature 134 |
| 25 → 119 | 0.99 | 1.36 | feature 25 | feature 119 |
| 30 → 119 | 0.99 | 1.37 | feature 30 | feature 119 |
| 22 → 119 | 0.99 | 1.36 | feature 22 | feature 119 |

_Duplicate pairs (rename/split candidates):_ 3≈22 (feature 3); 3≈25 (feature 3); 3≈30 (feature 3); 3≈168 (feature 3); 3≈213 (feature 3); 22≈25 (feature 22)

## Block B1  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 288 → 284 | 0.55 | 1.71 | feature 288 | feature 284 |

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 491 → 502 | 0.93 | 3.93 | feature 491 | feature 502 |

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
