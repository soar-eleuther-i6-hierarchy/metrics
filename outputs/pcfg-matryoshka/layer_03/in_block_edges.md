# In-block (same-level) directed edges

**Layer 3**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L3　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **888** directed edges, **232** duplicate pairs, 888 survive PMI>0; PolyFrac 86%, Gini 0.954.
- **S_res: 7/884** edges are genuine refinements (0 children untestable).
- In-block superparents: 6 (e.g. F17 _feature 17_: 120 children, fires 37%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 22 → 27 | 1.00 | 0.82 | feature 22 | feature 27 |
| 18 → 27 | 1.00 | 1.37 | feature 18 | feature 27 |
| 88 → 27 | 1.00 | 1.40 | feature 88 | feature 27 |
| 144 → 27 | 1.00 | 1.37 | feature 144 | feature 27 |
| 154 → 52 | 0.97 | 0.89 | feature 154 | feature 52 |
| 46 → 52 | 0.97 | 0.90 | feature 46 | feature 52 |
| 119 → 34 | 0.97 | 1.92 | feature 119 | feature 34 |
| 139 → 34 | 0.96 | 1.89 | feature 139 | feature 34 |

_Duplicate pairs (rename/split candidates):_ 3≈17 (feature 3); 3≈46 (feature 3); 3≈76 (feature 3); 3≈99 (feature 3); 3≈154 (feature 3); 5≈12 (feature 5)

## Block B1  (224 features)
- **4** directed edges, **0** duplicate pairs, 4 survive PMI>0; PolyFrac 50%, Gini 0.982.
- **S_res: 0/4** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 286 → 240 | 0.96 | 3.81 | feature 286 | feature 240 |
| 254 → 291 | 0.76 | 3.17 | feature 254 | feature 291 |
| 304 → 240 | 0.65 | 2.16 | feature 304 | feature 240 |
| 225 → 240 | 0.52 | 2.08 | feature 225 | feature 240 |

## Block B2  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1280 → 1158 | 0.57 | 5.09 | feature 1280 | feature 1158 |

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
