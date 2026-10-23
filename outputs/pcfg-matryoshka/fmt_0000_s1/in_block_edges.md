# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　1,022,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **206** directed edges, **7** duplicate pairs, 206 survive PMI>0; PolyFrac 35%, Gini 0.943.
- **S_res: 7/206** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 0 → 106 | 0.99 | 0.85 | feature 0 | feature 106 |
| 86 → 120 | 0.98 | 2.39 | feature 86 | feature 120 |
| 21 → 120 | 0.98 | 1.84 | feature 21 | feature 120 |
| 5 → 106 | 0.98 | 1.55 | feature 5 | feature 106 |
| 0 → 209 | 0.98 | 0.84 | feature 0 | feature 209 |
| 81 → 120 | 0.98 | 1.52 | feature 81 | feature 120 |
| 161 → 120 | 0.98 | 1.99 | feature 161 | feature 120 |
| 0 → 67 | 0.98 | 0.84 | feature 0 | feature 67 |

_Duplicate pairs (rename/split candidates):_ 21≈68 (feature 21); 21≈161 (feature 21); 35≈120 (feature 35); 35≈127 (feature 35); 35≈191 (feature 35); 120≈191 (feature 120)

## Block B1  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B2  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1156 → 1275 | 0.79 | 3.76 | feature 1156 | feature 1275 |

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
