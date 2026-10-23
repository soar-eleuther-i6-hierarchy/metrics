# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　1,022,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **45** directed edges, **4** duplicate pairs, 45 survive PMI>0; PolyFrac 80%, Gini 0.936.
- **S_res: 4/45** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 185 → 165 | 1.00 | 1.20 | feature 185 | feature 165 |
| 185 → 184 | 0.99 | 1.19 | feature 185 | feature 184 |
| 185 → 102 | 0.99 | 1.19 | feature 185 | feature 102 |
| 102 → 184 | 0.99 | 2.25 | feature 102 | feature 184 |
| 193 → 119 | 0.99 | 1.12 | feature 193 | feature 119 |
| 125 → 119 | 0.99 | 1.08 | feature 125 | feature 119 |
| 185 → 24 | 0.99 | 1.18 | feature 185 | feature 24 |
| 185 → 17 | 0.97 | 1.17 | feature 185 | feature 17 |

_Duplicate pairs (rename/split candidates):_ 84≈125 (feature 84); 84≈193 (feature 84); 125≈193 (feature 125); 154≈169 (feature 154)

## Block B1  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B2  (224 features)
- **2** directed edges, **0** duplicate pairs, 2 survive PMI>0; PolyFrac 100%, Gini 0.991.
- **S_res: 1/2** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 500 → 617 | 0.71 | 3.53 | feature 500 | feature 617 |
| 598 → 617 | 0.61 | 2.39 | feature 598 | feature 617 |

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **2** directed edges, **0** duplicate pairs, 2 survive PMI>0; PolyFrac 100%, Gini 0.991.
- **S_res: 0/2** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1141 → 1291 | 0.60 | 2.30 | feature 1141 | feature 1291 |
| 1264 → 1291 | 0.54 | 3.34 | feature 1264 | feature 1291 |

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
