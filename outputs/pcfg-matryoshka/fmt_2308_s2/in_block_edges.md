# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　646,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **66** directed edges, **4** duplicate pairs, 66 survive PMI>0; PolyFrac 20%, Gini 0.977.
- **S_res: 0/66** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 167 → 221 | 1.00 | 2.02 | feature 167 | feature 221 |
| 167 → 188 | 0.98 | 2.01 | feature 167 | feature 188 |
| 99 → 217 | 0.98 | 1.42 | feature 99 | feature 217 |
| 57 → 144 | 0.98 | 1.80 | feature 57 | feature 144 |
| 167 → 40 | 0.90 | 1.91 | feature 167 | feature 40 |
| 167 → 120 | 0.89 | 1.91 | feature 167 | feature 120 |
| 223 → 51 | 0.88 | 0.85 | feature 223 | feature 51 |
| 167 → 173 | 0.84 | 1.85 | feature 167 | feature 173 |

_Duplicate pairs (rename/split candidates):_ 13≈100 (feature 13); 13≈184 (feature 13); 25≈100 (feature 25); 173≈188 (feature 173)

## Block B1  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 318 → 247 | 0.82 | 3.10 | feature 318 | feature 247 |

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 590 → 553 | 0.57 | 1.85 | feature 590 | feature 553 |

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
