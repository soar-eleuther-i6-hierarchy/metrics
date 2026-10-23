# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　598,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **154** directed edges, **6** duplicate pairs, 154 survive PMI>0; PolyFrac 8%, Gini 0.987.
- **S_res: 3/154** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F83 _feature 83_: 116 children, fires 37%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 121 → 118 | 1.00 | 1.39 | feature 121 | feature 118 |
| 121 → 204 | 1.00 | 1.39 | feature 121 | feature 204 |
| 121 → 25 | 1.00 | 1.38 | feature 121 | feature 25 |
| 121 → 101 | 0.99 | 1.38 | feature 121 | feature 101 |
| 121 → 209 | 0.99 | 1.38 | feature 121 | feature 209 |
| 212 → 42 | 0.99 | 1.34 | feature 212 | feature 42 |
| 212 → 160 | 0.99 | 1.34 | feature 212 | feature 160 |
| 164 → 118 | 0.98 | 1.79 | feature 164 | feature 118 |

_Duplicate pairs (rename/split candidates):_ 9≈169 (feature 9); 101≈169 (feature 101); 121≈161 (feature 121); 121≈164 (feature 121); 161≈164 (feature 161); 164≈209 (feature 164)

## Block B1  (224 features)
- **5** directed edges, **0** duplicate pairs, 5 survive PMI>0; PolyFrac 50%, Gini 0.978.
- **S_res: 1/5** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 297 → 241 | 0.68 | 2.00 | feature 297 | feature 241 |
| 337 → 375 | 0.63 | 1.87 | feature 337 | feature 375 |
| 309 → 241 | 0.53 | 2.03 | feature 309 | feature 241 |
| 321 → 241 | 0.51 | 1.86 | feature 321 | feature 241 |
| 296 → 241 | 0.50 | 1.53 | feature 296 | feature 241 |

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 608 → 517 | 0.74 | 5.36 | feature 608 | feature 517 |

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/0** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1152 → 1207 | 0.97 | 3.92 | feature 1152 | feature 1207 |

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
