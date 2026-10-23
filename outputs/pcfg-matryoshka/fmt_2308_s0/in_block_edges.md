# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　646,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **80** directed edges, **7** duplicate pairs, 80 survive PMI>0; PolyFrac 68%, Gini 0.955.
- **S_res: 3/80** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 81 → 140 | 1.00 | 1.45 | feature 81 | feature 140 |
| 75 → 140 | 1.00 | 1.46 | feature 75 | feature 140 |
| 36 → 209 | 1.00 | 1.44 | feature 36 | feature 209 |
| 53 → 176 | 1.00 | 2.47 | feature 53 | feature 176 |
| 81 → 176 | 1.00 | 1.45 | feature 81 | feature 176 |
| 69 → 176 | 1.00 | 2.46 | feature 69 | feature 176 |
| 75 → 196 | 1.00 | 1.46 | feature 75 | feature 196 |
| 81 → 196 | 1.00 | 1.45 | feature 81 | feature 196 |

_Duplicate pairs (rename/split candidates):_ 7≈43 (feature 7); 23≈75 (feature 23); 23≈81 (feature 23); 23≈196 (feature 23); 53≈69 (feature 53); 75≈81 (feature 75)

## Block B1  (224 features)
- **9** directed edges, **1** duplicate pairs, 9 survive PMI>0; PolyFrac 100%, Gini 0.968.
- **S_res: 2/9** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 331 → 424 | 0.91 | 4.18 | feature 331 | feature 424 |
| 331 → 436 | 0.88 | 4.15 | feature 331 | feature 436 |
| 424 → 436 | 0.80 | 4.82 | feature 424 | feature 436 |
| 229 → 424 | 0.76 | 3.70 | feature 229 | feature 424 |
| 330 → 436 | 0.67 | 4.20 | feature 330 | feature 436 |
| 295 → 423 | 0.63 | 1.78 | feature 295 | feature 423 |
| 230 → 423 | 0.62 | 2.03 | feature 230 | feature 423 |
| 432 → 423 | 0.59 | 2.00 | feature 432 | feature 423 |

_Duplicate pairs (rename/split candidates):_ 229≈331 (feature 229)

## Block B2  (224 features)
- **4** directed edges, **0** duplicate pairs, 4 survive PMI>0; PolyFrac 33%, Gini 0.989.
- **S_res: 1/4** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 589 → 454 | 0.89 | 3.65 | feature 589 | feature 454 |
| 604 → 610 | 0.59 | 3.60 | feature 604 | feature 610 |
| 610 → 552 | 0.59 | 3.87 | feature 610 | feature 552 |
| 604 → 552 | 0.59 | 3.59 | feature 604 | feature 552 |

## Block B3  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/0** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1668 → 1683 | 0.91 | 3.14 | feature 1668 | feature 1683 |
