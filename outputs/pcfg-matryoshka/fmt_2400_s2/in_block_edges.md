# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　656,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **881** directed edges, **49** duplicate pairs, 811 survive PMI>0; PolyFrac 54%, Gini 0.888.
- **S_res: 22/811** edges are genuine refinements (0 children untestable).
- In-block superparents: 2 (e.g. F94 _feature 94_: 186 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 94 → 61 | 1.00 | 0.01 | feature 94 | feature 61 |
| 94 → 152 | 1.00 | 0.01 | feature 94 | feature 152 |
| 94 → 115 | 1.00 | 0.01 | feature 94 | feature 115 |
| 40 → 173 | 1.00 | 1.69 | feature 40 | feature 173 |
| 94 → 172 | 1.00 | 0.01 | feature 94 | feature 172 |
| 94 → 105 | 1.00 | 0.01 | feature 94 | feature 105 |
| 94 → 19 | 1.00 | 0.01 | feature 94 | feature 19 |
| 129 → 27 | 1.00 | 1.70 | feature 129 | feature 27 |

_Duplicate pairs (rename/split candidates):_ 8≈184 (feature 8); 17≈46 (feature 17); 17≈59 (feature 17); 17≈178 (feature 17); 17≈212 (feature 17); 26≈46 (feature 26)

## Block B1  (224 features)
- **30** directed edges, **4** duplicate pairs, 30 survive PMI>0; PolyFrac 62%, Gini 0.959.
- **S_res: 3/30** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 420 → 292 | 0.99 | 2.25 | feature 420 | feature 292 |
| 403 → 292 | 0.99 | 2.04 | feature 403 | feature 292 |
| 250 → 292 | 0.99 | 2.55 | feature 250 | feature 292 |
| 372 → 292 | 0.99 | 4.51 | feature 372 | feature 292 |
| 277 → 292 | 0.99 | 5.03 | feature 277 | feature 292 |
| 266 → 394 | 0.99 | 4.79 | feature 266 | feature 394 |
| 266 → 231 | 0.98 | 4.78 | feature 266 | feature 231 |
| 273 → 414 | 0.92 | 2.67 | feature 273 | feature 414 |

_Duplicate pairs (rename/split candidates):_ 224≈414 (feature 224); 231≈415 (feature 231); 266≈415 (feature 266); 310≈414 (feature 310)

## Block B2  (224 features)
- **12** directed edges, **0** duplicate pairs, 12 survive PMI>0; PolyFrac 43%, Gini 0.969.
- **S_res: 2/12** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 654 → 482 | 0.96 | 3.97 | feature 654 | feature 482 |
| 520 → 617 | 0.83 | 2.65 | feature 520 | feature 617 |
| 520 → 635 | 0.73 | 2.53 | feature 520 | feature 635 |
| 615 → 545 | 0.70 | 1.89 | feature 615 | feature 545 |
| 612 → 545 | 0.68 | 1.99 | feature 612 | feature 545 |
| 521 → 466 | 0.66 | 2.24 | feature 521 | feature 466 |
| 600 → 466 | 0.62 | 2.13 | feature 600 | feature 466 |
| 520 → 609 | 0.60 | 2.33 | feature 520 | feature 609 |

## Block B3  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 707 → 894 | 0.57 | 3.60 | feature 707 | feature 894 |

## Block B4  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 971 → 1022 | 0.85 | 4.64 | feature 971 | feature 1022 |

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
