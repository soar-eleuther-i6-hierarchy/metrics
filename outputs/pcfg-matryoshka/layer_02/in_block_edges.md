# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **128** directed edges, **19** duplicate pairs, 128 survive PMI>0; PolyFrac 39%, Gini 0.929.
- **S_res: 6/128** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 34 → 125 | 1.00 | 1.37 | feature 34 | feature 125 |
| 5 → 125 | 1.00 | 2.49 | feature 5 | feature 125 |
| 34 → 121 | 1.00 | 1.37 | feature 34 | feature 121 |
| 34 → 25 | 1.00 | 1.37 | feature 34 | feature 25 |
| 155 → 25 | 1.00 | 1.80 | feature 155 | feature 25 |
| 158 → 25 | 1.00 | 1.45 | feature 158 | feature 25 |
| 41 → 25 | 1.00 | 1.79 | feature 41 | feature 25 |
| 12 → 125 | 1.00 | 2.02 | feature 12 | feature 125 |

_Duplicate pairs (rename/split candidates):_ 5≈12 (feature 5); 12≈34 (feature 12); 12≈98 (feature 12); 34≈41 (feature 34); 34≈98 (feature 34); 34≈155 (feature 34)

## Block B1  (224 features)
- **4** directed edges, **0** duplicate pairs, 4 survive PMI>0; PolyFrac 50%, Gini 0.982.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 443 → 395 | 0.99 | 4.88 | feature 443 | feature 395 |
| 444 → 379 | 0.89 | 2.93 | feature 444 | feature 379 |
| 351 → 379 | 0.86 | 2.99 | feature 351 | feature 379 |
| 417 → 379 | 0.86 | 2.02 | feature 417 | feature 379 |

## Block B2  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

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
