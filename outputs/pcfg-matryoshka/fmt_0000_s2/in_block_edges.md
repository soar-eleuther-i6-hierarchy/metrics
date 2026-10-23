# In-block (same-level) directed edges

**Layer 2**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L2　·　1,792 latents in 8 blocks　·　1,022,000 tokens over 2000 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **213** directed edges, **6** duplicate pairs, 84 survive PMI>0; PolyFrac 24%, Gini 0.982.
- **S_res: 3/84** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F191 _feature 191_: 142 children, fires 56%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 222 → 152 | 1.00 | 1.52 | feature 222 | feature 152 |
| 186 → 152 | 1.00 | 1.55 | feature 186 | feature 152 |
| 186 → 30 | 0.99 | 1.55 | feature 186 | feature 30 |
| 222 → 30 | 0.99 | 1.52 | feature 222 | feature 30 |
| 99 → 176 | 0.99 | 1.16 | feature 99 | feature 176 |
| 9 → 30 | 0.99 | 2.57 | feature 9 | feature 30 |
| 152 → 30 | 0.99 | 4.10 | feature 152 | feature 30 |
| 123 → 30 | 0.99 | 1.57 | feature 123 | feature 30 |

_Duplicate pairs (rename/split candidates):_ 41≈181 (feature 41); 99≈161 (feature 99); 123≈142 (feature 123); 123≈186 (feature 123); 123≈222 (feature 123); 186≈222 (feature 186)

## Block B1  (224 features)
- **3** directed edges, **0** duplicate pairs, 3 survive PMI>0; PolyFrac 0%, Gini 0.987.
- **S_res: 1/3** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 424 → 330 | 0.70 | 5.50 | feature 424 | feature 330 |
| 430 → 317 | 0.63 | 3.48 | feature 430 | feature 317 |
| 283 → 336 | 0.57 | 1.88 | feature 283 | feature 336 |

## Block B2  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B3  (224 features)
- **2** directed edges, **0** duplicate pairs, 2 survive PMI>0; PolyFrac 100%, Gini 0.991.
- **S_res: 0/2** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 711 → 888 | 0.67 | 2.67 | feature 711 | feature 888 |
| 720 → 888 | 0.65 | 2.49 | feature 720 | feature 888 |

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B6  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B7  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.
