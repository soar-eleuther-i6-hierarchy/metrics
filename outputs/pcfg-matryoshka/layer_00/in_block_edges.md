# In-block (same-level) directed edges

**Layer 0**　·　PCFG toy 4L d_model=448 / pcfg　·　matryoshka_hook_resid_post_L0　·　1,792 latents in 8 blocks　·　1,016,600 tokens over 3400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (224 features)
- **763** directed edges, **59** duplicate pairs, 631 survive PMI>0; PolyFrac 94%, Gini 0.967.
- **S_res: 10/631** edges are genuine refinements (0 children untestable).
- In-block superparents: 4 (e.g. F129 _feature 129_: 156 children, fires 100%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 129 → 139 | 1.00 | 0.00 | feature 129 | feature 139 |
| 129 → 44 | 1.00 | 0.00 | feature 129 | feature 44 |
| 129 → 83 | 1.00 | 0.00 | feature 129 | feature 83 |
| 129 → 158 | 1.00 | 0.00 | feature 129 | feature 158 |
| 129 → 205 | 1.00 | 0.00 | feature 129 | feature 205 |
| 129 → 10 | 1.00 | 0.00 | feature 129 | feature 10 |
| 129 → 98 | 1.00 | 0.00 | feature 129 | feature 98 |
| 129 → 120 | 1.00 | 0.00 | feature 129 | feature 120 |

_Duplicate pairs (rename/split candidates):_ 10≈158 (feature 10); 13≈14 (feature 13); 13≈89 (feature 13); 14≈15 (feature 14); 14≈25 (feature 14); 14≈34 (feature 14)

## Block B1  (224 features)
- **8** directed edges, **0** duplicate pairs, 8 survive PMI>0; PolyFrac 50%, Gini 0.964.
- **S_res: 0/8** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 245 → 401 | 1.00 | 3.01 | feature 245 | feature 401 |
| 292 → 401 | 1.00 | 2.91 | feature 292 | feature 401 |
| 386 → 401 | 1.00 | 3.83 | feature 386 | feature 401 |
| 410 → 401 | 1.00 | 3.16 | feature 410 | feature 401 |
| 420 → 401 | 0.99 | 2.71 | feature 420 | feature 401 |
| 375 → 401 | 0.99 | 2.87 | feature 375 | feature 401 |
| 359 → 259 | 0.82 | 4.81 | feature 359 | feature 259 |
| 351 → 401 | 0.59 | 2.39 | feature 351 | feature 401 |

## Block B2  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 1/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 465 → 617 | 0.53 | 3.25 | feature 465 | feature 617 |

## Block B3  (224 features)
- **3** directed edges, **0** duplicate pairs, 3 survive PMI>0; PolyFrac 100%, Gini 0.987.
- **S_res: 2/3** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 674 → 798 | 0.91 | 4.46 | feature 674 | feature 798 |
| 758 → 798 | 0.88 | 4.29 | feature 758 | feature 798 |
| 830 → 798 | 0.85 | 4.71 | feature 830 | feature 798 |

## Block B4  (224 features)
- **0** directed edges, **0** duplicate pairs, 0 survive PMI>0; PolyFrac 0%, Gini 0.000.

## Block B5  (224 features)
- **5** directed edges, **0** duplicate pairs, 5 survive PMI>0; PolyFrac 25%, Gini 0.978.
- **S_res: 0/5** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1266 → 1291 | 0.86 | 5.26 | feature 1266 | feature 1291 |
| 1178 → 1121 | 0.82 | 5.55 | feature 1178 | feature 1121 |
| 1282 → 1121 | 0.76 | 4.81 | feature 1282 | feature 1121 |
| 1256 → 1182 | 0.71 | 5.09 | feature 1256 | feature 1182 |
| 1173 → 1230 | 0.71 | 5.18 | feature 1173 | feature 1230 |

## Block B6  (224 features)
- **1** directed edges, **0** duplicate pairs, 1 survive PMI>0; PolyFrac 0%, Gini 0.996.
- **S_res: 0/1** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1364 → 1567 | 0.52 | 5.26 | feature 1364 | feature 1567 |

## Block B7  (224 features)
- **2** directed edges, **0** duplicate pairs, 2 survive PMI>0; PolyFrac 0%, Gini 0.991.
- **S_res: 0/2** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1607 → 1634 | 0.90 | 6.16 | feature 1607 | feature 1634 |
| 1621 → 1573 | 0.88 | 5.69 | feature 1621 | feature 1573 |
