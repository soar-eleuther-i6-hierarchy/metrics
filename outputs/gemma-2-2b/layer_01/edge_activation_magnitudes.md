# Edge activation magnitudes

**Layer 1**　·　gemma-2-2b / 1-res-matryoshka-dc　·　blocks.1.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 725,235 | 1.30 | 5.12 | 26.00 |
| B1 | 436,246 | 1.14 | 2.03 | 9.30 |
| B2 | 712,190 | 1.11 | 1.77 | 7.33 |
| B3 | 924,468 | 1.13 | 1.90 | 5.82 |
| B4 | 1,024,787 | 1.12 | 1.81 | 5.25 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 2929 | 0.55 | 25% | 1.00 | 0% |
| 1->2 | 1861 | 1.06 | 55% | 0.91 | 40% |
| 2->3 | 16729 | 0.96 | 44% | 0.45 | 56% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 96 → 148 | 0.19 | 1.62 | sequences of formatting and style code … | complex numerical relationships and pat… |
| 73 → 219 | 0.21 | 12.89 | references to formal mathematical expre… | references to self-organization and eme… |
| 73 → 193 | 0.21 | 0.92 | references to formal mathematical expre… | examples of the word "such." |
| 17 → 148 | 0.22 | 0.37 | sequences of special characters and for… | complex numerical relationships and pat… |
| 111 → 148 | 0.24 | 1.29 | segments of text that are enclosed by q… | complex numerical relationships and pat… |
| 95 → 219 | 0.32 | 7.30 | references to color spaces, especially … | references to self-organization and eme… |
| 91 → 148 | 0.33 | 2.67 | expressions of nested parentheses | complex numerical relationships and pat… |
| 95 → 281 | 0.41 | 0.65 | references to color spaces, especially … | references to the concept of a "between… |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 403 → 1845 | 0.06 | 1.19 | patterns indicative of structured, form… | references to network-related technolog… |
| 403 → 668 | 0.06 | 1.02 | patterns indicative of structured, form… | references to specific gene names |
| 403 → 678 | 0.06 | 1.10 | patterns indicative of structured, form… | references to the defendant |
| 403 → 1846 | 0.06 | 1.15 | patterns indicative of structured, form… | judgments and related legal final decis… |
| 403 → 2036 | 0.06 | 0.96 | patterns indicative of structured, form… | references to executing commands or scr… |
| 403 → 1647 | 0.06 | 1.93 | patterns indicative of structured, form… | references to filtering or filtering de… |
| 403 → 955 | 0.06 | 1.58 | patterns indicative of structured, form… | calculations involving the variable l |
| 403 → 1791 | 0.06 | 1.40 | patterns indicative of structured, form… | structures related to tensor products i… |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1338 → 2092 | 0.08 | 0.72 | references to the word "that." | references to property-like entities wi… |
| 1338 → 2179 | 0.08 | 0.98 | references to the word "that." | references to court decisions or rulings |
| 1338 → 2261 | 0.08 | 0.73 | references to the word "that." | references to chest-related medical iss… |
| 1338 → 2310 | 0.08 | 0.70 | references to the word "that." | mathematical expressions and array-base… |
| 1338 → 2474 | 0.08 | 1.15 | references to the word "that." | references to the technique of pattern … |
| 1338 → 2485 | 0.08 | 0.73 | references to the word "that." | references to specific types of transpo… |
| 1338 → 2578 | 0.08 | 1.31 | references to the word "that." | references to measures of diversity or … |
| 1338 → 2595 | 0.08 | 1.06 | references to the word "that." | references to the scientific term "infl… |
