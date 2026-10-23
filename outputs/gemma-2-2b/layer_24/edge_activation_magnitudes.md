# Edge activation magnitudes

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 677,187 | 13.02 | 37.94 | 165.00 |
| B1 | 597,346 | 11.52 | 19.55 | 65.00 |
| B2 | 851,798 | 11.34 | 17.77 | 51.78 |
| B3 | 1,163,676 | 11.30 | 16.95 | 43.09 |
| B4 | 780,092 | 11.32 | 17.28 | 45.78 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 2273 | 0.45 | 6% | 1.01 | 0% |
| 1->2 | 424 | 0.74 | 26% | 1.47 | 5% |
| 2->3 | 4867 | 0.94 | 43% | 1.03 | 4% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 54 → 186 | 0.44 | 0.61 | Medical concepts related to arteries, b… | characters and symbols associated with … |
| 49 → 220 | 0.44 | 0.38 | what appears to be the start of a disjo… | references to figures, tables, and supp… |
| 32 → 493 | 0.45 | 1.09 | numbers within a document | mixed English and Russian text related … |
| 32 → 220 | 0.48 | 0.10 | numbers within a document | references to figures, tables, and supp… |
| 32 → 264 | 0.48 | 0.26 | numbers within a document | code-related tokens, especially within … |
| 85 → 156 | 0.50 | 1.58 | code and jargon related to technology a… | number sequences |
| 32 → 395 | 0.52 | 0.20 | numbers within a document | code syntax with underscores and hyphens |
| 32 → 231 | 0.52 | 0.24 | numbers within a document | capitalized words or acronyms separated… |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 387 → 1065 | 0.28 | 0.98 | contractions with the word "not." | math equations |
| 208 → 1114 | 0.33 | 0.63 | instances of the word "There" | math problems involving sequences, and … |
| 208 → 1170 | 0.33 | 0.43 | instances of the word "There" | occurrences of C# (C Sharp) code |
| 276 → 1065 | 0.34 | 0.98 | US legal and code citations. | math equations |
| 400 → 1065 | 0.34 | 0.99 | mentions of spans of time | math equations |
| 393 → 1065 | 0.35 | 0.77 | words used in electrical engineering co… | math equations |
| 505 → 1114 | 0.35 | 1.43 | language associated with death and prof… | math problems involving sequences, and … |
| 505 → 1170 | 0.35 | 0.98 | language associated with death and prof… | occurrences of C# (C Sharp) code |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1762 → 3200 | 0.21 | 1.03 | technical specifications for camera len… | sentences that describe the authors of … |
| 1762 → 4616 | 0.21 | 1.04 | technical specifications for camera len… | code pertaining to error messages |
| 1762 → 5445 | 0.21 | 0.90 | technical specifications for camera len… | LaTeX, code snippets, and/or data |
| 1762 → 7270 | 0.21 | 1.76 | technical specifications for camera len… | the letter "A" followed by a colon |
| 1762 → 7843 | 0.21 | 0.95 | technical specifications for camera len… | statistical analysis values, particular… |
| 928 → 2597 | 0.26 | 0.92 | code or text broken up into single char… | the description of baseball plays, spec… |
| 928 → 3113 | 0.26 | 1.07 | code or text broken up into single char… | technical documents about elements |
| 928 → 3411 | 0.26 | 1.05 | code or text broken up into single char… | mathematical expressions, especially th… |
