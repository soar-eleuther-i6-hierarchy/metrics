# Edge activation magnitudes

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 629,247 | 3.65 | 9.83 | 39.09 |
| B1 | 556,031 | 3.16 | 5.45 | 16.08 |
| B2 | 753,850 | 3.08 | 4.71 | 13.23 |
| B3 | 1,362,205 | 3.04 | 4.34 | 10.08 |
| B4 | 1,098,834 | 3.03 | 4.25 | 9.57 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 1473 | 0.56 | 8% | 1.03 | 0% |
| 1->2 | 621 | 0.82 | 32% | 1.07 | 6% |
| 2->3 | 1747 | 0.91 | 39% | 0.76 | 12% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 44 → 209 | 0.37 | 0.27 | technical terminology related to progra… | terms related to legal proceedings and … |
| 44 → 147 | 0.50 | 0.16 | technical terminology related to progra… | phrases associated with legal terminolo… |
| 44 → 369 | 0.52 | 0.21 | technical terminology related to progra… | references to legal terminology and cas… |
| 44 → 433 | 0.52 | 0.17 | technical terminology related to progra… | references to legal cases and citations… |
| 44 → 402 | 0.52 | 0.16 | technical terminology related to progra… | numbers and related numerical values or… |
| 44 → 287 | 0.53 | 0.14 | technical terminology related to progra… | structured data or code snippets with r… |
| 44 → 394 | 0.54 | 0.22 | technical terminology related to progra… | legal citations and references in judic… |
| 44 → 145 | 0.54 | 0.25 | technical terminology related to progra… | numerical values and their occurrences … |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 370 → 775 | 0.24 | 2.45 | proper names and specific identifiers w… | complex mathematical expressions, parti… |
| 196 → 775 | 0.30 | 2.05 | the word "with" and its variations in d… | complex mathematical expressions, parti… |
| 250 → 1466 | 0.32 | 1.26 | questions and interrogative phrases | mathematical expressions and calculatio… |
| 307 → 775 | 0.32 | 2.48 | expressions of necessity or requirement | complex mathematical expressions, parti… |
| 250 → 1659 | 0.32 | 0.88 | questions and interrogative phrases | legal terminology related to jurisdicti… |
| 250 → 1883 | 0.32 | 1.00 | questions and interrogative phrases | references to citations or legal docume… |
| 250 → 1001 | 0.32 | 1.10 | questions and interrogative phrases | mathematical expressions and notations … |
| 250 → 1052 | 0.33 | 0.83 | questions and interrogative phrases | mathematical symbols and notations used… |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 936 → 2168 | 0.20 | 1.31 | terms related to inventions and their d… | structured programming elements and syn… |
| 936 → 2209 | 0.20 | 1.06 | terms related to inventions and their d… | mathematical expressions and variables … |
| 936 → 3519 | 0.20 | 1.45 | terms related to inventions and their d… | modal verbs |
| 936 → 3671 | 0.20 | 1.68 | terms related to inventions and their d… | mathematical expressions and symbols re… |
| 936 → 3903 | 0.20 | 1.14 | terms related to inventions and their d… | character patterns and formats typicall… |
| 936 → 4094 | 0.20 | 1.83 | terms related to inventions and their d… | mathematical expressions involving vari… |
| 936 → 5145 | 0.20 | 1.01 | terms related to inventions and their d… | proper nouns, specifically names and ti… |
| 936 → 5889 | 0.20 | 1.01 | terms related to inventions and their d… | sections or markers within a structured… |
