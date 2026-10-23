# Edge activation magnitudes

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, co-fire ≥ 30, both endpoints fire ≥ 20

Per kept edge: endpoint mean activations on the tokens both fire on. `child/parent` compares the two means; `parent shared/all` is the parent's mean there against its mean over all its firings — well below 1 means the parent goes quiet exactly where the child fires, the regime where the reconstruction contribution filter can fail an active feature.

## Firing values per block

| block | firings | p10 | p50 | p90 |
|---|---|---|---|---|
| B0 | 679,452 | 1.63 | 4.97 | 24.36 |
| B1 | 622,767 | 1.47 | 2.76 | 10.16 |
| B2 | 988,106 | 1.44 | 2.49 | 7.54 |
| B3 | 1,105,615 | 1.42 | 2.22 | 6.18 |
| B4 | 806,099 | 1.41 | 2.18 | 6.30 |

## Kept edges per pair

| pair | edges | child/parent p50 | % child stronger | parent shared/all p50 | % parent halved |
|---|---|---|---|---|---|
| 0->1 | 1971 | 0.58 | 13% | 1.07 | 0% |
| 1->2 | 1387 | 0.88 | 39% | 1.11 | 5% |
| 2->3 | 4379 | 0.85 | 39% | 0.99 | 7% |

### 0->1: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 17 → 217 | 0.57 | 0.33 | words or phrases related to science and… | mathematical formulas and notation |
| 70 → 461 | 0.61 | 0.37 | proper nouns that have mixed upper and … | mathematical formulas with exponents, a… |
| 52 → 313 | 0.62 | 1.59 | sections from legal and scientific publ… | code snippets and structured data, part… |
| 111 → 184 | 0.62 | 0.58 | code, formatting and markup syntax, and… | uses of the word "first" and subsequent… |
| 70 → 194 | 0.63 | 0.45 | proper nouns that have mixed upper and … | decimal numbers |
| 70 → 325 | 0.64 | 0.27 | proper nouns that have mixed upper and … | computer code |
| 70 → 439 | 0.64 | 0.86 | proper nouns that have mixed upper and … | statistical data from scientific public… |
| 70 → 505 | 0.65 | 0.76 | proper nouns that have mixed upper and … | legal and scientific research article f… |

### 1->2: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 209 → 2021 | 0.24 | 0.96 | source code and equations | bold math symbols |
| 209 → 1593 | 0.24 | 1.36 | source code and equations | horizontal lines made of equals signs |
| 223 → 2021 | 0.26 | 0.83 | questions that ask "What are..." or "Wh… | bold math symbols |
| 223 → 954 | 0.26 | 0.92 | questions that ask "What are..." or "Wh… | the phrase "required by" |
| 223 → 1593 | 0.26 | 1.18 | questions that ask "What are..." or "Wh… | horizontal lines made of equals signs |
| 209 → 1678 | 0.27 | 2.04 | source code and equations | LaTeX code |
| 223 → 1678 | 0.28 | 1.89 | questions that ask "What are..." or "Wh… | LaTeX code |
| 209 → 954 | 0.28 | 0.93 | source code and equations | the phrase "required by" |

### 2->3: parents most weakened on their child's tokens

| parent → child | shared/all | child/parent | parent label | child label |
|---|---|---|---|---|
| 1766 → 2404 | 0.12 | 13.21 | the names of days of the week | question delimiters |
| 1766 → 4488 | 0.12 | 1.48 | the names of days of the week | legal citations of cases |
| 1766 → 4750 | 0.12 | 1.45 | the names of days of the week | the phrase "behind the" |
| 1766 → 4815 | 0.12 | 2.83 | the names of days of the week | the word "bottom" |
| 1766 → 5601 | 0.12 | 1.05 | the names of days of the week | asterisks |
| 1766 → 5679 | 0.12 | 1.79 | the names of days of the week | the letters "UT" or "ut" |
| 1766 → 5876 | 0.12 | 1.13 | the names of days of the week | positions or ordinals in mathematical t… |
| 1766 → 6815 | 0.12 | 1.68 | the names of days of the week | math multiple choice questions |
