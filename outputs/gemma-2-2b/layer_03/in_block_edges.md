# In-block (same-level) directed edges

**Layer 3**　·　gemma-2-2b / 3-res-matryoshka-dc　·　blocks.3.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **557** directed edges, **16** duplicate pairs, 411 survive PMI>0; PolyFrac 98%, Gini 0.946.
- **S_res: 0/411** edges are genuine refinements (0 children untestable).
- In-block superparents: 4 (e.g. F70 _proper nouns that have mixed upper and lowerc…_: 124 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 70 → 77 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | LaTeX math environments |
| 70 → 13 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | the ends of sentences or phrases that include… |
| 70 → 36 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | the word "to" |
| 70 → 26 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | first and third person pronouns along with re… |
| 70 → 10 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | symbols utilized in complex scientific public… |
| 70 → 5 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | equal signs followed by a space and a hexadec… |
| 70 → 121 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | possessive pronouns and adjectives. |
| 70 → 57 | 1.00 | 0.01 | proper nouns that have mixed upper and lowerc… | the article "a" |

_Duplicate pairs (rename/split candidates):_ 1≈17 (words related to groups…); 17≈99 (words or phrases relate…); 17≈111 (words or phrases relate…); 39≈75 (scientific terms, espec…); 39≈81 (scientific terms, espec…); 52≈70 (sections from legal and…)

## Block B1  (384 features)
- **275** directed edges, **1** duplicate pairs, 275 survive PMI>0; PolyFrac 12%, Gini 0.997.
- **S_res: 0/275** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F448 _a grab bag of proper nouns including names, p…_: 244 children, fires 48%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 505 → 439 | 0.74 | 5.70 | legal and scientific research article formatt… | statistical data from scientific publications |
| 249 → 439 | 0.74 | 4.15 | law references, including numbers, footnotes,… | statistical data from scientific publications |
| 448 → 330 | 0.71 | 0.39 | a grab bag of proper nouns including names, p… | exponents |
| 448 → 208 | 0.70 | 0.38 | a grab bag of proper nouns including names, p… | scientific and technical texts, especially th… |
| 448 → 496 | 0.70 | 0.38 | a grab bag of proper nouns including names, p… | terms related to animal models in neuroscienc… |
| 448 → 188 | 0.70 | 0.37 | a grab bag of proper nouns including names, p… | LaTeX code |
| 448 → 280 | 0.70 | 0.37 | a grab bag of proper nouns including names, p… | citations to published research and words ind… |
| 448 → 217 | 0.69 | 0.37 | a grab bag of proper nouns including names, p… | mathematical formulas and notation |

_Duplicate pairs (rename/split candidates):_ 276≈342 (strings of numbers)

## Block B2  (1536 features)
- **1707** directed edges, **7** duplicate pairs, 858 survive PMI>0; PolyFrac 3%, Gini 0.991.
- **S_res: 7/835** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F1457 _words used in official documents and scientif…_: 1403 children, fires 88%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1457 → 1281 | 1.00 | 0.13 | words used in official documents and scientif… | the word "indicated" and related terms like "… |
| 1457 → 1259 | 0.99 | 0.12 | words used in official documents and scientif… | uses of the word "whether" |
| 1457 → 1837 | 0.98 | 0.11 | words used in official documents and scientif… | medical articles talking about probabilities … |
| 1457 → 1032 | 0.98 | 0.11 | words used in official documents and scientif… | the `aligned` environment tag in LaTeX. |
| 1457 → 1265 | 0.98 | 0.11 | words used in official documents and scientif… | mathematical symbols, especially gamma |
| 1457 → 1816 | 0.98 | 0.11 | words used in official documents and scientif… | the digit 5 |
| 1457 → 1274 | 0.98 | 0.11 | words used in official documents and scientif… | the word "stress" |
| 1457 → 1061 | 0.97 | 0.10 | words used in official documents and scientif… | the word "video", sometimes followed by a hyp… |

_Duplicate pairs (rename/split candidates):_ 655≈1032 (mathematical or physics…); 655≈1533 (mathematical or physics…); 954≈1593 (the phrase "required by"); 954≈2021 (the phrase "required by"); 1032≈1533 (the `aligned` environme…); 1233≈1765 (references to US Suprem…)

## Block B3  (6144 features)
- **1640** directed edges, **142** duplicate pairs, 1640 survive PMI>0; PolyFrac 42%, Gini 0.980.
- **S_res: 35/1632** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 5692 → 6997 | 1.00 | 5.11 | code documentation containing parameter decla… | the word "spatial" |
| 7291 → 6997 | 1.00 | 3.91 | the names of cities, municipalities, and rela… | the word "spatial" |
| 7147 → 6997 | 1.00 | 5.34 | mentions of disagreement and debate | the word "spatial" |
| 7141 → 6997 | 1.00 | 3.74 | prepositions | the word "spatial" |
| 7721 → 6997 | 1.00 | 5.61 | DNA sequences with their directionality. | the word "spatial" |
| 5562 → 6997 | 1.00 | 3.93 | proper nouns referring to people, places, org… | the word "spatial" |
| 6542 → 6997 | 1.00 | 5.49 | occurrences of the word "work" in code-relate… | the word "spatial" |
| 4300 → 6997 | 1.00 | 3.46 | technical details in product descriptions, me… | the word "spatial" |

_Duplicate pairs (rename/split candidates):_ 2334≈4691 (the character "t" at th…); 2346≈2358 (sequences of numbers, e…); 2346≈2556 (sequences of numbers, e…); 2346≈2915 (sequences of numbers, e…); 2346≈3948 (sequences of numbers, e…); 2346≈4532 (sequences of numbers, e…)

## Block B4  (24576 features)
- **1529** directed edges, **285** duplicate pairs, 1529 survive PMI>0; PolyFrac 38%, Gini 0.996.
- **S_res: 43/1508** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 19137 → 22548 | 1.00 | 4.71 | words related to either medical contexts or g… | square brackets |
| 20705 → 22548 | 1.00 | 5.01 | mentions of fatty acids, especially omegas | square brackets |
| 26727 → 8809 | 1.00 | 5.01 | words and phrases that suggest changes and tr… | the letter 'Q' |
| 24644 → 20350 | 1.00 | 5.74 | the term "amazonaws" as well as mentions of "… | format specifiers in code |
| 24577 → 20350 | 1.00 | 5.72 | words associated with singular outcomes, cond… | format specifiers in code |
| 13510 → 26858 | 1.00 | 3.18 | periods ending scientific statements that are… | indented paragraphs |
| 22117 → 20350 | 1.00 | 5.65 | terms related to aircraft and flight | format specifiers in code |
| 8400 → 22548 | 1.00 | 5.23 | words referencing a legal inquiry into a death | square brackets |

_Duplicate pairs (rename/split candidates):_ 8361≈14703 (mentions of the numbers…); 8470≈24531 (the first person singul…); 8680≈8809 (LaTeX math symbols); 8680≈10054 (LaTeX math symbols); 8680≈11623 (LaTeX math symbols); 8680≈14728 (LaTeX math symbols)
