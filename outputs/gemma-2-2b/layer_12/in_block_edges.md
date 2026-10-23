# In-block (same-level) directed edges

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **417** directed edges, **3** duplicate pairs, 302 survive PMI>0; PolyFrac 99%, Gini 0.951.
- **S_res: 0/302** edges are genuine refinements (0 children untestable).
- In-block superparents: 2 (e.g. F44 _technical terminology related to programming …_: 125 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 44 → 25 | 1.00 | 0.01 | technical terminology related to programming … | letters and symbols, particularly those assoc… |
| 44 → 106 | 1.00 | 0.01 | technical terminology related to programming … | biological processes and mechanisms related t… |
| 44 → 113 | 1.00 | 0.01 | technical terminology related to programming … | technical terms related to electrical and mec… |
| 44 → 111 | 1.00 | 0.01 | technical terminology related to programming … | scientific terminology related to genetic ana… |
| 44 → 30 | 1.00 | 0.01 | technical terminology related to programming … | phrases related to healthcare and training in… |
| 44 → 40 | 1.00 | 0.01 | technical terminology related to programming … | expressions of personal experiences and emoti… |
| 44 → 92 | 1.00 | 0.01 | technical terminology related to programming … | mentions of legal proceedings and implication… |
| 44 → 101 | 1.00 | 0.01 | technical terminology related to programming … | terms related to health and healthcare resear… |

_Duplicate pairs (rename/split candidates):_ 27≈75 (mathematical expression…); 44≈114 (technical terminology r…); 82≈101 (technical terms related…)

## Block B1  (384 features)
- **77** directed edges, **0** duplicate pairs, 77 survive PMI>0; PolyFrac 9%, Gini 0.995.
- **S_res: 3/77** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 383 → 487 | 0.86 | 0.75 | terms related to mathematical proofs and stru… | mathematical notation and expressions related… |
| 383 → 350 | 0.81 | 0.69 | terms related to mathematical proofs and stru… | syntactical structures and special characters… |
| 383 → 499 | 0.80 | 0.67 | terms related to mathematical proofs and stru… | references to legal proceedings and outcomes |
| 383 → 133 | 0.77 | 0.63 | terms related to mathematical proofs and stru… | technical terms related to computer systems a… |
| 383 → 488 | 0.74 | 0.59 | terms related to mathematical proofs and stru… | topics related to indigenous peoples and thei… |
| 383 → 398 | 0.73 | 0.58 | terms related to mathematical proofs and stru… | incomplete or interrupted sentences and their… |
| 383 → 249 | 0.67 | 0.50 | terms related to mathematical proofs and stru… | references to legal cases and jurisdictions |
| 383 → 188 | 0.66 | 0.48 | terms related to mathematical proofs and stru… | references to legal terminology and court cas… |

## Block B2  (1536 features)
- **268** directed edges, **10** duplicate pairs, 268 survive PMI>0; PolyFrac 21%, Gini 0.963.
- **S_res: 9/268** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1011 → 1001 | 0.75 | 2.05 | technical descriptions of computer systems an… | mathematical expressions and notations relate… |
| 1925 → 1001 | 0.74 | 3.54 | CSS-related issues, particularly those involv… | mathematical expressions and notations relate… |
| 887 → 1001 | 0.73 | 3.02 | technical terms related to programming and so… | mathematical expressions and notations relate… |
| 2002 → 1001 | 0.73 | 5.17 | JSON-like structure elements and properties | mathematical expressions and notations relate… |
| 1916 → 1001 | 0.73 | 3.18 | legal and medical terms related to injuries a… | mathematical expressions and notations relate… |
| 761 → 1001 | 0.73 | 3.43 | references to data structures or programming … | mathematical expressions and notations relate… |
| 1110 → 1001 | 0.73 | 2.88 | key terms related to healthcare professionals… | mathematical expressions and notations relate… |
| 1075 → 1001 | 0.73 | 4.78 | JSON-like structured data elements and key-va… | mathematical expressions and notations relate… |

_Duplicate pairs (rename/split candidates):_ 1001≈1052 (mathematical expression…); 1001≈1466 (mathematical expression…); 1001≈1659 (mathematical expression…); 1001≈1883 (mathematical expression…); 1052≈1466 (mathematical symbols an…); 1052≈1659 (mathematical symbols an…)

## Block B3  (6144 features)
- **1844** directed edges, **258** duplicate pairs, 1844 survive PMI>0; PolyFrac 37%, Gini 0.986.
- **S_res: 28/1839** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 5636 → 6486 | 1.00 | 4.71 | mathematical notations and symbols | mathematical symbols and expressions, particu… |
| 3394 → 6486 | 1.00 | 4.68 | discussions related to cryptography and encry… | mathematical symbols and expressions, particu… |
| 6216 → 6486 | 0.98 | 5.32 | numerical data related to counts, measurement… | mathematical symbols and expressions, particu… |
| 3873 → 6486 | 0.98 | 3.80 | phrases related to medical guidelines and pro… | mathematical symbols and expressions, particu… |
| 7244 → 6486 | 0.98 | 4.84 | references to academic work and acknowledgmen… | mathematical symbols and expressions, particu… |
| 3145 → 6486 | 0.98 | 5.37 | medical terminologies related to diagnostic m… | mathematical symbols and expressions, particu… |
| 3179 → 6486 | 0.98 | 4.71 | phrases related to political campaigns and th… | mathematical symbols and expressions, particu… |
| 4609 → 6486 | 0.98 | 2.98 | terms related to magnetic properties and perm… | mathematical symbols and expressions, particu… |

_Duplicate pairs (rename/split candidates):_ 2168≈2209 (structured programming …); 2168≈3519 (structured programming …); 2168≈3541 (structured programming …); 2168≈3661 (structured programming …); 2168≈3671 (structured programming …); 2168≈3788 (structured programming …)

## Block B4  (24576 features)
- **1737** directed edges, **328** duplicate pairs, 1737 survive PMI>0; PolyFrac 10%, Gini 0.999.
- **S_res: 55/1677** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 12427 → 20071 | 1.00 | 5.88 | questions formatted in a specific way | references to monetary values or financial fi… |
| 10480 → 20071 | 1.00 | 4.22 | elements related to health and wellness servi… | references to monetary values or financial fi… |
| 25244 → 20071 | 1.00 | 4.08 | keywords and names related to science and gen… | references to monetary values or financial fi… |
| 19815 → 24367 | 1.00 | 5.83 | references to personal relationships and emot… | status and sequencing information |
| 18833 → 24367 | 1.00 | 5.40 | behaviors that indicate defiance or non-compl… | status and sequencing information |
| 18833 → 20071 | 1.00 | 5.40 | behaviors that indicate defiance or non-compl… | references to monetary values or financial fi… |
| 18200 → 24367 | 1.00 | 5.88 | phrases related to forensic evidence and phys… | status and sequencing information |
| 18200 → 20071 | 1.00 | 5.88 | phrases related to forensic evidence and phys… | references to monetary values or financial fi… |

_Duplicate pairs (rename/split candidates):_ 8474≈11385 (terms related to genera…); 8474≈11445 (terms related to genera…); 8474≈14389 (terms related to genera…); 8474≈15823 (terms related to genera…); 8474≈16734 (terms related to genera…); 8474≈16923 (terms related to genera…)
