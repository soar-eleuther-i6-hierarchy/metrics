# In-block (same-level) directed edges

**Layer 6**　·　gemma-2-2b / 6-res-matryoshka-dc　·　blocks.6.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **712** directed edges, **11** duplicate pairs, 468 survive PMI>0; PolyFrac 99%, Gini 0.940.
- In-block superparents: 5 (e.g. F15 _technical documentation-like language, includ…_: 123 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 15 → 8 | 1.00 | 0.01 | technical documentation-like language, includ… | code blocks and math formulas |
| 15 → 34 | 1.00 | 0.01 | technical documentation-like language, includ… | discourse markers and conjunctions like "that… |
| 15 → 0 | 1.00 | 0.01 | technical documentation-like language, includ… | numerical quantities |
| 15 → 19 | 1.00 | 0.01 | technical documentation-like language, includ… | possessive pronouns, especially "your," "our,… |
| 15 → 28 | 1.00 | 0.01 | technical documentation-like language, includ… | "is" or other forms of the verb "to be" |
| 15 → 91 | 1.00 | 0.01 | technical documentation-like language, includ… | terms related to medical and biological proce… |
| 15 → 99 | 1.00 | 0.01 | technical documentation-like language, includ… | words related to research, studies, and data … |
| 15 → 38 | 1.00 | 0.01 | technical documentation-like language, includ… | words frequently used in online forum posts o… |

_Duplicate pairs (rename/split candidates):_ 1≈15 (words related to scienc…); 1≈37 (words related to scienc…); 1≈42 (words related to scienc…); 15≈37 (technical documentation…); 15≈42 (technical documentation…); 15≈105 (technical documentation…)

## Block B1  (384 features)
- **15** directed edges, **1** duplicate pairs, 15 survive PMI>0; PolyFrac 7%, Gini 0.992.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 302 → 479 | 0.66 | 2.63 | source code and/or documents with very partic… | leading whitespace followed by numerical valu… |
| 236 → 159 | 0.60 | 4.57 | section headers | horizontal lines of dashes used within tables |
| 302 → 311 | 0.55 | 2.46 | source code and/or documents with very partic… | something, but it's too difficult to determin… |
| 311 → 479 | 0.55 | 2.90 | something, but it's too difficult to determin… | leading whitespace followed by numerical valu… |
| 401 → 370 | 0.54 | 0.37 | computing-related content, especially softwar… | possessive pronouns and the word it |
| 401 → 482 | 0.52 | 0.35 | computing-related content, especially softwar… | code snippets and programming terms |
| 401 → 208 | 0.52 | 0.34 | computing-related content, especially softwar… | mentions of children and adolescents along wi… |
| 401 → 133 | 0.52 | 0.34 | computing-related content, especially softwar… | the word "name" and its variants in programmi… |

_Duplicate pairs (rename/split candidates):_ 159≈462 (horizontal lines of das…)

## Block B2  (1536 features)
- **88** directed edges, **0** duplicate pairs, 88 survive PMI>0; PolyFrac 14%, Gini 0.966.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1554 → 1501 | 0.91 | 3.37 | the numeral 1 | numbers starting with 1 |
| 1191 → 1091 | 0.86 | 1.35 | scientific and technical writing, particularl… | names of people or places, abbreviations, and… |
| 1554 → 1694 | 0.82 | 3.27 | the numeral 1 | references to US legal cases |
| 1472 → 1694 | 0.82 | 4.69 | code-related terms, including both assembly l… | references to US legal cases |
| 1501 → 1694 | 0.82 | 4.21 | numbers starting with 1 | references to US legal cases |
| 1351 → 1694 | 0.78 | 3.88 | references to the World Meteorological Organi… | references to US legal cases |
| 1195 → 1694 | 0.78 | 3.98 | text related to automated technology and arti… | references to US legal cases |
| 1380 → 1694 | 0.78 | 1.83 | an assortment of common words and parts of wo… | references to US legal cases |

## Block B3  (6144 features)
- **566** directed edges, **19** duplicate pairs, 566 survive PMI>0; PolyFrac 45%, Gini 0.983.

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 5308 → 4624 | 0.97 | 5.28 | code syntax | mentions of asynchronous operations in code |
| 3791 → 4624 | 0.97 | 4.65 | java and android code | mentions of asynchronous operations in code |
| 7840 → 4624 | 0.97 | 4.33 | text fragments that are not standard and/or i… | mentions of asynchronous operations in code |
| 3099 → 4624 | 0.97 | 3.76 | names of places and people, especially in pol… | mentions of asynchronous operations in code |
| 5535 → 4624 | 0.97 | 4.89 | words and phrases related to inflammation and… | mentions of asynchronous operations in code |
| 4615 → 4624 | 0.94 | 5.75 | angle brackets followed by certain html input… | mentions of asynchronous operations in code |
| 2836 → 4624 | 0.94 | 2.91 | words and phrases that indicate a comparison,… | mentions of asynchronous operations in code |
| 7600 → 4624 | 0.94 | 5.14 | email headers and starting lines | mentions of asynchronous operations in code |

_Duplicate pairs (rename/split candidates):_ 2624≈5936 (LaTeX math notation); 3014≈4217 (decimal numbers); 3513≈3772 (words relating to scien…); 3513≈5838 (words relating to scien…); 3513≈5963 (words relating to scien…); 3717≈4476 (C++ documentation block…)
