# In-block (same-level) directed edges

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **345** directed edges, **1** duplicate pairs, 270 survive PMI>0; PolyFrac 86%, Gini 0.949.
- **S_res: 0/270** edges are genuine refinements (0 children untestable).
- In-block superparents: 3 (e.g. F89 _technical or scientific language related to d…_: 126 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 89 → 116 | 1.00 | 0.01 | technical or scientific language related to d… | words related to international treaties and c… |
| 89 → 113 | 1.00 | 0.01 | technical or scientific language related to d… | phrases about media, especially film and tele… |
| 89 → 16 | 1.00 | 0.01 | technical or scientific language related to d… | first-person singular pronouns and auxiliary … |
| 89 → 98 | 1.00 | 0.01 | technical or scientific language related to d… | terms from scientific papers about cellular a… |
| 89 → 73 | 1.00 | 0.01 | technical or scientific language related to d… | descriptions of events and observations, ofte… |
| 89 → 40 | 1.00 | 0.01 | technical or scientific language related to d… | code snippets and sequences |
| 89 → 95 | 1.00 | 0.01 | technical or scientific language related to d… | text referring to people's biographies, inclu… |
| 89 → 78 | 1.00 | 0.01 | technical or scientific language related to d… | phrases in scientific writing about cell and … |

_Duplicate pairs (rename/split candidates):_ 9≈89 (mentions of political, …)

## Block B1  (384 features)
- **375** directed edges, **0** duplicate pairs, 163 survive PMI>0; PolyFrac 1%, Gini 0.997.
- **S_res: 0/163** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F304 _words or phrases related to technical manuals…_: 369 children, fires 81%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 304 → 295 | 0.91 | 0.12 | words or phrases related to technical manuals… | clauses and question-like structures often in… |
| 304 → 135 | 0.90 | 0.11 | words or phrases related to technical manuals… | words and phrases related to scientific and t… |
| 304 → 229 | 0.90 | 0.11 | words or phrases related to technical manuals… | discussions of abstract existential problems … |
| 304 → 373 | 0.90 | 0.10 | words or phrases related to technical manuals… | arguments containing a series of related poin… |
| 304 → 363 | 0.90 | 0.10 | words or phrases related to technical manuals… | the word "be" in various forms |
| 304 → 507 | 0.90 | 0.10 | words or phrases related to technical manuals… | a mix of different things, including UI eleme… |
| 304 → 351 | 0.89 | 0.10 | words or phrases related to technical manuals… | words or phrases indicating proof or quality … |
| 304 → 453 | 0.89 | 0.10 | words or phrases related to technical manuals… | words that relate to societal values and recr… |

## Block B2  (1536 features)
- **385** directed edges, **5** duplicate pairs, 385 survive PMI>0; PolyFrac 32%, Gini 0.937.
- **S_res: 8/385** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 555 → 796 | 0.90 | 4.81 | phrases that include the word "state" or refe… | LaTeX commands |
| 1508 → 796 | 0.88 | 5.06 | mathematical notation and symbols | LaTeX commands |
| 1526 → 796 | 0.88 | 4.11 | words and phrases related to space missions a… | LaTeX commands |
| 2022 → 796 | 0.88 | 3.65 | a hodgepodge of mostly technical or scientifi… | LaTeX commands |
| 517 → 796 | 0.88 | 4.33 | LaTeX and mathematical notation | LaTeX commands |
| 729 → 796 | 0.88 | 5.71 | lines of code or data with identifying charac… | LaTeX commands |
| 522 → 796 | 0.88 | 5.06 | code blocks and possibly code syntax | LaTeX commands |
| 1484 → 796 | 0.88 | 3.80 | descriptions of genetic syndromes and mutatio… | LaTeX commands |

_Duplicate pairs (rename/split candidates):_ 592≈964 (references to figures); 592≈1746 (references to figures); 796≈900 (LaTeX commands); 900≈1524 (JavaDoc-style comments …); 964≈1746 (legal citations)

## Block B3  (6144 features)
- **6091** directed edges, **833** duplicate pairs, 6091 survive PMI>0; PolyFrac 9%, Gini 0.981.
- **S_res: 68/6064** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 6165 → 3712 | 0.98 | 5.69 | hyphenated sequences of letters and numbers | code documentation tags like "summary" |
| 5300 → 5454 | 0.97 | 5.16 | words related to antimicrobials and antibioti… | substrings that match the format "[@B##-#ijer… |
| 5884 → 8087 | 0.97 | 3.60 | code snippets and tags like html, css, javasc… | HTML document class declarations |
| 3563 → 5454 | 0.97 | 5.41 | intensifiers such as "very", "more" and "most… | substrings that match the format "[@B##-#ijer… |
| 7353 → 5454 | 0.97 | 4.34 | words related to literature, specifically boo… | substrings that match the format "[@B##-#ijer… |
| 4655 → 5490 | 0.97 | 4.45 | comparisons between groups. | LaTeX commands |
| 5805 → 5454 | 0.97 | 4.61 | content discussing making work or life more f… | substrings that match the format "[@B##-#ijer… |
| 3637 → 5454 | 0.97 | 3.89 | source code syntax | substrings that match the format "[@B##-#ijer… |

_Duplicate pairs (rename/split candidates):_ 2103≈2288 (the phrase "Challenge M…); 2103≈2291 (the phrase "Challenge M…); 2103≈2658 (the phrase "Challenge M…); 2103≈3067 (the phrase "Challenge M…); 2103≈3421 (the phrase "Challenge M…); 2103≈3712 (the phrase "Challenge M…)

## Block B4  (24576 features)
- **6986** directed edges, **8866** duplicate pairs, 6986 survive PMI>0; PolyFrac 62%, Gini 0.998.
- **S_res: 43/6961** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 11342 → 14048 | 1.00 | 5.15 | capitalized letters followed by lowercase let… | math problems asking about the relative sizes… |
| 27764 → 20907 | 1.00 | 5.56 | text about how the need to pay bills causes p… | references to legal cases |
| 29875 → 31591 | 1.00 | 4.97 | words in comparative contexts, especially of … | database configuration settings in XML |
| 16651 → 14048 | 1.00 | 5.70 | words surrounding expressions of physical or … | math problems asking about the relative sizes… |
| 18707 → 9651 | 1.00 | 5.73 | text depicting homoerotic or sexual acts | horizontal lines made of hyphens and other ch… |
| 20286 → 31591 | 1.00 | 5.86 | words associated with concert halls and opera | database configuration settings in XML |
| 17760 → 31591 | 1.00 | 5.31 | a variety of separators between content, such… | database configuration settings in XML |
| 23772 → 20907 | 1.00 | 5.33 | this neuron finds documents discussing childh… | references to legal cases |

_Duplicate pairs (rename/split candidates):_ 8560≈8771 (Java code snippets, esp…); 8560≈8899 (Java code snippets, esp…); 8560≈8900 (Java code snippets, esp…); 8560≈8933 (Java code snippets, esp…); 8560≈8971 (Java code snippets, esp…); 8560≈9047 (Java code snippets, esp…)
