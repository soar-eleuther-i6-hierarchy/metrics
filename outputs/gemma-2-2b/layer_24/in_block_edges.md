# In-block (same-level) directed edges

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

Parent→child *within* a block (asymmetric containment); co-extensive pairs are reported as duplicates (renames/splits), never edges.

## Block B0  (128 features)
- **640** directed edges, **13** duplicate pairs, 411 survive PMI>0; PolyFrac 100%, Gini 0.944.
- **S_res: 4/411** edges are genuine refinements (0 children untestable).
- In-block superparents: 6 (e.g. F32 _numbers within a document_: 123 children, fires 99%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 32 → 84 | 1.00 | 0.01 | numbers within a document | periods at the end of sentences, especially i… |
| 32 → 72 | 1.00 | 0.01 | numbers within a document | snippets of mathematical proofs |
| 32 → 58 | 1.00 | 0.01 | numbers within a document | LaTeX mathematical notation |
| 32 → 83 | 1.00 | 0.00 | numbers within a document | detailed storytelling of dramatic events invo… |
| 32 → 104 | 1.00 | 0.00 | numbers within a document | scientific jargon related to biological proce… |
| 32 → 126 | 1.00 | 0.00 | numbers within a document | dates with months and years |
| 32 → 88 | 1.00 | 0.00 | numbers within a document | words and symbols related to measurements in … |
| 32 → 51 | 1.00 | 0.00 | numbers within a document | words related to online sales and marketing |

_Duplicate pairs (rename/split candidates):_ 19≈32 (Japanese names or media…); 19≈49 (Japanese names or media…); 19≈85 (Japanese names or media…); 19≈92 (Japanese names or media…); 32≈49 (numbers within a docume…); 32≈85 (numbers within a docume…)

## Block B1  (384 features)
- **26** directed edges, **0** duplicate pairs, 26 survive PMI>0; PolyFrac 14%, Gini 0.980.
- **S_res: 2/26** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 471 → 484 | 0.91 | 1.96 | code snippets and related terms like "diff," … | words pertaining to social structures and org… |
| 226 → 186 | 0.64 | 1.63 | equations | characters and symbols associated with mathem… |
| 336 → 498 | 0.64 | 3.20 | numbers, especially as part of references, co… | single quotes and apostrophes |
| 207 → 310 | 0.63 | 1.90 | words related to biological research and medi… | mentions of antibodies and animal types used … |
| 207 → 426 | 0.61 | 1.87 | words related to biological research and medi… | text from scientific research papers about te… |
| 500 → 186 | 0.59 | 2.56 | words related to knowing, wondering, and tell… | characters and symbols associated with mathem… |
| 207 → 169 | 0.58 | 1.80 | words related to biological research and medi… | terms related to biological experiments and s… |
| 207 → 396 | 0.57 | 1.80 | words related to biological research and medi… | names of chemical structures and analysis |

## Block B2  (1536 features)
- **1155** directed edges, **6** duplicate pairs, 859 survive PMI>0; PolyFrac 7%, Gini 0.990.
- **S_res: 24/856** edges are genuine refinements (0 children untestable).
- In-block superparents: 1 (e.g. F2042 _words related to court cases, delays in proje…_: 934 children, fires 59%).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 1234 → 887 | 0.91 | 5.04 | Lithuanian words within a historical context,… | words in a Scandinavian language, possibly Da… |
| 1830 → 1323 | 0.86 | 4.98 | non-English words such as those used to descr… | code or programming discussion in Spanish |
| 730 → 1234 | 0.85 | 3.83 | names and occupations within a biographical c… | Lithuanian words within a historical context,… |
| 2042 → 1096 | 0.85 | 0.36 | words related to court cases, delays in proje… | complex clauses involving multiple people, re… |
| 2042 → 1313 | 0.84 | 0.35 | words related to court cases, delays in proje… | words and numbers somewhat randomly |
| 1475 → 545 | 0.84 | 3.83 | medical research papers and articles | authors' last names, publication years and ot… |
| 730 → 887 | 0.83 | 3.80 | names and occupations within a biographical c… | words in a Scandinavian language, possibly Da… |
| 2042 → 1904 | 0.82 | 0.33 | words related to court cases, delays in proje… | technical descriptions of power inverters |

_Duplicate pairs (rename/split candidates):_ 582≈1831 (punctuation and discour…); 1032≈1234 (French text where peopl…); 1114≈1170 (math problems involving…); 1117≈1660 (russian words, especial…); 1210≈1810 (code snippets and Japan…); 1823≈1890 (number comparisons and …)

## Block B3  (6144 features)
- **969** directed edges, **54** duplicate pairs, 969 survive PMI>0; PolyFrac 28%, Gini 0.980.
- **S_res: 113/946** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 3416 → 7729 | 0.96 | 5.83 | a seemingly random collection of abbreviation… | characters with diacritics or less common let… |
| 4210 → 3416 | 0.94 | 4.63 | mentions of funding and research institutions | a seemingly random collection of abbreviation… |
| 2533 → 3200 | 0.94 | 4.58 | chemical formulas, numbers, units, and codes | sentences that describe the authors of a study |
| 2056 → 3200 | 0.94 | 5.02 | research papers describing studies, especiall… | sentences that describe the authors of a study |
| 4210 → 7729 | 0.94 | 4.62 | mentions of funding and research institutions | characters with diacritics or less common let… |
| 5528 → 5354 | 0.93 | 4.58 | segments of a song lyric that expresses relat… | instances of confusion, failure and a general… |
| 5971 → 6106 | 0.93 | 4.31 | code snippets, particularly those defining re… | Java code related to executing queries using … |
| 6882 → 3200 | 0.92 | 4.99 | hexadecimal color codes and date formats | sentences that describe the authors of a study |

_Duplicate pairs (rename/split candidates):_ 2481≈2758 (text related to car spe…); 2481≈6767 (text related to car spe…); 2572≈5553 (words related to softwa…); 2597≈3113 (the description of base…); 2597≈3308 (the description of base…); 2597≈3411 (the description of base…)

## Block B4  (24576 features)
- **1089** directed edges, **720** duplicate pairs, 1089 survive PMI>0; PolyFrac 34%, Gini 0.998.
- **S_res: 30/1026** edges are genuine refinements (0 children untestable).

| parent → child | R | PMI | parent label | child label |
|---|---|---|---|---|
| 24968 → 17391 | 1.00 | 5.81 | words with the character sequences "wir", "wi… | numbers inside parenthesis |
| 26474 → 17391 | 1.00 | 5.92 | legal jargon related to court and legal proce… | numbers inside parenthesis |
| 11705 → 17414 | 1.00 | 2.54 | words and abbreviations used in scientific or… | words and phrases related to the Mormon and Q… |
| 18618 → 17391 | 1.00 | 5.51 | academic writing, especially regarding bioche… | numbers inside parenthesis |
| 18618 → 17414 | 1.00 | 5.51 | academic writing, especially regarding bioche… | words and phrases related to the Mormon and Q… |
| 29037 → 17391 | 1.00 | 5.94 | capital letters and abbreviations, often rela… | numbers inside parenthesis |
| 29037 → 17414 | 1.00 | 5.94 | capital letters and abbreviations, often rela… | words and phrases related to the Mormon and Q… |
| 19278 → 17391 | 1.00 | 5.95 | the keyword "in" at the beginning of a senten… | numbers inside parenthesis |

_Duplicate pairs (rename/split candidates):_ 8275≈8485 (references to figures a…); 8275≈13049 (references to figures a…); 8275≈14226 (references to figures a…); 8275≈14854 (references to figures a…); 8275≈15402 (references to figures a…); 8275≈15510 (references to figures a…)
