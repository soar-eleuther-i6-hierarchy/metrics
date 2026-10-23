# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia (not fetched - URLs only).

## Block pair 0->1

### survivor  (8)

- **73 -> 446**  `R=0.94 F=0.13 recon_gain=0.640 recon=Y surv=1.02 p_fires=13%`
    - parent [73](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/73): _technical academic language_
    - child  [446](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/446): _words related to events, changes, difficulties, or limitations_
- **103 -> 192**  `R=0.73 F=0.08 recon_gain=0.428 recon=Y surv=1.05 p_fires=4%`
    - parent [103](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/103): _Danish words or phrases, especially those that might be movie titles, as well as parenthetical year dates_
    - child  [192](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/192): _code snippets and questions about programming, some of which are in Russian_
- **21 -> 449**  `R=0.71 F=0.02 recon_gain=0.425 recon=Y surv=0.97 p_fires=27%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/21): _programming source code_
    - child  [449](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/449): _various syntax elements commonly found in different coding languages like C++, PHP, etc_
- **103 -> 493**  `R=0.91 F=0.15 recon_gain=0.393 recon=Y surv=1.04 p_fires=4%`
    - parent [103](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/103): _Danish words or phrases, especially those that might be movie titles, as well as parenthetical year dates_
    - child  [493](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/493): _mixed English and Russian text related to website coding_
- **21 -> 465**  `R=0.78 F=0.00 recon_gain=0.391 recon=Y surv=1.09 p_fires=27%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/21): _programming source code_
    - child  [465](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/465): _references to figures and tables_
- **124 -> 231**  `R=0.65 F=0.08 recon_gain=0.326 recon=Y surv=0.90 p_fires=9%`
    - parent [124](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/124): _code with names separated by underscores like "NG X"_
    - child  [231](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/231): _capitalized words or acronyms separated by underscores_
- **74 -> 491**  `R=0.81 F=0.16 recon_gain=0.315 recon=Y surv=0.97 p_fires=9%`
    - parent [74](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/74): _technical writing in the field of electrical engineering_
    - child  [491](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/491): _text related to patents, inventions, and technical writing_
- **33 -> 320**  `R=0.84 F=0.17 recon_gain=0.314 recon=Y surv=0.98 p_fires=7%`
    - parent [33](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/33): _words or phrases used in a specific, technical or legal context._
    - child  [320](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/320): _noun phrases followed by a number._

### reject:superparent  (4)

- **32 -> 365**  `R=1.00 F=0.00 recon_gain=0.573 recon=Y surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [365](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/365): _citations in research papers_
- **32 -> 193**  `R=1.00 F=0.00 recon_gain=1.102 recon=n surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [193](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/193): _language common to legal opinions._
- **32 -> 421**  `R=1.00 F=0.02 recon_gain=1.295 recon=Y surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [421](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/421): _text related to the game World of Warcraft._
- **32 -> 387**  `R=1.00 F=0.01 recon_gain=1.145 recon=Y surv=1.00 p_fires=99%`
    - parent [32](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/32): _numbers within a document_
    - child  [387](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/387): _contractions with the word "not."_

### reject:freq-driven  (4)

- **56 -> 460**  `R=0.82 F=0.23 recon_gain=0.084 recon=Y surv=0.15 p_fires=8%`
    - parent [56](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/56): _numerical data like version numbers or measurements_
    - child  [460](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/460): _numbers and citations_
- **0 -> 156**  `R=0.79 F=0.15 recon_gain=0.429 recon=Y surv=0.13 p_fires=2%`
    - parent [0](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/0): _numbers, especially those related to dates and statistics_
    - child  [156](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/156): _number sequences_
- **12 -> 308**  `R=0.78 F=0.08 recon_gain=0.041 recon=Y surv=0.20 p_fires=13%`
    - parent [12](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/12): _LaTeX mathematical expressions_
    - child  [308](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/308): _hyphens_
- **50 -> 186**  `R=0.75 F=0.07 recon_gain=0.122 recon=n surv=0.39 p_fires=5%`
    - parent [50](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/50): _math calculation problems involving multiplication, remainders, quotients, factors, and divisors_
    - child  [186](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/186): _characters and symbols associated with mathematical expressions and code._

### reject:no-recon  (4)

- **21 -> 173**  `R=0.85 F=0.18 recon_gain=0.142 recon=n surv=0.99 p_fires=27%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/21): _programming source code_
    - child  [173](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/173): _references to legal and scientific publications, including journals, acts, and institutions_
- **6 -> 435**  `R=0.82 F=0.00 recon_gain=0.050 recon=n surv=0.98 p_fires=43%`
    - parent [6](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/6): _words that are part of names of organisms, locations or people and abbreviations commonly used when referring to them._
    - child  [435](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/435): _code snippets and related terms including Russian_
- **21 -> 288**  `R=0.79 F=0.05 recon_gain=0.265 recon=n surv=1.01 p_fires=27%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/21): _programming source code_
    - child  [288](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/288): _numbers and date-like sequences, especially those found in log files or code_
- **6 -> 211**  `R=0.78 F=0.04 recon_gain=0.055 recon=n surv=1.00 p_fires=43%`
    - parent [6](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/6): _words that are part of names of organisms, locations or people and abbreviations commonly used when referring to them._
    - child  [211](https://www.neuronpedia.org/gemma-2-2b/24-res-matryoshka-dc/211): _words related to storytelling and imagination in quotes_
