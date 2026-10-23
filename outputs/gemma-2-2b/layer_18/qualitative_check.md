# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 18**　·　gemma-2-2b / 18-res-matryoshka-dc　·　blocks.18.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **107 -> 223**  `R=0.78 F=0.13 recon_gain=0.369 recon=Y surv=0.95 p_fires=10%`
    - parent [107](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/107): _text from patents or technical documents._
    - child  [223](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/223): _the beginning sections of technical documentation for inventions_
- **18 -> 281**  `R=0.69 F=0.43 recon_gain=0.309 recon=Y surv=0.67 p_fires=5%`
    - parent [18](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/18): _sentences that end with citations or numerical values in parentheses or brackets_
    - child  [281](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/281): _source code syntax_
- **82 -> 499**  `R=0.85 F=0.14 recon_gain=0.276 recon=Y surv=0.65 p_fires=13%`
    - parent [82](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/82): _small, common words such as prepositions and conjunctions_
    - child  [499](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/499): _the preposition "in"._
- **27 -> 420**  `R=0.94 F=0.04 recon_gain=0.248 recon=Y surv=1.00 p_fires=34%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/27): _snippets of code, mathematical formulas, and citation-like text_
    - child  [420](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/420): _blank lines_
- **53 -> 309**  `R=0.64 F=0.25 recon_gain=0.220 recon=Y surv=0.73 p_fires=5%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/53): _the verb "is" along with other auxiliary verbs ("are", "was", "be", "been")_
    - child  [309](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/309): _uses of the verb "to be" in different tenses._
- **27 -> 475**  `R=0.64 F=0.01 recon_gain=0.210 recon=Y surv=0.90 p_fires=34%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/27): _snippets of code, mathematical formulas, and citation-like text_
    - child  [475](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/475): _math formulas including derivatives and functions with variables_
- **83 -> 317**  `R=0.60 F=0.13 recon_gain=0.196 recon=Y surv=1.04 p_fires=4%`
    - parent [83](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/83): _months and years_
    - child  [317](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/317): _identifies months of the year, especially when followed by a number_
- **56 -> 458**  `R=0.88 F=0.11 recon_gain=0.191 recon=Y surv=1.07 p_fires=32%`
    - parent [56](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/56): _words and phrases related to documentation, data, and specifications in technical fields like programming, engineering, or science._
    - child  [458](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/458): _words related to academic, sporting, governmental, technological, and musical professions and endeavors._

### reject:superparent  (4)

- **89 -> 263**  `R=1.00 F=0.02 recon_gain=1.949 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [263](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/263): _mentions of different kinds of government and municipal departments_
- **89 -> 132**  `R=1.00 F=0.02 recon_gain=1.302 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [132](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/132): _hyphens used to connect words_
- **89 -> 249**  `R=1.00 F=0.05 recon_gain=2.136 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [249](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/249): _references to political parties and elections, especially regarding the Democratic and Green parties._
- **89 -> 261**  `R=1.00 F=0.07 recon_gain=2.682 recon=Y surv=1.00 p_fires=99%`
    - parent [89](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/89): _technical or scientific language related to data processing, analysis, or modeling_
    - child  [261](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/261): _passages with emotional and personal reflections about poignant life events_

### reject:freq-driven  (4)

- **61 -> 397**  `R=0.72 F=0.23 recon_gain=0.086 recon=Y surv=0.38 p_fires=4%`
    - parent [61](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/61): _commas and various punctuation marks._
    - child  [397](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/397): _commas within longer sentences and paragraphs._
- **70 -> 183**  `R=0.69 F=0.33 recon_gain=0.173 recon=Y surv=0.44 p_fires=5%`
    - parent [70](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/70): _indented code snippets and other code-related artifacts_
    - child  [183](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/183): _code or configuration snippets_
- **82 -> 246**  `R=0.68 F=0.17 recon_gain=0.148 recon=Y surv=0.28 p_fires=13%`
    - parent [82](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/82): _small, common words such as prepositions and conjunctions_
    - child  [246](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/246): _the word "of" and the phrase "of the."_
- **21 -> 456**  `R=0.65 F=0.26 recon_gain=0.049 recon=Y surv=0.28 p_fires=5%`
    - parent [21](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/21): _numbers and related symbols in tables_
    - child  [456](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/456): _the number 4_

### reject:no-recon  (4)

- **56 -> 152**  `R=0.75 F=0.13 recon_gain=0.095 recon=n surv=1.14 p_fires=32%`
    - parent [56](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/56): _words and phrases related to documentation, data, and specifications in technical fields like programming, engineering, or science._
    - child  [152](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/152): _medical or scientific terms, experiments, and study results, often also flagging numbers and percentages._
- **95 -> 411**  `R=0.71 F=0.08 recon_gain=0.052 recon=n surv=0.91 p_fires=28%`
    - parent [95](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/95): _text referring to people's biographies, including names, locations, organizations and timeline data_
    - child  [411](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/411): _clauses in historical narrative sentences_
- **60 -> 142**  `R=0.69 F=0.07 recon_gain=0.033 recon=n surv=1.03 p_fires=39%`
    - parent [60](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/60): _technical descriptions of equipment used in scientific experiments._
    - child  [142](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/142): _words and phrases related to liquids and gasses flowing through pipes/tanks/reservoirs._
- **60 -> 335**  `R=0.68 F=0.02 recon_gain=0.017 recon=n surv=1.00 p_fires=39%`
    - parent [60](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/60): _technical descriptions of equipment used in scientific experiments._
    - child  [335](https://www.neuronpedia.org/gemma-2-2b/18-res-matryoshka-dc/335): _code snippets from various different programming languages_
