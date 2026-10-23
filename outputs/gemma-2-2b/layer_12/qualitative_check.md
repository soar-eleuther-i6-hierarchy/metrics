# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 12**　·　gemma-2-2b / 12-res-matryoshka-dc　·　blocks.12.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **27 -> 410**  `R=0.53 F=0.05 recon_gain=0.288 recon=Y surv=0.61 p_fires=6%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/27): _mathematical expressions or equations related to inequalities_
    - child  [410](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/410): _references to financial transactions or economic concepts_
- **42 -> 318**  `R=0.70 F=0.37 recon_gain=0.271 recon=Y surv=0.94 p_fires=5%`
    - parent [42](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/42): _items related to programming and database operations_
    - child  [318](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/318): _code-related constructs and syntax_
- **54 -> 466**  `R=0.77 F=0.04 recon_gain=0.267 recon=Y surv=0.95 p_fires=8%`
    - parent [54](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/54): _formal legal terminology and procedural language_
    - child  [466](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/466): _phrases that contain the word "In" or its variations, indicating a focus on contextual information_
- **69 -> 174**  `R=0.62 F=0.18 recon_gain=0.205 recon=Y surv=0.53 p_fires=11%`
    - parent [69](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/69): _phrases that indicate relationships, particularly in the context of implementation and effect_
    - child  [174](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/174): _the presence of the word "in" across different contexts_
- **69 -> 466**  `R=0.75 F=0.03 recon_gain=0.177 recon=Y surv=0.76 p_fires=11%`
    - parent [69](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/69): _phrases that indicate relationships, particularly in the context of implementation and effect_
    - child  [466](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/466): _phrases that contain the word "In" or its variations, indicating a focus on contextual information_
- **54 -> 232**  `R=0.70 F=0.10 recon_gain=0.168 recon=Y surv=0.67 p_fires=8%`
    - parent [54](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/54): _formal legal terminology and procedural language_
    - child  [232](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/232): _definitive articles and the presence of the word "the" in a document_
- **10 -> 370**  `R=0.74 F=0.17 recon_gain=0.164 recon=Y surv=1.16 p_fires=7%`
    - parent [10](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/10): _names of individuals in various contexts_
    - child  [370](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/370): _proper names and specific identifiers within text_
- **22 -> 426**  `R=0.55 F=0.27 recon_gain=0.143 recon=Y surv=0.61 p_fires=4%`
    - parent [22](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/22): _mentions of specific metrics or numerical data related to studies or evaluations_
    - child  [426](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/426): _quantitative data regarding measurements and statistics_

### reject:superparent  (4)

- **44 -> 318**  `R=1.00 F=0.03 recon_gain=1.665 recon=Y surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [318](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/318): _code-related constructs and syntax_
- **44 -> 203**  `R=1.00 F=0.00 recon_gain=1.007 recon=n surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [203](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/203): _mathematical symbols and expressions_
- **44 -> 283**  `R=1.00 F=0.00 recon_gain=1.693 recon=Y surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [283](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/283): _terms related to cell biology, specifically focusing on cell types and cellular activities_
- **44 -> 272**  `R=1.00 F=0.01 recon_gain=1.445 recon=n surv=1.00 p_fires=99%`
    - parent [44](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/44): _technical terminology related to programming and coding concepts_
    - child  [272](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/272): _expressions of encouragement and support in conversations_

### reject:freq-driven  (4)

- **75 -> 354**  `R=0.86 F=0.23 recon_gain=0.217 recon=Y surv=0.44 p_fires=5%`
    - parent [75](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/75): _phrases indicating technical specifications or performance outcomes_
    - child  [354](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/354): _significant emotional experiences or pivotal moments in life narratives_
- **27 -> 354**  `R=0.85 F=0.21 recon_gain=0.275 recon=Y surv=0.32 p_fires=6%`
    - parent [27](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/27): _mathematical expressions or equations related to inequalities_
    - child  [354](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/354): _significant emotional experiences or pivotal moments in life narratives_
- **50 -> 285**  `R=0.72 F=0.33 recon_gain=0.403 recon=Y surv=0.49 p_fires=4%`
    - parent [50](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/50): _punctuation marks used in academic writing_
    - child  [285](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/285): _instances of strong emotional or impactful statements_
- **69 -> 291**  `R=0.71 F=0.22 recon_gain=0.249 recon=Y surv=0.39 p_fires=11%`
    - parent [69](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/69): _phrases that indicate relationships, particularly in the context of implementation and effect_
    - child  [291](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/291): _elements and relationships in technical and engineering contexts_

### reject:no-recon  (4)

- **40 -> 272**  `R=0.89 F=0.03 recon_gain=0.119 recon=n surv=0.95 p_fires=24%`
    - parent [40](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/40): _expressions of personal experiences and emotions related to relationships and social interactions_
    - child  [272](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/272): _expressions of encouragement and support in conversations_
- **100 -> 417**  `R=0.81 F=0.03 recon_gain=0.041 recon=n surv=0.98 p_fires=13%`
    - parent [100](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/100): _statistical data and numerical values_
    - child  [417](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/417): _mathematical formulas, terms, and structural elements related to functional analysis and perturbation theory_
- **94 -> 495**  `R=0.80 F=0.13 recon_gain=0.048 recon=n surv=0.96 p_fires=40%`
    - parent [94](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/94): _discussions about financial or budgetary issues_
    - child  [495](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/495): _terms related to historical events and their significance_
- **94 -> 131**  `R=0.79 F=0.10 recon_gain=0.043 recon=n surv=0.96 p_fires=40%`
    - parent [94](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/94): _discussions about financial or budgetary issues_
    - child  [131](https://www.neuronpedia.org/gemma-2-2b/12-res-matryoshka-dc/131): _terms related to military actions and strategies_
