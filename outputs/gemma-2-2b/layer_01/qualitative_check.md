# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)

**Layer 1**　·　gemma-2-2b / 1-res-matryoshka-dc　·　blocks.1.hook_resid_post　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

For each block pair we compare edges the metrics KEEP (survivors) against edges they REJECT despite passing the crude coverage test. Read the parent/child labels: survivors should be semantically related; rejected edges should look like frequency / co-occurrence artifacts. Labels from Neuronpedia.

## Block pair 0->1

### survivor  (8)

- **94 -> 277**  `R=0.66 F=0.29 recon_gain=1.994 recon=Y surv=0.58 p_fires=2%`
    - parent [94](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/94): _references to citation symbols, such as parentheses, brackets, and similar punctuation_
    - child  [277](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/277): _syntax elements related to code blocks and structure_
- **3 -> 151**  `R=0.78 F=0.27 recon_gain=1.828 recon=Y surv=0.50 p_fires=4%`
    - parent [3](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/3): _patterns related to the structure and content of numerical data and formatted information_
    - child  [151](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/151): _references to dates, times, and numerical data_
- **111 -> 344**  `R=0.66 F=0.10 recon_gain=1.823 recon=Y surv=0.82 p_fires=3%`
    - parent [111](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/111): _segments of text that are enclosed by quotation marks_
    - child  [344](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/344): _distinctions between different types of biological or synthetic particles_
- **83 -> 151**  `R=0.73 F=0.16 recon_gain=1.823 recon=Y surv=0.83 p_fires=6%`
    - parent [83](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/83): _snippets of text related to technical or code-related information, particularly error messages, command syntax, or configuration details_
    - child  [151](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/151): _references to dates, times, and numerical data_
- **96 -> 252**  `R=0.75 F=0.06 recon_gain=1.801 recon=Y surv=1.07 p_fires=3%`
    - parent [96](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/96): _sequences of formatting and style code in document markup_
    - child  [252](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/252): _references to hierarchical or structured data patterns, particularly related to nested or separated elements_
- **125 -> 252**  `R=0.89 F=0.08 recon_gain=1.315 recon=Y surv=1.04 p_fires=3%`
    - parent [125](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/125): _references to scientific or medical imaging findings and measurements within documents_
    - child  [252](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/252): _references to hierarchical or structured data patterns, particularly related to nested or separated elements_
- **96 -> 371**  `R=0.67 F=0.14 recon_gain=1.194 recon=Y surv=1.19 p_fires=3%`
    - parent [96](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/96): _sequences of formatting and style code in document markup_
    - child  [371](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/371): _detected regions or features in complex data representations_
- **9 -> 403**  `R=0.58 F=0.11 recon_gain=1.014 recon=Y surv=1.26 p_fires=3%`
    - parent [9](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/9): _patterns of code structure and specific syntax elements used in programming files_
    - child  [403](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/403): _patterns indicative of structured, formatted data, such as tables, listings, or code blocks_

### reject:superparent  (4)

- **53 -> 437**  `R=1.00 F=0.01 recon_gain=0.601 recon=Y surv=1.00 p_fires=100%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/53): _linguistic expressions related to organizing, planning, and evaluating group activities or discussions_
    - child  [437](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/437): _code components related to declaring access modifiers and property definitions in object-oriented programming_
- **86 -> 219**  `R=1.00 F=0.02 recon_gain=1.021 recon=Y surv=1.00 p_fires=99%`
    - parent [86](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/86): _descriptions of technological or mechanical modes or settings_
    - child  [219](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/219): _references to self-organization and emergent states in physical systems_
- **53 -> 169**  `R=1.00 F=0.03 recon_gain=2.041 recon=Y surv=1.00 p_fires=100%`
    - parent [53](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/53): _linguistic expressions related to organizing, planning, and evaluating group activities or discussions_
    - child  [169](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/169): _references to the concept of time or time-related information within a text_
- **86 -> 242**  `R=1.00 F=0.01 recon_gain=0.404 recon=Y surv=1.00 p_fires=99%`
    - parent [86](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/86): _descriptions of technological or mechanical modes or settings_
    - child  [242](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/242): _references to the current time or moments indicating immediacy_

### reject:freq-driven  (4)

- **3 -> 230**  `R=0.95 F=0.49 recon_gain=2.034 recon=Y surv=0.42 p_fires=4%`
    - parent [3](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/3): _patterns related to the structure and content of numerical data and formatted information_
    - child  [230](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/230): _numerical data related to measurements, statistics, or sensor outputs_
- **64 -> 374**  `R=0.77 F=0.07 recon_gain=0.901 recon=Y surv=0.42 p_fires=9%`
    - parent [64](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/64): _references to statistical measures and numerical data_
    - child  [374](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/374): _mentions of specific people, especially individuals' names or personal identifiers_
- **17 -> 219**  `R=0.77 F=0.43 recon_gain=1.080 recon=Y surv=0.21 p_fires=4%`
    - parent [17](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/17): _sequences of special characters and formatted code snippets_
    - child  [219](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/219): _references to self-organization and emergent states in physical systems_
- **33 -> 374**  `R=0.76 F=0.18 recon_gain=1.498 recon=Y surv=0.39 p_fires=3%`
    - parent [33](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/33): _references to licensing information, legal notices, or copyright statements_
    - child  [374](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/374): _mentions of specific people, especially individuals' names or personal identifiers_

### reject:no-recon  (4)

- **6 -> 138**  `R=0.67 F=0.05 recon_gain=0.027 recon=n surv=1.01 p_fires=27%`
    - parent [6](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/6): _references to fuel injection components and their functions within internal combustion engines_
    - child  [138](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/138): _CSS style properties and formatting syntax_
- **99 -> 258**  `R=0.63 F=0.27 recon_gain=0.070 recon=n surv=1.03 p_fires=33%`
    - parent [99](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/99): _code snippets related to defining classes and methods_
    - child  [258](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/258): _references to detailed descriptions of deities' bodily divisions and their associated spiritual concepts_
- **6 -> 166**  `R=0.60 F=0.01 recon_gain=0.024 recon=n surv=1.18 p_fires=27%`
    - parent [6](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/6): _references to fuel injection components and their functions within internal combustion engines_
    - child  [166](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/166): _references to the mathematical concept of variance within Gaussian processes_
- **6 -> 153**  `R=0.59 F=0.02 recon_gain=0.018 recon=n surv=1.08 p_fires=27%`
    - parent [6](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/6): _references to fuel injection components and their functions within internal combustion engines_
    - child  [153](https://www.neuronpedia.org/gemma-2-2b/1-res-matryoshka-dc/153): _symbols and notation used in mathematical expressions_
