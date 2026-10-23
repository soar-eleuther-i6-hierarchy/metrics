# Exp 0 - metrics report

**Layer 24**　·　gemma-2-2b / 24-res-matryoshka-dc　·　blocks.24.hook_resid_post　·　48,571 tokens over 400 docs　·　edge: reverse coverage ≥ 0.5, both endpoints fire ≥ 20

## Block pair 0->1  -  2273 candidate edges

- **Out-degree**: 64 parents, 384 children, 384 multi-parented (PolyFrac 100.0%); top-1 parent holds 16.9% of edges, Gini 0.915, max out-degree 384.
- **Superparents** (out-degree flag): 6 (6 also pass the old fire-rate AND-gate) — e.g. feature 32 _numbers within a document_: 384 children, fires on 99.5% of tokens
- **Independence null**: mean edge PMI 0.30; 1810 edges (79.6%) at chance level (PMI < 0.5). 8 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 1673/2273 edges pass (73.6%).
- **Frequency control**: mean survival 0.994 over 2273 testable edges; 49 (2.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.060 over 47 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 0 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.544.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 32 -> 387 | 1.00 | 0.01 | 0.01 | 1.15/0.06 | Y | 1.00 | 0.01 | numbers within a document | contractions with the word "not." |
| 32 -> 193 | 1.00 | 0.00 | 0.01 | 1.10/0.00 | n | 1.00 | 0.01 | numbers within a document | language common to legal opinions. |
| 32 -> 201 | 1.00 | 0.03 | 0.01 | 1.15/0.02 | Y | 1.00 | 0.01 | numbers within a document | names of people and titles or abbreviat… |
| 32 -> 365 | 1.00 | 0.00 | 0.01 | 0.57/0.04 | Y | 1.00 | 0.01 | numbers within a document | citations in research papers |
| 32 -> 336 | 1.00 | 0.03 | 0.01 | 1.24/0.16 | Y | 1.00 | 0.01 | numbers within a document | numbers, especially as part of referenc… |
| 32 -> 421 | 1.00 | 0.02 | 0.01 | 1.29/0.03 | Y | 1.00 | 0.01 | numbers within a document | text related to the game World of Warcr… |
| 32 -> 502 | 1.00 | 0.11 | 0.00 | 1.88/0.02 | Y | 1.00 | 0.01 | numbers within a document | words and phrases associated with polit… |
| 32 -> 146 | 1.00 | 0.07 | 0.00 | 2.14/0.01 | n | 1.00 | 0.01 | numbers within a document | words related to teaching and education |

## Block pair 1->2  -  424 candidate edges

- **Out-degree**: 143 parents, 271 children, 70 multi-parented (PolyFrac 25.8%); top-1 parent holds 8.3% of edges, Gini 0.808, max out-degree 35.
- **Superparents** (out-degree flag): 0 (0 also pass the old fire-rate AND-gate)
- **Independence null**: mean edge PMI 2.51; 16 edges (3.8%) at chance level (PMI < 0.5). 44 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 221/424 edges pass (52.1%).
- **Frequency control**: mean survival 0.799 over 396 testable edges; 80 (20.2%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.133 over 81 parents; 0 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 3 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.239.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 345 -> 1547 | 0.97 | 0.10 | 4.93 | 0.39/0.21 | Y | 0.97 | 0.11 | HTML closing tags | HTML closing tags |
| 180 -> 1810 | 0.95 | 0.06 | 3.15 | 0.10/0.03 | Y | 1.00 | 0.06 | text having to do with dictionaries, en… | mentions of the Chinese placename "Lius… |
| 218 -> 1323 | 0.94 | 0.17 | 4.18 | 0.79/0.05 | Y | 0.99 | 0.36 | code snippets that decrement a counter … | code or programming discussion in Spani… |
| 226 -> 928 | 0.94 | 0.10 | 2.01 | 0.05/0.07 | Y | 1.00 | 0.07 | equations | code or text broken up into single char… |
| 180 -> 1210 | 0.92 | 0.08 | 3.12 | 0.10/0.74 | Y | 1.02 | 0.06 | text having to do with dictionaries, en… | code snippets and Japanese error messag… |
| 493 -> 887 | 0.92 | 0.28 | 4.94 | 0.41/0.05 | Y | 1.03 | 0.26 | mixed English and Russian text related … | words in a Scandinavian language, possi… |
| 491 -> 2044 | 0.91 | 0.09 | 3.93 | 0.16/0.31 | Y | 1.02 | 0.17 | text related to patents, inventions, an… | section headers, particularly those rel… |
| 493 -> 1234 | 0.90 | 0.81 | 4.92 | 0.32/0.39 | Y | 1.02 | 0.26 | mixed English and Russian text related … | Lithuanian words within a historical co… |

## Block pair 2->3  -  4867 candidate edges

- **Out-degree**: 509 parents, 3190 children, 774 multi-parented (PolyFrac 24.3%); top-1 parent holds 59.3% of edges, Gini 0.937, max out-degree 2887.
- **Superparents** (out-degree flag): 1 (1 also pass the old fire-rate AND-gate) — e.g. feature 2042 _words related to court cases, delays in projects, the audit…_: 2887 children, fires on 59.4% of tokens
- **Independence null**: mean edge PMI 1.44; 2887 edges (59.3%) at chance level (PMI < 0.5). 1329 edges dropped by the joint-support guard (n_joint < 30).
- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): 699/4867 edges pass (14.4%).
- **Frequency control**: mean survival 0.886 over 4709 testable edges; 475 (10.1%) are frequency-driven (survival < 0.5).
- **Sibling redundancy** (global Jaccard — confounded proxy, not the splitting verdict; the parent-conditioned version is in the stage-03 second pass): mean 0.176 over 315 parents; 38 over the 0.5 global threshold.
- **Joint-child (exact union, parents with edges)**: R_supp mean 1.000, R_mass mean 1.000; 10 parents with one child holding >=90% of their energy (rename candidates).
- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children co-fire, kept only for contrast with the exact union): 0.349.

| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |
|---|---|---|---|---|---|---|---|---|---|
| 1210 -> 2572 | 1.00 | 0.30 | 5.64 | 0.78/0.18 | Y | 1.00 | 0.22 | code snippets and Japanese error messag… | words related to software code errors a… |
| 1210 -> 4253 | 1.00 | 0.26 | 5.64 | 1.11/0.00 | n | 1.00 | 0.22 | code snippets and Japanese error messag… | the Japanese phrase 「いうと[十返]{じゅうかえ}ってくる」 |
| 1210 -> 5553 | 1.00 | 0.30 | 5.64 | 0.75/0.19 | Y | 1.00 | 0.22 | code snippets and Japanese error messag… | code-related terms |
| 1210 -> 5770 | 1.00 | 0.27 | 5.64 | 1.38/0.06 | Y | 1.00 | 0.22 | code snippets and Japanese error messag… | code-related terms in Japanese mixed wi… |
| 1505 -> 2925 | 0.98 | 0.11 | 4.50 | 0.03/0.04 | Y | 0.99 | 0.32 | words around the topic of live theatre. | classical music references including co… |
| 1810 -> 5553 | 0.98 | 0.43 | 5.99 | 0.04/0.19 | Y | 1.00 | 0.25 | mentions of the Chinese placename "Lius… | code-related terms |
| 1582 -> 2940 | 0.98 | 0.32 | 5.14 | 0.24/0.08 | Y | 1.00 | 0.38 | words and phrases related to competitiv… | mixed martial arts fight descriptions |
| 1299 -> 5540 | 0.97 | 0.34 | 4.11 | 0.18/0.04 | Y | 0.99 | 0.18 | statistical medical research about infa… | words related to the experience of chil… |
