# Pure metrics, block pair 0->1

Every number read from the run's metrics_report.json (stage 02) and second_pass.json (stage 03). No pathology mapping. PCFG cells are mean ± sd over the formatting-sweep runs; Temporal SAE toy cells are mean ± sd over its seeds. The Matryoshka toy's probe pass comes from trained_toy_calibration.json; the Temporal SAE toy has no probe pass (the toy adapter caches no residuals), shown as n/a.

| Metric | Quantity | Matryoshka, gemma-2-2b L12 | Temporal SAE, gemma-2-2b L12 | Matryoshka, Bussmann toy | Temporal SAE, Bussmann toy (3 seeds) | Matryoshka, PCFG L2 (12 runs) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 Activation coverage | candidate edges | 1,473 | 1,621 | 9 | 9 ± 1 | 158 ± 200 |
|  | edge density (edges / P x C) | 3.0e-02 | 3.8e-05 | 3.3e-01 | 3.6e-01 | 3.1e-03 |
| 2 Independence null | share at chance level (PMI < 0.5) | 0.759 | 0.000 | 0.000 | 0.000 ± 0.000 | 0.327 ± 0.257 |
|  | mean PMI over edges | 0.366 | 4.154 | 1.898 | 1.812 ± 0.149 | 1.166 ± 0.572 |
| 3 Token-frequency control | frequency-driven share (survival < 0.5) | 0.018 | 0.082 | 0.000 | 0.000 ± 0.000 | 0.051 ± 0.085 |
|  | mean survival on rare tokens | 0.998 | 0.909 | 1.000 | 1.000 ± 0.000 | 0.939 ± 0.078 |
| 4 Reconstruction contribution | pass share | 0.539 | 0.986 | 1.000 | 1.000 ± 0.000 | 0.961 ± 0.046 |
| 5 Probe rank | pass share | 0.020 | 0.584 | 1.000 | n/a | 0.057 ± 0.044 |
| 6 Out-degree | dense parents | 2 | 0 | 3 | 3 ± 0 | 0 ± 1 |
|  | multi-parenting share | 0.997 | 0.287 | 0.000 | 0.083 ± 0.144 | 0.391 ± 0.271 |
|  | out-degree Gini | 0.906 | 0.928 | 0.000 | 0.022 ± 0.038 | 0.959 ± 0.027 |
| 7 Sibling redundancy | mean pairwise Jaccard | 0.033 | 0.091 | 0.000 | 0.054 ± 0.093 | 0.084 ± 0.085 |
| 9 Joint-child coverage | mean support coverage | 0.506 | 0.381 | 0.601 | 0.658 ± 0.099 | 0.248 ± 0.122 |

Tokens: Matryoshka, gemma-2-2b L12: 48,571 tokens, Temporal SAE, gemma-2-2b L12: 48,571 tokens, Matryoshka, Bussmann toy: 199,936 tokens, Temporal SAE, Bussmann toy (3 seeds): 199,936 tokens, Matryoshka, PCFG L2 (12 runs): 1,022,000 tokens.

Block shapes (parents x children): Matryoshka, gemma-2-2b L12: 128 x 384, Temporal SAE, gemma-2-2b L12: 3276 x 13108, Matryoshka, Bussmann toy: 3 x 9, Temporal SAE, Bussmann toy (3 seeds): 3 x 9, Matryoshka, PCFG L2 (12 runs): 224 x 224.
