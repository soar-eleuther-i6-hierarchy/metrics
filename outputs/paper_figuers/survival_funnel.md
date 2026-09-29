# Survival funnel, block pair 0->1

Cumulative counts: edges confirmed by every gate up to that stage, in the order of the hierarchy rule. Unmeasurable edges leave the funnel at the gate that cannot measure them (counts in survival_funnel.json). A bracketed cell is an interval the reports bound when the statistics file is not on disk.

| Setting | 1 candidates | 2 PMI ≥ 0.5 | 3 survival ≥ 0.5 | 4 reconstruction | 5 direction alignment | final share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matryoshka, gemma-2-2b L12 | 1,473 | 355 | 329 | 188 | 16 | 0.011 |
| Temporal SAE, gemma-2-2b L12 | 1,621 | 1,621 | [1,191, 1,191] | [1,168, 1,191] | [317, 770] | 0.196–0.475 |
| Matryoshka, Bussmann toy | 9 | 9 | 9 | 9 | 9 | 1.000 |
| Temporal SAE, Bussmann toy (3 seeds) | 9.3 ± 0.6 | 9.3 ± 0.6 | 9.3 ± 0.6 | 9.3 ± 0.6 | n/a | 1.000 |
| Matryoshka, PCFG L2 (12 runs) | 157.8 ± 200.2 | 97.3 ± 140.0 | 75.2 ± 109.1 | 71.9 ± 105.4 | 4.2 ± 4.1 | 0.042 |
