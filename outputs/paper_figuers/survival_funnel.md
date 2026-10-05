# Survival funnel, block pair 0->1

Cumulative counts: edges confirmed by every core criterion up to that stage, in the order of the hierarchy rule. Unmeasurable edges leave the funnel at the stage that cannot measure them (counts in survival_funnel.json). A bracketed cell is an interval the reports bound when the statistics file is not on disk.

| Setting | 1 candidates | 2 PMI ≥ 0.5 | 3 survival ≥ 0.5 | 4 reconstruction | 5 probe rank | final share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matryoshka, gemma-2-2b L12 | 1,473 | 355 | 329 | 188 | 16 | 0.011 |
| Temporal SAE, gemma-2-2b L12 | 1,621 | 1,621 | 1,191 | 1,171 | 555 | 0.342 |
| Matryoshka, Bussmann toy | 9 | 9 | 9 | 9 | 9 | 1.000 |
| Temporal SAE, Bussmann toy (3 seeds) | 9.3 ± 0.6 | 9.3 ± 0.6 | 9.3 ± 0.6 | 9.3 ± 0.6 | n/a | 1.000 |
| Matryoshka, PCFG L2 (12 runs) | 157.8 ± 200.2 | 97.3 ± 140.0 | 75.2 ± 109.1 | 71.9 ± 105.4 | 4.2 ± 4.1 | 0.042 |
| PCFG density 0.00 (3 seeds) | 96.3 ± 79.2 | 52.0 ± 28.6 | 52.0 ± 28.6 | 49.7 ± 30.7 | 3.7 ± 4.0 | 0.028 |
| PCFG density 0.17 (3 seeds) | 83.0 ± 46.5 | 35.0 ± 10.8 | 35.0 ± 10.8 | 32.0 ± 6.0 | 1.7 ± 1.2 | 0.024 |
| PCFG density 0.23 (3 seeds) | 48.0 ± 19.0 | 36.3 ± 14.5 | 34.0 ± 12.1 | 32.7 ± 9.8 | 3.0 ± 1.7 | 0.080 |
| PCFG density 0.24 (3 seeds) | 403.7 ± 298.2 | 266.0 ± 222.6 | 180.0 ± 205.3 | 173.3 ± 198.0 | 8.3 ± 5.9 | 0.039 |

## Supporting criteria on the surviving edges

Read on the edges that pass every core criterion (stage 5, or stage 4 where no probe was run). None of these is a cut.

| Setting | edges | children with ≥ 2 parents | dense parents (≥ 30% of block) | sibling overlap (mean Jaccard) | parent coverage R_supp (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matryoshka, gemma-2-2b L12 | 16 | 0.071 | 0.000 | 0.064 | 1.000 |
| Temporal SAE, gemma-2-2b L12 | 555 | 0.165 | 0.000 | 0.082 | 1.000 |
| Matryoshka, Bussmann toy | 9 | 0.000 | 1.000 | 0.000 | 0.671 |
| Temporal SAE, Bussmann toy (3 seeds) | 9.3 | 0.083 ± 0.144 | 1.000 ± 0.000 | 0.054 ± 0.093 | 0.671 ± 0.000 |
| Matryoshka, PCFG L2 (12 runs) | 4.2 | 0.312 ± 0.247 | 0.000 ± 0.000 | 0.157 ± 0.110 | 0.953 ± 0.081 |

## The same funnel under `rule_hierarchy` (ruleset version 2)

Clause by clause: support and strict containment (R ≥ τ, R_rev < τ), PMI > 0, survival ≥ 0.5, probe rank. No reconstruction clause; PMI cut 0 rather than 0.5. `keep_edges` is the core funnel's stage 1 (R ≥ τ only); `rule_containment` is the rule's first clause.

| Setting | 1 support ∧ strict containment | 2 PMI > 0 | 3 survival ≥ 0.5 | 4 probe rank | final share | keep_edges | rule_containment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matryoshka, gemma-2-2b L12 | 1,472 | 1,110 | 1,084 | 20 | 0.014 | 1,473 | 1,472 |
| Temporal SAE, gemma-2-2b L12 | 1,511 | 1,511 | 1,085 | 547 | 0.362 | 1,621 | 1,511 |
| Matryoshka, Bussmann toy | 9 | 9 | 9 | 9 | 1.000 | 9 | 9 |
| Temporal SAE, Bussmann toy (3 seeds) | 9.3 ± 0.6 | 9.3 ± 0.6 | 9.3 ± 0.6 | n/a | 1.000 | 9.3 | 9.3 |
| Matryoshka, PCFG L2 (12 runs) | 155.6 ± 196.4 | 139.2 ± 174.9 | 117.2 ± 139.1 | 4.1 ± 3.4 | 0.048 | 157.8 | 155.6 |
