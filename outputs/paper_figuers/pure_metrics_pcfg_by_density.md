# PCFG formatting sweep, block pair 0->1, mean ± sd over seeds

| Metric | Quantity | density 0.0000 (n=3) | density 0.1667 (n=3) | density 0.2308 (n=3) | density 0.2400 (n=3) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 Activation coverage | candidate edges | 96 ± 79 | 83 ± 47 | 48 ± 19 | 404 ± 298 |
|  | edge density (edges / P x C) | 1.9e-03 | 1.7e-03 | 9.6e-04 | 8.0e-03 |
| 2 Independence null | share at chance level (PMI < 0.5) | 0.356 ± 0.194 | 0.456 ± 0.372 | 0.182 ± 0.316 | 0.312 ± 0.173 |
|  | mean PMI over edges | 0.833 ± 0.263 | 0.776 ± 0.270 | 1.651 ± 0.536 | 1.401 ± 0.735 |
| 3 Token-frequency control | frequency-driven share (survival < 0.5) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.042 ± 0.048 | 0.163 ± 0.101 |
|  | mean survival on rare tokens | 0.999 ± 0.008 | 0.954 ± 0.073 | 0.950 ± 0.048 | 0.851 ± 0.091 |
| 4 Reconstruction contribution | pass share | 0.942 ± 0.068 | 0.976 ± 0.042 | 0.975 ± 0.044 | 0.949 ± 0.043 |
| 5 Probe rank | pass share | 0.063 ± 0.008 | 0.024 ± 0.014 | 0.087 ± 0.067 | 0.055 ± 0.055 |
| 6 Out-degree | dense parents | 0 ± 1 | 1 ± 1 | 0 ± 0 | 1 ± 1 |
|  | multi-parenting share | 0.336 ± 0.100 | 0.121 ± 0.103 | 0.618 ± 0.409 | 0.488 ± 0.099 |
|  | out-degree Gini | 0.978 ± 0.009 | 0.980 ± 0.008 | 0.945 ± 0.029 | 0.932 ± 0.023 |
| 7 Sibling redundancy | mean pairwise Jaccard | 0.065 ± 0.072 | 0.021 ± 0.006 | 0.066 ± 0.073 | 0.183 ± 0.085 |
| 9 Joint-child coverage | mean support coverage | 0.226 ± 0.070 | 0.167 ± 0.031 | 0.201 ± 0.110 | 0.397 ± 0.130 |
