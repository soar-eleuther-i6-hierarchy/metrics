# Split detectors on the collaborator's shard corruption (toy `only_isa`, seeds [0, 1, 2], fraction 0.5, 50,000 tokens)

Shards of one feature share its direction and partition its tokens. Firing-overlap detectors read them as disjoint siblings; decoder cosine reads them as one direction.

| k | shard Jaccard (global) | shard Jaccard (within parent) | shard decoder cosine | null cosine |abs| (unsplit parents) | null Jaccard | P(shard \| child), median | cosine ≥ 0.9 detector: precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.065 ± 0.000 (max 0.120 ± 0.000) | 0.099 ± 0.000 | 0.500 ± 0.000 | 1.00 / 1.00 |
| 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.065 ± 0.000 (max 0.120 ± 0.000) | 0.099 ± 0.000 | 0.333 ± 0.000 | 1.00 / 1.00 |
| 4 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.065 ± 0.000 (max 0.120 ± 0.000) | 0.099 ± 0.000 | 0.250 ± 0.000 | 1.00 / 1.00 |
| 6 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.065 ± 0.000 (max 0.120 ± 0.000) | 0.099 ± 0.000 | 0.166 ± 0.000 | 1.00 / 1.00 |
