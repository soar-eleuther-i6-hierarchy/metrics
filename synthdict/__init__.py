"""synthdict — synthetic SAE dictionaries for latent-side metric stress tests.

Constructs dictionaries with planted, dial-parameterized pathologies and scores them through
the UNMODIFIED benchmark pipeline (`scoring.benchmark.run_read`, `read="synthetic"`). No SAE
training anywhere; activations are NNLS strengths on the planted support.

Damages: `absorption` (decoder carry + co-fire firing hole, L = F), `hedging` (a child's row
deleted and mixed into its parent's, L < F), `split` (a feature carried by k shards, L > F),
`composition` (a combination latent for two independent features, L > F).

Lives outside `scoring/` / `toygen/` / `metrics/` on purpose: `manifest.EVALUATOR_SOURCES`
content-hashes those trees, so adding a file there would change `evaluator_sha256` and break
freeze verification. This package only IMPORTS them.
"""
