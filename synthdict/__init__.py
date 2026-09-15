"""synthdict — synthetic SAE dictionaries for latent-side metric stress tests.

Constructs dictionaries with planted, dial-parameterized pathologies and scores them through
the UNMODIFIED benchmark pipeline (`scoring.benchmark.run_read`, `read="synthetic"`). No SAE
training anywhere.

Damages: `absorption` (decoder carry + firing hole, L = F), `noise` (all rows perturbed,
L = F), `missing` (a feature's row deleted, L < F), `split` (a feature carried by k shards,
L > F). Merging is deliberately absent: it needs a sibling world, and `PlantedMap` refuses a
latent claimed by two features.

Only absorption has been pre-registered and run. The rest is machinery.

Lives outside `scoring/` / `toygen/` / `metrics/` on purpose: `manifest.EVALUATOR_SOURCES`
content-hashes those trees, so adding a file there would change `evaluator_sha256` and break
freeze verification. This package only IMPORTS them.

Scope, dials, predictions, and claim wording are frozen in `SYNTH_PRECOMMIT.md`.
"""
