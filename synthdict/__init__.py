"""synthdict — synthetic SAE dictionaries for latent-side metric stress tests.

Constructs dictionaries with planted, dial-parameterized pathologies (round 1: absorption)
and scores them through the UNMODIFIED benchmark pipeline (`scoring.benchmark.run_read`,
`read="synthetic"`). No SAE training anywhere.

Lives outside `scoring/` / `toygen/` / `metrics/` on purpose: `manifest.EVALUATOR_SOURCES`
content-hashes those trees, so adding a file there would change `evaluator_sha256` and break
freeze verification. This package only IMPORTS them.

Scope, dials, predictions, and claim wording are frozen in `SYNTH_PRECOMMIT.md`.
"""
