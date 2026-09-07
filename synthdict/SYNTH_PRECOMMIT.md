# SYNTH_PRECOMMIT - synthetic-dictionary latent-side stress test, round 1 (absorption)

Status: FINALIZED 2026-09-06 (all four reserved approvals settled by Chidaksh; see the plan file).
This is a development-stage study, separate from the benchmark freeze.
Nothing here changes any metric, expression, threshold, or the frozen evaluator; deviations from this document require re-approval before scoring.

## Question

Do the registered metrics and frozen expressions respond to latent-side absorption as operationalized below, with what dose-response, and is the response attributable to the metric or to the recovery pipeline?

## Claim wording (fixed in advance)

Licensed: "metric M / expression E responds to operationalized absorption (decoder-carry, firing hole) with the reported dose-response f(realized severity), under the pure-toy nulls."
Never claimed: "metrics detect absorption in SAEs"; nothing about prevalence of absorption under real training.
A flat or non-monotone response is a reportable negative, not a bug to fix.

## Operationalization (generator knobs, independent of every detector/census threshold)

For each corrupted containment edge (p, c) of the world's tree:

- Decoder carry (dial `beta`): child raw decoder row becomes `unit(g_c + beta * rbar * g_p)`; `rbar = 1.0` analytically (flat strength design, `toygen/strengths.py`: mean active strength is `q*sqrt(E0)` for every feature).
- Firing hole (dial `eta`): the parent p is removed from the planted support on `round(eta * n_c)` of the tokens where c fires, chosen by a dedicated deterministic RNG seeded from (world seed, draw seed, edge); never from the world's sampling streams.
- Edge selection (dial `edge_fraction` f): a deterministic fraction of the tree's containment edges, seeded from the world seed only, so the corrupted edge set is identical across the matching/scoring/probe draws.
- Magnitudes: re-solved per token by ridge least-squares on the planted support against the real `h` (lam = 1e-4, the `oracle_encode` convention); the parent latent stays present (hole, not deletion).
- Severity is reported as realized per-edge `cos(d_c', g_p)`; `beta` is only the generator knob.

Tautology guard: `corruptions.py` imports neither `ABSORPTION_CONSTANTS` nor `scoring.core.registry.CONSTANTS` (enforced by test).
Census (`classify_dictionary`) output is annotation only: it shares definitions with the pathology, so it is a manipulation check, never validation.

## Round-1 grid (approved)

- Toys: only_isa, only_firing. Seed: 0. n_tokens: 200,000. Both match modes (identity, hungarian) per dial point.
- Diagonal slice: beta = eta in {0, 0.2, 0.4, 0.6, 0.8, 1.0} x f in {0.1, 1.0}.
- Decoupled controls: (beta=0.6, eta=0) "decoder-only carry" and (beta=0, eta=0.6) "hole-only", at f = 0.1, both toys.
- Bridge slice: only_firing, f = 11/120, beta tuned so realized severity ~ 0.48.
- Venue: server (`exp0_remote.sh`), dial points as parallel jobs; one timed dial point before the grid.

## Pre-registered predictions (approved verbatim)

- P1: G and S_res on corrupted edges rise monotonically with realized severity.
- P2: `orthogonal_v6` recall falls on corrupted only_firing edges; `overlap_v6` leaks onto them.
- P3: coverage_R / pmi / asymmetry_R are flat in beta at eta = 0 (specificity control).
- P4: coverage_R(p->c) scales ~ (1 - eta); C-recall falls at high eta.
- P5: Hungarian parent recovery falls with eta; the identity-vs-hungarian delta attributes metric vs pipeline failure.

## Bridge conditional (pre-registered)

IF the bridge slice lands the s_res median on corrupted edges in [0.15, 0.30] AND the same detectors elevate as in the trained artifact (`outputs_archive/pilot_seed0/only_firing/`: s_res 0 -> 0.227, band-precision collapse), THEN the report may claim "the corruption model reproduces the measured training phenomenology on only_firing".
Otherwise the mismatch is reported as a bound on the corruption model.
No post-hoc upgrade of the claim either way.

## Gates (block everything downstream)

1. Passthrough anchor: the uncorrupted synthetic read (acts = true A, identity match) reproduces `oracle_read`'s metric vectors bit-for-bit.
2. Ridge anchor at beta = 0: per-feature corr(acts, A) >= 0.99, FVU ~ noise floor, support-flip rate measured; if flips > 0.5% of support entries, switch to NNLS (no silent clamping).
3. One timed dial point end-to-end before any grid.

## Out of scope

Splitting / merging / missing-latent / dead / duplicate / interference-noise corruptions (registry slots exist); the sibling world; seeds 1-2; mixed-pathology grids; any change under `scoring/`, `toygen/`, or `metrics/` (would change `evaluator_sha256`).
