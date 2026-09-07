# FINDINGS - synthetic-dictionary absorption stress test, round 1

**Study:** planted, dial-parameterized absorption in synthetic SAE dictionaries (no training), scored through the unmodified frozen benchmark evaluator.
**Scope, dials, predictions, and claim wording:** frozen in `SYNTH_PRECOMMIT.md` before any scoring; the two mid-study deviations (a second activation model; the bridge framing) were user-approved before interpretation, and are recorded in section 6.
**Runs:** seed 0, n_tokens 200k, toys {only_isa, only_firing}, both match modes, two activation models: `ridge` tag `SYNTH-R1` (29 dial points) and `clean` tag `SYNTH-R1-CLEAN` (17 points, the f=0.1 subset + bridge), server artifacts + `dose_response.csv` + `REPORT.md` per tag.
**Verification state:** 50 tests, 32/32 mutation anchors killed; passthrough anchor bit-identical to the real `oracle_read`; three independent reviews (adversarial correctness, instrument validity, test quality) closed; the frozen `tests_local` suite unaffected.

Claim wording (fixed in advance): "metric M / expression E responds to operationalized absorption (decoder-carry, firing hole) with the reported dose-response, under the pure-toy nulls".
Never claimed: "metrics detect absorption in SAEs", or anything about prevalence under real training.

## 1. Headline results

**The G-based expressions are extremely sensitive to decoder carry - a step, not a slope.**
On only_firing (f=0.1, identity, both activation models): at beta=0.2 (realized severity 0.196 against a null q99 of ~0.12), `orthogonal_v6` drops every corrupted edge (pass 1.00 -> 0.00) and `overlap_v6` picks all of them up (0.00 -> 1.00).
The registered dose-response is therefore saturation above one threshold crossing; recall-given-recovery moves 1.000 -> 0.900 exactly as the 12/120 corrupted edges change class.
On only_isa, `overlap_v6` keeps corrupted edges through beta<=0.6 and loses them at beta>=0.8, where the diagonal's hole (coverage = 1-eta = 0.2) breaks the C clause - with the clean activation model the attribution to the hole is exact.

**S_res (the deployed self-label probe) does not respond to absorption at all - and reads elevated on a PERFECT dictionary.**
S_res corrupted-edge median is 0.323 and intact 0.312 at every beta including beta=0 on the f=0.1 grid, in both activation models, both match modes (only_isa: 0.380; the f=1.0 population reads 0.314).
So the probe's "elevation" on co-firing edges is produced by the probe mechanism itself (a child's probe direction picks up the parent via co-firing; `PRECOMMIT.md` s5 warned of exactly this; it is now measured), not by any dictionary property.
The flatness is exact across activation models, and for a verified reason: `S_res` and `G` are byte-identical between the ridge and clean grids at all 34 shared dial points, because the probe binarizes its fitting labels and pair (p, c) uses the CHILD's probe while the hole only ever touches the parent column - the entire ridge-vs-clean delta lands on coverage_R / pmi / asymmetry_R / recon_2a / FVU.
P1's S_res half is a clean negative: this instrument cannot carry an absorption claim on co-firing edges.

**The recovery pipeline is robust to absorption in-range - P5 is refuted, informatively.**
Hungarian recovery stays 240/240 at every dial point, including beta=eta=1.0 with every edge corrupted, in both activation models; identity and hungarian columns are identical everywhere.
At child_p_edge=0.32, a fully holed parent still keeps 68% of its firing, so its activation correlation never approaches rho_star=0.5.
The matcher becomes the failure point only when the child owns most of the parent's support (forced in-test at child_p_edge=0.9), which quantifies where "pipeline failure" begins - far outside these worlds' regime.
This is consistent with the trained-side superparent result: non-recovery there was non-emergence, not matcher fragility.

**Absorption's explaining-away, quantified as the difference of two activation models.**
Under the unconstrained ridge (deployed-encoder-like), decoder carry alone suppresses the parent on co-firing tokens: parent-side coverage reads 0.850 at (beta=0.6, eta=0) where the construct said 1.0 - beta leaks into the firing channel (found by review before interpretation).
Under the clean, firing-preserving model the same dial reads coverage exactly 1.000, pmi 1.71, asymmetry 0.680 (identical to intact), flip rate 0.
The 0.85-vs-1.00 gap at identical dials (only_firing; only_isa reads 0.847) is a measured size of the reconstruction-driven explaining-away, which parallels - as an analogy, not a measured identity - how absorption starves a parent in a trained SAE; the clean model's FVU exceeds the ridge model's at all 17 matched dial points (paired delta +0.0020 to +0.0086) - the hedging-style price of forbidding it.

## 2. Prediction verdicts (pre-registered in SYNTH_PRECOMMIT.md)

| Prediction | Verdict | Evidence |
| --- | --- | --- |
| P1: G and S_res rise monotonically with realized severity on corrupted edges | SPLIT | G: yes, but definitionally (G__corrupted == severity by construction in identity mode; a manipulation check, not a response - instrument-audit F2). S_res: NO - flat at 0.323 at every severity, both models. |
| P2: orthogonal_v6 falls on corrupted only_firing edges; overlap_v6 leaks onto them | CONFIRMED | Step at beta=0.2: ortho pass_corrupted 1.00->0.00, overlap 0.00->1.00; both 0.00 at beta>=0.8 when C collapses. |
| P3: coverage_R / pmi / asymmetry_R flat in beta at eta=0 | CONFIRMED on the clean model (1.000 / 1.71 / 0.680, == intact); REFUTED under the ridge model (only_firing 0.850 / 1.60 / 0.565; only_isa 0.847 / 1.60 / 0.562) - a property of the activation solver, not of the metrics (section 6). |
| P4: coverage_R(p->c) ~ (1-eta); C-recall falls at high eta | CONFIRMED exactly (clean diagonal: 1.0, 0.8, 0.6, 0.4, 0.2, 0.0); C pass_corrupted holds through eta<=0.6 and fails at eta>=0.8. The coverage identity is a manipulation check anchoring the eta axis. |
| P5: hungarian recovery falls with eta; identity-vs-hungarian delta attributes failures | REFUTED in-range | Recovery 240/240 everywhere; delta identically zero. The refutation is the result: the matcher is not the fragile link for absorption at these containment rates. |

## 3. The bridge (pre-registered conditional): FAIL, and the comparator itself was wrong

Pre-registered: IF the bridge slice (only_firing, f=11/120, realized severity ~0.48) lands the S_res corrupted-edge median in [0.15, 0.30], THEN the corruption model may be claimed to reproduce the trained phenomenology.
Measured: S_res corrupted median **0.323**, outside the band, in both activation models.
Per the pre-registration this is reported as a bound on the corruption model, with no claim in either direction and no post-hoc upgrade.

The instrument audit additionally found the conditional was mis-specified before it was evaluated:
- the trained 0.227 it targeted is the CLASS-WIDE probe median over all 120 firing_only pairs (probe mode, matryoshka seed-0);
- the 11 census-absorbed trained edges sit at median ~0.141 (range 0.121-0.253), the LOW tail - the hole degrades the parent-side probe correlation and `s_res_from_directions` takes the min over endpoints - while the 109 un-absorbed edges sit at ~0.229 (per-pair mapping by parent id; an inference, not a recorded key - but a near-certain one: the 11 census-absorbed parents occupy exactly the 11 lowest coverage_R slots of 120 with a clean gap, and spearman(coverage_R, census R_P) over them is 0.964);
- so `outputs_archive/pilot_seed0/only_firing/FINDINGS.md` section 5's causal sentence ("s_res rises to 0.227 BECAUSE the 11 absorbed child latents pull the parent direction into their decoder rows") is contradicted by its own per-pair data and should be corrected in a follow-up pass (the archived file is not edited here).

Corrected per-population comparison (reported as an audit finding, not a pre-registered claim):
synthetic corrupted 0.323 / intact 0.312, versus trained absorbed ~0.141 / un-absorbed ~0.229.
The corruption model does not reproduce the trained probe phenomenology in either population; what the synthetic grid adds is the leading candidate mechanism for the trained class-wide elevation (section 1: the probe reads ~0.32 on co-firing edges of a perfect dictionary), though the synthetic level (0.323) and the trained level (0.227) do not match, so the identification is not complete.
Why the synthetic S_res does not drop on holed edges the way the trained one does is an open mechanism question for round 2 (candidates: the trained probe was fitted on SAE activations that themselves carry the hole; our probe direction is fitted per child and scored against decoders whose parent row is unchanged).

## 4. Census (annotation, not validation)

The census counts only planted edges - zero false positives outside the planted set in all 92 artifact rows - once the carry clears its span-calibrated eps (0.342-0.368 in every census.json, both tags).
The clearing point differs by toy: 0 flagged at beta=0.2 everywhere; on only_firing it flags the full planted set from beta>=0.4, on only_isa from beta>=0.6 (the designed alpha overlap raises the bar); the bridge reads exactly 11/11 in both activation models.
The hole gate works as designed and corroborates deviation 1: at (beta=0.6, eta=0) the clean model reads 0 absorbed (no hole, so no absorption per the definition) while the ridge model reads 6 (beta manufactured real holes).
Instrument facts recorded along the way: the eps is span-rank dependent (~0.65 in a rank-24 test world, hiding beta<=0.8 carry; ~0.35 at grid scale), and `theta_hat` is radians while severity is a cosine (severity = sin(theta_hat) on an orthogonal edge).

## 5. Side observations (recorded, not chased)

- Every eval-null FPR stays tiny across all dials (max 0.00063 for the G rules and C), except `topical_v6`, which reads ~0.010 on BOTH toys' eval-nulls at every dial including zero corruption (only_firing ~0.0104, only_isa ~0.0102, both models) - i.e. its off-world FPR violation is world-independent on these reads, a stronger statement than the benchmark's per-world records (PRECOMMIT s3: only_firing 318/28560 and only_superparent 80/7140).
- At f=1.0 the per-read calibration null is itself corrupted: on only_firing the G q99 drifts 0.118 -> 0.151 (+28%; only_isa is near-flat at +1.8% and non-monotone) and the coverage q99 0.186 -> 0.128 (-31%, both toys) along the diagonal, with the intact control empty; f=0.1 rows are the primary dose-response (reading rule in REPORT.md).
- `token_freq_survival` was flat everywhere, and licenses nothing: random holes are frequency-uniform by construction and these worlds have no token-frequency structure.
- orthogonal_v6 at f=1.0 reads recall 0.000 from beta=0.2 on: with every edge corrupted there is no intact control and the whole class fails IN-BAND(G) - the instrument-as-deployed on a fully absorbed dictionary.

## 6. Deviations from SYNTH_PRECOMMIT.md (all user-approved before interpretation)

1. **Second activation model.** Review found the unconstrained ridge lets beta manufacture firing holes (the CRITICAL above); decision: keep both models - ridge as the deployed-encoder-like variant, clean as the dial-independent construct the precommit promised. P3's verdict is given per model.
2. **Bridge framing.** Decision: report the pre-registered FAIL as frozen, plus the corrected per-population comparison and the probe-mechanism finding, and flag the archived FINDINGS sentence for correction.
3. The support-flip NNLS escape hatch (gate 2) was written for the beta=0 anchor and never triggered there; the beta>0 flips were the construct leak of deviation 1 and were handled by it, not by NNLS.

## 7. Scope and caveats

- Single seed (0), two worlds, one corruption family; nothing here estimates variance or prevalence.
- `G__corrupted` vs severity and `coverage_R__corrupted` vs (1-eta) are definitional identities (in both match modes here, whose columns are byte-identical) - dose-axis manipulation checks; the informative content is threshold crossings, expression verdicts, intact/null controls, and the hungarian pipeline.
- The trained absorbed-edge medians in section 3 rest on an index-mapping inference (the rank-0..10 coverage_R separation plus spearman(coverage_R, R_P)=0.964 over the absorbed set), not a recorded pair key.
- The eta-hole selects random token subsets; real encoder holes are magnitude-ordered, so magnitude-reading detectors (recon_2a, joint_child_mass, S_res, matching corr) transfer directionally, not at exact breakpoints.

## 8. Follow-ups (parked; nothing here blocks round 1)

- Round 2 pathologies via the registry: splitting, merging (needs a sibling world), duplicates, missing latents, interference noise.
- The S_res hole-response mechanism question (section 3).
- Correction pass on the archived only_firing FINDINGS s5 causal sentence.
- Seeds 1-2 replication if any round-1 number is to be quoted with uncertainty.
