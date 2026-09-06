# Seed-0 regression gate: `scoring/benchmark/` against the pilot evaluator

The pilot's expression numbers came from `tests_local/gw_check.py`, untracked scratch with two
confirmed defects.
This package replaces it.
Before any fresh seed is opened, the new evaluator has to reproduce the pilot on seed 0 except
where a fix is supposed to change something, and every difference has to be accounted for.

The predicted-difference list was written **before** the run, in the plan:

1. the `superparent` rule may move, because NaN no longer passes `NOT-HIGH`;
2. evaluation-null FPRs shift slightly, because the calibration split now keys on true feature IDs;
3. **recall counts must not move at all**;
4. anything else moving is investigated before proceeding.

An independent audit re-derived every number below from the artifacts and found four claims in
the first version of this file that were wrong or overstated.
They are corrected here, and each correction is marked.

## Result

Prediction 3 was stated imprecisely and the gate caught it.
What must not move is **`N_pass`**, and `N_pass` did not move on a single target-class row.
What did move is `recall_given_recovery`, because its denominator changed from `N_recovered` to
the clause-specific `N_scorable` - which is the fix from `PRECOMMIT.md` s7 box 2 doing exactly
its job.

| Toy | Expression | Class | Pilot | New | Cause |
| --- | --- | --- | --- | --- | --- |
| only_isa | `overlap_v6` | `is_a` | 2/112 | 2/112 | unchanged |
| only_firing | `orthogonal_v6` | `firing_only` | 95/120 | 95/120 | unchanged |
| only_isa | `containment_baseline` | `is_a` | 112/112 | 112/112 | unchanged |
| only_firing | `containment_baseline` | `firing_only` | 117/120 | 117/120 | unchanged |
| only_frequency | `frequency_v6` | `frequency` | 236/**240** = 0.983 | 236/**236** = 1.000 | 4 target pairs have a NaN `token_freq_survival` and were counted as REJECTIONS; they are now unscorable |
| only_topical | `topical_v6` | `topical` | 18/**56** = 0.321 | 18/**54** = 0.333 | 2 target pairs, same cause |
| only_superparent | `superparent_v5` | `superparent` | (absent from the pilot JSON) | UNTESTABLE, 0/726 recovered | see the note below |

The audit traced the two denominator changes to the individual pair: `only_frequency` has
exactly 4 NaN `token_freq_survival` cells and **all 4 are on `frequency`-class pairs**;
`only_topical` has 6, of which 2 are on `topical` and 4 on `unrelated`.
`pmi` is finite and its thresholds usable throughout, so `token_freq_survival` is the only
possible source.

**Correction.** The `superparent` row is not a pilot-vs-new comparison.
The pilot evaluator skips absent classes outright (`if n == 0: continue`), so `only_superparent`'s
superparent row does not exist in the pilot JSON - only `unrelated` and `unrelated_eval` do.
The new side genuinely reports `verdict: UNTESTABLE`, `N_recovered: 0`, `N_total: 726`.
The first version of this file presented "pilot: UNTESTABLE" as if it had been read off an
artifact; it was an inference.

Four rows did move on `N_pass`, all of them `unrelated` / `unrelated_eval` rows in `only_isa`:

| Expression | Class | Pilot | New |
| --- | --- | --- | --- |
| `containment_baseline` | `unrelated` | 1 | 3 |
| `orthogonal_v6` | `unrelated` | 1 | 3 |
| `topical_v6` | `unrelated` | 518 | 584 |
| `topical_v6` | `unrelated_eval` | 254 | 324 |

**Correction, previously undisclosed.** The evaluation-half POPULATION moved too, not only the
pass counts.
In `only_isa` the evaluation null went from 26,684 to 26,708 ordered pairs and the calibration
half from 26,684 to 26,660.
So `topical_v6 unrelated_eval 254 -> 324` is a comparison over different denominators:
254/26,684 = 0.952% against 324/26,708 = 1.213%.
Same cause, but the rates are what should be compared, not the counts.

## The four moved rows are the split fix, verified causally rather than argued

The pilot's split differs from the new one in **two** ways, not one:

- it keys on recovered POSITIONS rather than true feature ids;
- it permutes over the null keys PRESENT IN THE SCORED FRAME, so `len(uniq)` - and therefore the
  entire `torch.randperm` draw - depends on how many features survived recovery.

`PRECOMMIT.md` s6.4 requires assignment on true ids **before** intersecting with recovered
endpoints, which fixes both at once.

**Correction.** The first version said `only_isa` is the only world with partial recovery.
That is wrong: `only_superparent` recovers 120 of 123.
The right statement is that `only_isa` is the only world whose recovered ids are
**non-contiguous**.
`only_superparent`'s 120 recovered features are true ids 0..119, so positions and true ids
coincide there and the two splits agree.
Measured against the saved `split` arrays, the number of pairs that change half is
26,808/53,592 in `only_isa` and **0** in each of the other four worlds.

The check: re-implement the pilot's exact split inside the new evaluator and rerun `only_isa`
with nothing else varying.

- Every threshold came back **bit-identical** to the pilot's saved thresholds.
- Every `N_pass` reproduced exactly. **0 rows unexplained.**

**Correction on the strength of that evidence.** The diagnostic compared
`set(pilot_thresholds) & set(new_thresholds)`, which is **11 metrics, not 12**: the pilot named
the geometry metric `G_W` and this package names it `G`, so the rename silently dropped it from
the intersection - and `G`'s Q99 is the threshold `overlap_v6` and `orthogonal_v6` rest on.
The audit checked `G` separately under the name mapping and it **is** bit-identical
(q01 -0.19987319679640184, q99 0.1672850048936937 on both sides), so 12/12 does hold.
But as run, the automated check covered 11 and missed the load-bearing one.

A first attempt at this check patched only the lookup key and did NOT reproduce the pilot's
thresholds, which is what revealed the second difference.
That earlier attempt is not evidence for anything and is superseded by the faithful one.

## Prediction 1 did not materialise, and the reason matters

`superparent_v5` is `LOW(wide) AND NOT-HIGH(pmi)`, so the NaN-passes-`NOT-HIGH` defect can only
fire where `pmi` is missing.
Read off the saved arrays, `pmi` is finite on **every scored pair in all five toys, on both
reads**, because `pmi` uses fixed Laplace smoothing: `detectors.py:100-104` computes
`log((cofire+l)*N / ((fire_p+l)(fire_c+l)))` with `pmi_laplace = 1.0`, so the numerator is at
least `N > 0` and the denominator at least 1, and `compute_all` maps any infinity to NaN while
`_nan_diag` NaNs only the diagonal, which the pair frame excludes.

So the defect is real in the code and was **not triggered by seed 0**.
It is fixed, and it stays fixed, but seed 0 supplies no evidence either way about its impact.
It would fire wherever a required score goes missing - which the fresh seeds and the complex toy
may well produce.

Where missingness actually occurs, **on the trained read** (the oracle read has different pair
counts and a different `sibling_redundancy` pattern; the `pmi` result above holds on both):

| Toy | pairs | `token_freq_survival` NaN | `sibling_redundancy` NaN | every other metric |
| --- | --- | --- | --- | --- |
| only_isa | 53,592 | 1,736 | 53,592 | none |
| only_firing | 57,360 | 46 | 57,360 | none |
| only_superparent | 14,280 | 0 | 14,280 | none |
| only_frequency | 18,360 | 4 | 18,225 | none |
| only_topical | 20,592 | 6 | 20,020 | none |

`token_freq_survival` is the only missingness that touches a registered expression, and it is
exactly what moved the two recall denominators above.
`sibling_redundancy` is near-totally undefined but no registered expression reads it.

## Harness gate

Every trained read reproduces its checkpoint's own saved scoring arrays element-wise across all
nine non-`s_res` detectors, with length and finite-pattern equality required:

| Toy | max abs diff | keys checked |
| --- | --- | --- |
| only_isa | 1.97e-07 | 81 |
| only_firing | 9.89e-08 | 81 |
| only_superparent | 2.98e-08 | 81 |
| only_frequency | 1.34e-07 | 81 |
| only_topical | 2.25e-07 | 81 |

Tolerance 1e-5; the saved arrays are float32.
These are **identical to the pilot's own gate values to every printed digit, on the same worst
key** - an exact reproduction, not merely the same order of magnitude, which is how the first
version of this file put it.

## Provenance caveat on the baseline

The pilot side of this comparison is `outputs_local/gw_check/*.json`, which were written
2026-09-06 06:30 UTC by a same-day re-run of `tests_local/gw_check.py` (mtime 06:17 UTC).
They are not the original pilot artifacts, and both the script and the JSONs are untracked, so
the pilot side of this gate has no immutable record.
They do agree with the pilot counts independently recorded in `PRECOMMIT.md` s3 - 112/112,
117/120, 62/240, 18/56, 2/112, 22/120, 110/112, 95/120, 236/240 - which is the strongest
available check on them.

Separately, the new evaluator reproduces the probe counts `PRECOMMIT.md` s11 recorded before
this package existed: the `firing_only` `NOT-HIGH` probe clause passes **0/120 oracle and 13/120
trained**, exactly as the document states.

## Verdict

The gate passes.
Every difference from the pilot is attributable to one of the two intended fixes, and the
attribution was verified by exact reproduction rather than inferred.

## How to reproduce

`exp0_remote.sh` lives in the PARENT directory of this repo, not at the repo root.
`run_seed0.sh` is tracked alongside this file and is pushed with the rest of the tree.

```
cd ..            # soar/, where exp0_remote.sh lives
./exp0_remote.sh push
ssh soar-gpu 'cd ~/exp0-chidaksh && bash scoring/benchmark/run_seed0.sh'
```

Then write and check the freeze record:

```
ssh soar-gpu 'cd ~/exp0-chidaksh && ~/.local/bin/uv run python -m scoring.benchmark.run_benchmark \
    --write-manifest --tag <TAG> --out outputs_local/benchmark --seeds 0 \
    --toys only_isa,only_firing,only_superparent,only_frequency,only_topical'
```

Ten reads, all in parallel: the last measured wall time was **104 s** end to end, with the
slowest single read (`only_firing` oracle) at 102.5 s.

The diagnostic scripts that produced the pilot-comparison tables (`regdiff.py`, `why_moved2.py`,
`missingness.py`, on the server) hard-code the `REGRESSION-SEED0` tag directory, which has since
been removed; they would need their path updated to `SEED0-PREFREEZE` before they would run
again.
Every number above reproduces against the current `SEED0-PREFREEZE` artifacts.

---

# Phase B2: the second gate, SEED0-CORRECTED against SEED0-PREFREEZE

Everything above describes the FIRST gate, which compared this package against the pilot scratch
evaluator under **report schema 1**.
Two of its findings were reversed by B2 and the section stays as written, as the record of what
was true then:

- the `recall_given_recovery` denominator moved from `N_recovered` to `N_scorable` in phase B, and
  B2.3 moved it back to `N_recovered`, where it now carries the bar.
  The pilot's original 236/240 and 18/56 figures are therefore restored.
- the `fpr` key is now `fpr_given_scorable`, with `fpr_over_half` beside it.

`registry.REPORT_SCHEMA` exists so the two contracts cannot be pooled by accident.

## What B2 changed, and what it was allowed to move

Five freeze blockers, found by an external review and all reproduced:

1. `oracle_read` left geometry at seed 0 whatever seed was requested, so oracle reads at seeds
   1/2/3 would have used seed-0 geometry against trained reads that used the real geometry, both
   labelled the same seed. The existing test passed because it asserted on `pmi`, which moves
   with the token draw; only an assertion on `g` or on `G` can see it.
2. `--verify-manifest` rebuilt the manifest instead of loading it, so it verified the evaluator
   against itself and passed on a tree with no `MANIFEST.json` at all.
3. the recall bar rode on a denominator whose name did not match it.
4. the support floor guarded only the target, not the negative evidence.
5. the probe was fitted on the same draw it scored.

## The predicted-difference list, written before the run

| Prediction | Outcome |
| --- | --- |
| every trained non-`s_res` array holds | HELD: 22 arrays identical in all five toys, 0 moved |
| harness gate unchanged | HELD: identical to every printed digit, 81 keys, all five toys |
| oracle `G` holds (pure geometry, `cfg.seed` is 0 either way at seed 0) | HELD in all five toys |
| oracle co-firing detectors move (new scoring draw) | MOVED in all five toys |
| `S_res` moves on BOTH reads (separate fitting draw) | MOVED, all ten reads |
| `null_split_sha256`, `n_pairs`, `F`, class populations hold | HELD |
| exactly two trained recall cells move, no verdict flips | HELD: `frequency_v6` 1.000 -> 0.983 and `topical_v6` 0.333 -> 0.321, `N_pass` unchanged on both |
| named flip risk: a bar-adjacent oracle FPR may cross 0.01 | HAPPENED: `only_topical/oracle/topical_v6` MET -> DID NOT MEET, FPR 0.00818 -> 0.01091 |

`tests_local/corrected_diff.py` checks both directions and reports GATE PASSED: every difference is
on the list and every prediction held.
The one verdict that moved was called in advance, so it is a reportable coin-flip on a
bar-adjacent number rather than a regression.

## The bridge run, which is the only gate B2.2 gets

`harness_gate` deliberately excludes `s_res`, so nothing in a normal run would catch a mistake in
the probe rewiring.
`SEED0-BRIDGE` was run with the fitting draw set EQUAL to the scoring draw:

- all five TRAINED `S_res` arrays came back **bit-identical** to the pilot, max diff exactly 0.0;
- `PRECOMMIT.md` s11's `probe_orthogonal_v3` count on `firing_only` held at 13/120.

The ORACLE side is deliberately NOT compared to the pilot, and claiming otherwise would be false:
B2.1 moved the oracle scoring draw, so its probe is fitted and scored on a different corpus than
the pilot's, and s11's 0/120 oracle figure is no longer a valid comparison.
The oracle rewiring is pinned instead by
`test_benchmark_probe_draw.py::test_fitting_on_the_scoring_draw_reproduces_the_old_behaviour_exactly`,
which proves the split is the identity when fit == score, whichever draw that is.

## What the cross-world verdict shows

A rule's target lives in one world and its leakage in the others, so no within-world verdict can
see it. On the trained read, three rules disagree with their own within-world verdict, and only
`frequency_v6` survives benchmark-wide - which is what `PRECOMMIT.md` s3 already said in prose.

## Reproduce

```
cd ..                                  # soar/, where exp0_remote.sh lives
./exp0_remote.sh push
ssh soar-gpu 'cd ~/exp0-chidaksh && bash scoring/benchmark/run_seed0_tagged.sh SEED0-BRIDGE --probe-fit-seed 10000'
ssh soar-gpu 'cd ~/exp0-chidaksh && ~/.local/bin/uv run python tests_local/bridge_check.py'
ssh soar-gpu 'cd ~/exp0-chidaksh && bash scoring/benchmark/run_seed0_tagged.sh SEED0-CORRECTED'
ssh soar-gpu 'cd ~/exp0-chidaksh && ~/.local/bin/uv run python tests_local/corrected_diff.py'
```

`bridge_check.py` and `corrected_diff.py` live in `tests_local/`, which is deliberately not
tracked, so they reach the server by rsync rather than by git.
