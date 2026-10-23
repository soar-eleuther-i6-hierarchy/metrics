# `tests/` — unit tests for the pipeline code

Distinct from [`validation/`](../validation/), which scores the *metrics* against known answers.
Nothing here measures hierarchy or grades an edge. These guard claims the **code** makes about
itself — the kind that break silently, with every downstream number still looking reasonable.

| File | Guards | Cost |
| ---- | ------ | ---- |
| [`test_collect_generic.py`](test_collect_generic.py) | `collect_statistics.collect()` runs on a source that is not gemma | no network, no GPU, ~1s |
| [`test_dashboards_generic.py`](test_dashboards_generic.py) | stages 02 → 03 → 04 grade and *render* a source that is not gemma | no network, no GPU, ~13s |
| [`test_calibration_covers_metrics.py`](test_calibration_covers_metrics.py) | every function in `metrics.__all__` is actually called by the Tier-1 calibration | AST only, ~0.1s |
| [`test_metric_math.py`](test_metric_math.py) | every metric equals its own definition, recomputed independently | no network, no GPU, ~2s |
| [`test_site_links.py`](test_site_links.py) | every `href` and `src` on the generated site resolves | ~1s |
| [`test_metrics_v2.py`](test_metrics_v2.py) | every `metrics_v2` name is the identical object as its `metrics` original; the new functions match their definitions | ~1s |
| [`test_rules.py`](test_rules.py) | every gate and rule of `metrics/rules/` pinned on a 2x2 frame worked out by hand, with and without a probe; the one-sided `either_endpoint` case | ~0.1s |

```bash
python3 -m tests.test_collect_generic
python3 -m tests.test_dashboards_generic
python3 -m tests.test_calibration_covers_metrics
python3 -m tests.test_metric_math
python3 -m tests.test_site_links
```

**Why a math test, when `validation/` already scores every metric.** Tier 1 grades *behaviour*:
it runs a metric and checks that it separates the class it should keep from the class it should
reject. The production function is what produces the numbers Tier 1 scores, so a formula that is
consistently wrong still separates the classes and still passes — divide coverage by the parent
instead of the child everywhere, and genuine edges still out-score pathological ones.
`test_metric_math.py` recomputes each metric from the definition in its own docstring, by explicit
loops where possible, and asserts equality. Verified by injecting four real errors — dropping the
factor of 2 in the ablation gain, swapping coverage's denominator, forgetting `N` in the
independence null, and making the in-block graph cyclic — each of which it caught and none of which
Tier 1 would have.

## Why this one exists

Stage 01 used to load gemma-2-2b, the released Matryoshka SAE and pile-10k, then accumulate
statistics from them, all in one function. `collect()` is the accumulation split out, so an adapter
can feed it a PCFG transformer or a trained toy instead.

That claim is cheap to make and easy to break. One `config` global left in the accumulation
loop — `C.D_SAE`, `C.BLOCK_RANGES`, `utils.sae_utils.block_slice` — reintroduces gemma's 32768
latents in 5 blocks with no symptom at all: a 1792-latent dictionary gets sliced at the wrong
boundaries and every statistic downstream is computed from the wrong columns. Wrong statistics do
not crash. They produce plausible numbers.

So the test builds a stub model, a stub SAE and a config with 28 features in 3 blocks — nothing like
gemma — and asserts the result is a well-formed schema-v2 stats file whose shapes came from the
config it was handed. When the umbrella repo is checked out beside this one it also runs
`contracts/validate_stats.py` against its own output.

## And why the second one

`collect()` being source-agnostic bought nothing while the stages that *read* its output were not.
`run_token_metrics.py` and `reporting/visualize.py` both sliced `config.BLOCK_RANGES` directly, so a
PCFG file (1792 latents in 8 blocks) was graded and drawn against gemma's five — silently for the
first four pairs, because every index is in range there, and only then with an error. Stage 03 also
loaded the *released gemma decoder* to score S_res, whatever dictionary the cache came from.

The second test runs a PCFG-shaped file (8 blocks) through 02 → 03 → 04 and asserts the pages
describe the file they were built from: all seven pairs graded, the header naming that run's model
and dictionary rather than gemma's, S_res computed from the run's own `w_dec.pt`, and no layer pill
lit on a run that is not a gemma layer.

## The captions guard

`test_captions_match_panels.py` + `panel_fingerprint.py` cover the one part of the generated site
that cannot check itself. In `reporting/visualize.py` every **number** in a dashboard caption is
read from JSON when the page is written, so it cannot disagree with the data. The **sentences**
around those numbers are hand-written, and they can: change what a panel plots and the numbers
update themselves while the prose keeps describing the old panel.

The test does **not** verify that a caption is correct — nothing can. It extracts each dashboard's
`subplot_titles` statically with `ast` (no data, no cache, so it runs in a bare clone) and pins a
hash. When a panel changes the hash changes, the test fails, and the message names the
`captions_*()` function to re-read.

**When it fails, do not re-pin first.** Open the builder, see what moved, check every sentence in
its caption function still describes the panel it names, and only then update the hash. Re-pinning
without reading is the same as deleting the test, and it is much easier to do by accident.

A second check, `test_every_caption_builder_is_wired`, catches the opposite mistake: a caption
function that exists but was never passed to `write_page()`, which publishes a page with no
captions and no error.
