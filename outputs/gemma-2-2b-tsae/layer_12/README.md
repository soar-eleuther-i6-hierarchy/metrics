# Temporal SAE, gemma-2-2b, layer 12

The released checkpoint `alex-oesterling/temporal-saes`, `trainer_0`, run through the same
metric code as the Matryoshka SAE on the same corpus and layer. The adapter is
`adapters/from_tsae_gemma.py`; the metrics themselves are unchanged.

| file | what it is |
| --- | --- |
| `metrics_report.json` | the battery, with the probe results merged in |
| `metrics_report_pre_sres.json` | the same report before the probe pass; kept to show what the merge added |
| `second_pass.json` | the probe test, per edge |
| `threshold_sweep.json` | the acceptance thresholds swept on this source |
| `checkpoint_config.json` | the released training config, as published |
| `token_cache_meta.json` | the residual cache the probe test ran off |
| `metrics_report.md` | the same report, readable |
| `run_logs/` | console output of every run |

## Two things that would be easy to misread

**The block partition comes from the checkpoint**, `[3276, 13108]`, not from `config.py`, which
assumes gemma's five nested prefixes. Passing those would have sliced this dictionary at the
wrong boundaries and still produced a full report.

**The layer is the output of block 12**, which HuggingFace indexes as `hidden_states[13]`. It
was settled by reconstruction rather than by reading the convention. Sparsity does not settle
it: neighbouring residual activations have similar scale, and a BatchTopK threshold responds
mostly to scale.

## The Matryoshka baseline is not here

It lives with the rest of the Matryoshka results, in `gemma-2-2b/layer_12/`, as
`metrics_report.json`. That file and this one both report 48,571 tokens and both record
`min_joint: 30` and `bos_excluded: true`, so the two architectures were graded under the same
rules.

For one day the baseline pointed instead at a report graded before BOS positions were excluded
from the co-firing counts. BOS is an attention sink, so that report admitted almost every pair
in the dictionary and its edge counts were far too high. It is quarantined in
`gemma-2-2b/layer_12/withdrawn/`, with the details.

Claims: `findings/I6-F007`, `I6-F008`, `I6-F009`. Log: `EXPERIMENT_LOG.md`, 2026-09-11 17:40
and 21:05.
