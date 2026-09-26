# Withdrawn: a pre-BOS-fix report

`metrics_report_node_stats.json` and its `.md` were graded before BOS positions were
excluded from the co-firing counts. They are kept for the record and must not be read as
results.

## What is wrong with them

BOS is an attention sink. With 400 documents, every feature pair in the dictionary
collected 400 joint firings from the BOS position alone, which clears `MIN_JOINT = 30` on
its own. The co-fire guard therefore passed almost everything, and every metric downstream
graded a candidate set that should not have existed.

The project already withdrew two pages for this reason in commit `cf96fd4`.

## How to recognise a file with this problem

| signal | affected file | current files |
| --- | --- | --- |
| `total_tokens` | 48,971 | 48,571 |
| `config.bos_excluded` | absent | `true` |
| `config.min_joint` | absent | `30` |
| candidate edges, 2->1 | 430,547 | 1,747 |

The token gap is 400, one BOS position per document. It is not one extra document.

## What this cost

These files were used as the Matryoshka baseline in the Temporal SAE comparison, in
`reporting/make_tsae_report.py` and in findings I6-F008 and I6-F009. That comparison graded
the Temporal SAE with BOS excluded against a Matryoshka baseline with BOS included, so the
edge counts were not comparable. The baseline now reads `../metrics_report.json`, which
carries the same guards as the Temporal SAE run.

A fresh run on the GPU node on 2026-09-11, with the current code and `--docs 400`,
reproduced `../metrics_report.json` exactly: 48,571 tokens and 1,473 / 621 / 1,747
candidate edges. That run is in `../layer_12_b3b4/`.

## Also withdrawn: `threshold_sweep_pre_bos.json`

The threshold sweep for this layer was first run on the node against
`experiment_0/outputs/layer_12/exp0_stats.pt`, the same pre-BOS cache as the report above.
Its coverage count at the operating point, 3,262, is that report's 3,260 candidates plus the
2 that `MIN_JOINT` drops. The sweep file records no token count or config, which is why the
comparability audit could not flag it.

Re-run on the current cache (48,571 tokens, `bos_excluded: true`), the coverage count is
1,475, which is the committed report's 1,473 plus 2. The shape of the result is unchanged:
the chance share still rises with the edge threshold and reaches 1.00 at 0.9. The edge
counts differ (771 accepted at the operating point rather than 519), and the tables that
carried the old counts were corrected on 2026-09-12. `threshold_sweep.py` now writes the token
count and guards into its output so this cannot recur silently.
