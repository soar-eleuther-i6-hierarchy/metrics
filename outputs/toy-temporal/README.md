# Toy model with a time axis

The Bussmann toy draws independent samples, so it cannot train any architecture whose loss has
a temporal term: on a flat loader that term is a constant 0.0 and the run still saves a
checkpoint. `toy_model.TemporalTreeSampler` adds the missing axis by depth-scaled persistence.

The construction leaves the per-token distribution unchanged. Expected L0 stays at the tree's
1.12 under every setting, which is what makes `--persistence 0` a control rather than a second
dataset.

| file | what it is |
| --- | --- |
| `tsae_results.json` | Temporal SAE, 24 runs across two trees |
| `priors_in_time_results.json` | Priors in Time, 9 runs |
| `priors_in_time_boundary.json` | the event-boundary test, per seed, with the predictions that were stated before the run |

One file per architecture, named for what is in it. They were one file until the name stopped
telling a reader where to look.

## Two trees

`configs/tree.json` has parents at 0.15 and distractors at 0.05, so parents are also the most
frequent features and the two properties cannot be told apart.
`configs/tree_decorrelated.json` reverses that, parents at 0.04 and distractors at 0.115, with
expected L0 held at 1.112. Anything that changes between them is about frequency, not depth.

Claims: `findings/I6-F001`, `I6-F003`, `I6-F004`, `I6-F011`.
