# Priors in Time, gemma-2-2b, layer 12

The released checkpoint `ekdeepslubana/temporalSAEs_gemma` (the `temporal` variant), measured
with the authors' own forward pass rather than our port.

`results.json` holds everything: the depth sweep, the dictionary quality at the winning depth,
and the provenance fields. `run_logs/` holds the console output of the runs that produced it.

## There is no metrics report here, and that is deliberate

This architecture produces no parent-to-child feature edges. Its dictionary is flat, 9,216
features with no block structure, and its hierarchy is a clustering of token positions into
events rather than of features into parents and children. Its released code contains no
clustering at all.

Running the metric battery would mean inventing a block partition. That would be our
construction rather than the method's, and grading it would be grading something we made up.
See `findings/I6-F005`.

## What is measured instead

The split the method does claim: how much of a token's representation is predicted from its
context and how much is new. At the output of block 12, the predictive component carries 90.4%
of the reconstruction energy.

## The layer

`conf.yaml` records `block_id: 1` while the repository README lists layer 12. The two cannot
both be a layer index. The depth was settled by reconstruction, since an SAE rebuilds the layer
it was fitted to best: the peak is `hidden_states[13]`, the output of block 12, matching the
README. `block_id` refers to something else, most likely a position in a list of precomputed
activation layers.

Full entry: `research-log/EXPERIMENT_LOG.md`, 2026-09-11 23:40. Claim: `findings/I6-F010`.
