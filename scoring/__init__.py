"""Score toy SAEs against the toy generator's ground truth.

`core` holds the checkpoint-free pieces, `oracle` checks detectors on true inputs, `trained` loads
checkpoints and classifies absorption, and `benchmark` evaluates the rules per toy, seed and read.
"""