"""Synthetic SAE dictionaries with planted, dial-controlled damage, scored by the benchmark.

Damages are absorption, hedging, split and composition (see `corruptions.py`); activations are
NNLS strengths on the planted support, with no SAE trained. Kept outside the trees
`manifest.EVALUATOR_SOURCES` hashes, so edits here do not move `evaluator_sha256`.
"""
