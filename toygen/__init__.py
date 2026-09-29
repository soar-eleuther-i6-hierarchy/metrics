"""Synthetic toy worlds: a declared feature forest, its activations, and a per-pair answer key.

Pipeline: spec -> tree -> geometry -> strengths -> sample. A world is fully determined by its
`ToyConfig` and sampling seed, so it is rebuilt rather than stored. Named configs are in `spec.CONFIGS`.
"""
