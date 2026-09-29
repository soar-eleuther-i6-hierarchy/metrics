"""Detector names and signs. Their settings are `metrics.rules.SYNTHETIC_TOYS` and
`metrics.rules.METRIC_SETTINGS`."""

from __future__ import annotations

# +1 if a higher raw value is more is-a-like, -1 if the detector is negated.
DETECTOR_SIGN: dict[str, int] = {
    "coverage_R": 1, "asymmetry_R": 1, "joint_child_J": 1, "pmi": 1,
    "token_freq_survival": 1, "recon_2a": 1,
    "sibling_redundancy": -1, "joint_child_mass": 1, "outdegree": -1,
    "recon_child_gain": 1,         # child half of the reconstruction condition
    "joint_child_supp": 1,         # exact union form of joint_child_J
    "sibling_redundancy_pc": -1,   # parent-conditioned sibling overlap; high means splitting
}

# Per-ordered-pair detector scalars, in a fixed order.
DETECTORS: tuple[str, ...] = tuple(DETECTOR_SIGN)
