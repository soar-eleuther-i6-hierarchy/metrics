"""Detector names and signs, the scored pair columns, and the named constants detectors and gates read."""

from __future__ import annotations

from dataclasses import replace

from metrics.rules import RULESET_VERSION, SYNTHETIC_TOYS, GateConstants

# Per-ordered-pair detector scalars, in a fixed order.
DETECTORS: tuple[str, ...] = (
    "coverage_R", "asymmetry_R", "joint_child_J", "pmi", "token_freq_survival",
    "recon_2a", "s_res", "sibling_redundancy", "joint_child_mass", "outdegree",
    "recon_child_gain", "joint_child_supp", "sibling_redundancy_pc",
)

# +1 if a higher raw value is more is-a-like, -1 if the detector is negated.
DETECTOR_SIGN: dict[str, int] = {
    "coverage_R": 1, "asymmetry_R": 1, "joint_child_J": 1, "pmi": 1,
    "token_freq_survival": 1, "recon_2a": 1, "s_res": 1,
    "sibling_redundancy": -1, "joint_child_mass": 1, "outdegree": -1,
    "recon_child_gain": 1,         # child half of the reconstruction condition
    "joint_child_supp": 1,         # exact union form of joint_child_J
    "sibling_redundancy_pc": -1,   # parent-conditioned sibling overlap; high means splitting
}

# Confound columns scored against is_a.
SCORED_COLUMNS: tuple[str, ...] = (
    "firing_only", "sibling", "superparent", "frequency", "topical",
    "transitive", "reversed", "unrelated",
)

# Dictionary-damage columns, labelled by absorption classification rather than pair_labels.
LATENT_COLUMNS: tuple[str, ...] = ("absorbed", "merged")

POSITIVE_LABEL: str = "is_a"

# Symmetric in (parent, child), so CIs count (a, b) and (b, a) as one pair.
SYMMETRIC_DETECTORS: tuple[str, ...] = ("pmi",)

# The named gate-constant set; it fills the gate keys of CONSTANTS below.
GATE_CONSTANT_SET = SYNTHETIC_TOYS
_SET = GATE_CONSTANT_SET


def ruleset_stamp() -> dict:
    """The ruleset version and gate-constant set name, written beside every artifact."""
    return {"ruleset_version": RULESET_VERSION, "gate_constant_set": GATE_CONSTANT_SET.name}


def gate_constants(constants: dict) -> GateConstants:
    """`GateConstants` built from a CONSTANTS-shaped dict, so an override reaches `score_pairs`."""
    return replace(GATE_CONSTANT_SET, edge_tau=constants["edge_tau"],
                   min_fire_count=constants["min_fire_count"],
                   min_joint=constants["support_min_joint"],
                   recon_rel_gain_min=constants["recon_rel_gain_min"],
                   sres_rank_top_k=constants["sres_rank_top_k"],
                   superparent_outdeg_frac=constants["superparent_outdeg_frac"],
                   freq_survival_min=constants["freq_survival_min_raw"])


# Numeric knobs the detectors and scorer read by name.
CONSTANTS: dict[str, float] = {
    "fire_thresh": _SET.fire_threshold,   # firing means activation > 0
    "edge_tau": _SET.edge_tau,           # reverse-coverage cut for the inferred edge set
    "min_fire_count": _SET.min_fire_count,   # both endpoints must fire this often to form an edge
    "min_joint": 30,           # min co-firing tokens for a supported edge
    # support gate's joint floor; separate from min_joint, which token_freq_survival also reads
    "support_min_joint": _SET.min_joint,
    "recon_rel_gain_min": _SET.recon_rel_gain_min,   # >=1% relative error increase
    "superparent_outdeg_frac": _SET.superparent_outdeg_frac,   # child share that flags high out-degree
    # raw ratio; the detector reports x / (1 + x), so square_pair_stats marks it squashed
    "freq_survival_min_raw": _SET.freq_survival_min,
    "pmi_laplace": 1.0,        # +1 smoothing in the PMI ratio
    "coverage_eps": 1e-6,      # coverage denominator floor
    "r_disp_m": 5,             # top-m competitors for the dispersion readout
    "freq_high_mass": 0.5,     # top corpus-mass cut for the high-frequency token bucket
    "freq_mid_mass": 0.4,      # mid bucket cut
    "n_freq_buckets": 3,
    "freq_min_fire_low": 5,    # below this, the rare-token survival cell is underpowered
    "rho_star": 0.5,           # recovery threshold the recovered universe is built on
    "auroc_clamp": 1e-6,       # clamp AUROC to [clamp, 1-clamp] before the logit CI
    # --- probe s_res ---
    # both decoders must rank in the child probe's top k
    # 2 here, 5 in config.py and Tree SAE; the departure is recorded in PRECOMMIT.md s4
    "sres_rank_top_k": _SET.sres_rank_top_k,
    "sres_min_probe_pos": 50,  # min child-firing tokens to train a probe, else NaN
    "sres_neg_ratio": 4,       # negatives sampled per positive
    "sres_max_probe_tokens": 20000,  # cap on (pos + neg) tokens per probe
    "sres_min_neg": 10,        # fewer negatives than this -> child untestable (no probe)
    "sres_steps": 300,         # probe Adam steps
    "sres_lr": 0.05,           # probe Adam lr
}