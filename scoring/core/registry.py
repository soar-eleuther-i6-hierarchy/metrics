"""
Configuration for the retrieval scorer.

Defines the detector set and their orientation, the confound columns scored
against is-a, and the numeric constants the detectors look up by name.
"""

from __future__ import annotations

# The per-ordered-pair detector scalars, in a fixed order. The last three were added when the
# package adopted `metrics/`'s definitions: each is a quantity `metrics/` computes and the
# frozen ten did not.
DETECTORS: tuple[str, ...] = (
    "coverage_R", "asymmetry_R", "joint_child_J", "pmi", "token_freq_survival",
    "recon_2a", "s_res", "sibling_redundancy", "joint_child_mass", "outdegree",
    "recon_child_gain", "joint_child_supp", "sibling_redundancy_pc",
)

# Orientation per detector: +1 if higher raw value already means "more is-a-like", -1 if negated.
DETECTOR_SIGN: dict[str, int] = {
    "coverage_R": 1, "asymmetry_R": 1, "joint_child_J": 1, "pmi": 1,
    "token_freq_survival": 1, "recon_2a": 1, "s_res": 1,
    "sibling_redundancy": -1, "joint_child_mass": 1, "outdegree": -1,
    # the child half of the reconstruction condition: more damage from ablating the child
    # means the child carries something of its own, so higher is more is-a-like
    "recon_child_gain": 1,
    # exact union form of joint_child_J, same orientation as the bound it replaces
    "joint_child_supp": 1,
    # parent-conditioned sibling overlap: high means the children are near-copies (splitting),
    # so the sign flips exactly as it does for the global form
    "sibling_redundancy_pc": -1,
}

# The confound columns scored against is_a
SCORED_COLUMNS: tuple[str, ...] = (
    "firing_only", "sibling", "superparent", "frequency", "topical",
    "transitive", "reversed", "unrelated",
)

# Latent-side scoring columns (dictionary damage), not generative pair_label classes; positives come from absorption classification, not the answer key. `split` is per-latent, so it's a side readout, not a pair column.
LATENT_COLUMNS: tuple[str, ...] = ("absorbed", "merged")

POSITIVE_LABEL: str = "is_a"

# Detectors symmetric in (parent, child): a symmetric negative class counts (a,b) and (b,a) as two identical negatives.
SYMMETRIC_DETECTORS: tuple[str, ...] = ("pmi",)

# Numeric knobs the detectors and scorer read by name.
CONSTANTS: dict[str, float] = {
    "fire_thresh": 0.0,        # firing := activation > 0 (BatchTopK nonzero == top-k)
    "edge_tau": 0.5,           # reverse-coverage cut for the inferred edge set
    "min_fire_count": 20,      # both endpoints must fire this often to form an edge
    "min_joint": 30,           # min co-firing tokens for a supported edge
    # The SCORABILITY guard's own joint floor, deliberately named apart from `min_joint` even
    # though it starts at the same value. `min_joint` already has two readers (`edge_mask` and
    # `token_freq_survival`, which applies it internally); tuning the mask through that name
    # would silently retune the frequency detector as well.
    "support_min_joint": 30,
    "recon_rel_gain_min": 0.01,   # config.RECON_REL_GAIN_MIN: >=1% relative error increase
    "superparent_outdeg_frac": 0.30,   # config.SUPERPARENT_OUTDEG_FRAC; the flag is out-degree alone
    # config.FREQ_SURVIVAL_MIN, stated on the RAW ratio. `token_freq_survival` reports the
    # squashed ratio x/(1+x), so it must go through `gates.squash` before any comparison.
    "freq_survival_min_raw": 0.5,
    "pmi_laplace": 1.0,        # +1 smoothing in the PMI ratio
    "coverage_eps": 1e-6,      # coverage denominator floor
    "r_disp_m": 5,             # top-m competitors for the dispersion readout
    "freq_high_mass": 0.5,     # top corpus-mass cut for the high-frequency token bucket
    "freq_mid_mass": 0.4,      # mid bucket cut
    "n_freq_buckets": 3,
    "freq_min_fire_low": 5,    # below this, the rare-token survival cell is underpowered
    "rho_star": 0.5,           # recovery threshold the recovered universe is built on
    "auroc_clamp": 1e-6,       # clamp AUROC to [clamp, 1-clamp] before the logit CI
    # --- probe s_res, mirror config.py so compute_all is self-contained ---
    # Both decoders in the top-k probe correlations. `config.SRES_RANK_TOP_K` and Tree SAE are
    # 5; this benchmark uses 2, decided 2026-09-19. The probe is fitted on the child's own
    # firing, so the child's decoder is rank 1 by construction and k=5 admits any parent in
    # ranks 2-5 -- measured at the oracle ceiling that passed 100% of is_a AND 100% of
    # firing_only, i.e. the rule separated nothing. k=2 asks that the parent be the child's
    # single strongest competitor after itself. Departure from config.py is deliberate and is
    # recorded in PRECOMMIT.md s4.
    "sres_rank_top_k": 2,
    "sres_min_probe_pos": 50,  # min child-firing tokens to train probe, below this the column is NaN
    "sres_neg_ratio": 4,       # negatives sampled per positive
    "sres_max_probe_tokens": 20000,  # cap on (pos + neg) tokens per probe
    "sres_min_neg": 10,        # fewer negatives than this -> child untestable (no probe)
    "sres_steps": 300,         # probe Adam steps (calibration knob)
    "sres_lr": 0.05,           # probe Adam lr (calibration knob)
}