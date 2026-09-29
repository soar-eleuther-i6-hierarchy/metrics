"""The shared `metrics.rules` gates in scoring's square [R, R] frame of recovered features.

Self-pairs are NaN, reverse coverage is `R.T`, out-degree counts the other R - 1 features, and
survival arrives squashed as x / (1 + x).
"""

from __future__ import annotations

import torch

from metrics import rules as MR

DT = torch.float64

GATE_NAMES: tuple[str, ...] = MR.GATE_NAMES

# CONSTANTS keys the gates read; stamped on every artifact.
GATE_CONSTANT_KEYS: tuple[str, ...] = (
    "edge_tau", "min_fire_count", "support_min_joint", "recon_rel_gain_min",
    "sres_rank_top_k", "superparent_outdeg_frac", "freq_survival_min_raw",
)

# Metrics each gate decides on, so a metric can inherit a rule's target.
GATE_SOURCES: dict[str, tuple[str, ...]] = {
    "gate_support": (),
    "gate_contains": ("coverage_R",),
    "gate_strictly_contains": ("coverage_R", "asymmetry_R"),
    "gate_mutually_contains": ("coverage_R", "asymmetry_R"),
    "gate_recon": ("recon_2a", "recon_child_gain"),
    "gate_sres_rank": ("S_res",),
    "gate_high_outdegree": ("outdegree", "wide"),
    "gate_freq_survives": ("token_freq_survival",),
}
assert set(GATE_SOURCES) == set(GATE_NAMES), "a gate has no declared source metrics"


def support_mask(cofire: torch.Tensor, fire: torch.Tensor, min_fire: int,
                 min_joint: int) -> tuple[torch.Tensor, dict]:
    """Bool [R, R] of pairs with enough firing and co-firing to score, plus exclusion counts by cause.

    Counts leave out the diagonal."""
    keep = MR.supported(cofire, fire, fire, min_fire, min_joint)
    enough = fire >= float(min_fire)
    R = int(keep.shape[0])
    off = ~torch.eye(R, dtype=torch.bool, device=keep.device)
    n_pairs = int(off.sum())
    n_supported = int((keep & off).sum())
    low_fire = ~(enough.reshape(-1, 1) & enough.reshape(1, -1)) & off
    low_joint = (cofire < float(min_joint)) & off
    counts = {
        "n_pairs": n_pairs,
        "n_supported": n_supported,
        "n_excluded": n_pairs - n_supported,
        "frac_excluded": (n_pairs - n_supported) / n_pairs if n_pairs else float("nan"),
        # by cause; a pair can fail both, so these overlap
        "n_excluded_low_fire": int(low_fire.sum()),
        "n_excluded_low_joint": int(low_joint.sum()),
        "n_excluded_both_causes": int((low_fire & low_joint).sum()),
        "min_fire": int(min_fire), "min_joint": int(min_joint),
    }
    return keep, counts


def square_pair_stats(cofire: torch.Tensor, fire: torch.Tensor, R_mat: torch.Tensor,
                      edge_mask: torch.Tensor, parent_gain: torch.Tensor,
                      child_gain: torch.Tensor, survival: torch.Tensor,
                      constants: MR.GateConstants,
                      probe_directions: torch.Tensor | None = None,
                      probe_available: torch.Tensor | None = None,
                      W_unit: torch.Tensor | None = None) -> MR.PairStats:
    """`metrics.rules.PairStats` for the square frame, ready for `score_pairs`."""
    R = int(R_mat.shape[0])
    wide = MR.high_outdegree(edge_mask, fire, max(R - 1, 1), constants.superparent_outdeg_frac,
                             constants.min_fire_count)
    corr = ids = None
    if probe_directions is not None and probe_available is not None:
        Wu = W_unit.to(DT)
        corr = torch.zeros((R, R), dtype=DT, device=Wu.device)
        for c in range(R):
            if bool(probe_available[c]):
                corr[c] = Wu @ probe_directions[c].to(Wu.device).to(DT)
        ids = torch.arange(R, device=Wu.device)
    return MR.PairStats(cofire=cofire, fire_p=fire, fire_c=fire, R=R_mat, R_rev=R_mat.T,
                        parent_gain=parent_gain, child_gain=child_gain,
                        survival=survival, survival_scale="squashed",
                        high_outdeg_p=wide, high_outdeg_c=wide,
                        probe_corr=corr, parent_ids=ids, child_ids=ids,
                        probe_available=probe_available, self_pairs=True)
