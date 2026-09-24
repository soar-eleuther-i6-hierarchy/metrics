"""
gates — the shared fixed-threshold gates (`metrics/rules/gates.py`), in the square `[R, R]` frame.

Every decision is made by `metrics.rules`; this module only adapts it to scoring's frame, so the
two pipelines cannot drift apart. What the adapter adds:

  * SELF-PAIRS. The frame is one set of features against itself, so every gate NaNs its
    diagonal: a feature is not its own parent, and `gate_mutually_contains` would otherwise
    read True there (a feature covers itself in both directions).
  * REVERSE COVERAGE is `R.T`, since both orderings of a pair live in the same matrix.
  * OUT-DEGREE is counted over `R - 1` candidate children, and a pair is flagged when EITHER
    endpoint is wide, the fixed-threshold form of the `wide` transformation.
  * THE RANK POOL is the scored frame, not the whole dictionary. On a trained read with R < L
    the bar is top-k of R, weaker than top-k of L; on oracle and synthetic reads R == F.
  * SURVIVAL is reported squashed (`x / (1 + x)`) by `detectors.token_freq_survival`, so the
    gate is called with `scale="squashed"` and the raw constant.

Gates are float64 tristates over {1.0, 0.0, NaN}, never booleans, for three separately silent
reasons: `fill_diagonal_(nan)` on a bool tensor writes True, `torch.isfinite` on a bool tensor is
all-True, and multiplying a flag by a `DETECTOR_SIGN` is meaningless.

Constants come from `scoring.core.registry.CONSTANTS`, whose gate keys are the named set
`metrics.rules.SYNTHETIC_TOYS`.
"""

from __future__ import annotations

import torch

from metrics import rules as MR

DT = torch.float64
_NAN = float("nan")

GATE_NAMES: tuple[str, ...] = MR.GATE_NAMES

# The constants the gates compare against. ONE list: the freeze record and every artifact's
# `__meta__` both stamp it.
GATE_CONSTANT_KEYS: tuple[str, ...] = (
    "edge_tau", "min_fire_count", "support_min_joint", "recon_rel_gain_min",
    "sres_rank_top_k", "superparent_outdeg_frac", "freq_survival_min_raw",
)

# Which REGISTERED METRICS each gate decides on, so `evaluate.constant_flags` can attach a
# target to them. `gate_support` lists nothing: it is built from raw counts, which are not
# registered metrics.
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

nan_diag = MR.nan_self_pairs
squash = MR.squash


def _square(m: torch.Tensor) -> torch.Tensor:
    return nan_diag(m) if m.ndim == 2 and m.shape[0] == m.shape[1] else m


def tristate(flag: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    """`metrics.rules.tristate`, with the diagonal NaN'd on a square frame."""
    return _square(MR.tristate(flag, defined))


def support_mask(cofire: torch.Tensor, fire: torch.Tensor, min_fire: int,
                 min_joint: int) -> tuple[torch.Tensor, dict]:
    """`(bool [R, R], counts)`: which ordered pairs carry enough evidence to be scored at all.

    The mask is `metrics.rules.supported`. The counts are returned because the excluded
    fraction is what separates "the rule rejected these pairs" from "the rule could not see
    them". They EXCLUDE the diagonal, which is not a pair.
    """
    keep = MR.supported(cofire, fire, fire, min_fire, min_joint)
    enough = fire >= float(min_fire)
    R = int(keep.shape[0])
    off = ~torch.eye(R, dtype=torch.bool, device=keep.device)
    n_pairs = int(off.sum())
    n_supported = int((keep & off).sum())
    counts = {
        "n_pairs": n_pairs,
        "n_supported": n_supported,
        "n_excluded": n_pairs - n_supported,
        "frac_excluded": (n_pairs - n_supported) / n_pairs if n_pairs else float("nan"),
        # By cause. NOT A PARTITION: a pair can fail both, so these overlap.
        "n_excluded_low_fire": int(((~(enough.reshape(-1, 1) & enough.reshape(1, -1))) & off).sum()),
        "n_excluded_low_joint": int(((cofire < float(min_joint)) & off).sum()),
        "n_excluded_both_causes": int(
            ((~(enough.reshape(-1, 1) & enough.reshape(1, -1)))
             & (cofire < float(min_joint)) & off).sum()),
        "min_fire": int(min_fire), "min_joint": int(min_joint),
    }
    return keep, counts


def support_gate(support: torch.Tensor) -> torch.Tensor:
    """`gate_support`, defined on the whole frame."""
    return _square(MR.support_gate(support))


def directed_coverage_gates(R_mat: torch.Tensor, support: torch.Tensor,
                            tau: float) -> dict[str, torch.Tensor]:
    """`{gate_contains, gate_strictly_contains, gate_mutually_contains}` on supported pairs.

    `R_mat` is `detectors.coverage_R`, NaN on the diagonal and for a dead child; `NaN >= tau` is
    False, and `support` already requires both endpoints to fire, so the reverse comparison is
    never reading a NaN inside the domain that survives.
    """
    g = MR.coverage_gates(R_mat, R_mat.transpose(0, 1), support, tau)
    return {k: _square(v) for k, v in g.items()}


def recon_contributes(parent_gain: torch.Tensor, child_gain: torch.Tensor,
                      rel_gain_min: float) -> torch.Tensor:
    """`gate_recon`. `child_gain` is the per-child vector, broadcast down each column."""
    return _square(MR.recon_gate(parent_gain, child_gain, rel_gain_min))


def sres_rank_gate(P: torch.Tensor, available: torch.Tensor, W_unit: torch.Tensor,
                   top_k: int) -> torch.Tensor:
    """`gate_sres_rank`, ranking each child's probe against the scored frame's decoders.

    Row c of the correlation matrix is child c's probe against every decoder in the frame; a
    child whose probe did not train gets a NaN column.
    """
    Wu = W_unit.to(DT)
    R = int(Wu.shape[0])
    corr = torch.zeros((R, R), dtype=DT, device=Wu.device)
    for c in range(R):
        if bool(available[c]):
            corr[c] = Wu @ P[c].to(Wu.device).to(DT)
    ids = torch.arange(R, device=Wu.device)
    return _square(MR.sres_rank_gate(corr, ids, ids, available, top_k))


def high_outdegree_flag(em: torch.Tensor, fire: torch.Tensor, outdeg_frac: float,
                        min_fire: int) -> torch.Tensor:
    """`gate_high_outdegree`: EITHER endpoint holds `outdeg_frac` of its `R - 1` candidates.

    Defined where both endpoints fire at least `min_fire` times: below that no edge can form,
    so an out-degree of 0 is arithmetic, not evidence of a narrow parent.
    """
    R = int(em.shape[0])
    t = MR.high_outdegree(em, fire, max(R - 1, 1), outdeg_frac, min_fire)
    return _square(MR.either_endpoint(t, t))


def freq_survives_gate(survival: torch.Tensor, min_ratio_raw: float) -> torch.Tensor:
    """`gate_freq_survives` on `detectors.token_freq_survival`, which is squashed."""
    return _square(MR.freq_survives_gate(survival, min_ratio_raw, scale="squashed"))
