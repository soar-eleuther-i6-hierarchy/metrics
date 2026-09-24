"""
Gates: fixed-threshold pass/fail decisions on candidate edges, shared by every pipeline.

Frame. Every gate is a matrix over candidate pairs [P, C]: row p is a candidate parent, column
c a candidate child. In a cross-block frame P and C are different blocks; in a square frame
(one set of features against itself) P == C and the caller removes self-pairs with
`nan_self_pairs`. The gates never assume either.

Tristate. A gate is float64 over {1.0, 0.0, NaN}:

    1.0   the rule holds for this pair
    0.0   the rule does not hold
    NaN   the pair was never measurable, so neither answer was earned

A boolean has no way to say "no evidence", so an unmeasurable pair would silently become a
rejection. `defined` is therefore a required argument everywhere, never defaulted.

Constants come from a named set in `constants.py`; the gates take plain numbers so any
pipeline can call them.
"""

from __future__ import annotations

import torch

DT = torch.float64
_NAN = float("nan")

GATE_NAMES: tuple[str, ...] = (
    "gate_support",            # both endpoints fire enough and co-fire enough
    "gate_contains",           # P(p | c) >= tau
    "gate_strictly_contains",  # P(p | c) >= tau and P(c | p) < tau
    "gate_mutually_contains",  # both directions >= tau
    "gate_recon",              # parent and child both carry reconstruction mass on c's tokens
    "gate_sres_rank",          # both decoders in the child probe's top-k
    "gate_high_outdegree",     # either endpoint holds a large share of the candidate children
    "gate_freq_survives",      # coverage holds up once frequent tokens are removed
)


def tristate(flag: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    """`{1.0, 0.0, NaN}` from a pass flag and the domain on which it was measurable."""
    if flag.shape != defined.shape:
        raise ValueError(f"flag {tuple(flag.shape)} and defined {tuple(defined.shape)} "
                         f"describe different frames")
    return torch.where(defined, flag.to(DT),
                       torch.full(flag.shape, _NAN, dtype=DT, device=flag.device))


def nan_self_pairs(m: torch.Tensor) -> torch.Tensor:
    """NaN on the diagonal of a square frame: a feature is not its own parent.

    Float only. On a bool tensor `fill_diagonal_(nan)` silently writes True.
    """
    if not m.is_floating_point():
        raise TypeError("nan_self_pairs needs a float tensor; on a bool tensor "
                        "fill_diagonal_(nan) silently writes True")
    m = m.clone()
    m.fill_diagonal_(_NAN)
    return m


def squash(ratio: float) -> float:
    """`x / (1 + x)`, the transform some pipelines report the survival ratio under."""
    x = float(ratio)
    return x / (1.0 + x)


# --------------------------------------------------------------------------
# support
# --------------------------------------------------------------------------
def supported(cofire: torch.Tensor, fire_p: torch.Tensor, fire_c: torch.Tensor,
              min_fire: int, min_joint: int) -> torch.Tensor:
    """Bool [P, C]: both endpoints fire at least `min_fire` times and co-fire at least
    `min_joint` times. The guard in `metrics.coverage.keep_edges`, used as the domain on which
    the coverage gates are defined."""
    return ((fire_p >= float(min_fire)).reshape(-1, 1)
            & (fire_c >= float(min_fire)).reshape(1, -1)
            & (cofire >= float(min_joint)))


def support_gate(support: torch.Tensor) -> torch.Tensor:
    """`gate_support`. Defined on every pair, since firing counts exist for every pair."""
    return tristate(support, torch.ones_like(support, dtype=torch.bool))


# --------------------------------------------------------------------------
# containment  (metrics/coverage.py, metrics/in_block.py)
# --------------------------------------------------------------------------
def coverage_gates(R: torch.Tensor, R_rev: torch.Tensor, support: torch.Tensor,
                   tau: float) -> dict[str, torch.Tensor]:
    """The three containment gates, defined on supported pairs.

        R[p, c]     = P(p fires | c fires)   (reverse coverage)
        R_rev[p, c] = P(c fires | p fires)   (forward coverage; R.T in a square frame)

        contains           R >= tau                     her cross-block edge (keep_edges)
        strictly_contains  R >= tau and R_rev <  tau    her in-block parent_of
        mutually_contains  R >= tau and R_rev >= tau    her in-block duplicate

    NaN >= tau is False, so a NaN coverage cannot pass.
    """
    ge = R >= float(tau)
    ge_rev = R_rev >= float(tau)
    return {
        "gate_contains": tristate(ge, support),
        "gate_strictly_contains": tristate(ge & ~ge_rev, support),
        "gate_mutually_contains": tristate(ge & ge_rev, support),
    }


# --------------------------------------------------------------------------
# reconstruction  (metrics/reconstruction.py)
# --------------------------------------------------------------------------
def recon_gate(parent_gain: torch.Tensor, child_gain: torch.Tensor,
               rel_gain_min: float) -> torch.Tensor:
    """`gate_recon`: parent_gain[p, c] >= min AND child_gain[c] >= min.

    `child_gain` is per child [C] and is broadcast down each COLUMN. Defined where both gains
    are finite; `edge_reconstruction_condition` instead clamps the denominator, so a child that
    never fires reads there as a confident failure.
    """
    if child_gain.ndim == 1:
        child_gain = child_gain.reshape(1, -1).expand_as(parent_gain)
    flag = (parent_gain >= float(rel_gain_min)) & (child_gain >= float(rel_gain_min))
    defined = torch.isfinite(parent_gain) & torch.isfinite(child_gain)
    return tristate(flag, defined)


# --------------------------------------------------------------------------
# S_res rank  (metrics/sres.py)
# --------------------------------------------------------------------------
def sres_rank_gate(corr: torch.Tensor, parent_ids: torch.Tensor, child_ids: torch.Tensor,
                   available: torch.Tensor, top_k: int) -> torch.Tensor:
    """`gate_sres_rank`: Tree SAE's rank rule, as in `metrics.sres.sres_rank_check`.

    corr[j, f]  = child j's probe direction against decoder f, over the whole competitor pool
    parent_ids  = [P] pool index of each candidate parent
    child_ids   = [C] pool index of each candidate child
    available   = [C] whether child j's probe trained

    Passes when both the parent's and the child's own decoder rank inside the top k of the
    pool. Which pool (whole dictionary or the scored frame) is the caller's choice, and is
    recorded as `GateConstants.sres_rank_pool`. A child with no probe gets a NaN column.
    """
    k = int(top_k)
    P, C = int(parent_ids.numel()), int(child_ids.numel())
    flag = torch.zeros((P, C), dtype=torch.bool, device=corr.device)
    defined = torch.zeros((P, C), dtype=torch.bool, device=corr.device)
    for j in range(C):
        if not bool(available[j]):
            continue
        order = torch.argsort(corr[j], descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.numel(), device=order.device)
        in_top = ranks < k
        flag[:, j] = in_top[parent_ids] & in_top[int(child_ids[j])]
        defined[:, j] = True
    return tristate(flag, defined)


# --------------------------------------------------------------------------
# out-degree  (metrics/outdegree.py)
# --------------------------------------------------------------------------
def high_outdegree(edge_mask: torch.Tensor, fire_p: torch.Tensor, n_candidates: int,
                   outdeg_frac: float, min_fire: int) -> torch.Tensor:
    """Per-parent tristate [P]: the parent's out-degree is at least
    `outdeg_frac * n_candidates`, as in `metrics.outdegree.find_superparents`.

    `n_candidates` is how many children the out-degree was counted over: the child-block size
    in a cross-block frame, R - 1 in a square one. A parent firing fewer than `min_fire` times
    cannot form an edge at all, so its out-degree of 0 is not evidence: NaN.
    """
    outdeg = edge_mask.to(DT).sum(dim=1)
    testable = fire_p >= float(min_fire)
    flag = (outdeg >= float(outdeg_frac) * n_candidates) & testable
    return tristate(flag, testable)


def either_endpoint(t_p: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
    """Pair tristate [P, C] from per-feature tristates: passes when EITHER endpoint passes,
    defined only when both endpoints are."""
    fp, fc = t_p > 0.5, t_c > 0.5                       # NaN > 0.5 is False
    dp, dc = torch.isfinite(t_p), torch.isfinite(t_c)
    return tristate(fp.reshape(-1, 1) | fc.reshape(1, -1), dp.reshape(-1, 1) & dc.reshape(1, -1))


# --------------------------------------------------------------------------
# frequency survival  (metrics/token_control.py)
# --------------------------------------------------------------------------
def freq_survives_gate(survival: torch.Tensor, min_ratio: float, scale: str) -> torch.Tensor:
    """`gate_freq_survives`: survival >= the cut, equality surviving.

    `min_ratio` is always stated on the raw ratio R_rare / R_all. `scale` says which scale
    `survival` is on: "raw", or "squashed" (x / (1 + x)), in which case the cut is squashed
    too. Comparing a squashed ratio against the raw 0.5 would demand a raw ratio of 1.0.
    Defined where `survival` is finite.
    """
    if scale == "raw":
        cut = float(min_ratio)
    elif scale == "squashed":
        cut = squash(min_ratio)
    else:
        raise ValueError(f"scale must be 'raw' or 'squashed', got {scale!r}")
    return tristate(survival >= cut, torch.isfinite(survival))
