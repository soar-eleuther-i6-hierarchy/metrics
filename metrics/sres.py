"""
Metric 2b - Probe-based reconstruction condition S_res, rank-scored
(Tree SAE).

    S_res(p, c) = min( (d_c*)^T d_c , (d_c*)^T d_p )

where d_c* is a linear-probe direction for the CHILD CONCEPT trained on the
residual stream, and d_c / d_p are the learned decoder vectors. The parent
decoder must point toward the child concept (refinement); the child decoder
must too (sanity).

Scoring is Tree SAE's operational RANK rule: both decoders must be in the
top-k probe correlations over ALL dictionary features — never a threshold.
Healthy pairs have d_p ⟂ d_c, which caps min(.,.) at 1/sqrt(2) ~ 0.707, so
any tau above that rejects every healthy pair by construction.

Circularity caveat (inherited from Tree SAE):
the probe target 1[f_c > 0] is a SELF-LABEL — a corrupted (absorbed/split)
latent yields a corrupted probe that then validates the corruption. Numbers
from this module are a self-consistency check, not ground truth; report them
as "self-labeled S_res". The negative-class parent composition is reported
per probe because a parent-rich negative class can suppress the shared
parent component of the probe (severity grows with rho_p).
"""

from __future__ import annotations

import torch


def train_probe(
    resid: torch.Tensor,        # [n, d] residual-stream activations
    pos_mask: torch.Tensor,     # [n] bool: child-firing tokens (self-label!)
    seed: int = 0,
    neg_ratio: int = 4,
    max_tokens: int = 20_000,
    steps: int = 300,
    lr: float = 0.05,
    min_neg: int = 10,
) -> torch.Tensor | None:
    """Unit-norm logistic-probe direction separating child-firing tokens from
    sampled negatives. Centering only (no per-dim scaling) so the returned
    direction lives in the original residual basis.

    Returns None when fewer than min_neg negative tokens exist: with too few
    negatives the probe has no classification signal and collapses toward a
    zero direction, so the rank check would score noise. The caller marks such
    children untestable instead of trusting a degenerate direction."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
    neg_idx = torch.nonzero(~pos_mask, as_tuple=False).flatten()
    if neg_idx.numel() < min_neg or pos_idx.numel() == 0:
        return None

    n_pos = min(pos_idx.numel(), max_tokens // (1 + neg_ratio))
    n_neg = min(neg_idx.numel(), n_pos * neg_ratio)
    # permutations are drawn on CPU (deterministic across devices) and moved to
    # wherever the masks live — CUDA tensors reject CPU index tensors here
    perm_p = torch.randperm(pos_idx.numel(), generator=g).to(pos_idx.device)
    perm_n = torch.randperm(neg_idx.numel(), generator=g).to(neg_idx.device)
    pos_idx = pos_idx[perm_p[:n_pos]]
    neg_idx = neg_idx[perm_n[:n_neg]]

    idx = torch.cat([pos_idx, neg_idx])
    X = resid[idx].float()
    y = torch.zeros(idx.numel(), dtype=torch.float32, device=X.device)
    y[: n_pos] = 1.0
    X = X - X.mean(dim=0, keepdim=True)

    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    # class weight balances the neg_ratio-skewed sample
    pos_weight = torch.tensor(float(n_neg) / max(n_pos, 1), device=X.device)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        ) + 1e-4 * (w * w).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        return (w / w.norm().clamp(min=1e-12)).detach()


def sres_rank_check(
    corr: torch.Tensor,         # [F] probe . W_dec[f] over ALL dictionary features
    parent_idx: int,
    child_idx: int,
    top_k: int = 5,
) -> tuple[bool, dict]:
    """Tree SAE rank rule: the edge passes iff BOTH the child decoder and the
    parent decoder are among the top_k probe correlations over all features.
    Returns (passes, detail) with ranks and values for reporting."""
    order = torch.argsort(corr, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel())
    r_p, r_c = int(ranks[parent_idx]), int(ranks[child_idx])
    detail = {
        "parent_rank": r_p,
        "child_rank": r_c,
        "parent_corr": float(corr[parent_idx]),
        "child_corr": float(corr[child_idx]),
        "s_res": float(min(corr[parent_idx], corr[child_idx])),
    }
    return (r_p < top_k) and (r_c < top_k), detail


def sres_scores(
    probe_corr: torch.Tensor,   # [C, pool] each child's probe . every pool decoder
    parent_ids: torch.Tensor,   # [P] pool index of each parent
    child_ids: torch.Tensor,    # [C] pool index of each child
    probe_available: torch.Tensor | None = None,   # [C] bool; False means no probe
) -> torch.Tensor:
    """[P, C] S_res values, min(probe . parent decoder, probe . child decoder), with the child's
    probe: the all-pairs form of `sres_rank_check`'s `s_res`. A child without a probe is NaN."""
    parent_corr = probe_corr[:, parent_ids]                         # [C, P]
    child_corr = probe_corr.gather(1, child_ids.reshape(-1, 1))     # [C, 1]
    out = torch.minimum(parent_corr, child_corr).transpose(0, 1)    # [P, C]
    if probe_available is not None:
        keep = probe_available.to(out.device).reshape(1, -1)
        out = torch.where(keep, out, torch.full_like(out, float("nan")))
    return out


def negative_parent_composition(
    neg_mask: torch.Tensor,     # [n] bool: tokens used as probe negatives
    fires_p: torch.Tensor,      # [n] bool: parent-firing tokens
) -> float:
    """Fraction of the negative class on which the parent fires — reported per
    probe (a parent-rich negative class suppresses the shared parent
    component of a discriminative probe)."""
    n = int(neg_mask.sum())
    return float((neg_mask & fires_p).sum()) / n if n else 0.0
