"""
Metric 2a - Reconstruction-ablation CONTRIBUTION FILTER (Tree-SAE-INSPIRED
baseline).

Honest-name note: this is NOT Tree SAE's S_res.
It tests that both features carry reconstruction mass on the child's tokens —
a contribution filter. It tests neither refinement (P2) nor semantic
coherence (P3): two strong but unrelated co-firing features pass both sides,
and its pass rate scales with block energy, mechanically suppressing deep
blocks. The probe-based, rank-scored S_res proper lives in metrics/sres.py
(second pass). Keep this one as a cheap baseline.

The underlying idea: a parent-child edge should IMPROVE reconstruction, not
just co-activate. We test that on the child's firing tokens:

    ablating the parent from the SAE reconstruction increases error, and
    ablating the child does too (the child adds something beyond the parent).

Ablating feature f on a token only changes the reconstruction by its own
contribution a_f * d_f, so the error delta has a closed form:

    err_without_f - err  =  2 a_f <d_f, x - x_hat>  +  a_f^2 ||d_f||^2
                         =: g_f  (per token; 0 when f doesn't fire)

The streaming pass accumulates, per candidate edge (p, c), sums over the
child's firing tokens of: err (base), g_p, g_c. This module turns those sums
into pass/fail per edge.

Coverage cannot tell a genuine refinement from a frequency coincidence;
this metric can, because a coincidental parent contributes ~nothing to
reconstructing the child's tokens.
"""

from __future__ import annotations

import torch


def edge_reconstruction_condition(
    err_sum_c: torch.Tensor,     # [C]    sum of base recon error over c's firing tokens
    g_parent_sum: torch.Tensor,  # [P, C] sum over c's firing tokens of g_p
    g_child_sum: torch.Tensor,   # [C]    sum over c's firing tokens of g_c
    rel_gain_min: float = 0.01,
    *,
    undefined: float | None = None,
) -> dict[str, torch.Tensor]:
    """Relative reconstruction gains and the pass mask, all [P, C] (or [C]).

    parent_gain[p, c] = (sum g_p over c-tokens) / (sum err over c-tokens)
        how much worse reconstruction on the child's tokens gets (relatively)
        if the parent is ablated.
    child_gain[c]     = same for ablating the child itself.

    An edge passes when BOTH gains >= rel_gain_min.
    With `undefined` given, a child with zero error sum gets that value instead of a clamped
    denominator, and a tiny error sum keeps its true ratio.
    """
    denom = err_sum_c.double().clamp(min=1e-12)          # [C]
    if undefined is not None:
        denom = err_sum_c.double()
    parent_gain = g_parent_sum.double() / denom.unsqueeze(0)   # [P, C]
    child_gain = g_child_sum.double() / denom                  # [C]
    if undefined is not None:
        defined = denom > 0
        parent_gain = torch.where(defined.unsqueeze(0), parent_gain, undefined)
        child_gain = torch.where(defined, child_gain, undefined)

    passes = (parent_gain >= rel_gain_min) & (child_gain >= rel_gain_min).unsqueeze(0)
    return {
        "parent_gain": parent_gain,
        "child_gain": child_gain,
        "passes": passes,
    }


@torch.no_grad()
def per_token_ablation_gain(
    feats: torch.Tensor,      # [n, D] SAE feature activations (post-nonlinearity)
    resid_err: torch.Tensor,  # [n, d_model] x - x_hat
    W_dec: torch.Tensor,      # [D, d_model] decoder directions
) -> torch.Tensor:
    """g[n, D]: per-token error increase from ablating each feature.

    Used by the streaming pass (real SAE) and the synthetic tests alike.
        g_f = 2 a_f <d_f, x - x_hat> + a_f^2 ||d_f||^2
    """
    dot = resid_err @ W_dec.T                         # [n, D]
    d_norm2 = (W_dec * W_dec).sum(dim=1)              # [D]
    return 2.0 * feats * dot + feats.pow(2) * d_norm2
