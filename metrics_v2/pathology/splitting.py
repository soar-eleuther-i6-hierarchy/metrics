"""Splitting: one feature carried by several latents. Two forms, two detectors.

Copies: the latents fire on the SAME tokens. Real children partition the
parent's firing set, copies co-fire, so firing overlap sees them:

    J(ci, cj) = cofire / (fire_i + fire_j - cofire)     mean over sibling pairs
    within-parent form: the same, restricted to the parent's tokens
    co-extensive pairs: R >= tau both ways inside one block (a rename)

Shards: the latents share one DIRECTION and partition the tokens
(`synthdict.corruptions.split`). Firing overlap reads shards at 0.00, below
even unrelated pairs, so it is blind to them by construction; decoder cosine
reads them at 1.00 against at most 0.12 for unrelated directions in the
synthetic world (`metrics_v2/validation/split_detectors.py`). `direction_duplicates`
is that detector. Not calibrated on a trained SAE, where equal directions become
near-equal; used by no pipeline.

Same objects as `metrics.sibling_redundancy` / `metrics.in_block`, plus the new one.
"""

from __future__ import annotations

import torch

from metrics.in_block import duplicate_pairs as coextensive_pairs
from metrics.sibling_redundancy import parent_conditioned_redundancy as sibling_overlap_within_parent
from metrics.sibling_redundancy import sibling_redundancy as sibling_overlap

__all__ = ["sibling_overlap", "sibling_overlap_within_parent", "coextensive_pairs",
           "direction_duplicates"]


def direction_duplicates(
    W_dec: torch.Tensor,        # [L, d] decoder rows of one block
    *,
    cos_min: float,             # 0.9 recovered every planted shard pair with no false positive
    eps: float = 1e-12,
) -> dict[str, object]:
    """Pairs of latents in one block whose decoder directions (almost) coincide.

    Returns {"cosine" [L, L] float64 with NaN on the diagonal, "pairs" [(i, j), ...] with i < j
    and cosine >= cos_min}. Sign is ignored: a shard and its negation carry the same direction.
    """
    W = W_dec.double()
    W = W / W.norm(dim=1, keepdim=True).clamp(min=eps)
    cos = (W @ W.T).abs()
    cos.fill_diagonal_(float("nan"))
    iu = torch.triu_indices(W.shape[0], W.shape[0], offset=1)
    hit = cos[iu[0], iu[1]] >= float(cos_min)
    pairs = [(int(i), int(j)) for i, j in zip(iu[0][hit].tolist(), iu[1][hit].tolist())]
    return {"cosine": cos, "pairs": pairs}
