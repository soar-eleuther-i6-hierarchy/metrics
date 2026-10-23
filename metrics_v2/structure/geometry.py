"""Geometry: does the parent's decoder point toward the child's concept?

Two readings of decoder geometry live here.

1. The probe route (Tree SAE's S_res, rank-scored). A linear probe fitted on the
   child's own firing gives a direction d_c*; the edge passes when both decoders
   rank in the probe's top k. The probe target is a self-label, so this is a
   self-consistency check. Same objects as `metrics.sres`.

2. The decoder route (new). `decoder_cosine` is the plain cosine between the
   two decoder rows. Bussmann et al. (2025) read absorption off exactly this
   matrix in their toy: parent/child similarity is high in a vanilla SAE and the
   matrix is diagonal in a Matryoshka SAE. It needs no probe and no token cache.
   Not calibrated, used by no pipeline; `pathology.absorption_signature` reads it.
"""

from __future__ import annotations

import torch

from metrics.sres import negative_parent_composition as probe_negative_parent_share
from metrics.sres import sres_rank_check as refinement_rank
from metrics.sres import sres_scores as refinement_scores
from metrics.sres import train_probe as child_probe

__all__ = [
    "child_probe",
    "refinement_rank",
    "refinement_scores",
    "probe_negative_parent_share",
    "decoder_cosine",
]


def decoder_cosine(
    W_dec_p: torch.Tensor,   # [P, d] parent decoder rows
    W_dec_c: torch.Tensor,   # [C, d] child decoder rows
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """[P, C] cosine similarity between every parent and child decoder row.

    Rows are normalised here, so unnormalised decoders are fine. A zero row gives
    cosine 0 against everything, not NaN.
    """
    p = W_dec_p.double()
    c = W_dec_c.double()
    p = p / p.norm(dim=1, keepdim=True).clamp(min=eps)
    c = c / c.norm(dim=1, keepdim=True).clamp(min=eps)
    return p @ c.T
