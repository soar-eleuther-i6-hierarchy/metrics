"""Absorption signature (new, uncalibrated, used by no pipeline).

Absorption, as injected in the paper's Section 4.1: a firing hole (the parent is
set to 0 on a share eta of the tokens where it co-fires with the child) and a
decoder carry (the child's decoder gains beta times the parent's direction).

The hole lowers R = P(p | c). The carry raises the decoder cosine between p and
c. A healthy edge is the opposite: R high, cosine low (healthy pairs are close
to orthogonal, which is why the rank rule caps at 1/sqrt(2)). So the pair that
coverage would *drop* and geometry would *keep* is the absorption candidate:

    flag[p, c] = cos[p, c] >= cos_min  and  R[p, c] < tau  and  supported

This is the only route to absorption that does not go through the candidate
set, because an absorbed edge has R = 0.00 there and never enters it. It is a
report, not a cut: nothing downstream reads it. Calibration against the
absorption toy (eta, beta sweep) is still to do.
"""

from __future__ import annotations

import torch


def absorption_signature(
    R: torch.Tensor,          # [P, C] reverse coverage P(p | c)
    cos: torch.Tensor,        # [P, C] decoder cosine (structure.decoder_cosine)
    *,
    tau: float,               # the containment cut the edge would have to clear
    cos_min: float,           # decoder alignment above which a pair counts as aligned
    support: torch.Tensor | None = None,   # [P, C] bool: both endpoints measurable
) -> dict[str, torch.Tensor]:
    """{"flag" [P, C] bool, "R", "cos"}: aligned decoders with failed containment."""
    if R.shape != cos.shape:
        raise ValueError(f"R {tuple(R.shape)} and cos {tuple(cos.shape)} describe different frames")
    flag = (cos.double() >= float(cos_min)) & (R.double() < float(tau))
    if support is not None:
        flag = flag & support.bool()
    return {"flag": flag, "R": R, "cos": cos}
