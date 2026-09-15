"""Honest activations for an arbitrary synthetic dictionary: per-token ridge on a PLANTED support.

Generalizes `oracle_encode` step 2 (`scoring/oracle/validate_metrics.py`): there the support
comes from a tied unit-g JumpReLU gate; here the support is PLANTED (true `A > 0` plus the
corruption's hole transform), which is the whole point — the firing structure is a dial, not
an emergent property.

`acts[t, S] = argmin_a ||h_t - a . W_S||^2 + lam ||a||^2`, zero off the support, same
`lam = 1e-4` convention as `oracle_encode` (RIDGE_LAMBDA — fixed, not tuned to anything).

W1 accounting: an unconstrained ridge can assign a NONPOSITIVE magnitude on a planted-support
entry, silently turning planted firing OFF for every `> 0`-thresholded detector. That rate is
measured (`support_flip_rate`), gated in Phase 0, and stamped into every artifact's meta —
realized support is what the detectors see, exactly as for a real SAE.
"""

from __future__ import annotations

import torch

from scoring.oracle.validate_metrics import RIDGE_LAMBDA


def ridge_acts(h: torch.Tensor, W_raw: torch.Tensor, support: torch.Tensor,
               lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """[n, S] float64 magnitudes: per-token ridge least-squares on the support, zero elsewhere.

    `support` is LATENT-space: one column per decoder row, which is one per feature only while
    the planted map is 1-1.
    """
    Wd = W_raw.double()
    hd = h.double()
    n, S = support.shape
    # The output width comes from the SUPPORT, so a feature-space support against an [L, D]
    # dictionary would return a plausible [n, F] array built from the first F rows — no error,
    # every metric finite, all of them scoring the wrong latents.
    if int(Wd.shape[0]) != S:
        raise ValueError(
            f"dictionary width {int(Wd.shape[0])} != support width {S}; the support must be "
            f"latent-space (expand it through the planted map before solving)")
    acts = torch.zeros(n, S, dtype=torch.float64, device=Wd.device)
    for t in range(n):
        idx = support[t].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        Ws = Wd[idx]                                       # [k, D]
        gram = Ws @ Ws.transpose(0, 1)
        gram = gram + lam * torch.eye(idx.numel(), dtype=torch.float64, device=Wd.device)
        acts[t, idx] = torch.linalg.solve(gram, Ws @ hd[t])
    return acts


def support_flip_rate(acts: torch.Tensor, support: torch.Tensor) -> float:
    """Fraction of planted-support entries whose ridge magnitude is <= 0 (fire_thresh = 0.0,
    the detectors' firing convention) — i.e. planted firing the solve turned off.

    LATENT-space, like the support it is given: at L != F this is a rate over decoder rows, not
    over features, and a split feature contributes one entry per shard.
    """
    on = support.sum()
    if int(on) == 0:
        return 0.0
    flipped = (acts[support] <= 0.0).sum()
    return float(flipped) / float(on)
