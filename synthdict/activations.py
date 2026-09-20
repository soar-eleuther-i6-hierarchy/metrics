"""Activations for an arbitrary synthetic dictionary: per-token NNLS on a PLANTED support.

The support is planted (true `A > 0` plus the damage's own firing transform), so which
latents fire is a dial, not an emergent property. For each token the strengths of the firing
latents are the best non-negative fit of that token's activation vector from their decoder
rows:

  acts[t, S] = argmin_{a >= 0} ||h_t - a . W_S||^2 + lam ||a||^2,   zero off the support

Least squares on a known support is sparse coding's oracle estimator (Candes & Tao), and SAE
codes are non-negative, so this is an ideal encoder for the dictionary it is given. The
regularizer is the same lam = 1e-4 as `oracle_encode` (RIDGE_LAMBDA), solved as the
augmented system [W_S^T; sqrt(lam) I] a = [h_t; 0].

A planted latent the fit sets to 0 is off for every `> 0`-thresholded detector; that rate is
measured (`zeroed_rate`) and stamped into every artifact.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import nnls

from scoring.oracle.validate_metrics import RIDGE_LAMBDA


def nnls_acts(h: torch.Tensor, W_raw: torch.Tensor, support: torch.Tensor,
              lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """[n, L] float64 non-negative strengths on the support, zero elsewhere.

    `support` is LATENT-space: one column per decoder row.
    """
    Wd = W_raw.double().cpu().numpy()
    hd = h.double().cpu().numpy()
    n, S = support.shape
    # The output width comes from the SUPPORT, so a feature-space support against an [L, D]
    # dictionary would return a plausible array built from the wrong rows.
    if int(Wd.shape[0]) != S:
        raise ValueError(
            f"dictionary width {int(Wd.shape[0])} != support width {S}; the support must be "
            f"latent-space (expand it through the planted map before solving)")
    D = Wd.shape[1]
    root_lam = math.sqrt(float(lam))
    sup = support.cpu().numpy()
    acts = np.zeros((n, S), dtype=np.float64)
    b = np.zeros(D + S, dtype=np.float64)
    for t in range(n):
        idx = np.flatnonzero(sup[t])
        k = idx.size
        if k == 0:
            continue
        M = np.zeros((D + k, k), dtype=np.float64)
        M[:D] = Wd[idx].T
        M[D:] = root_lam * np.eye(k)
        b[:D] = hd[t]
        acts[t, idx] = nnls(M, b[:D + k])[0]
    return torch.from_numpy(acts)


def zeroed_rate(acts: torch.Tensor, support: torch.Tensor) -> float:
    """Fraction of planted-support entries whose strength is <= 0 (fire_thresh = 0.0, the
    detectors' firing convention): planted firing the fit turned off. Latent-space, so a
    split feature contributes one entry per shard."""
    on = support.sum()
    if int(on) == 0:
        return 0.0
    return float((acts[support] <= 0.0).sum()) / float(on)
