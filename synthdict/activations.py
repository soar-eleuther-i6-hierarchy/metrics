"""Activations for a synthetic dictionary: per-token NNLS on a planted support.

acts[t, S] = argmin over a >= 0 of ||h_t - a . W_S||^2 + lam ||a||^2, zero off the support S.
A planted latent the fit sets to 0 is off for every `> 0` detector; `zeroed_rate` measures that.
`reconstruction_fvu` is the fit's unexplained variance.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import nnls

from scoring.config import BENCHMARK

_TINY = 1e-12


def nnls_acts(h: torch.Tensor, W_raw: torch.Tensor, support: torch.Tensor,
              lam: float = BENCHMARK.ridge_lambda) -> torch.Tensor:
    """[n, L] float64 non-negative strengths on the support, zero elsewhere.

    `support` is latent-space: one column per decoder row.
    """
    Wd = W_raw.double().cpu().numpy()
    hd = h.double().cpu().numpy()
    n, S = support.shape
    # a feature-space support would return a plausible array built from the wrong rows
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
    """Share of planted-support entries the fit set to <= 0. Latent-space, so a split feature
    counts once per shard."""
    on = support.sum()
    if int(on) == 0:
        return 0.0
    return float((acts[support] <= 0.0).sum()) / float(on)


def reconstruction_fvu(h: torch.Tensor, acts: torch.Tensor, g: torch.Tensor) -> float:
    """`||h - acts @ g||^2 / ||h||^2` with the raw decoder `g`, not mean-centered."""
    h_hat = acts.double() @ g.double()
    err = h.double() - h_hat
    return float((err ** 2).sum() / (h.double() ** 2).sum().clamp_min(_TINY))
