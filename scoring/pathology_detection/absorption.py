"""Per-token absorption scores.

Reimplements SAEBench's `feature_absorption_calculator.py::calculate_absorption` and
`_is_full_absorption` (commit 8042bb3).

A latent's probe projection on a token is its activation times its decoder's cosine with the probe;
the token's probe projection is h . p_unit. Absorption is probe projection that the concept's main
latents miss and other probe-aligned latents carry.
"""

from __future__ import annotations

import torch

from .settings import FullAbsorptionThresholds, ThresholdSet

_TINY = 1e-12


def latent_probe_cos(W_dec: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """[L] cosine of each raw decoder row with the probe direction; a zero row gives 0."""
    W, p = W_dec.double(), p.double()
    return (W @ p) / (W.norm(dim=1) * p.norm()).clamp_min(_TINY)


def _projections(h: torch.Tensor, z: torch.Tensor, cos: torch.Tensor, p: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    p = p.double()
    p = p / p.norm().clamp_min(_TINY)
    return h.double() @ p, z.double() * cos.double()[None, :]


def absorption_fraction(h: torch.Tensor, z: torch.Tensor, cos: torch.Tensor,
                        p_unit: torch.Tensor, main: list[int], ts: ThresholdSet) -> torch.Tensor:
    """[n] share of each token's probe projection carried by absorbing latents instead of `main`."""
    probe_proj, lat = _projections(h, z, cos, p_unit)
    n, L = lat.shape
    main = [int(j) for j in main]
    main_proj = lat[:, main].sum(dim=1) if main else torch.zeros(n, dtype=lat.dtype)
    allowed = torch.ones(L, dtype=torch.bool)
    if main:
        allowed[main] = False
    allowed &= cos.double() >= ts.probe_cos_min
    cand = torch.where(allowed[None, :] & (lat > 0), lat, torch.zeros_like(lat))
    if ts.max_absorbers is None:
        total = cand.sum(dim=1)
    else:
        total = cand.topk(min(int(ts.max_absorbers), L), dim=1).values.sum(dim=1)
    share = total / probe_proj
    a = torch.minimum(total, probe_proj - main_proj)
    frac = (a / (a + main_proj)).clamp(0.0, 1.0)
    none = (main_proj >= probe_proj) | (share < ts.projection_share_min)
    out = torch.where(main_proj <= 0, torch.ones_like(frac), frac)
    return torch.where(none, torch.zeros_like(out), out)


def full_absorption(h: torch.Tensor, z: torch.Tensor, cos: torch.Tensor, p_unit: torch.Tensor,
                    main: list[int], fa: FullAbsorptionThresholds, topk: int = 10
                    ) -> torch.Tensor:
    """[n] bool: every main latent is off and one probe-aligned latent carries the projection."""
    probe_proj, lat = _projections(h, z, cos, p_unit)
    n, L = lat.shape
    main = [int(j) for j in main]
    main_off = ((z[:, main] < fa.main_off_below).all(dim=1) if main
                else torch.ones(n, dtype=torch.bool))
    top = lat.topk(min(int(topk), L), dim=1).indices[:, 0]
    top_proj = lat[torch.arange(n), top]
    return (main_off & (cos.double()[top] >= fa.top_cos_min) & (probe_proj >= 0)
            & (top_proj / probe_proj >= fa.projection_share_min))
