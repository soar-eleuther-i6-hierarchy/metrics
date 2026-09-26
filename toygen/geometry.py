"""Feature directions: residual directions `u`, concept directions `g`, and the change of basis `Lam`.

`Lam` is read off the construction, not computed as `U^T G`; the two differ once siblings are involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .spec import ToyConfig
from .tree import Tree

DT = torch.float64


@dataclass
class Geometry:
    u: torch.Tensor            # [F, D] residual directions
    g: torch.Tensor            # [F, D] concept directions
    Lam: torch.Tensor          # [F, F] constructive change of basis
    coherence: float           # realised max |cos(u_a, u_b)| after orthogonalisation
    # coherence <= max_unrelated_cos; a soft flag, expected False when F >> D
    coherence_ok: bool


def _repel(x: torch.Tensor, max_unrelated_cos: float, steps: int = 60) -> torch.Tensor:
    """Push unit vectors apart until pairwise |cos| approaches `max_unrelated_cos`.

    Rejection sampling fails when F >> D, so this repels instead; the realised value is reported,
    not asserted.
    """
    x = x / x.norm(dim=1, keepdim=True)
    for _ in range(steps):
        gram = x @ x.T
        gram.fill_diagonal_(0.0)
        over = gram.abs() > max_unrelated_cos
        if not over.any():
            break
        push = (gram * over.to(DT)) @ x
        x = x - 0.10 * push
        x = x / x.norm(dim=1, keepdim=True)
    return x


def build_directions(cfg: ToyConfig, tree: Tree, seed: int | None = None) -> Geometry:
    """Build `u`, `g` and `Lam`: the first of 8 draws under `max_unrelated_cos`, else the least coherent."""
    gen = torch.Generator().manual_seed(cfg.seed if seed is None else seed)
    F, D = tree.F, cfg.D

    best: Geometry | None = None
    for _ in range(8):
        # step 1: low-coherence seed directions
        ut = torch.randn(F, D, generator=gen, dtype=DT)
        ut = _repel(ut, cfg.max_unrelated_cos)

        u = torch.zeros(F, D, dtype=DT)
        g = torch.zeros(F, D, dtype=DT)
        Lam = torch.zeros(F, F, dtype=DT)

        # step 2, parents first: u_k is ut_k orthogonalised against the ancestors' g
        for k in tree.topology_ordering:
            anc = sorted(tree.ancestors[k])
            if anc:
                basis = torch.linalg.qr(torch.stack([g[a] for a in anc], dim=1))[0]
                r = ut[k] - basis @ (basis.T @ ut[k])
            else:
                r = ut[k].clone()
            if float(r.norm()) < 1e-8:
                raise ValueError(f"u_{k} undefined: the drawn direction lies in its ancestor span")
            u[k] = r / r.norm()

            # g_k mixes the parent's residual direction (weight alpha) with u_k.
            par = tree.parent_of(k)
            a_k = float(tree.alpha_of(k))
            if par is None or a_k == 0.0:
                g[k] = u[k]                                   # root, or firing_only edge: no parent mix
                Lam[k, k] = 1.0
            else:
                w = u[par]                                    # unit-norm parent residual direction
                g[k] = a_k * w + math.sqrt(1.0 - a_k ** 2) * u[k]
                Lam[par, k] = a_k
                Lam[k, k] = math.sqrt(1.0 - a_k ** 2)

        gram = u @ u.T
        gram.fill_diagonal_(0.0)
        coherence = float(gram.abs().max())
        cand = Geometry(u=u, g=g, Lam=Lam, coherence=coherence, coherence_ok=coherence <= cfg.max_unrelated_cos)
        if coherence <= cfg.max_unrelated_cos:
            return cand
        if best is None or coherence < best.coherence:
            best = cand

    assert best is not None
    return best
