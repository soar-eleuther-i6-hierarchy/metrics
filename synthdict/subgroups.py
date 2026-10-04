"""Sub-groups planted on split features, so each split piece follows a group visible in h.

A split feature f fires as g_f plus, on tokens of sub-group j, a component s * u_{f,j}, with u_{f,.}
orthonormal and orthogonal to g_f. Piece j's decoder row is unit(g_f + s * u_{f,j}) and it fires
exactly on sub-group j, so the split dictionary reconstructs the planted world. Sub-groups are the
token partition the split has always used; k = 1 or s = 0 plants nothing.
"""

from __future__ import annotations

import dataclasses

import torch

# Private RNG offsets, away from the other seed offsets in `synthdict.corruptions`.
SHARD_SEED_OFFSET = 41_213
SUBGROUP_SEED_OFFSET = 19_391


def shard_generator_seed(world_seed: int, sample_seed: int, feature: int) -> int:
    """Per-(world, draw, feature) seed for the token->piece partition."""
    return (SHARD_SEED_OFFSET
            + 1_000_003 * int(world_seed)
            + 7_919 * int(sample_seed)
            + 613 * int(feature))


def subgroup_generator_seed(world_seed: int, feature: int) -> int:
    """Per-(world, feature) seed for the sub-group directions; every draw shares them."""
    return SUBGROUP_SEED_OFFSET + 1_000_003 * int(world_seed) + 613 * int(feature)


def split_partition(fires: torch.Tensor, k: int, world_seed: int, sample_seed: int,
                    feature: int) -> torch.Tensor:
    """[n] long: the piece in 0..k-1 of each firing token, -1 elsewhere. Equal shares of a seeded
    permutation, with cumulative bounds so the pieces sum to the firing count exactly."""
    tok = fires.nonzero(as_tuple=True)[0]
    n_f = int(tok.numel())
    gen = torch.Generator().manual_seed(shard_generator_seed(world_seed, sample_seed, feature))
    perm = torch.randperm(n_f, generator=gen)
    bounds, cum = [], 0.0
    for s in [1.0 / int(k)] * int(k):
        cum += s
        bounds.append(int(round(cum * n_f)))
    bounds[-1] = n_f
    piece = torch.full((int(fires.shape[0]),), -1, dtype=torch.long, device=fires.device)
    start = 0
    for i, end in enumerate(bounds):
        piece[tok[perm[start:end]]] = i
        start = end
    return piece


def subgroup_directions(g_f: torch.Tensor, k: int, world_seed: int, feature: int
                        ) -> torch.Tensor:
    """[k, D] float64 rows, orthonormal and orthogonal to g_f (QR of [g_f, Gaussian draws])."""
    g = g_f.double()
    D = int(g.shape[0])
    if not 1 <= int(k) <= D - 1:
        raise ValueError(f"k = {k} sub-group directions do not fit beside g_f in D = {D}")
    gen = torch.Generator().manual_seed(subgroup_generator_seed(world_seed, feature))
    M = torch.cat([(g / g.norm())[:, None],
                   torch.randn(D, int(k), generator=gen, dtype=torch.float64)], dim=1)
    Q = torch.linalg.qr(M)[0]
    return Q[:, 1:].T.contiguous()


def plant_subgroups(bundle, corruption, world_seed: int, sample_seed: int):
    """The world with each split feature's sub-group component added to h; the same object when
    nothing is planted (no split, k = 1 or s = 0)."""
    if corruption is None or corruption.kind != "split":
        return bundle
    k, s = int(corruption.dials.k), float(corruption.dials.subgroup_strength)
    if k < 2 or s == 0.0:
        return bundle
    h = bundle.h.clone()
    for f in corruption.corrupted_features:
        piece = split_partition(bundle.A[:, f] > 0, k, world_seed, sample_seed, f)
        on = (piece >= 0).nonzero(as_tuple=True)[0]
        U = subgroup_directions(bundle.g[f], k, world_seed, f).to(h.dtype)
        h[on] += (bundle.A[on, f] * s)[:, None] * U[piece[on]]
    return dataclasses.replace(bundle, h=h)
