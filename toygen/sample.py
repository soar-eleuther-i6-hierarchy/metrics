"""Sampler: documents, topics, token ids, coefficients and activations.

Containment and sibling exclusivity hold on every token, and firing rates match their design
as long as each exclusive parent's child p_edges sum to <= 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import Geometry
from .spec import ToyConfig
from .strengths import StrengthSpec, topic_rates
from .tree import Tree

DT = torch.float64


@dataclass
class World:
    A: torch.Tensor            # [n, F] true coefficients
    Atilde: torch.Tensor       # [n, F] coefficients re-expressed in the concept basis (= A @ Lam.T)
    h: torch.Tensor            # [n, D] activations
    noise: torch.Tensor        # [n, D]
    tokens: torch.Tensor       # [n] token ids
    docs: torch.Tensor         # [n] document index
    topics: torch.Tensor       # [n] topic of the token's document
    token_sets: dict[int, torch.Tensor]


def _zipf_probs(vocab: int, s: float) -> torch.Tensor:
    r = torch.arange(1, vocab + 1, dtype=DT)
    w = r ** (-s)
    return w / w.sum()


def _strength(n: int, mean_strength: float, cv: float, gen: torch.Generator) -> torch.Tensor:
    """Lognormal strengths with mean `mean_strength` and sd `cv * mean_strength`.

    Lognormal, not Gamma: `torch.distributions.Gamma` takes no `generator`.
    """
    if n == 0:
        return torch.zeros(0, dtype=DT)
    sig2 = math.log(1.0 + cv ** 2)
    mean_log = math.log(mean_strength) - sig2 / 2.0
    z = torch.randn(n, generator=gen, dtype=DT)
    return torch.exp(mean_log + math.sqrt(sig2) * z)


def sample_world(cfg: ToyConfig, tree: Tree, strengths: StrengthSpec, geo: Geometry, n_tokens: int, seed: int | None = None) -> World:
    # +1000 keeps this stream apart from geometry's when both get the same seed
    gen = torch.Generator().manual_seed((cfg.seed if seed is None else seed) + 1000)
    F = tree.F

    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be positive; got {n_tokens}")
    n_docs = math.ceil(n_tokens / cfg.doc_len)
    n = n_tokens
    docs = torch.arange(n, dtype=torch.long) // cfg.doc_len
    doc_topic = torch.randint(0, cfg.Z, (n_docs,), generator=gen)
    topics = doc_topic[docs]

    probs = _zipf_probs(cfg.vocab, cfg.zipf_s)
    tokens = torch.multinomial(probs, n, replacement=True, generator=gen)

    # token-bound pairs share one top-frequency id set; token groups carry their own ids
    token_sets: dict[int, torch.Tensor] = {
        k: torch.tensor(tree.token_ids[k], dtype=torch.long) for k in range(F) if tree.token_bound[k]
    }

    fires = torch.zeros(n, F, dtype=torch.bool)

    def root_fire(k: int) -> torch.Tensor:
        p_k = float(strengths.p[k])
        rate = tree.cause_rate.get(k)
        if tree.token_bound[k]:
            ids = token_sets[k]
            inside = torch.isin(tokens, ids)
            if rate is not None:                          # token group: fixed rate inside the cause
                return inside & (torch.rand(n, generator=gen, dtype=DT) < rate)
            frac = float(inside.double().mean())
            r = min(1.0, p_k / max(frac, 1e-12))
            return inside & (torch.rand(n, generator=gen, dtype=DT) < r)
        if rate is not None:                              # topic register: fixed rate inside its topic
            return (topics == tree.topic[k]) & (torch.rand(n, generator=gen, dtype=DT) < rate)
        if tree.kappa[k] > 0.0:
            rates = topic_rates(p_k, tree.kappa[k], tree.topic[k], strengths.topic_prior)
            return torch.rand(n, generator=gen, dtype=DT) < rates[topics]
        return torch.rand(n, generator=gen, dtype=DT) < p_k

    for k in tree.topology_ordering:
        par = tree.parent_of(k)
        if par is None:
            fires[:, k] = root_fire(k)
            continue
        if tree.exclusive.get(par, False):
            continue                                      # handled per-parent below
        fires[:, k] = fires[:, par] & (torch.rand(n, generator=gen, dtype=DT) < tree.p_edge_of(k))

    # exclusive siblings: one uniform draw per token, split into p_edge-sized bands.
    # Must run in topological order since it reads fires[:, par].
    for par in tree.topology_ordering:
        kids = tree.children.get(par, [])
        if not tree.exclusive.get(par, False) or not kids:
            continue
        u = torch.rand(n, generator=gen, dtype=DT)
        lo = torch.zeros(n, dtype=DT)
        for c in kids:
            hi = lo + tree.p_edge_of(c)
            fires[:, c] = fires[:, par] & (u >= lo) & (u < hi)
            lo = hi

    A = torch.zeros(n, F, dtype=DT)
    for k in range(F):
        idx = fires[:, k].nonzero(as_tuple=True)[0]
        if idx.numel():
            A[idx, k] = _strength(idx.numel(), float(strengths.mean_strength[k]), cfg.strength_spread, gen)

    Atilde = A @ geo.Lam.T
    noise = cfg.noise_sigma * torch.randn(n, cfg.D, generator=gen, dtype=DT)
    h = A @ geo.g + noise
    return World(A=A, Atilde=Atilde, h=h, noise=noise, tokens=tokens, docs=docs, topics=topics, token_sets=token_sets)
