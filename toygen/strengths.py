"""Firing rates and activation strengths.

Firing rate walks down the forest (p_child = p_parent * p_edge), and every feature has the
same mean strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .spec import ToyConfig
from .tree import Tree

DT = torch.float64


@dataclass
class StrengthSpec:
    p: torch.Tensor              # [F] firing rate per feature
    mean_strength: torch.Tensor  # [F] mean active strength, flat at q * sqrt(E0)
    topic_prior: torch.Tensor    # [Z] uniform prior over document topics


def firing_rates(tree: Tree) -> torch.Tensor:
    """Firing rate p_k, walking the forest downward: p_child = p_parent * p_edge."""
    p = torch.zeros(tree.F, dtype=DT)
    for k in tree.topology_ordering:
        par = tree.parent_of(k)
        p[k] = tree.root_p[k] if par is None else p[par] * tree.p_edge_of(k)
    return p


def target_l0(tree: Tree) -> float:
    """Expected L0 (mean active features per token) = sum of firing rates; world.choose_k uses it."""
    return float(firing_rates(tree).sum())


def topic_rates(p_i: float | torch.Tensor, kappa: float, z_i: int | None,
                pi: torch.Tensor) -> torch.Tensor:
    """Per-topic firing rates that average back to `p_i` under a uniform topic prior.

    Centring by `pi` makes the average exact; without it the rates drift yet still look like
    valid probabilities.
    """
    Z = pi.numel()
    if z_i is None or kappa == 0.0:
        return torch.full((Z,), float(p_i), dtype=DT)
    onehot = torch.zeros(Z, dtype=DT)
    onehot[z_i] = 1.0
    return float(p_i) * (1.0 + kappa * (onehot - pi))


def build_strengths(cfg: ToyConfig, tree: Tree) -> StrengthSpec:
    """Firing rates, one mean strength for every feature, and a uniform topic prior.

    mean_strength = q * sqrt(E0) with q = 1 / sqrt(1 + strength_spread^2), so the mean squared
    active strength is E0.
    """
    p = firing_rates(tree)
    q = 1.0 / math.sqrt(1.0 + cfg.strength_spread ** 2)
    mean_strength = torch.full((tree.F,), q * math.sqrt(cfg.E0), dtype=DT)
    topic_prior = torch.full((cfg.Z,), 1.0 / cfg.Z, dtype=DT)
    return StrengthSpec(p=p, mean_strength=mean_strength, topic_prior=topic_prior)
