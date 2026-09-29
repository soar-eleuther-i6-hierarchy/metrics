"""Rebuild a toy world from a checkpoint's saved config, and orient decoder rows without ground truth."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from toygen import geometry, labels, sample, spec, strengths
from toygen import tree as tree_mod

from scoring.core.recovery import child_direction_dispersion

_TINY = 1e-12


@dataclass(frozen=True)
class WorldBundle:
    """One regenerated toy world: activations, true coefficients and directions, and the answer key."""

    A: torch.Tensor                # [n, F] true coefficients (g basis)
    Atilde: torch.Tensor           # [n, F] healthy coefficients (u basis)
    h: torch.Tensor                # [n, D] activations the SAE saw
    g: torch.Tensor                # [F, D] concept directions
    u: torch.Tensor                # [F, D] residual directions
    Lam: torch.Tensor              # [F, F] change of basis, g == Lam.T @ u
    tree: tree_mod.Tree
    pair_labels: torch.Tensor      # [F, F] int8 class index per ordered pair (toygen.labels)
    p: torch.Tensor                # [F] firing rate per feature
    CONT: list[tuple[int, int]]    # direct parent -> child edges
    ISA: list[tuple[int, int]]     # the alpha > 0 (is-a) subset of CONT
    cfg: spec.ToyConfig
    tokens: torch.Tensor | None = None   # [n] token id per row


def regenerate_world(resolved_config: dict, sample_seed: int, n_tokens: int) -> WorldBundle:
    """Rebuild a checkpoint's toy world and draw `n_tokens` fresh tokens under `sample_seed`.

    Geometry comes from `cfg.seed` in `resolved_config`, as in training. Unknown config keys are dropped."""
    known = {f.name for f in dataclasses.fields(spec.ToyConfig)}
    cfg = spec.ToyConfig(**{k: v for k, v in resolved_config.items() if k in known})

    tree = tree_mod.build_tree(cfg)
    strength = strengths.build_strengths(cfg, tree)
    geo = geometry.build_directions(cfg, tree, seed=cfg.seed)
    world = sample.sample_world(cfg, tree, strength, geo, n_tokens=n_tokens, seed=sample_seed)

    cont = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa = [(p, c) for (p, c) in cont if tree.alpha_of(c) > 0.0]
    return WorldBundle(
        A=world.A, Atilde=world.Atilde, h=world.h,
        g=geo.g, u=geo.u, Lam=geo.Lam, tree=tree,
        pair_labels=labels.pair_label(tree), p=strength.p,
        CONT=cont, ISA=isa, cfg=cfg, tokens=world.tokens,
    )


def dispersion_clean_floor(bundle: WorldBundle, m: int = 5) -> torch.Tensor:
    """Child-direction dispersion of the true `g` used as the decoder: the floor for this config.

    The floor moves with the config, so compare a trained dispersion against it, not a fixed number."""
    gn = bundle.g / bundle.g.norm(dim=1, keepdim=True).clamp_min(_TINY)
    F = bundle.g.shape[0]
    isa_children = [c for _, c in bundle.ISA]
    return child_direction_dispersion(gn, bundle.g, torch.arange(F), isa_children, m=m)


def check_world_invariants(bundle: WorldBundle, atol: float = 1e-6) -> None:
    """Raise unless `A @ g == Atilde @ u` and `g == Lam.T @ u`, i.e. the regenerated world is consistent."""
    err1 = float((bundle.A @ bundle.g - bundle.Atilde @ bundle.u).abs().max())
    err2 = float((bundle.g - bundle.Lam.transpose(0, 1) @ bundle.u).abs().max())
    if err1 > atol or err2 > atol:
        raise ValueError(
            f"world invariants violated: |A@g - Atilde@u|={err1:.2e}, "
            f"|g - Lam.T@u|={err2:.2e} (atol={atol:.1e})")


def signed_normalized_decoder(W_dec: torch.Tensor, acts: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Unit-normalize decoder rows, each signed by `sign(sum_i acts[i, j] * (h_i . w_j))`.

    Uses no ground truth and does not depend on the input row's sign."""
    hw = h @ W_dec.transpose(0, 1)                       # [n, S]
    s = (acts * hw).sum(dim=0)                            # [S]
    sign = torch.ones_like(s)
    sign[s < 0] = -1.0
    unit = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)
    return unit * sign[:, None]
