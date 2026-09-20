"""
Feasibility guards for a built toy.

Reject any config whose sampled world would silently disagree with the ground-truth answer
key. Each check raises `ValueError` at build time, so an infeasible config fails loudly
instead of producing a wrong `truth.pt` without erroring. These are structural checks only;
the old energy-feasibility constraints are gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .spec import EDGE_TAU_REFERENCE, ToyConfig
from .strengths import firing_rates, topic_rates
from .tree import token_mass

if TYPE_CHECKING:
    from .tree import Tree

DT = torch.float64


def validate_config(cfg: ToyConfig, tree: Tree) -> None:
    """Raise `ValueError` if the built tree/config is infeasible; return None if OK.

    Called at the end of `build_tree`. Checks only the structural feasibility that the sampler
    and the answer key both rely on.
    """
    # Basic config sanity -- fail fast on degenerate values before any per-feature work (e.g. Z=0 would divide by zero below).
    if cfg.Z < 1:
        raise ValueError(f"Z (number of topics) must be >= 1 (got {cfg.Z})")
    if cfg.vocab < cfg.n_bind_ids:
        raise ValueError(
            f"vocab ({cfg.vocab}) must be >= n_bind_ids ({cfg.n_bind_ids}); the "
            f"token-bound id set would fall outside the vocabulary and never fire")
    if cfg.doc_len <= 0:
        raise ValueError(f"doc_len must be > 0 (got {cfg.doc_len})")
    if cfg.D <= 0:
        raise ValueError(f"D (activation dimension) must be > 0 (got {cfg.D})")
    if cfg.n_topic_registers_per_topic not in (0, 1):
        raise ValueError(
            f"n_topic_registers_per_topic must be 0 or 1 (got {cfg.n_topic_registers_per_topic})")

    if cfg.noise_sigma <= 0:
        raise ValueError(
            f"noise_sigma must be > 0 (got {cfg.noise_sigma}); a zero-noise world "
            f"gives the reconstruction metric a zero residual to divide by")

    if not (0.0 < cfg.strength_spread < 0.5):
        raise ValueError(
            f"strength_spread must be in (0, 0.5) (got {cfg.strength_spread}); "
            f"outside it the firing-strength floor goes non-positive")

    # Causes are read only for roots: a child fires from its parent, so a cause on a child is never used.
    for k in range(tree.F):
        if tree.parent_of(k) is None:
            continue
        if (tree.kappa.get(k, 0.0) > 0.0 or tree.topic.get(k) is not None
                or tree.token_bound.get(k, False) or tree.cause_rate.get(k) is not None):
            raise ValueError(
                f"feature {k} is a non-root with a cause (topic, token binding or cause_rate); "
                f"the sampler draws a child from its parent and ignores the cause")

    # Every parent declares exclusivity, and a non-exclusive parent may not sit under an exclusive one: exclusive children are drawn after all non-exclusive ones, so its children would read an all-False parent column.
    for par, kids in tree.children.items():
        if not kids:
            continue
        if par not in tree.exclusive:
            raise ValueError(f"parent {par} has children but no explicit `exclusive` entry")
        gp = tree.parent_of(par)
        if gp is not None and not tree.exclusive[par] and tree.exclusive.get(gp, False):
            raise ValueError(
                f"non-exclusive parent {par} sits under exclusive parent {gp}; the sampler draws "
                f"exclusive children last, so {par}'s children would never fire")

    # Firing rates must be probabilities; outside (0, 1] `rand < rate` never or always fires.
    for k, rp in tree.root_p.items():
        if not (0.0 < rp <= 1.0):
            raise ValueError(f"feature {k} has root_p {rp}; firing rates must lie in (0, 1]")
    for k in range(tree.F):
        r = tree.cause_rate.get(k)
        if r is not None and not (0.0 < r <= 1.0):
            raise ValueError(f"feature {k} has cause_rate {r}; firing rates must lie in (0, 1]")

    _validate_fixed_rate_causes(cfg, tree)

    # Per-edge firing probability and per-feature loading bounds, keeping the eps_p / eps_alpha safety margins.
    for k in range(tree.F):
        if tree.parent_of(k) is None:
            continue
        pe = tree.p_edge_of(k)
        if pe > 1.0 - cfg.eps_p:
            raise ValueError(
                f"child_p_edge must be <= 1 - eps_p = {1.0 - cfg.eps_p:.4f}; "
                f"feature {k} has p_edge {pe}")
        a = tree.alpha_of(k)
        if a > 1.0 - cfg.eps_alpha:
            raise ValueError(
                f"alpha must be <= 1 - eps_alpha = {1.0 - cfg.eps_alpha:.4f}; "
                f"feature {k} has alpha {a}")

    # Exclusive siblings split one draw by p_edge, so per-edge probs must sum to <= 1 -- above that the sampler truncates the realized rate while firing_rates keeps the nominal one. Non-exclusive parents carry no such budget and are skipped.
    for par, kids in tree.children.items():
        if not tree.exclusive.get(par, False) or not kids:
            continue
        tot = sum(tree.p_edge_of(c) for c in kids)
        if tot > 1.0 + 1e-9:
            detail = ", ".join(f"{c}:{tree.p_edge_of(c):.3f}" for c in kids)
            raise ValueError(
                f"sibling budget: exclusive parent {par} has {len(kids)} children "
                f"with sum p_edge = {tot:.4f} > 1 (children p_edge -> {detail}); the "
                f"sampler would truncate the last sibling's realized rate while "
                f"firing_rates keeps the nominal rate, so truth.pt would disagree "
                f"with the sampled world")

    # Topical admissibility: per-topic firing rate must stay in [0, 1], or the "averages back to exactly p_i" guarantee breaks silently (`rand < rate` just never/always fires).
    p = firing_rates(tree)
    pi = torch.full((cfg.Z,), 1.0 / cfg.Z, dtype=DT)
    for k in range(tree.F):
        if tree.kappa[k] <= 0.0:
            continue
        rates = topic_rates(float(p[k]), tree.kappa[k], tree.topic[k], pi)
        if float(rates.min()) < 0.0 or float(rates.max()) > 1.0:
            raise ValueError(
                f"topical admissibility: feature {k} has per-topic firing rates "
                f"outside [0, 1] (min {float(rates.min()):.4f}, "
                f"max {float(rates.max()):.4f}); reduce kappa or increase Z")

    # Token-bound admissibility: firing rate can't exceed the top-n_bind_ids id set's Zipf mass, or the sampler silently caps the realized rate while firing_rates keeps the nominal p_k.
    if any(tree.token_bound[k] for k in range(tree.F)):
        ranks = torch.arange(1, cfg.vocab + 1, dtype=DT)
        w = ranks ** (-cfg.zipf_s)
        bind_mass = float(w[:cfg.n_bind_ids].sum() / w.sum())
        for k in range(tree.F):
            if not tree.token_bound[k] or tree.cause_rate.get(k) is not None:
                continue
            if float(p[k]) > bind_mass + 1e-9:
                raise ValueError(
                    f"token-bound admissibility: feature {k} has firing rate "
                    f"p = {float(p[k]):.4f} above the top-{cfg.n_bind_ids} Zipf mass "
                    f"{bind_mass:.4f}; the sampler would cap the realized rate below "
                    f"p, so truth.pt would disagree with the sampled world "
                    f"(raise n_bind_ids or lower the token-bound rate)")


def _validate_fixed_rate_causes(cfg: ToyConfig, tree: Tree) -> None:
    """Check roots that fire at a fixed rate inside one cause (token groups, topic registers).

    Each needs exactly one cause, token ids inside the design top-frequency bucket or a valid
    topic, and root_p equal to the cause's design mass times its rate. Given the cause, features
    fire independently, so P(a | b) = cause_rate[a] for two features on one cause; each cause
    must therefore plant containment: a partner and some rate >= the hierarchy cut.
    """
    groups: dict[tuple, list[int]] = {}
    cum = None
    for k in range(tree.F):
        rate = tree.cause_rate.get(k)
        if rate is None:
            continue
        on_tokens, on_topic = tree.token_bound.get(k, False), tree.topic.get(k) is not None
        if on_tokens == on_topic or tree.kappa.get(k, 0.0) > 0.0:
            raise ValueError(
                f"feature {k} has a cause_rate but not exactly one cause (token ids or a topic, "
                f"with no kappa modulation)")
        if on_tokens:
            ids = tree.token_ids.get(k, ())
            if not ids or min(ids) < 0 or max(ids) >= cfg.vocab:
                raise ValueError(f"feature {k} has token ids {ids} outside the vocabulary [0, {cfg.vocab})")
            if cum is None:
                w = torch.arange(1, cfg.vocab + 1, dtype=DT) ** (-cfg.zipf_s)
                cum = torch.cumsum(w / w.sum(), dim=0)
            if float(cum[max(ids)]) > cfg.freq_high_mass:
                raise ValueError(
                    f"feature {k} fires on token id {max(ids)}, outside the design top-frequency "
                    f"bucket (cumulative mass {float(cum[max(ids)]):.4f} > freq_high_mass "
                    f"{cfg.freq_high_mass}); a frequency control would not remove its whole cause")
            want, key = token_mass(cfg, ids) * rate, ("tokens", tuple(sorted(ids)))
        else:
            z = tree.topic[k]
            if not (0 <= z < cfg.Z):
                raise ValueError(f"feature {k} has topic {z} outside [0, {cfg.Z})")
            want, key = rate / cfg.Z, ("topic", z)
        if abs(tree.root_p[k] - want) > 1e-12:
            raise ValueError(
                f"feature {k} has root_p {tree.root_p[k]} but its cause gives {want}; "
                f"firing_rates would disagree with the sampled world")
        groups.setdefault(key, []).append(k)

    for key, feats in groups.items():
        top = max(tree.cause_rate[k] for k in feats)
        if len(feats) < 2 or top < EDGE_TAU_REFERENCE:
            raise ValueError(
                f"cause {key} plants no containment: {len(feats)} feature(s), highest within-cause "
                f"rate {top} (needs a partner and a rate >= {EDGE_TAU_REFERENCE})")
