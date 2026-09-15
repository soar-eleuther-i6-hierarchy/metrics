"""Latent-side corruption generators — physical dials only, no instrument thresholds.

Four damages, each built to its literature definition:

  absorption   (Chanin et al. 2409.14507) the parent latent fails where it should fire and
               the child latent carries a component of the parent direction.
                 child row      unit(g_c + beta * rbar * g_p)
                 firing hole    parent removed on round(eta * n) of the tokens where it
                                co-fires with any of its absorbed children
                 edges          a deterministic `edge_fraction` of the eligible edges:
                                tree parent -> child, token container -> member,
                                topic register -> member                          L = F
  hedging      (Chanin, Dulka, Garriga-Alonso 2505.11756) too few latents: a child has no
               latent and its parent's latent mixes in the child direction.
                 child row      deleted
                 parent row     unit(g_p + gamma * g_c), gamma = gamma_rel * gamma*
                 edges          a deterministic `edge_fraction` of the tree edges    L < F
  split        (Bricken et al.; Chanin et al.) a feature carried by k latents sharing its
               direction and partitioning its firing tokens; chosen within declared roles.
                                                                                   L > F
  composition  (Anders et al. 2024; Leask et al. 2502.04878) a latent for the CONJUNCTION of
               two independent co-occurring features.
                 own rows       kept
                 combination    unit(g_p + g_r), firing on a share `pi` of the tokens where
                                p and r both fire; p's and r's own latents are off there
                                                                                   L > F

TAUTOLOGY GUARD (tested): this module imports neither `ABSORPTION_CONSTANTS` (the census's
thresholds) nor `scoring.core.registry.CONSTANTS` (the detectors').

Determinism contract — the DICTIONARY is a property of the world, the TOKEN assignments are
per draw, and no stream touches the global RNG:
  * selected edges, features, partners, shard structure and rows depend on (world_seed,
    dials) only, so every draw of one world sees the same dictionary;
  * the hole, the token->shard partition and the combination tokens depend on
    (world_seed, sample_seed, the features involved).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from synthdict.planted import PlantedMap
from toygen.strengths import build_strengths

_TINY = 1e-12

# Dedicated seed offsets for the private RNG streams, away from the repo's other derivations
# (toygen STRUCTURE_SEED_OFFSET=9973, held-out +10000, probe fit +20000). One per stream, so
# two damages sharing a world seed do not select the same edges or features.
EDGE_SEED_OFFSET = 61_211
HOLE_SEED_OFFSET = 77_003
HEDGE_SEED_OFFSET = 34_919
SPLIT_SEED_OFFSET = 52_711
SHARD_SEED_OFFSET = 41_213
COMPOSE_SEED_OFFSET = 88_547
COMBINATION_SEED_OFFSET = 23_417

# What `realized_severity` MEASURES, per damage:
#   edge_cos_parent       one entry per absorbed EDGE — cos(child row, g_p)
#   child_own_component   one entry per hedged EDGE — the parent row's component along the
#                         child's own direction (g_c minus its g_p component, unit)
#   none                  no severity axis; the dose is the dial (split k, composition pi)
SEVERITY_KINDS = ("none", "edge_cos_parent", "child_own_component")

# How `corrupted_pair` is derived:
#   ordered_edge              (p, c) is a damaged edge                        absorption
#   either_endpoint_feature   either feature is damaged                        hedging
#   candidate_parent_feature  the candidate parent (first feature) is damaged  split, composition
PAIR_RULES = ("ordered_edge", "either_endpoint_feature", "candidate_parent_feature")

# Declared candidate-parent roles a split is aimed at. `parent` is any feature with children
# that carries none of the other role tags, so the roles are disjoint.
SPLIT_ROLES = ("parent", "superparent", "dense_parent", "token_container", "topic_register")

# Eligible edge sources per edge damage, in the order they are enumerated.
EDGE_SOURCES = ("tree_edge", "token_group", "topic_register")
EDGE_SOURCES_BY_KIND = {"absorption": EDGE_SOURCES, "hedging": ("tree_edge",)}


def _empty_severity() -> torch.Tensor:
    return torch.zeros(0, dtype=torch.float64)


def _check_fraction(name: str, v: float, allow_zero: bool) -> None:
    lo_ok = float(v) >= 0.0 if allow_zero else float(v) > 0.0
    if not (lo_ok and float(v) <= 1.0):
        rng = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {rng}, got {v}")


@dataclass(frozen=True)
class AbsorptionDials:
    """`rbar` is the parent/child mean-magnitude ratio; 1.0 is analytic for this generator
    (toygen/strengths.py builds a FLAT mean strength)."""

    KIND = "absorption"

    beta: float
    eta: float
    edge_fraction: float = 1.0
    rbar: float = 1.0

    def __post_init__(self) -> None:
        if float(self.beta) < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        _check_fraction("eta", self.eta, allow_zero=True)
        _check_fraction("edge_fraction", self.edge_fraction, allow_zero=False)
        if float(self.rbar) <= 0.0:
            raise ValueError(f"rbar must be positive, got {self.rbar}")


@dataclass(frozen=True)
class HedgingDials:
    """`gamma_rel` scales the reference mixing gamma* (see `hedging_gamma_star`); 0.0 is a
    plain missing child."""

    KIND = "hedging"

    gamma_rel: float
    edge_fraction: float = 1.0

    def __post_init__(self) -> None:
        if float(self.gamma_rel) < 0.0:
            raise ValueError(f"gamma_rel must be non-negative, got {self.gamma_rel}")
        _check_fraction("edge_fraction", self.edge_fraction, allow_zero=False)


@dataclass(frozen=True)
class SplitDials:
    """One feature carried by `k` latents that share its direction and partition its firing.

    `roles` names the candidate-parent populations split; `fraction` applies within each role.
    `skew` tilts the token shares geometrically; 0.0 is an equal split.
    """

    KIND = "split"

    k: int
    roles: tuple[str, ...]
    skew: float = 0.0
    fraction: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))    # the CLI hands a list
        if isinstance(self.k, bool) or not isinstance(self.k, int):
            raise ValueError(f"k must be an int, got {self.k!r}")
        if int(self.k) < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not (0.0 <= float(self.skew) < 1.0):
            raise ValueError(f"skew must be in [0, 1), got {self.skew}")
        _check_fraction("fraction", self.fraction, allow_zero=True)
        if not self.roles:
            raise ValueError(f"split needs at least one declared role from {SPLIT_ROLES}")
        unknown = [r for r in self.roles if r not in SPLIT_ROLES]
        if unknown:
            raise ValueError(f"unknown split role(s) {unknown}; roles are {SPLIT_ROLES}")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError(f"a split role is declared twice: {self.roles}")


@dataclass(frozen=True)
class CompositionDials:
    """`pi` is the share of co-firing tokens carried by the combination latent (0.0 = never);
    `fraction` is the share of the possible partner pairs composed."""

    KIND = "composition"

    pi: float
    fraction: float = 1.0

    def __post_init__(self) -> None:
        _check_fraction("pi", self.pi, allow_zero=True)
        _check_fraction("fraction", self.fraction, allow_zero=False)


@dataclass(frozen=True)
class Corruption:
    """One synthetic dictionary's corruption record.

    dials              the damage's own dial dataclass; its `KIND` names the damage
    W_raw              [L, D] float64 decoder rows (unit-norm; undamaged rows are g unchanged)
    planted_map        the feature->latent correspondence this dictionary was built with
    corrupted_edges    ordered (parent, child) edges — absorption and hedging
    corrupted_features the features the pair rule reads
    hedged_children    features left with no latent — hedging
    composition_pairs  (p, r) pairs sharing a combination latent — composition
    realized_severity  the severity axis, whose population `severity_kind` declares
    details            derived values recorded with the run (JSON-able)
    """

    dials: object
    W_raw: torch.Tensor
    planted_map: PlantedMap
    corrupted_edges: tuple[tuple[int, int], ...] = ()
    corrupted_features: tuple[int, ...] = ()
    hedged_children: tuple[int, ...] = ()
    composition_pairs: tuple[tuple[int, int], ...] = ()
    realized_severity: torch.Tensor = field(default_factory=_empty_severity)
    severity_kind: str = "none"
    corrupted_pair_rule: str = "ordered_edge"
    details: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """The damage name, DERIVED from the dials type, so it cannot disagree with them."""
        return type(self.dials).KIND

    def touched_features(self) -> tuple[int, ...]:
        """Every feature whose row or firing the damage changed: the damaged features plus both
        endpoints of every damaged edge. A pair touching one is not an intact control."""
        return tuple(sorted(set(self.corrupted_features)
                            | {int(x) for e in self.corrupted_edges for x in e}))

    def __post_init__(self) -> None:
        if self.severity_kind not in SEVERITY_KINDS:
            raise ValueError(f"unknown severity_kind {self.severity_kind!r}; "
                             f"registered populations are {SEVERITY_KINDS}")
        if self.corrupted_pair_rule not in PAIR_RULES:
            raise ValueError(f"unknown corrupted_pair_rule {self.corrupted_pair_rule!r}; "
                             f"registered rules are {PAIR_RULES}")
        if int(self.W_raw.shape[0]) != int(self.planted_map.n_latents):
            raise ValueError(
                f"the dictionary has {int(self.W_raw.shape[0])} rows but the planted map "
                f"declares {self.planted_map.n_latents} latents; every gather downstream "
                f"would still succeed and score other latents")
        has_sev = int(self.realized_severity.numel()) > 0
        if has_sev != (self.severity_kind != "none"):
            raise ValueError(
                f"severity_kind={self.severity_kind!r} but {self.realized_severity.numel()} "
                f"severities were recorded; the population of that column must be declared, "
                f"not inferred from its length")


# --------------------------------------------------------------------------
# populations: eligible edges, roles, partners
# --------------------------------------------------------------------------
def _tagged(tree, tag: str) -> list[int]:
    return [k for k in range(tree.F) if tag in tree.tags[k]]


def eligible_edges(tree, kind: str) -> tuple[tuple[tuple[int, int], ...], dict[str, int]]:
    """(edges, count per source) an edge damage may select from, in a fixed order.

    tree_edge       every direct parent -> child edge, in `WorldBundle.CONT` order
    token_group     a token container -> each member of its own group
    topic_register  a topic register -> each member of its own topic
    """
    sources = EDGE_SOURCES_BY_KIND[kind]
    found: dict[str, list[tuple[int, int]]] = {s: [] for s in EDGE_SOURCES}
    found["tree_edge"] = [(int(p), c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]

    def group(k: int) -> set[str]:
        return {t for t in tree.tags[k] if t.startswith("tokgroup")}

    members = _tagged(tree, "token_member")
    found["token_group"] = [(a, b) for a in _tagged(tree, "token_container") for b in members
                            if group(a) == group(b)]
    topic_members = _tagged(tree, "topic_member")
    found["topic_register"] = [(a, b) for a in _tagged(tree, "topic_register")
                               for b in topic_members if tree.topic[a] == tree.topic[b]]
    edges = tuple(e for s in EDGE_SOURCES if s in sources for e in found[s])
    counts = {s: (len(found[s]) if s in sources else 0) for s in EDGE_SOURCES}
    return edges, counts


def split_roles(tree) -> dict[str, tuple[int, ...]]:
    """Features per declared split role, ascending. The roles are disjoint."""
    tag_roles = SPLIT_ROLES[1:]
    out: dict[str, list[int]] = {r: [] for r in SPLIT_ROLES}
    for k in range(tree.F):
        tags = tree.tags[k]
        hit = [r for r in tag_roles if r in tags]
        for r in hit:
            out[r].append(k)
        if tree.children.get(k) and not hit:
            out["parent"].append(k)
    return {r: tuple(v) for r, v in out.items()}


def _root(tree, k: int) -> int:
    return min(a for a in tree.ancestors[k] | {k} if not tree.parents.get(a))


def composition_partners(tree, fraction: float, world_seed: int
                         ) -> tuple[tuple[tuple[int, int], ...], dict]:
    """One-to-one partner pairs, deterministic from the world seed.

    A superparent with a dense parent where the world has both; otherwise a parent with a
    parent of ANOTHER tree. `fraction` is the share of the largest possible set of pairs.
    """
    roles = split_roles(tree)
    sp, dp, par = roles["superparent"], roles["dense_parent"], roles["parent"]
    gen = torch.Generator().manual_seed(int(world_seed) + COMPOSE_SEED_OFFSET)
    if sp and dp:
        rule, n_eligible = "superparent_with_dense_parent", len(sp) + len(dp)
        max_pairs = min(len(sp), len(dp))
        n = max(1, round(float(fraction) * max_pairs))
        a = [sp[i] for i in torch.randperm(len(sp), generator=gen)[:n].tolist()]
        b = [dp[i] for i in torch.randperm(len(dp), generator=gen)[:n].tolist()]
        pairs = list(zip(a, b))
        role_counts = {"superparent": {"n_eligible": len(sp), "n_composed": n},
                       "dense_parent": {"n_eligible": len(dp), "n_composed": n}}
    else:
        rule, n_eligible = "parent_with_other_tree_parent", len(par)
        order = [par[i] for i in torch.randperm(len(par), generator=gen).tolist()]
        groups: dict[int, list[int]] = {}
        for k in order:
            groups.setdefault(_root(tree, k), []).append(k)
        rank = {r: i for i, r in enumerate(groups)}
        max_pairs = (min(len(par) // 2, len(par) - max(len(v) for v in groups.values()))
                     if par else 0)
        if max_pairs == 0:
            raise ValueError(
                "composition: no partner features in this world; composition pairs a "
                "superparent with a dense parent, or a parent with another tree's parent")
        n = max(1, round(float(fraction) * max_pairs))
        pairs = []
        # Pairing from the two trees with the most unpaired parents reaches the largest
        # cross-tree matching; pairing in plain order can strand parents of one tree.
        while len(pairs) < n:
            live = sorted((r for r in groups if groups[r]), key=lambda r: (-len(groups[r]), rank[r]))
            a, b = groups[live[0]].pop(0), groups[live[1]].pop(0)
            pairs.append((min(a, b), max(a, b)))
        role_counts = {"parent": {"n_eligible": len(par), "n_composed": 2 * n}}
    return tuple(pairs), {"partner_rule": rule, "n_eligible_features": n_eligible,
                          "max_pairs": max_pairs, "role_counts": role_counts}


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def select_edges(cont_edges, edge_fraction: float, world_seed: int,
                 offset: int = EDGE_SEED_OFFSET) -> tuple[tuple[int, int], ...]:
    """A deterministic `edge_fraction` subset of the edges, by world seed only (no sample
    seed, by contract: every draw sees the same dictionary). f=1.0 keeps the given order."""
    _check_fraction("edge_fraction", edge_fraction, allow_zero=False)
    edges = [(int(p), int(c)) for p, c in cont_edges]
    if edge_fraction >= 1.0:
        return tuple(edges)
    k = max(1, round(edge_fraction * len(edges)))
    gen = torch.Generator().manual_seed(int(world_seed) + int(offset))
    perm = torch.randperm(len(edges), generator=gen).tolist()
    return tuple(edges[i] for i in sorted(perm[:k]))


def _select_eligible_edges(world, kind: str, edge_fraction: float, world_seed: int,
                           offset: int) -> tuple[tuple[tuple[int, int], ...], dict]:
    eligible, counts = eligible_edges(world.tree, kind)
    if not eligible:
        raise ValueError(
            f"{kind}: no eligible edges in this world (sources {EDGE_SOURCES_BY_KIND[kind]}); "
            f"the corrupted arm would be empty")
    edges = select_edges(eligible, edge_fraction, world_seed, offset)
    return edges, {"eligible_edge_sources": counts, "n_eligible_edges": len(eligible)}


def select_features(candidates, fraction: float, world_seed: int, offset: int
                    ) -> tuple[int, ...]:
    """A deterministic `fraction` of `candidates`, by world seed only; 0.0 selects nothing."""
    cand = tuple(sorted(int(x) for x in candidates))
    _check_fraction("fraction", fraction, allow_zero=True)
    if fraction <= 0.0 or not cand:
        return ()
    if fraction >= 1.0:
        return cand
    k = max(1, round(fraction * len(cand)))
    gen = torch.Generator().manual_seed(int(world_seed) + int(offset))
    perm = torch.randperm(len(cand), generator=gen).tolist()
    return tuple(sorted(cand[i] for i in perm[:k]))


def shard_shares(k: int, skew: float = 0.0) -> tuple[float, ...]:
    """The `k` token shares of one split feature, DESCENDING and summing to 1.

    Descending order makes `feature_to_latents[f][0]` the strongest shard by declaration.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0.0 <= float(skew) < 1.0):
        raise ValueError(f"skew must be in [0, 1), got {skew}")
    w = [(1.0 - float(skew)) ** i for i in range(int(k))]
    total = sum(w)
    return tuple(x / total for x in w)


def hole_generator_seed(world_seed: int, sample_seed: int, p: int, c: int) -> int:
    """Per-(world, draw, edge) seed for the hole; distinct multipliers per input."""
    return (HOLE_SEED_OFFSET
            + 1_000_003 * int(world_seed)
            + 7_919 * int(sample_seed)
            + 613 * int(p)
            + int(c))


def shard_generator_seed(world_seed: int, sample_seed: int, feature: int) -> int:
    """Per-(world, draw, feature) seed for the token->shard partition."""
    return (SHARD_SEED_OFFSET
            + 1_000_003 * int(world_seed)
            + 7_919 * int(sample_seed)
            + 613 * int(feature))


def composition_generator_seed(world_seed: int, sample_seed: int, p: int, r: int) -> int:
    """Per-(world, draw, pair) seed for the combination latent's tokens."""
    return (COMBINATION_SEED_OFFSET
            + 1_000_003 * int(world_seed)
            + 7_919 * int(sample_seed)
            + 613 * int(p)
            + 3 * int(r))


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def absorb(world, dials: AbsorptionDials, world_seed: int, readout: str = "identity"
           ) -> Corruption:
    """Carry `beta * rbar` of the parent direction into each selected child's row, unit norm.

    Undamaged rows pass through EXACTLY, so beta=0 or an unselected edge changes nothing.
    """
    g = world.g
    edges, details = _select_eligible_edges(world, "absorption", dials.edge_fraction,
                                            world_seed, EDGE_SEED_OFFSET)
    children = [c for _, c in edges]
    if len(set(children)) != len(children):
        # Each row is rebuilt FROM g[c], so a second parent would silently drop the first carry.
        raise ValueError("absorb: a child appears on more than one corrupted edge; "
                         "multi-parent carry is not defined for this corruption")
    W = g.double().clone()
    sev = torch.empty(len(edges), dtype=torch.float64)
    for i, (p, c) in enumerate(edges):
        if dials.beta != 0.0:
            d = g[c].double() + float(dials.beta) * float(dials.rbar) * g[p].double()
            W[c] = d / d.norm().clamp_min(_TINY)
        gp = g[p].double()
        sev[i] = float((W[c] @ gp) / (W[c].norm() * gp.norm()).clamp_min(_TINY))
    return Corruption(dials=dials, W_raw=W,
                      planted_map=PlantedMap.identity(int(g.shape[0]), readout=readout),
                      corrupted_edges=edges, realized_severity=sev,
                      severity_kind="edge_cos_parent", corrupted_pair_rule="ordered_edge",
                      details=details)


def hedging_gamma_star(world, p: int, c: int) -> float:
    """The reference mixing of child c into parent p's decoder row: the least-squares row
    when the parent latent's code is held at p's TRUE strength.

    That row is d = g_p + (E[A_p A_c] / E[A_p^2]) g_c. Strengths are drawn independently with
    mean m and sd spread*m, so E[A_p A_c] = P(c|p) m_p m_c and E[A_p^2] = P(p) m_p^2 (1 + spread^2)
    per firing token:

        gamma* = P(c|p) * m_c / (m_p * (1 + spread^2))

    It is not the MSE optimum once the code is refit by NNLS, where the optimum is the top
    principal direction of the parent's tokens and sits higher; gamma_rel is relative to this
    fixed reference, so read curves against the realized severity.
    """
    tree, cfg = world.tree, world.cfg
    if tree.parent_of(c) != p:
        raise ValueError(f"({p}, {c}) is not a direct tree edge")
    st = build_strengths(cfg, tree)
    m_p, m_c = float(st.mean_strength[p]), float(st.mean_strength[c])
    return tree.p_edge_of(c) * m_c / (m_p * (1.0 + float(cfg.strength_spread) ** 2))


def hedge(world, dials: HedgingDials, world_seed: int, readout: str = "identity"
          ) -> Corruption:
    """Delete each selected child's row and mix gamma of it into its parent's row.

    Rows and the map are built from ONE `keep` list, so a deletion cannot shift the latent id
    of a later feature between the two.
    """
    g = world.g.double()
    F = int(g.shape[0])
    edges, details = _select_eligible_edges(world, "hedging", dials.edge_fraction,
                                            world_seed, HEDGE_SEED_OFFSET)
    parents = [p for p, _ in edges]
    children = [c for _, c in edges]
    if len(set(parents)) != len(parents) or set(parents) & set(children):
        # Each parent row is rebuilt FROM g[p], so a second hedged child would silently replace
        # the first carry, and a hedged child cannot also carry a hedge of its own.
        raise ValueError("hedging: a feature sits on more than one hedged edge; mixing several "
                         "children into one row is not defined for this corruption")
    star = [hedging_gamma_star(world, p, c) for p, c in edges]
    gamma = [float(dials.gamma_rel) * s for s in star]
    W = g.clone()
    sev = torch.empty(len(edges), dtype=torch.float64)
    for i, (p, c) in enumerate(edges):
        if gamma[i] != 0.0:
            d = g[p] + gamma[i] * g[c]
            W[p] = d / d.norm().clamp_min(_TINY)
        gp = g[p] / g[p].norm().clamp_min(_TINY)
        own = g[c] - (g[c] @ gp) * gp
        own = own / own.norm().clamp_min(_TINY)
        sev[i] = float(W[p] @ own)
    drop = set(children)
    keep = [f for f in range(F) if f not in drop]
    f2l: list[tuple[int, ...]] = [()] * F
    for pos, f in enumerate(keep):
        f2l[f] = (pos,)
    details |= {"gamma_star": star, "gamma": gamma}
    return Corruption(dials=dials, W_raw=W[torch.tensor(keep, dtype=torch.long)],
                      planted_map=PlantedMap(feature_to_latents=tuple(f2l), readout=readout,
                                             n_latents=len(keep)),
                      corrupted_edges=edges, corrupted_features=tuple(sorted(parents)),
                      hedged_children=tuple(sorted(children)), realized_severity=sev,
                      severity_kind="child_own_component",
                      corrupted_pair_rule="either_endpoint_feature", details=details)


def split(world, dials: SplitDials, world_seed: int, readout: str = "identity") -> Corruption:
    """Carry each chosen feature on `k` latents that SHARE its direction: L = F + (k-1)*n.

    Latent ids are feature-major and CONTIGUOUS: `support[t].nonzero()` returns ascending ids,
    so a token's active rows arrive in the unsplit order, which with identical rows and
    disjoint firing makes the `union` readout reproduce the unsplit firing channel exactly.
    """
    g = world.g
    F = int(g.shape[0])
    roles = split_roles(world.tree)
    chosen: set[int] = set()
    role_counts = {}
    for r in dials.roles:
        if not roles[r]:
            raise ValueError(f"split role {r!r} has no features in this world; a declared role "
                             f"with nothing to split would leave its arm empty")
        sel = select_features(roles[r], float(dials.fraction), world_seed,
                              SPLIT_SEED_OFFSET + 101 * SPLIT_ROLES.index(r))
        role_counts[r] = {"n_eligible": len(roles[r]), "n_selected": len(sel)}
        chosen |= set(sel)
    rows: list[int] = []
    f2l: list[tuple[int, ...]] = []
    for f in range(F):
        k = int(dials.k) if f in chosen else 1
        f2l.append(tuple(range(len(rows), len(rows) + k)))
        rows.extend([f] * k)
    W = g.double()[torch.tensor(rows, dtype=torch.long)]
    return Corruption(dials=dials, W_raw=W,
                      planted_map=PlantedMap(feature_to_latents=tuple(f2l), readout=readout,
                                             n_latents=len(rows)),
                      corrupted_features=tuple(sorted(chosen)),
                      corrupted_pair_rule="candidate_parent_feature",
                      details={"roles": role_counts})


def compose(world, dials: CompositionDials, world_seed: int, readout: str = "identity"
            ) -> Corruption:
    """Append one combination latent unit(g_p + g_r) per partner pair; own rows unchanged.

    Own latent ids stay equal to feature ids and combination latents follow at F, F+1, ...
    """
    g = world.g.double()
    F = int(g.shape[0])
    pairs, details = composition_partners(world.tree, float(dials.fraction), world_seed)
    comb = [g[a] + g[b] for a, b in pairs]
    W = torch.cat([g, torch.stack([v / v.norm().clamp_min(_TINY) for v in comb])])
    f2l: list[tuple[int, ...]] = [(f,) for f in range(F)]
    entries = []
    for i, (a, b) in enumerate(pairs):
        f2l[a] = (a, F + i)
        f2l[b] = (b, F + i)
        entries.append((a, b, F + i))
    # P(r | p) and P(p | r) for independent partners: the partner's design firing rate
    details |= {"partner_density": [[float(world.p[b]), float(world.p[a])] for a, b in pairs]}
    return Corruption(dials=dials, W_raw=W,
                      planted_map=PlantedMap(feature_to_latents=tuple(f2l), readout=readout,
                                             n_latents=F + len(pairs),
                                             composition=tuple(entries)),
                      corrupted_features=tuple(sorted(x for pr in pairs for x in pr)),
                      composition_pairs=pairs,
                      corrupted_pair_rule="candidate_parent_feature", details=details)


# --------------------------------------------------------------------------
# per-draw firing transforms
# --------------------------------------------------------------------------
def apply_hole(support: torch.Tensor, corruption: Corruption, world_seed: int,
               sample_seed: int) -> tuple[torch.Tensor, dict[int, int]]:
    """Remove each absorbing PARENT on `round(eta * n)` of the tokens where it co-fires with
    any of its absorbed children. Returns (new support [n, F] bool, holed tokens per parent).

    One hole per parent over the union of its children's co-fire tokens: drawn per edge, a
    container's co-firing members would stack their holes on one column and remove more than
    eta. Co-fire rather than child tokens, because a member also fires without its container.
    On strict hierarchy with one child per parent this is the child-token hole. Only parent
    columns change; private generator per parent.
    """
    eta = float(corruption.dials.eta)
    out = support.clone()
    children_of: dict[int, list[int]] = {}
    for p, c in corruption.corrupted_edges:
        children_of.setdefault(int(p), []).append(int(c))
    n_holed: dict[int, int] = {}
    for p, cs in children_of.items():
        child_fires = torch.zeros(support.shape[0], dtype=torch.bool, device=support.device)
        for c in cs:
            child_fires |= support[:, c]
        cofire = (child_fires & support[:, p]).nonzero(as_tuple=True)[0]
        want = round(eta * int(cofire.numel()))
        n_holed[p] = want
        if want <= 0:
            continue
        gen = torch.Generator().manual_seed(hole_generator_seed(world_seed, sample_seed, p, cs[0]))
        perm = torch.randperm(int(cofire.numel()), generator=gen)
        out[cofire[perm[:want]], p] = False
    return out, n_holed


def expand_support(support: torch.Tensor, corruption: Corruption, world_seed: int,
                   sample_seed: int) -> torch.Tensor:
    """`[n, F]` feature-space support -> `[n, L]` latent-space support, through the planted map.

    A split feature's firing tokens are PARTITIONED over its shards (randperm plus contiguous
    slices at cumulative counts, so disjoint and exactly covering by construction). A composed
    pair's co-firing tokens go, for a share `pi`, to the combination latent, with both own
    latents off there.
    """
    pmap = corruption.planted_map
    n, F = support.shape
    if F != pmap.F:
        raise ValueError(f"support is {F} features wide but the planted map covers {pmap.F}")
    out = torch.zeros(n, pmap.n_latents, dtype=torch.bool, device=support.device)
    comb = pmap.combination_latents()
    for f, lats in enumerate(pmap.feature_to_latents):
        lats = tuple(j for j in lats if j not in comb)
        if not lats:
            continue                                   # hedged child: no latent fires
        if len(lats) == 1:
            out[:, lats[0]] = support[:, f]
            continue
        if not isinstance(corruption.dials, SplitDials):
            raise ValueError(
                f"{corruption.kind} declares feature {f} on {len(lats)} latents but carries no "
                f"share policy; a partition needs declared shares, not a default")
        tok = support[:, f].nonzero(as_tuple=True)[0]
        n_f = int(tok.numel())
        gen = torch.Generator().manual_seed(shard_generator_seed(world_seed, sample_seed, f))
        perm = torch.randperm(n_f, generator=gen)
        shares = shard_shares(len(lats), float(corruption.dials.skew))
        # Cumulative bounds, so the shard counts sum to n_f exactly.
        bounds, cum = [], 0.0
        for s in shares:
            cum += s
            bounds.append(int(round(cum * n_f)))
        bounds[-1] = n_f
        start = 0
        for i, end in enumerate(bounds):
            out[tok[perm[start:end]], lats[i]] = True
            start = end
    for a, b, lat in pmap.composition:
        if not isinstance(corruption.dials, CompositionDials):
            raise ValueError(f"{corruption.kind} declares a combination latent but no pi")
        both = (support[:, a] & support[:, b]).nonzero(as_tuple=True)[0]
        want = round(float(corruption.dials.pi) * int(both.numel()))
        if want <= 0:
            continue
        gen = torch.Generator().manual_seed(composition_generator_seed(world_seed, sample_seed,
                                                                       a, b))
        chosen = both[torch.randperm(int(both.numel()), generator=gen)[:want]]
        out[chosen, lat] = True
        out[chosen, pmap.feature_to_latents[a][0]] = False
        out[chosen, pmap.feature_to_latents[b][0]] = False
    return out


# Damage registry: kind -> builder. `build_corruption` is the only dispatch.
CORRUPTIONS = {
    "absorption": absorb,
    "hedging": hedge,
    "split": split,
    "composition": compose,
}


def build_corruption(world, dials, world_seed: int, readout: str = "identity") -> Corruption:
    """The dictionary this dials object describes for this world. The damage is chosen by the
    dials TYPE, so there is no way to ask for one damage with another's knobs."""
    kind = type(dials).KIND
    builder = CORRUPTIONS.get(kind)
    if builder is None:
        raise ValueError(f"no corruption builder registered for kind {kind!r}; "
                         f"registered damages are {tuple(CORRUPTIONS)}")
    return builder(world, dials, world_seed, readout)
