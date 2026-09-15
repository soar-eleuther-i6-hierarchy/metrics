"""
Builds the feature forest from a ToyConfig: parent/child edges, their transitive closure,
and per-feature tags.

`parents` holds only direct edges; `ancestors`/`descendents` hold the transitive closure --
a grandparent-grandchild pair is contained but not a direct edge, so it's labelled
`transitive` and scored separately. Every feature has exactly one parent, so firing rate is
a clean product down its chain: p_child = p_parent * p_edge.

`root_p` is always a root's marginal firing rate. Roots driven by an observed or latent cause
(token groups, topic registers) also carry `cause_rate`, their firing rate inside the cause;
their `root_p` is the cause's design mass times that rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .spec import (
    GUARDED_STRUCTURE_CLASSES,
    F_MIN,
    MIN_PAIRS_PER_CLASS,
    STRUCTURE_MAX_ATTEMPTS,
    STRUCTURE_SEED_OFFSET,
    ToyConfig,
)


@dataclass
class Tree:
    F: int
    parents: dict[int, list[tuple[int, float, float]]]   # child -> [(parent, p_edge, alpha_span)]
    children: dict[int, list[int]]
    exclusive: dict[int, bool]
    ancestors: dict[int, set[int]]
    descendents: dict[int, set[int]]
    topology_ordering: list[int]
    root_p: dict[int, float]                             # roots only; the marginal firing rate
    # --- tags -------------------------------------------------------------
    kappa: dict[int, float] = field(default_factory=dict)       # topic modulation, 0 = off
    topic: dict[int, int | None] = field(default_factory=dict)
    token_bound: dict[int, bool] = field(default_factory=dict)
    token_ids: dict[int, tuple[int, ...]] = field(default_factory=dict)  # token-bound features: the ids they fire on
    cause_rate: dict[int, float | None] = field(default_factory=dict)    # firing rate inside the cause; None = not cause-driven at a fixed rate
    tags: dict[int, set[str]] = field(default_factory=dict)

    def parent_of(self, k: int) -> int | None:
        ps = self.parents.get(k)
        return ps[0][0] if ps else None

    def p_edge_of(self, k: int) -> float:
        ps = self.parents.get(k)
        return ps[0][1] if ps else 1.0
    
    def alpha_of(self, k: int) -> float:
        ps = self.parents.get(k)
        return ps[0][2] if ps else 0.0


def _close(parents: dict[int, list[tuple[int, float, float]]], F: int
) -> tuple[dict[int, set[int]], dict[int, set[int]], list[int]]:
    """Transitive closure of the direct-edge relation, plus a topological order."""
    anc = {k: set() for k in range(F)}
    for k in range(F):
        stack = [p for p, _, _ in parents.get(k, [])]
        while stack:
            a = stack.pop()
            if a not in anc[k]:
                anc[k].add(a)
                stack += [p for p, _, _ in parents.get(a, [])]
    desc = {k: set() for k in range(F)}
    for k in range(F):
        for a in anc[k]:
            desc[a].add(k)
    depth = {k: 0 for k in range(F)}
    topo = sorted(range(F), key=lambda k: (len(anc[k]), k))
    for k in topo:
        ps = parents.get(k, [])
        depth[k] = 1 + max((depth[p] for p, _, _ in ps), default=-1) if ps else 0
    topo = sorted(range(F), key=lambda k: (depth[k], k))
    return anc, desc, topo


def build_tree(cfg: ToyConfig) -> Tree:
    """Build the feature forest, returning only a config the sampler can actually honour.

    `randomize_structure=False` (default) reproduces the fixed lattice exactly. When True the
    backbone is drawn from `cfg.seed` and retried deterministically until it clears both the
    structural floors and `validate_config`. The confound set is identical in both modes.
    """
    from .validate import validate_config     # lazy: avoid a tree <-> strengths import cycle

    if not cfg.randomize_structure:
        tree = _assemble_tree(cfg, gen=None)
        validate_config(cfg, tree)
        return tree

    _check_randomize_feasible(cfg)     # fail fast on params no draw could satisfy
    last_reason = "(no attempt ran)"
    for attempt in range(STRUCTURE_MAX_ATTEMPTS):
        # Multiply into the sub-seed so two seeds' attempt-streams can't overlap; cfg.seed itself is never advanced.
        gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + STRUCTURE_SEED_OFFSET + attempt)
        tree = _assemble_tree(cfg, gen=gen)
        ok, last_reason = _structure_guards_pass(tree)
        if not ok:
            continue
        try:
            validate_config(cfg, tree)
        except ValueError as e:            # infeasible draw: retry with the next sub-seed
            last_reason = f"validate_config: {e}"
            continue
        return tree
    raise ValueError(
        f"randomize_structure: no feasible backbone in {STRUCTURE_MAX_ATTEMPTS} attempts "
        f"(seed={cfg.seed}); last rejection: {last_reason}. Loosen F_MIN / "
        f"MIN_PAIRS_PER_CLASS or the branching-depth ranges.")


def _assemble_tree(cfg: ToyConfig, gen: "torch.Generator | None") -> Tree:
    """Construct one tree (backbone + confounds + closure) WITHOUT validation.

    `gen is None` -> the deterministic lattice backbone; otherwise a seed-varied backbone
    drawn from `gen`. The confound layer and the closure are identical either way.
    """
    parents: dict[int, list[tuple[int, float, float]]] = {}
    children: dict[int, list[int]] = {}
    exclusive: dict[int, bool] = {}
    root_p: dict[int, float] = {}
    kappa: dict[int, float] = {}
    topic: dict[int, int | None] = {}
    token_bound: dict[int, bool] = {}
    token_ids: dict[int, tuple[int, ...]] = {}
    cause_rate: dict[int, float | None] = {}
    tags: dict[int, set[str]] = {}

    nxt = 0

    def new(tag: str = "backbone") -> int:
        nonlocal nxt
        k = nxt
        nxt += 1
        kappa[k] = 0.0
        topic[k] = None
        token_bound[k] = False
        token_ids[k] = ()
        cause_rate[k] = None
        tags[k] = {tag}
        children[k] = []
        return k

    # --- backbone forest --------------------------------------------------
    if gen is None:
        _backbone_lattice(cfg, new, parents, children, exclusive, root_p)
    else:
        _backbone_random(cfg, gen, new, parents, children, exclusive, root_p)

    # --- confounds (the per-archetype firing rates below are hand-tuned constants) ----
    # Counts/rates are identical in both backbone modes; confound ids may shift under randomization but counts/rates/topics stay seed-invariant, so they form a fixed control group.
    if cfg.confounds:
        # superparent: an always-on childless feature, with a broad parent as its foil.
        for _ in range(cfg.n_superparent):
            s = new("superparent")
            root_p[s] = cfg.superparent_p
        for _ in range(cfg.n_broad_parent):
            b = new("broad_parent")
            root_p[b] = 0.40
            exclusive[b] = False
            for _ in range(cfg.broad_children):
                c = new("broad_child")
                parents[c] = [(b, cfg.child_p_edge, cfg.broad_alpha)]
                children[b].append(c)
        # dense true parent: as dense as a superparent, with one exactly orthogonal child.
        for _ in range(cfg.n_dense_parents):
            d = new("dense_parent")
            root_p[d] = cfg.superparent_p
            exclusive[d] = False
            c = new("dense_child")
            parents[c] = [(d, cfg.child_p_edge, 0.0)]
            children[d].append(c)

        # frequency-coincidence pairs: token-bound features co-fire only via shared high-frequency token ids, with no declared edge.
        for i in range(cfg.n_token_bound_pairs):
            for j in (0, 1):
                k = new("token_bound")
                root_p[k] = 0.10
                token_bound[k] = True
                token_ids[k] = tuple(range(cfg.n_bind_ids))
                tags[k].add(f"tokpair{i}")

        # token groups: a container and members fire only on the group's token ids, independently given the token.
        if cfg.n_token_groups > 0:
            group_ids, group_mass = _token_groups(cfg)
            for i, ids in enumerate(group_ids):
                roles = [("token_container", cfg.token_container_rate)]
                roles += [("token_member", cfg.token_member_rate)] * cfg.token_group_members
                for role, rate in roles:
                    k = new("token_bound")
                    tags[k] |= {role, f"tokgroup{i}"}
                    token_bound[k] = True
                    token_ids[k] = ids
                    cause_rate[k] = rate
                    root_p[k] = group_mass[i] * rate

        # topical pairs: co-occur via a shared document topic z -- correlated overall, independent once z is known.
        for i in range(cfg.n_topical_pairs):
            z = i % cfg.Z
            for j in (0, 1):
                k = new("topical")
                root_p[k] = 0.09
                kappa[k] = cfg.kappa
                topic[k] = z
                tags[k].add(f"toppair{i}")

        # topic registers: a register and members fire only in documents of one topic, independently given the topic.
        if cfg.n_topic_registers_per_topic > 0:
            for z in range(cfg.Z):
                roles = [("topic_register", cfg.topic_register_rate)] * cfg.n_topic_registers_per_topic
                roles += [("topic_member", cfg.topic_member_rate)] * cfg.topic_members
                for role, rate in roles:
                    k = new("topical")
                    tags[k].add(role)
                    topic[k] = z
                    cause_rate[k] = rate
                    root_p[k] = rate / cfg.Z

    F = nxt
    ancestors, descendents, topology_ordering = _close(parents, F)

    return Tree(
        F=F, parents=parents, children=children, exclusive=exclusive,
        ancestors=ancestors, descendents=descendents, topology_ordering=topology_ordering,
        root_p=root_p, kappa=kappa, topic=topic, token_bound=token_bound,
        token_ids=token_ids, cause_rate=cause_rate, tags=tags,
    )


def _token_groups(cfg: ToyConfig) -> tuple[list[tuple[int, ...]], list[float]]:
    """Split design-Zipf ids 1..n_token_groups*token_ids_per_group into equal-mass groups.

    Id 0 is excluded: its mass alone exceeds a group's. Greedy: heaviest id first, into the
    currently lightest group. Masses come from the design Zipf, never from realized counts.
    """
    from .sample import _zipf_probs     # lazy: sample imports tree
    n_ids = cfg.n_token_groups * cfg.token_ids_per_group
    if n_ids > cfg.vocab - 1:
        raise ValueError(
            f"token groups need ids 1..{n_ids} but vocab is {cfg.vocab}; lower n_token_groups "
            f"or token_ids_per_group")
    probs = _zipf_probs(cfg.vocab, cfg.zipf_s)
    groups: list[list[int]] = [[] for _ in range(cfg.n_token_groups)]
    mass = [0.0] * cfg.n_token_groups
    for i in sorted(range(1, n_ids + 1), key=lambda i: (-float(probs[i]), i)):
        j = min(range(cfg.n_token_groups), key=lambda j: (mass[j], j))
        groups[j].append(i)
        mass[j] += float(probs[i])
    ids = [tuple(sorted(g)) for g in groups]
    return ids, [token_mass(cfg, g) for g in ids]


def token_mass(cfg: ToyConfig, ids: tuple[int, ...]) -> float:
    """Design-Zipf probability that a token falls in `ids`."""
    from .sample import _zipf_probs     # lazy: sample imports tree
    if not ids:
        return 0.0
    return float(_zipf_probs(cfg.vocab, cfg.zipf_s)[list(ids)].sum())


def _backbone_lattice(cfg, new, parents, children, exclusive, root_p) -> None:
    """The fixed lattice backbone: `n_roots` roots, uniform `branching` and `depth`, with
    firing_only on every `alpha_zero_every`-th edge (counted across the whole forest).
    `alpha_zero_every=0` disables firing_only, so every edge is a real alpha>0 is-a edge."""
    edge_i = 0
    frontier = []
    for _ in range(cfg.n_roots):
        r = new()
        root_p[r] = cfg.root_p
        frontier.append(r)
    for _ in range(cfg.depth):
        nxt_frontier = []
        for p in frontier:
            exclusive[p] = cfg.exclusive_siblings
            for _ in range(cfg.branching):
                c = new()
                a = 0.0 if (cfg.alpha_zero_every > 0 and edge_i % cfg.alpha_zero_every == 0) else cfg.alpha
                edge_i += 1
                parents[c] = [(p, cfg.child_p_edge, a)]
                children[p].append(c)
                nxt_frontier.append(c)
        frontier = nxt_frontier


def _backbone_random(cfg, gen, new, parents, children, exclusive, root_p) -> None:
    """A seed-varied backbone: ragged per-root depth/branching, mass-preserving edge
    probabilities, and a balanced is-a/firing_only split.

    Edge probs are Dirichlet-drawn and scaled to a fixed budget `p_tot`, so total firing mass
    (and L0) stays roughly constant regardless of child count, and the exclusive-sibling budget
    holds automatically. alpha is assigned after the backbone is built so firing_only edges land
    at exactly a 1/alpha_zero_every fraction without clumping.
    """
    max_branch = max(1, int(1.0 / cfg.child_p_edge))    # floor(1/p_edge): the exclusive-budget cap
    lo_branch = max(1, max_branch - 1)
    lo_depth = max(1, cfg.depth - 1)
    p_tot = min(1.0 - 1e-9, cfg.branching * cfg.child_p_edge)

    edge_parent: dict[int, int] = {}
    edge_pedge: dict[int, float] = {}
    edge_order: list[int] = []                          # child ids in creation order

    roots = []
    for _ in range(cfg.n_roots):
        r = new()
        root_p[r] = cfg.root_p
        roots.append(r)
    for r in roots:
        depth_r = int(torch.randint(lo_depth, cfg.depth + 1, (1,), generator=gen).item())
        frontier = [r]
        for _ in range(depth_r):
            nxt_frontier = []
            for p in frontier:
                n_c = int(torch.randint(lo_branch, max_branch + 1, (1,), generator=gen).item())
                exclusive[p] = cfg.exclusive_siblings
                w = torch.empty(n_c, dtype=torch.float64).exponential_(generator=gen)
                pe = (w / w.sum() * p_tot)
                # nudge the realized sum down against float drift so validate_config's budget holds
                s = float(pe.sum())
                if s > 1.0 - 1e-9:
                    pe = pe * ((1.0 - 1e-9) / s)
                for j in range(n_c):
                    c = new()
                    edge_parent[c] = p
                    edge_pedge[c] = float(pe[j])
                    edge_order.append(c)
                    children[p].append(c)
                    nxt_frontier.append(c)
            frontier = nxt_frontier

    # balanced is-a / firing_only: exactly round(n_edges / alpha_zero_every) edges get alpha=0.
    n_edges = len(edge_order)
    n_zero = round(n_edges / cfg.alpha_zero_every) if (n_edges and cfg.alpha_zero_every > 0) else 0
    zero_children: set[int] = set()
    if n_zero > 0:
        perm = torch.randperm(n_edges, generator=gen)
        zero_children = {edge_order[int(i)] for i in perm[:n_zero]}
    for c in edge_order:
        a = 0.0 if c in zero_children else cfg.alpha
        parents[c] = [(edge_parent[c], edge_pedge[c], a)]


def _structure_guards_pass(tree: Tree) -> tuple[bool, str]:
    """Reject a randomized draw that would starve the dictionary or a scored class.

    Returns (ok, reason); `reason` is empty on success. Checks total feature count against
    `F_MIN` and each guarded class against `MIN_PAIRS_PER_CLASS` ordered pairs -- stronger
    than validate_config's global budget check.
    """
    if tree.F < F_MIN:
        return False, f"F={tree.F} < F_MIN={F_MIN}"
    from .labels import _index, pair_label
    pl = pair_label(tree)
    # Count off-diagonal cells only: a self-pair is never a real relation, so floors must be measured over genuine ordered pairs.
    off_diag = ~torch.eye(tree.F, dtype=torch.bool)
    for name in GUARDED_STRUCTURE_CLASSES:
        n = int(((pl == _index(name)) & off_diag).sum())
        if n < MIN_PAIRS_PER_CLASS:
            return False, f"class {name!r} has {n} ordered pairs < MIN_PAIRS_PER_CLASS={MIN_PAIRS_PER_CLASS}"
    return True, ""


def _check_randomize_feasible(cfg: ToyConfig) -> None:
    """Reject randomize params no draw could satisfy -- fail fast instead of burning every attempt.

    `child_p_edge >= 0.5` caps children-per-parent at 1, making 'sibling' (a guarded class)
    unreachable. `depth < 1` leaves the per-root depth draw with an empty randint range.
    """
    max_branch = max(1, int(1.0 / cfg.child_p_edge))
    if "sibling" in GUARDED_STRUCTURE_CLASSES and max_branch < 2:
        raise ValueError(
            f"randomize_structure needs child_p_edge < 0.5 so a parent can have >= 2 children "
            f"(siblings); got child_p_edge={cfg.child_p_edge} -> max children/parent = {max_branch}. "
            f"Lower child_p_edge, or drop 'sibling' from GUARDED_STRUCTURE_CLASSES.")
    if cfg.depth < 1:
        raise ValueError(
            f"randomize_structure needs depth >= 1 for a non-empty per-root depth draw; got "
            f"depth={cfg.depth}.")
