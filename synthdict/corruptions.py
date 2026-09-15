"""Latent-side corruption generators — physical dials only, no instrument thresholds.

ABSORPTION (round 1, pre-registered in `SYNTH_PRECOMMIT.md`):

  decoder carry   d_c' = unit(g_c + beta * rbar * g_p)     (rbar = 1.0: flat-strength design)
  firing hole     parent removed from the planted support on round(eta * n_c) child tokens
  edge selection  a deterministic `edge_fraction` of the tree's containment edges

The round-2 damages (machinery only — nothing registered, nothing run):

  noise    W'_j = unit(g_j + sigma * z_j), z unit-norm so sigma is cosine-scale.  L = F
  missing  a deterministic fraction of features lose their decoder row.           L < F
  split    a feature is carried by k latents sharing its direction and            L > F
           partitioning its firing tokens.

Absorption is the only EDGE-level damage; the other three are feature-level, which is why the
corrupted-pair rule travels with the corruption instead of being assumed.

TAUTOLOGY GUARD (tested): this module imports neither `ABSORPTION_CONSTANTS` (the census's
thresholds) nor `scoring.core.registry.CONSTANTS` (the detectors'). A corruption defined in
the instrument's own vocabulary would make "the instrument responds" true by construction.

Severity is REALIZED per edge as `cos(d_c', g_p)`; `beta` is only the generator knob
(PRECOMMIT: the dose-response is reported against realized severity).

Determinism contract — the DICTIONARY is a property of the world, the TOKEN assignments are
per draw, and no stream ever touches the global RNG or the world's own:
  * the corrupted EDGE SET, the damaged FEATURE SET, the shard structure (k, shares, latent
    ids) and the noise perturbation depend on (world_seed, dials) only, so every draw of one
    world sees the same dictionary;
  * the HOLE token subset depends on (world_seed, sample_seed, edge) and the token->SHARD
    partition on (world_seed, sample_seed, feature) — deterministic per draw, different
    across draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from synthdict.planted import PlantedMap

_TINY = 1e-12

# Dedicated seed offsets for the private RNG streams. Chosen away from the repo's other
# derivations (toygen STRUCTURE_SEED_OFFSET=9973, held-out +10000, probe fit +20000). One per
# stream, so two damages sharing a world seed do not damage the same features.
EDGE_SEED_OFFSET = 61_211
HOLE_SEED_OFFSET = 77_003
SPLIT_SEED_OFFSET = 52_711
MISSING_SEED_OFFSET = 34_919
NOISE_SEED_OFFSET = 88_547
SHARD_SEED_OFFSET = 41_213

# What `realized_severity` MEASURES. Same column name, different population per damage:
#   edge_cos_parent  one entry per corrupted EDGE  — cos(d_c', g_p)   (absorption)
#   row_cos_true     one entry per decoder ROW     — cos(W'_j, g_j)   (noise)
#   none             no severity axis exists for this damage          (splitting, missing)
SEVERITY_KINDS = ("none", "edge_cos_parent", "row_cos_true")

# How `corrupted_pair` is derived. Absorption damages an ORDERED edge, so only (p, c) is
# corrupted; a feature-level damage corrupts every pair touching a damaged feature. Without
# this the edge rule silently returns an all-False mask for split/missing/noise, which makes
# `intact` the whole target class in report.py and the dose-response a comparison with itself.
PAIR_RULES = ("ordered_edge", "either_endpoint_feature", "all")


def _empty_severity() -> torch.Tensor:
    return torch.zeros(0, dtype=torch.float64)


@dataclass(frozen=True)
class AbsorptionDials:
    """The physical knobs. `rbar` is the parent/child mean-magnitude ratio; 1.0 is analytic
    for this generator (toygen/strengths.py builds a FLAT mean strength q*sqrt(E0))."""

    KIND = "absorption"

    beta: float
    eta: float
    edge_fraction: float = 1.0
    rbar: float = 1.0


@dataclass(frozen=True)
class SplitDials:
    """One feature carried by `k` latents that share its direction and partition its firing.

    `skew` tilts the token shares geometrically; 0.0 is an equal split. `fraction` is the
    deterministic share of features split, so a split run still has an intact control.
    """

    KIND = "split"

    k: int
    skew: float = 0.0
    fraction: float = 1.0

    def __post_init__(self) -> None:
        if int(self.k) < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not (0.0 <= float(self.skew) < 1.0):
            raise ValueError(f"skew must be in [0, 1), got {self.skew}")
        if not (0.0 <= float(self.fraction) <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


@dataclass(frozen=True)
class MissingDials:
    """A deterministic `fraction` of features have NO latent at all — the row is deleted.

    Note the shape of the dose axis: `fraction` is both the dose and the selection, so the zero
    point has an empty corrupted arm by construction and is read against the intact and null
    columns rather than against a same-world corrupted side.
    """

    KIND = "missing"

    fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.fraction) <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


@dataclass(frozen=True)
class NoiseDials:
    """Every decoder row perturbed toward a random direction and renormalized.

    `sigma` is a COSINE-scale knob, not a vector magnitude: the perturbation is drawn unit-norm,
    so cos(W'_j, g_j) ~= 1/sqrt(1 + sigma^2) at any D. A raw standard normal would make sigma
    mean something different at every dimensionality (its norm grows as sqrt(D)).
    """

    KIND = "noise"

    sigma: float

    def __post_init__(self) -> None:
        if float(self.sigma) < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")


@dataclass(frozen=True)
class Corruption:
    """One synthetic dictionary's corruption record.

    dials              the damage's own dial dataclass; its `KIND` names the damage
    W_raw              [L, D] float64 synthetic raw decoder rows (unit-norm; uncorrupted
                       rows are the true g rows unchanged). L == F only while the map is 1-1.
    planted_map        the feature->latent correspondence this dictionary was built with
    corrupted_edges    ordered (parent, child) tuples — edge-level damages only
    corrupted_features true feature ids damaged — feature-level damages only
    realized_severity  the severity axis, whose population `severity_kind` declares
    """

    dials: object
    W_raw: torch.Tensor
    planted_map: PlantedMap
    corrupted_edges: tuple[tuple[int, int], ...] = ()
    corrupted_features: tuple[int, ...] = ()
    realized_severity: torch.Tensor = field(default_factory=_empty_severity)
    severity_kind: str = "none"
    corrupted_pair_rule: str = "ordered_edge"

    @property
    def kind(self) -> str:
        """The damage name, DERIVED from the dials type. Never a separate field: a kind string
        that can disagree with its dials would mislabel the artifact path, the report grouping
        and the census caveat while the dictionary carried something else."""
        return type(self.dials).KIND

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


def select_edges(cont_edges, edge_fraction: float, world_seed: int
                 ) -> tuple[tuple[int, int], ...]:
    """A deterministic `edge_fraction` subset of the containment edges.

    Depends on the WORLD seed only (no sample_seed parameter, by contract): the corrupted
    edge set is a property of the synthetic dictionary, which must be identical across the
    matching, scoring, and probe-fitting draws. f=1.0 returns the edges in tree order.
    """
    if not (0.0 < edge_fraction <= 1.0):
        raise ValueError(f"edge_fraction must be in (0, 1], got {edge_fraction}")
    edges = [(int(p), int(c)) for p, c in cont_edges]
    if edge_fraction >= 1.0:
        return tuple(edges)
    k = max(1, round(edge_fraction * len(edges)))
    gen = torch.Generator().manual_seed(int(world_seed) + EDGE_SEED_OFFSET)
    perm = torch.randperm(len(edges), generator=gen).tolist()
    chosen = sorted(perm[:k])
    return tuple(edges[i] for i in chosen)


def select_features(F: int, fraction: float, world_seed: int, offset: int
                    ) -> tuple[int, ...]:
    """A deterministic `fraction` of the world's features, by world seed only.

    The feature-level twin of `select_edges`, and a separate function rather than a reuse for
    two reasons: `fraction = 0` must be expressible (it is the no-op dial, where `select_edges`
    insists on at least one edge), and each damage passes its own `offset` so two damages
    sharing a world seed do not land on the same features.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if fraction <= 0.0:
        return ()
    if fraction >= 1.0:
        return tuple(range(int(F)))
    k = max(1, round(fraction * int(F)))
    gen = torch.Generator().manual_seed(int(world_seed) + int(offset))
    perm = torch.randperm(int(F), generator=gen).tolist()
    return tuple(sorted(perm[:k]))


def shard_shares(k: int, skew: float = 0.0) -> tuple[float, ...]:
    """The `k` token shares of one split feature, DESCENDING and summing to 1.

    Geometric decay in `skew`; 0.0 is an equal split. Descending order is the contract that
    makes `feature_to_latents[f][0]` the strongest shard by declaration — the `strongest_shard`
    readout reads that ordering and never inspects an activation.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0.0 <= float(skew) < 1.0):
        raise ValueError(f"skew must be in [0, 1), got {skew}")
    w = [(1.0 - float(skew)) ** i for i in range(int(k))]
    total = sum(w)
    return tuple(x / total for x in w)


def shard_generator_seed(world_seed: int, sample_seed: int, feature: int) -> int:
    """Deterministic per-(world, draw, feature) seed for the token->shard partition.

    Same form as `hole_generator_seed`, and the same contract: the shard STRUCTURE is fixed by
    (world_seed, dials) and identical across draws, while WHICH tokens land on which shard is
    per-draw. Distinct multipliers, so dropping any input collapses distinct seeds.
    """
    return (SHARD_SEED_OFFSET
            + 1_000_003 * int(world_seed)
            + 7_919 * int(sample_seed)
            + 613 * int(feature))


def absorb(g: torch.Tensor, cont_edges, dials: AbsorptionDials, world_seed: int,
           readout: str = "identity") -> Corruption:
    """Build the absorbed dictionary: carry `beta * rbar` of the parent direction into each
    corrupted child's decoder row, renormalized to unit norm.

    `g` rows are unit-norm by generator construction (tested); uncorrupted rows pass through
    EXACTLY (`torch.equal`), so beta=0 or an unselected edge changes nothing at all.

    Absorption preserves one latent per feature, so the planted map is the identity; `readout`
    only records which reduction policy the run declared (a no-op on a 1-1 map).
    """
    edges = select_edges(cont_edges, dials.edge_fraction, world_seed)
    children = [c for _, c in edges]
    if len(set(children)) != len(children):
        # Each iteration rebuilds the child row FROM g[c], so a second corrupted parent would
        # silently discard the first carry and orphan its recorded severity. Impossible in the
        # round-1 depth-1 worlds; guarded for the registry's future pathologies.
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
                      severity_kind="edge_cos_parent", corrupted_pair_rule="ordered_edge")


def hole_generator_seed(world_seed: int, sample_seed: int, p: int, c: int) -> int:
    """Deterministic per-(world, draw, edge) seed for the hole's private RNG stream.

    Mixes all four inputs with distinct multipliers so dropping any one of them (the
    mutation this anchors) collapses distinct seeds.
    """
    return (HOLE_SEED_OFFSET
            + 1_000_003 * int(world_seed)
            + 7_919 * int(sample_seed)
            + 613 * int(p)
            + int(c))


def apply_hole(support: torch.Tensor, corruption: Corruption, world_seed: int,
               sample_seed: int) -> tuple[torch.Tensor, dict[tuple[int, int], int]]:
    """Remove each corrupted edge's PARENT from `round(eta * n_c)` of the child's firing
    tokens. Returns (new support [n, F] bool, per-edge holed-token counts).

    Only parent columns change; the child's own firing is untouched (absorption starves the
    parent, not the child). Uses a private torch.Generator per edge — never the global RNG.
    """
    eta = float(corruption.dials.eta)
    out = support.clone()
    n_holed: dict[tuple[int, int], int] = {}
    for (p, c) in corruption.corrupted_edges:
        child_tok = support[:, c].nonzero(as_tuple=True)[0]
        want = round(eta * int(child_tok.numel()))
        n_holed[(p, c)] = want
        if want <= 0:
            continue
        gen = torch.Generator().manual_seed(hole_generator_seed(world_seed, sample_seed, p, c))
        perm = torch.randperm(int(child_tok.numel()), generator=gen)
        out[child_tok[perm[:want]], p] = False
    return out, n_holed


def noise(g: torch.Tensor, cont_edges, dials: NoiseDials, world_seed: int,
          readout: str = "identity") -> Corruption:
    """Perturb EVERY decoder row toward a random unit direction, renormalized.

    At `sigma = 0` the rows are not touched at all — not renormalized, since re-normalizing an
    already-unit row moves it by ~6e-17 and the origin of a dose-response curve has to be an
    exact no-op. Renormalization elsewhere keeps `recon_2a` from moving through ||d||^2 for
    reasons unrelated to the geometry the dial perturbs.
    """
    F = int(g.shape[0])
    W = g.double().clone()
    if float(dials.sigma) != 0.0:
        gen = torch.Generator().manual_seed(int(world_seed) + NOISE_SEED_OFFSET)
        z = torch.randn(g.shape, generator=gen, dtype=torch.float64)
        z = z / z.norm(dim=1, keepdim=True).clamp_min(_TINY)   # unit: sigma is cosine-scale
        for j in range(F):
            d = g[j].double() + float(dials.sigma) * z[j]
            W[j] = d / d.norm().clamp_min(_TINY)
    gd = g.double()
    sev = ((W * gd).sum(dim=1)
           / (W.norm(dim=1) * gd.norm(dim=1)).clamp_min(_TINY))
    return Corruption(dials=dials, W_raw=W,
                      planted_map=PlantedMap.identity(F, readout=readout),
                      corrupted_features=tuple(range(F)), realized_severity=sev,
                      severity_kind="row_cos_true", corrupted_pair_rule="all")


def missing(g: torch.Tensor, cont_edges, dials: MissingDials, world_seed: int,
            readout: str = "identity") -> Corruption:
    """Delete the decoder row of a deterministic fraction of features: L = F - n_dropped.

    The dictionary and the map are built from ONE `keep` list, so the post-deletion offset is
    computed exactly once. Built from two passes they drift silently: after deleting row d
    every later feature's latent id shifts down by one, and feature f would be scored against
    feature f+1's row everywhere past the first deletion.

    A kept-but-unclaimed row would be a DEAD LATENT — a different pathology, and one that moves
    the census's span-calibrated eps rather than the recovery count.
    """
    F = int(g.shape[0])
    drop = set(select_features(F, float(dials.fraction), world_seed, MISSING_SEED_OFFSET))
    keep = [f for f in range(F) if f not in drop]
    W = g.double()[torch.tensor(keep, dtype=torch.long)]
    f2l: list[tuple[int, ...]] = [()] * F
    for pos, f in enumerate(keep):
        f2l[f] = (pos,)
    return Corruption(dials=dials, W_raw=W,
                      planted_map=PlantedMap(feature_to_latents=tuple(f2l), readout=readout,
                                             n_latents=len(keep)),
                      corrupted_features=tuple(sorted(drop)),
                      corrupted_pair_rule="either_endpoint_feature")


def split(g: torch.Tensor, cont_edges, dials: SplitDials, world_seed: int,
          readout: str = "identity") -> Corruption:
    """Carry each chosen feature on `k` latents that SHARE its direction: L = F + (k-1)*n.

    Latent ids are feature-major and CONTIGUOUS, which is load-bearing rather than cosmetic:
    `support[t].nonzero()` returns ascending ids, so monotone numbering makes a token's active
    rows arrive in the same order as the unsplit solve. Together with shards sharing one
    direction and firing disjointly, that makes the per-token Gram matrix elementwise identical
    to the unsplit one — which is why the `union` readout reproduces the uncorrupted read
    bit-for-bit in the firing channel.
    """
    F = int(g.shape[0])
    chosen = set(select_features(F, float(dials.fraction), world_seed, SPLIT_SEED_OFFSET))
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
                      corrupted_pair_rule="either_endpoint_feature")


def expand_support(support: torch.Tensor, corruption: Corruption, world_seed: int,
                   sample_seed: int) -> torch.Tensor:
    """`[n, F]` feature-space support -> `[n, L]` latent-space support, through the planted map.

    The one new seam of round 2, applied immediately before the ridge solve. `apply_hole` stays
    feature-space above it on purpose (it carries six mutation anchors, and the approved cell
    matrix never composes absorption with splitting).

    A split feature's firing tokens are PARTITIONED: `randperm` plus contiguous slices at
    cumulative counts, so disjointness and exact coverage are true by construction rather than
    on average. Per-token sampling would give both only in expectation, and two shards live on
    one token duplicates a Gram row — a ~5e-5 relative magnitude error, finite and plausible
    and invisible to any tolerance looser than 1e-6.
    """
    pmap = corruption.planted_map
    n, F = support.shape
    if F != pmap.F:
        raise ValueError(f"support is {F} features wide but the planted map covers {pmap.F}")
    out = torch.zeros(n, pmap.n_latents, dtype=torch.bool, device=support.device)
    for f, lats in enumerate(pmap.feature_to_latents):
        if not lats:
            continue                                   # deleted feature: no latent fires
        if len(lats) == 1:
            out[:, lats[0]] = support[:, f]
            continue
        if not isinstance(corruption.dials, SplitDials):
            raise ValueError(
                f"{corruption.kind} declares feature {f} on {len(lats)} latents but carries no "
                f"share policy; a partition needs declared shares, not a default")
        tok = support[:, f].nonzero(as_tuple=True)[0]
        n_f = int(tok.numel())
        gen = torch.Generator().manual_seed(
            shard_generator_seed(world_seed, sample_seed, f))
        perm = torch.randperm(n_f, generator=gen)
        shares = shard_shares(len(lats), float(corruption.dials.skew))
        # Cumulative bounds, not per-shard counts: rounding each share independently would not
        # sum to n_f and would silently drop or duplicate tokens.
        bounds, cum = [], 0.0
        for s in shares:
            cum += s
            bounds.append(int(round(cum * n_f)))
        bounds[-1] = n_f
        start = 0
        for i, end in enumerate(bounds):
            out[tok[perm[start:end]], lats[i]] = True
            start = end
    return out


# Damage registry: kind -> builder. `build_corruption` is the only dispatch, so a dials class
# and its builder cannot disagree about which damage is being applied.
CORRUPTIONS = {
    "absorption": absorb,
    "split": split,
    "missing": missing,
    "noise": noise,
}


def build_corruption(g: torch.Tensor, cont_edges, dials, world_seed: int,
                     readout: str = "identity") -> Corruption:
    """The dictionary this dials object describes. The damage is chosen by the dials TYPE, so
    there is no way to ask for one damage with another's knobs."""
    kind = type(dials).KIND
    builder = CORRUPTIONS.get(kind)
    if builder is None:
        raise ValueError(f"no corruption builder registered for kind {kind!r}; "
                         f"registered damages are {tuple(CORRUPTIONS)}")
    return builder(g, cont_edges, dials, world_seed, readout)
