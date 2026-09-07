"""Latent-side corruption generators — physical dials only, no instrument thresholds.

Round 1 implements ABSORPTION per `SYNTH_PRECOMMIT.md`:

  decoder carry   d_c' = unit(g_c + beta * rbar * g_p)     (rbar = 1.0: flat-strength design)
  firing hole     parent removed from the planted support on round(eta * n_c) child tokens
  edge selection  a deterministic `edge_fraction` of the tree's containment edges

TAUTOLOGY GUARD (tested): this module imports neither `ABSORPTION_CONSTANTS` (the census's
thresholds) nor `scoring.core.registry.CONSTANTS` (the detectors'). A corruption defined in
the instrument's own vocabulary would make "the instrument responds" true by construction.

Severity is REALIZED per edge as `cos(d_c', g_p)`; `beta` is only the generator knob
(PRECOMMIT: the dose-response is reported against realized severity).

Determinism contract:
  * the corrupted EDGE SET depends on (world_seed, dials) only — every draw of one world
    corrupts the same edges;
  * the HOLE token subset depends on (world_seed, sample_seed, edge) — deterministic per
    draw, different across draws, and never consumes the global RNG or the world's streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_TINY = 1e-12

# Dedicated seed offsets for the two private RNG streams. Chosen away from the repo's other
# derivations (toygen STRUCTURE_SEED_OFFSET=9973, held-out +10000, probe fit +20000).
EDGE_SEED_OFFSET = 61_211
HOLE_SEED_OFFSET = 77_003


@dataclass(frozen=True)
class AbsorptionDials:
    """The physical knobs. `rbar` is the parent/child mean-magnitude ratio; 1.0 is analytic
    for this generator (toygen/strengths.py builds a FLAT mean strength q*sqrt(E0))."""

    beta: float
    eta: float
    edge_fraction: float = 1.0
    rbar: float = 1.0


@dataclass(frozen=True)
class Corruption:
    """One synthetic dictionary's corruption record.

    W_raw              [F, D] float64 synthetic raw decoder rows (unit-norm; uncorrupted
                       rows are the true g rows unchanged)
    corrupted_edges    ordered (parent, child) tuples, a deterministic subset of the tree's
                       containment edges
    realized_severity  [n_corrupted] cos(d_c', g_p) per corrupted edge, in edge order
    """

    kind: str
    dials: AbsorptionDials
    corrupted_edges: tuple[tuple[int, int], ...]
    W_raw: torch.Tensor
    realized_severity: torch.Tensor


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


def absorb(g: torch.Tensor, cont_edges, dials: AbsorptionDials, world_seed: int) -> Corruption:
    """Build the absorbed dictionary: carry `beta * rbar` of the parent direction into each
    corrupted child's decoder row, renormalized to unit norm.

    `g` rows are unit-norm by generator construction (tested); uncorrupted rows pass through
    EXACTLY (`torch.equal`), so beta=0 or an unselected edge changes nothing at all.
    """
    edges = select_edges(cont_edges, dials.edge_fraction, world_seed)
    W = g.double().clone()
    sev = torch.empty(len(edges), dtype=torch.float64)
    for i, (p, c) in enumerate(edges):
        if dials.beta != 0.0:
            d = g[c].double() + float(dials.beta) * float(dials.rbar) * g[p].double()
            W[c] = d / d.norm().clamp_min(_TINY)
        gp = g[p].double()
        sev[i] = float((W[c] @ gp) / (W[c].norm() * gp.norm()).clamp_min(_TINY))
    return Corruption(kind="absorption", dials=dials, corrupted_edges=edges,
                      W_raw=W, realized_severity=sev)


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


# Registry for later pathologies (splitting, merging, ... — out of scope in round 1;
# SYNTH_PRECOMMIT.md). Each entry: name -> builder returning a Corruption.
CORRUPTIONS = {
    "absorption": absorb,
}
