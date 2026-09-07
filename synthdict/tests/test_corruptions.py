"""corruptions.absorb — decoder carry, edge selection, the eta-hole, and the tautology guard.

Every test here is also a mutation anchor: the harness in `mutation_synthdict.py` flips the
specific behavior and expects the named test to fail.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from synthdict.corruptions import (AbsorptionDials, absorb, apply_hole, hole_generator_seed,
                                   select_edges)

_TINY = 1e-12


def _dials(beta=0.6, eta=0.6, f=1.0):
    return AbsorptionDials(beta=beta, eta=eta, edge_fraction=f)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(_TINY))


# --------------------------------------------------------------------------
# decoder carry
# --------------------------------------------------------------------------
def test_g_rows_are_unit_norm(isa_world):
    # Construction assumption of the whole study; if the generator ever changes this,
    # the rbar=1 analytic ratio and the severity formula both need revisiting.
    norms = isa_world.g.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)


def test_zero_beta_leaves_every_decoder_row_identical(isa_world):
    c = absorb(isa_world.g, isa_world.CONT, _dials(beta=0.0, eta=0.6), world_seed=0)
    assert torch.equal(c.W_raw, isa_world.g.double())


def test_uncorrupted_rows_untouched_at_partial_fraction(isa_world):
    c = absorb(isa_world.g, isa_world.CONT, _dials(beta=0.8, f=0.5), world_seed=0)
    corrupted_children = {ch for _, ch in c.corrupted_edges}
    for k in range(isa_world.g.shape[0]):
        if k not in corrupted_children:
            assert torch.equal(c.W_raw[k], isa_world.g[k].double()), f"row {k} moved"


def test_corrupted_rows_are_unit_norm(isa_world):
    c = absorb(isa_world.g, isa_world.CONT, _dials(beta=0.8), world_seed=0)
    for _, ch in c.corrupted_edges:
        assert abs(float(c.W_raw[ch].norm()) - 1.0) < 1e-8


def test_carry_direction_is_g_c_plus_beta_g_p(isa_world):
    beta = 0.6
    c = absorb(isa_world.g, isa_world.CONT, _dials(beta=beta), world_seed=0)
    p, ch = c.corrupted_edges[0]
    expect = isa_world.g[ch].double() + beta * isa_world.g[p].double()
    expect = expect / expect.norm()
    assert torch.allclose(c.W_raw[ch], expect, atol=1e-10)


def test_realized_severity_is_cos_with_parent_not_child(isa_world):
    # At beta=0 severity == alpha (0.48 here), NOT ~1. Measuring against g_c would read
    # cos(d', g_c) = 1 at beta=0 — the exact mutation this anchors.
    c = absorb(isa_world.g, isa_world.CONT, _dials(beta=0.0), world_seed=0)
    alpha = 0.48
    assert torch.allclose(c.realized_severity,
                          torch.full_like(c.realized_severity, alpha), atol=1e-6)


def test_realized_severity_zero_at_beta0_on_orthogonal_world(firing_world):
    c = absorb(firing_world.g, firing_world.CONT, _dials(beta=0.0), world_seed=0)
    assert float(c.realized_severity.abs().max()) < 1e-8


def test_realized_severity_increases_with_beta(firing_world):
    sev = []
    for beta in (0.2, 0.5, 0.9):
        c = absorb(firing_world.g, firing_world.CONT, _dials(beta=beta), world_seed=0)
        sev.append(float(c.realized_severity.median()))
        # analytic on an orthogonal edge: beta / sqrt(1 + beta^2)
        assert abs(sev[-1] - beta / math.sqrt(1 + beta ** 2)) < 1e-6
    assert sev[0] < sev[1] < sev[2]


# --------------------------------------------------------------------------
# edge selection
# --------------------------------------------------------------------------
def test_edge_fraction_selects_the_right_count(isa_world):
    n_edges = len(isa_world.CONT)
    for f in (0.1, 0.5, 1.0):
        sel = select_edges(isa_world.CONT, f, world_seed=0)
        assert len(sel) == max(1, round(f * n_edges))
    assert select_edges(isa_world.CONT, 1.0, world_seed=0) == tuple(isa_world.CONT)


def test_edge_selection_deterministic_and_seed_sensitive(isa_world):
    a = select_edges(isa_world.CONT, 0.3, world_seed=0)
    b = select_edges(isa_world.CONT, 0.3, world_seed=0)
    c = select_edges(isa_world.CONT, 0.3, world_seed=7)
    assert a == b
    assert a != c


def test_edge_selection_ignores_sample_seed(isa_world):
    # The corrupted edge set is a property of the WORLD (dial + world seed): every draw of the
    # same world must corrupt the same edges, or the matching/scoring/probe draws would see
    # three different dictionaries.
    import inspect

    sig = inspect.signature(select_edges)
    assert "sample_seed" not in sig.parameters


# --------------------------------------------------------------------------
# the eta-hole
# --------------------------------------------------------------------------
def _hole_setup(world, eta=0.5, f=1.0):
    c = absorb(world.g, world.CONT, _dials(beta=0.6, eta=eta, f=f), world_seed=0)
    support = world.A > 0
    holed, n_holed = apply_hole(support, c, world_seed=0, sample_seed=123)
    return c, support, holed, n_holed


def test_hole_removes_parent_on_eta_fraction_of_child_tokens(isa_world):
    c, support, holed, n_holed = _hole_setup(isa_world, eta=0.5)
    for (p, ch) in c.corrupted_edges:
        child_tok = support[:, ch]
        n_child = int(child_tok.sum())
        want = round(0.5 * n_child)
        # exactly `want` tokens lost parent firing, all of them child-firing tokens
        lost = support[:, p] & ~holed[:, p]
        assert int(lost.sum()) == want
        assert bool((lost & ~child_tok).any()) is False
        assert n_holed[(p, ch)] == want


def test_hole_touches_only_the_parent_column(isa_world):
    c, support, holed, _ = _hole_setup(isa_world, eta=0.7)
    parents = {p for p, _ in c.corrupted_edges}
    for k in range(support.shape[1]):
        if k not in parents:
            assert torch.equal(support[:, k], holed[:, k]), f"column {k} moved"


def test_hole_never_touches_the_child_column(isa_world):
    # Anchors the "hole applied to child instead of parent" mutation, including the case
    # where a feature is both a parent and a child (not in these depth-1 toys, but the
    # function must not rely on that).
    c, support, holed, _ = _hole_setup(isa_world, eta=1.0)
    for (_, ch) in c.corrupted_edges:
        assert torch.equal(support[:, ch], holed[:, ch])


def test_eta_zero_is_identity(isa_world):
    c, support, holed, n_holed = _hole_setup(isa_world, eta=0.0)
    assert torch.equal(support, holed)
    assert all(v == 0 for v in n_holed.values())


def test_eta_one_removes_parent_on_all_child_tokens(isa_world):
    c, support, holed, _ = _hole_setup(isa_world, eta=1.0)
    for (p, ch) in c.corrupted_edges:
        assert not bool((holed[:, p] & support[:, ch]).any())


def test_hole_deterministic_and_draw_dependent(isa_world):
    c = absorb(isa_world.g, isa_world.CONT, _dials(eta=0.5), world_seed=0)
    support = isa_world.A > 0
    h1, _ = apply_hole(support, c, world_seed=0, sample_seed=123)
    h2, _ = apply_hole(support, c, world_seed=0, sample_seed=123)
    h3, _ = apply_hole(support, c, world_seed=0, sample_seed=124)
    assert torch.equal(h1, h2)
    assert not torch.equal(h1, h3)      # a different draw holes a different token subset


def test_hole_seed_depends_on_world_seed(isa_world):
    # "per-world deterministic RNG": dropping the world seed from the derivation is the
    # mutation this catches.
    s0 = hole_generator_seed(world_seed=0, sample_seed=5, p=1, c=2)
    s1 = hole_generator_seed(world_seed=9, sample_seed=5, p=1, c=2)
    s2 = hole_generator_seed(world_seed=0, sample_seed=6, p=1, c=2)
    s3 = hole_generator_seed(world_seed=0, sample_seed=5, p=2, c=1)
    assert len({s0, s1, s2, s3}) == 4


def test_hole_does_not_consume_global_rng(isa_world):
    c = absorb(isa_world.g, isa_world.CONT, _dials(eta=0.5), world_seed=0)
    support = isa_world.A > 0
    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    apply_hole(support, c, world_seed=0, sample_seed=123)
    after = torch.rand(3)
    assert torch.equal(before, after)


# --------------------------------------------------------------------------
# tautology guard
# --------------------------------------------------------------------------
def test_generator_imports_no_detector_or_census_constants():
    """corruptions.py must be parameterized by physical dials, never by the thresholds of the
    instruments it will be measured with (the 33-tautologies lesson)."""
    src = (Path(__file__).resolve().parents[1] / "corruptions.py").read_text()
    tree = ast.parse(src)
    banned_names = {"ABSORPTION_CONSTANTS", "CONSTANTS"}
    banned_modules = {"scoring.trained.absorption", "scoring.core.registry"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in banned_modules, f"imports {node.module}"
            for a in node.names:
                assert a.name not in banned_names, f"imports {a.name}"
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name not in banned_modules, f"imports {a.name}"


def test_rbar_scales_the_carry(isa_world):
    # rbar defaults to the analytic 1.0; a dropped rbar factor is invisible at the default,
    # so this pins it at rbar=2.
    d = AbsorptionDials(beta=0.5, eta=0.0, edge_fraction=1.0, rbar=2.0)
    c = absorb(isa_world.g, isa_world.CONT, d, world_seed=0)
    p, ch = c.corrupted_edges[0]
    expect = isa_world.g[ch].double() + 0.5 * 2.0 * isa_world.g[p].double()
    expect = expect / expect.norm()
    assert torch.allclose(c.W_raw[ch], expect, atol=1e-10)
