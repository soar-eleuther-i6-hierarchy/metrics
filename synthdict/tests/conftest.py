"""Test bootstrap: put the experiment_0 root on sys.path (the tests_local convention).

Also provides the small shared world fixtures. Worlds are shrunk via `spec.replace` on the
real configs (fewer roots, fewer tokens) so the unit tests run in seconds on CPU; the
geometry/sampling code paths are the real ones.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


def small_world(toy: str, n_roots: int = 12, n_tokens: int = 3000, seed: int = 0,
                sample_seed: int | None = None):
    """A shrunken but real toy world (same generator code, fewer features/tokens)."""
    from scoring.core.world import regenerate_world
    from toygen import spec
    from toygen.world import resolve_config

    cfg = spec.replace(resolve_config(toy), seed=seed, n_roots=n_roots)
    rc = dataclasses.asdict(cfg)
    return regenerate_world(rc, sample_seed=seed if sample_seed is None else sample_seed,
                            n_tokens=n_tokens)


@pytest.fixture(scope="session")
def isa_world():
    return small_world("only_isa")


@pytest.fixture(scope="session")
def firing_world():
    return small_world("only_firing")
