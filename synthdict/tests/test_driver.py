"""run_synth + census — end-to-end on a small world into a tmp tree, schema-checked.

One expensive integration test on purpose: it exercises the exact writer path the grid uses
(write_artifacts' REQUIRED_META + report_schema enforcement included), then re-reads its own
artifacts the way report.py will.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from synthdict.corruptions import AbsorptionDials, absorb
from synthdict.run_synth import dial_dirname, run_dial_point, synthdict_sha256

SMALL = {"n_roots": 12}
N_TOK = 3000


def test_dial_point_end_to_end(tmp_path):
    # ASYMMETRIC dials on purpose: a beta/eta swap anywhere in the path must fail loudly
    dials = AbsorptionDials(beta=0.6, eta=0.3, edge_fraction=0.5)
    written = run_dial_point("only_isa", 0, dials, N_TOK, tmp_path, "TEST",
                             readout="identity",
                             with_probe=False, with_census=True,
                             cfg_overrides=SMALL)
    assert len(written) == 1
    for d in written:
        npz = np.load(d / "scores.npz", allow_pickle=True)
        meta = json.loads(str(npz["__meta__"]))
        assert meta["read"] == "synthetic"
        assert meta["report_schema"] == 2
        assert meta["checkpoint"] == "synthetic:none"
        assert meta["dials"]["beta"] == 0.6 and meta["dials"]["eta"] == 0.3
        assert meta["n_corrupted_edges"] == round(0.5 * 12)
        assert len(meta["checkpoint_weights_sha256"]) == 64
        assert len(meta["synthdict_sha256"]) == 64
        assert "corrupted_pair" in npz.files and "realized_severity" in npz.files
        assert npz["corrupted_pair"].sum() == meta["n_corrupted_edges"]

        report = json.loads((d / "expressions.json").read_text())
        assert len(report["expressions"]) == 8

        cen = json.loads((d / "census.json").read_text())
        assert "caveat" in cen and "counts" in cen
        assert cen["n_planted_corrupted_edges"] == meta["n_corrupted_edges"]
        # IDENTITY of the planted set, not just its count: the census rebuilds the corruption
        # deterministically, and a rebuild under the wrong seed matches on count at any f
        assert cen["corrupted_edges_sha256"] == meta["corrupted_edges_sha256"]
        assert len(cen["absorption_classifier_sha256"]) == 64

    assert written[0].name == "identity"          # the readout, now the only path segment
    # the literal, not dial_dirname(dials): comparing the path against the function that
    # built it can never fail (audit: half compare-with-self)
    assert written[0].parent.name == "beta0.6-eta0.3-f0.5"


def test_census_counts_planted_absorption_at_high_dials(firing_world):
    # Manipulation check of the manipulation check: strong planted absorption should register
    # in the census; zero dials must register nothing. Directional only — the census's eps is
    # its own instrument and its exact count is annotation, not a claim.
    #
    # MEASURED small-world instrument fact (recorded, not worked around silently): the
    # census's Bonferroni eps is span-calibrated, and in this 24-feature world the span has
    # rank ~24, putting eps at ~0.65 — so beta=0.8 carry (resid 0.45-0.63) reads CLEAN here.
    # Only beta=1.2 on the orthogonal world (resid 0.77) clears it. At grid scale (F=240,
    # D=128) eps sits far lower; the timed dial point checks census sensitivity there.
    import dataclasses

    from synthdict.census import run_census
    from toygen import spec
    from toygen.world import resolve_config

    rc = dataclasses.asdict(spec.replace(resolve_config("only_firing"), seed=0, n_roots=12))
    hot = absorb(firing_world.g, firing_world.CONT,
                 AbsorptionDials(beta=1.2, eta=0.9, edge_fraction=1.0), world_seed=0)
    cold = absorb(firing_world.g, firing_world.CONT,
                  AbsorptionDials(beta=0.0, eta=0.0, edge_fraction=1.0), world_seed=0)
    hot_c = run_census(rc, hot, 0, n_tokens=3000, readout="identity")
    cold_c = run_census(rc, cold, 0, n_tokens=3000, readout="identity")
    assert cold_c["counts"]["absorbed"] == 0
    assert hot_c["counts"]["absorbed"] > 0
    assert hot_c["n_census_absorbed_among_planted"] == hot_c["n_census_absorbed"]


def test_synthdict_sha_changes_with_source(tmp_path):
    # the content hash is the server-side provenance (git_sha is "unavailable" there)
    a = synthdict_sha256()
    pkg = tmp_path / "synthdict"
    pkg.mkdir()
    (pkg / "x.py").write_text("x = 1\n")
    b = synthdict_sha256(root=tmp_path)
    assert a != b and len(a) == 64


def test_census_handles_a_map_with_a_missing_feature(firing_world):
    """Census must index the planted map by FEATURE id, not by scored position.

    With a gap in the map the two differ in length, and the position-indexed version walks off
    the end inside `classify_dictionary` (which does `match[c]` for a true child id). Round-1
    identity maps cannot tell them apart, so this is the test that does.
    """
    import dataclasses as dc

    from synthdict.census import run_census
    from synthdict.planted import PlantedMap
    from toygen import spec
    from toygen.world import resolve_config

    rc = dc.asdict(spec.replace(resolve_config("only_firing"), seed=0, n_roots=12))
    base = absorb(firing_world.g, firing_world.CONT,
                  AbsorptionDials(beta=0.6, eta=0.5, edge_fraction=1.0), world_seed=0)
    F = int(firing_world.g.shape[0])
    gapped = PlantedMap(
        feature_to_latents=tuple(() if f == 1 else (f,) for f in range(F)),
        readout="identity", n_latents=F)
    cen = run_census(rc, dc.replace(base, planted_map=gapped), 0,
                     n_tokens=3000, readout="identity")
    assert cen["n_recovered"] == F - 1
