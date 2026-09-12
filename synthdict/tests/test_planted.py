"""PlantedMap — the synthetic path's feature->latent correspondence, planted not inferred.

Each test names the silent failure it catches; shape/type assertions are deliberately absent
(those crash loudly on their own).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from synthdict.planted import READOUTS, PlantedMap


# --------------------------------------------------------------------------
# the identity map (every round-1 damage)
# --------------------------------------------------------------------------
def test_identity_columns_are_in_feature_order():
    # Silent failure: columns() returning a set/sorted-by-something order would score
    # feature f against another feature's latent — every metric still finite, all wrong.
    pm = PlantedMap.identity(5)
    assert torch.equal(pm.columns(), torch.arange(5))
    assert pm.n_latents == 5


def test_identity_recovers_everything():
    # Declarative recovery: a planted feature is present because we planted it.
    pm = PlantedMap.identity(4)
    assert torch.equal(pm.recovered(), torch.ones(4, dtype=torch.bool))


def test_columns_follow_the_declared_map_not_position():
    # Silent failure: ignoring feature_to_latents and returning arange() regardless — which
    # is correct for identity and silently wrong for any permuted or round-2 map.
    pm = PlantedMap(feature_to_latents=((2,), (0,), (1,)), readout="identity", n_latents=3)
    assert torch.equal(pm.columns(), torch.tensor([2, 0, 1]))


def test_recovered_is_false_exactly_where_a_feature_has_no_latent():
    # Round 2's missing-latent damage is the only thing that makes an entry empty; if this
    # read all-True regardless, a deleted feature would be scored against latent 0.
    pm = PlantedMap(feature_to_latents=((0,), (), (1,)), readout="identity", n_latents=2)
    assert torch.equal(pm.recovered(), torch.tensor([True, False, True]))


def test_identity_readout_refuses_a_multi_latent_feature():
    # The trap this whole design exists to avoid: silently picking shard[0] as "the" latent.
    # A sharded feature under the identity readout must fail loudly, not choose for us.
    pm = PlantedMap(feature_to_latents=((0,), (1, 2)), readout="identity", n_latents=3)
    with pytest.raises(ValueError, match="readout"):
        pm.columns()


# --------------------------------------------------------------------------
# the round-2 seam
# --------------------------------------------------------------------------
def test_unimplemented_readouts_raise_rather_than_guess():
    # A readout that silently fell back to identity would publish numbers under a policy
    # nobody registered.
    for readout in ("strongest_shard", "union"):
        pm = PlantedMap(feature_to_latents=((0,), (1, 2)), readout=readout, n_latents=3)
        with pytest.raises(NotImplementedError):
            pm.columns()


def test_unknown_readout_is_rejected_at_construction():
    with pytest.raises(ValueError):
        PlantedMap(feature_to_latents=((0,),), readout="argmax", n_latents=1)
    assert "identity" in READOUTS


def test_a_latent_id_outside_the_dictionary_is_rejected():
    # Silent failure: an out-of-range column would index-error deep inside torch with a
    # message that says nothing about the planted map.
    with pytest.raises(ValueError):
        PlantedMap(feature_to_latents=((0,), (7,)), readout="identity", n_latents=2)


def test_sha256_distinguishes_different_maps():
    # Provenance: two runs whose maps differ must be distinguishable on disk.
    a = PlantedMap.identity(3)
    b = PlantedMap(feature_to_latents=((0,), (2,), (1,)), readout="identity", n_latents=3)
    assert a.sha256() != b.sha256()
    assert a.sha256() == PlantedMap.identity(3).sha256()
    assert len(a.sha256()) == 64


# --------------------------------------------------------------------------
# the structural guard — "matcher-free" enforced, not asserted
# --------------------------------------------------------------------------
BANNED_NAMES = frozenset({"match_features", "activation_corr", "reduce_to_recovered"})
BANNED_MODULES = frozenset({"scoring.core.recovery"})


def scan_for_matcher(paths) -> list[str]:
    """Names/imports from the matcher found in `paths`. Separated from the test so it can be
    driven by a POSITIVE CONTROL: on a clean tree the real scan finds nothing, so without one
    the scanner could be emptied and still look green (the trap the repo's doc-ref guard hit)."""
    offenders = []
    for path in paths:
        tree = ast.parse(Path(path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in BANNED_MODULES:
                    offenders.append(f"{Path(path).name}: from {node.module}")
                offenders += [f"{Path(path).name}: {a.name}" for a in node.names
                              if a.name in BANNED_NAMES]
            elif isinstance(node, ast.Import):
                offenders += [f"{Path(path).name}: {a.name}" for a in node.names
                              if a.name in BANNED_MODULES]
            elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
                offenders.append(f"{Path(path).name}: {node.id}()")
    return offenders


def test_the_matcher_scanner_actually_detects_each_banned_form(tmp_path):
    """Positive control for the guard below. Each banned form must be caught on its own."""
    forms = {
        "from_module.py": "from scoring.core.recovery import match_features\n",
        # only the MODULE branch catches this one: the name imported is not itself banned,
        # so a name-only scanner would wave the matcher module straight through
        "from_module_alias.py": "from scoring.core.recovery import MatchResult\n",
        "import_module.py": "import scoring.core.recovery\n",
        "bare_call.py": "def f(c, a):\n    return match_features(c, a)\n",
        "reduce.py": "from scoring.core.grid import reduce_to_recovered\n",
    }
    for name, body in forms.items():
        f = tmp_path / name
        f.write_text(body)
        assert scan_for_matcher([f]), f"scanner missed {name}: {body!r}"
    clean = tmp_path / "clean.py"
    clean.write_text("from synthdict.planted import PlantedMap\nx = PlantedMap.identity(2)\n")
    assert scan_for_matcher([clean]) == []        # and does not cry wolf


def test_no_matcher_anywhere_in_synthdict():
    """No module under `synthdict/` may import or call the matcher.

    Matching solves an INVERSE problem (whose dictionary is it?); synthesis has no inverse
    problem. This is the mechanical version of that claim - same pattern as the tautology
    guard in test_corruptions.py. Tests are scanned too: a test that reaches for the matcher
    would mean the production path could.
    """
    pkg = Path(__file__).resolve().parents[1]
    paths = [p for p in sorted(pkg.rglob("*.py")) if p.name != Path(__file__).name]
    offenders = scan_for_matcher(paths)
    assert not offenders, f"matcher reached the synthetic path: {offenders}"


def test_a_latent_shared_by_two_features_is_rejected():
    # A shared latent is a MERGE and needs a declared readout; allowing it here would score
    # two features on one column with every metric finite and both wrong.
    with pytest.raises(ValueError, match="merge"):
        PlantedMap(feature_to_latents=((0,), (0,)), readout="identity", n_latents=1)


def test_feature_lookup_is_feature_indexed_not_position_indexed():
    # `classify_dictionary` does `match[c]` with a TRUE FEATURE id. Handing it the
    # position-indexed columns() silently scores feature 2 against feature 3's latent as soon
    # as any feature is missing - finite, plausible, wrong.
    pm = PlantedMap(feature_to_latents=((0,), (), (2,), (3,)), readout="identity", n_latents=4)
    assert torch.equal(pm.feature_lookup(), torch.tensor([0, -1, 2, 3]))
    assert torch.equal(pm.columns(), torch.tensor([0, 2, 3]))     # positions, shorter than F
    assert pm.feature_lookup().shape[0] == pm.F


def test_resolve_map_rejects_a_map_from_a_different_world():
    from synthdict.planted import resolve_map

    class _Corruption:
        planted_map = PlantedMap.identity(3)

    with pytest.raises(ValueError, match="different world"):
        resolve_map(_Corruption(), F=5)
