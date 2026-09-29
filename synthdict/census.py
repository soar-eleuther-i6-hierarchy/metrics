"""Census annotation: the standard dictionary-damage classification, run on a synthetic dictionary.

A manipulation check that the planting registered, not validation: `classify_dictionary` shares
its definitions with the planted damage. Its blind spots are in `CAVEAT`, stamped on every output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from scoring.core.world import regenerate_world, signed_normalized_decoder
from scoring.trained.absorption import classify_dictionary, tree_edges_and_siblings

from synthdict.corruptions import Corruption
from synthdict.planted import resolve_map
from synthdict.read import synth_encode

CAVEAT = ("annotation only: shares definitions with the planted pathology (manipulation "
          "check, not validation); eps is span-calibrated, so counts at edge_fraction~1.0 "
          "can undercount planted carry; the conjunction test runs on sibling pairs only, so "
          "a planted composition is never flagged; a split past k=4 equal shards reads clean; "
          "only tree edges are classified, so container/register absorption is invisible")


def absorption_classifier_sha256() -> str:
    """Content hash of `scoring/trained/absorption.py`, which neither evaluator_sha256 nor
    synthdict_sha256 covers."""
    import scoring.trained.absorption as _absorption_mod

    return hashlib.sha256(Path(_absorption_mod.__file__).read_bytes()).hexdigest()


def run_census(rc: dict, corruption: Corruption | None, seed: int, n_tokens: int,
               readout: str) -> dict:
    """Classify the synthetic dictionary against ground truth on the in-sample draw
    (`sample_seed = seed`)."""
    sample_seed = int(seed)
    inw = regenerate_world(rc, sample_seed=sample_seed, n_tokens=n_tokens)
    acts, _support, _holed = synth_encode(inw, corruption, seed, sample_seed)
    W_raw = corruption.W_raw if corruption is not None else inw.g.double()
    oriented = signed_normalized_decoder(W_raw, acts, inw.h)
    F = int(inw.g.shape[0])
    # `match` comes from the planted map, not a matcher; `classify_dictionary` never reads
    # `matched_corr`, so a constant is safe.
    pmap = resolve_map(corruption, F, readout)
    # One latent per feature under every readout, so the strongest shard stands in; absorption
    # spread over a split child's shards can be missed, hence `census_latent_policy`.
    match = pmap.representative_lookup()
    census_latent_policy = "representative=strongest_declared_shard"
    matched_corr = torch.ones(F, dtype=torch.float64)
    recovered = pmap.recovered()

    cont_edges, sibling_pairs, isa_child, descendants = tree_edges_and_siblings(inw.tree)
    cls = classify_dictionary(inw.g, oriented, inw.A, acts, match, matched_corr, recovered,
                              cont_edges, sibling_pairs, isa_child, descendants)

    corrupted = set(corruption.corrupted_edges) if corruption is not None else set()
    # planted edges the classifier never looks at, kept apart so they do not read as misses
    outside_tree = corrupted - set(cont_edges)
    absorbed_edges = cls.get("absorbed_edges", [])
    absorbed_set = {(int(e["parent"]), int(e["child"])) for e in absorbed_edges}
    # A rebuild with the wrong seed matches the driver's meta on count but not on this hash.
    edges_list = list(corruption.corrupted_edges) if corruption is not None else []
    edges_sha = hashlib.sha256(json.dumps(edges_list).encode()).hexdigest()  # == the driver's form
    classifier_sha = absorption_classifier_sha256()
    return {
        "caveat": CAVEAT,
        "corrupted_edges_sha256": edges_sha,
        "absorption_classifier_sha256": classifier_sha,
        "readout": readout,
        "census_latent_policy": census_latent_policy,
        "planted_map_sha256": pmap.sha256(),
        "counts": cls["counts"],
        "absorbed_by_relation": cls["absorbed_by_relation"],
        "n_planted_corrupted_edges": len(corrupted),
        "n_census_absorbed": len(absorbed_set),
        "n_planted_edges_outside_tree": len(outside_tree),
        "n_census_absorbed_among_planted": (None if outside_tree
                                            else len(absorbed_set & corrupted)),
        "n_census_absorbed_outside_planted": len(absorbed_set - corrupted),
        "sample_seed": sample_seed,
        "theta_hat": sorted(float(e.get("theta_hat", float("nan"))) for e in absorbed_edges),
        "eps": cls.get("eps"),
        "n_recovered": int(recovered.sum()),
    }
