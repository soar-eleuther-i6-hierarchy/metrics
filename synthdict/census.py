"""Census annotation: the standard dictionary-damage classification, run on a synthetic dictionary.

ANNOTATION, NOT VALIDATION. `classify_dictionary` shares its definitions with the planted
pathology (decoder-carry + firing hole IS absorption's operational definition), so "the census
counts what we planted" is a manipulation check that the planting registered — never evidence
that the metrics work. The claims of the study are about the registered expressions, which are
a different code path (`SYNTH_PRECOMMIT.md`).

Known caveat recorded with every output (W8 in the plan): the census's eps is null-calibrated
INSIDE the decoder span, so at edge_fraction ~ 1.0 the span itself carries the parent
directions and the calibrated eps shifts — counts at high f can undercount planted carry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from scoring.core.recovery import activation_corr, match_features
from scoring.core.registry import CONSTANTS
from scoring.core.world import regenerate_world, signed_normalized_decoder
from scoring.trained.absorption import classify_dictionary, tree_edges_and_siblings

from synthdict.corruptions import Corruption
from synthdict.read import synth_encode

CAVEAT = ("annotation only: shares definitions with the planted pathology (manipulation "
          "check, not validation); eps is span-calibrated, so counts at edge_fraction~1.0 "
          "can undercount planted carry")


def run_census(rc: dict, corruption: Corruption | None, seed: int, n_tokens: int,
               match_mode: str, acts_mode: str = "ridge") -> dict:
    """Classify the synthetic dictionary against ground truth on the MATCHING draw
    (parity with `scoring/trained/retrieval.py`'s in-sample census)."""
    inw = regenerate_world(rc, sample_seed=int(seed), n_tokens=n_tokens)
    acts, _support, _holed = synth_encode(inw, corruption, seed, int(seed), acts_mode)
    W_raw = corruption.W_raw if corruption is not None else inw.g.double()
    oriented = signed_normalized_decoder(W_raw, acts, inw.h)
    F = int(inw.g.shape[0])
    if match_mode == "identity":
        match = torch.arange(F)
        matched_corr = torch.ones(F, dtype=torch.float64)
        recovered = torch.ones(F, dtype=torch.bool)
    else:
        res = match_features(activation_corr(inw.A, acts), inw.g, oriented,
                             rho=CONSTANTS["rho_star"])
        match, matched_corr, recovered = res.match, res.matched_corr, res.recovered

    cont_edges, sibling_pairs, isa_child, descendants = tree_edges_and_siblings(inw.tree)
    cls = classify_dictionary(inw.g, oriented, inw.A, acts, match, matched_corr, recovered,
                              cont_edges, sibling_pairs, isa_child, descendants)

    corrupted = set(corruption.corrupted_edges) if corruption is not None else set()
    absorbed_edges = cls.get("absorbed_edges", [])
    absorbed_set = {(int(e["parent"]), int(e["child"])) for e in absorbed_edges}
    # Provenance the driver's meta can be cross-checked against: WHICH planted set this census
    # counted against (a rebuild with the wrong seed matches on count but not on identity),
    # and WHICH classifier source produced the counts (scoring/trained is outside both
    # evaluator_sha256 and synthdict_sha256, and the server copy has no git).
    import scoring.trained.absorption as _absorption_mod
    edges_list = list(corruption.corrupted_edges) if corruption is not None else []
    edges_sha = hashlib.sha256(json.dumps(edges_list).encode()).hexdigest()  # == the driver's form
    classifier_sha = hashlib.sha256(
        Path(_absorption_mod.__file__).read_bytes()).hexdigest()
    return {
        "caveat": CAVEAT,
        "corrupted_edges_sha256": edges_sha,
        "absorption_classifier_sha256": classifier_sha,
        "match_mode": match_mode,
        "counts": cls["counts"],
        "absorbed_by_relation": cls["absorbed_by_relation"],
        "n_planted_corrupted_edges": len(corrupted),
        "n_census_absorbed": len(absorbed_set),
        "n_census_absorbed_among_planted": len(absorbed_set & corrupted),
        "n_census_absorbed_outside_planted": len(absorbed_set - corrupted),
        "theta_hat": sorted(float(e.get("theta_hat", float("nan"))) for e in absorbed_edges),
        "eps": cls.get("eps"),
        "n_recovered": int(recovered.sum()),
    }
