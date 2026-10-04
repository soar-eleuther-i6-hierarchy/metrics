"""Pathology detectors (`scoring.pathology_detection`) run on a synthetic dictionary.

The toy supplies what a detector needs and a trained SAE does not: concept labels from the world's
A > 0, each concept's planted latents, and which concepts the damage touched. Concepts are the toy's
candidate parents (every feature with a split role); each is in the damaged arm (a split feature, or
the parent of an absorbed edge) or the intact arm, so a detector reads as a difference within one
dictionary. Each detector family writes one section of the result.
"""

from __future__ import annotations

import torch

from scoring.pathology_detection import evaluate_absorption_splitting

from synthdict.corruptions import split_roles

# The detectors read NNLS latents; hedging and composition have no detector yet.
DETECTED_KINDS = ("none", "absorption", "split")

DRAW_NOTE = ("probes fitted on the probe-fit draw and scored on the scoring draw, in place of "
             "SAEBench's 80/20 split of one token set")


def pathology_detection_applies(kind: str, acts_mode: str) -> bool:
    """Whether a point of this kind and encoder gets the pathology detectors."""
    return acts_mode == "nnls" and kind in DETECTED_KINDS


def candidate_parents(tree) -> list[int]:
    """Every feature carrying a split role, ascending."""
    return sorted({f for fs in split_roles(tree).values() for f in fs})


def concept_arms(concepts: list[int], corruption) -> dict[int, str]:
    """Damaged for a split feature or an absorbing parent, intact otherwise."""
    damaged: set[int] = set()
    if corruption is not None and corruption.kind == "split":
        damaged = set(corruption.corrupted_features)
    elif corruption is not None and corruption.kind == "absorption":
        damaged = {int(p) for p, _ in corruption.corrupted_edges}
    return {int(c): ("damaged" if int(c) in damaged else "intact") for c in concepts}


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _summary(rows: list[dict], mode: str, names: list[str]) -> dict:
    tested = [r for r in rows if r["testable"]]
    gated = [r for r in tested if r["gated_in"] and r[mode] is not None]
    out = {"n_concepts": len(rows), "n_testable": len(tested), "n_gated_in": len(gated),
           "absorption_fraction": {n: _mean([r[mode]["absorption_fraction"][n] for r in gated])
                                   for n in names},
           "full_absorption_rate": _mean([r[mode]["full_absorption_rate"] for r in gated])}
    if mode == "ksparse_main":
        # SAEBench's k-sparse helper counts extra latents; its headline counts every latent
        out["num_split_features"] = _mean([r[mode]["num_split_features"] for r in gated])
        out["n_main_latents"] = _mean([r[mode]["n_main_latents"] for r in gated])
        out["n_unstable"] = sum(1 for r in gated if not r[mode]["stable"])
    return out


def detect_absorption_splitting(score, fw, acts_ho, acts_f, W_raw, corruption, pmap, settings,
                                seed: int) -> dict:
    """SAEBench's absorption and splitting evaluation on the candidate parents, with per-arm means
    over gated-in concepts."""
    concepts = candidate_parents(score.tree)
    idx = torch.tensor(concepts, dtype=torch.long)
    res = evaluate_absorption_splitting(
        h_fit=fw.h, z_fit=acts_f, Y_fit=fw.A[:, idx] > 0,
        h_eval=score.h, z_eval=acts_ho, Y_eval=score.A[:, idx] > 0,
        W_dec=W_raw, concepts=concepts,
        planted_main={f: list(pmap.feature_to_latents[f]) for f in concepts},
        settings=settings, seed=int(seed))
    arms = concept_arms(concepts, corruption)
    names = [ts.name for ts in settings.threshold_sets]
    summary = {}
    for arm in ("damaged", "intact"):
        rows = [r for r in res["concepts"] if arms[r["concept"]] == arm]
        summary[arm] = {mode: _summary(rows, mode, names)
                        for mode in ("ksparse_main", "planted_main")}
    return res | {"arms": {str(c): a for c, a in arms.items()}, "summary": summary,
                  "draw_note": DRAW_NOTE}


def detect_pathologies(score, fw, acts_ho, acts_f, W_raw, corruption, pmap, settings,
                       seed: int) -> dict:
    """Every detector's section for one dictionary, keyed by detector family. `settings` is the
    absorption and splitting settings, the only family so far."""
    return {"absorption_splitting": detect_absorption_splitting(
        score, fw, acts_ho, acts_f, W_raw, corruption, pmap, settings, seed)}
