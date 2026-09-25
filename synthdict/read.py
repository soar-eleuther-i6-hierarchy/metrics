"""synthetic_read — a benchmark `Read` built from a synthetic dictionary. No SAE anywhere.

Mirrors `scoring/benchmark/reads.py::oracle_read` at its seams, with one deliberate
difference: NO MATCHER. Matching solves an inverse problem — whose dictionary is this? —
and synthesis has none, because we built it. The feature->latent correspondence is the
PLANTED map (`synthdict.planted`), and `readout` declares how a feature carried by several
latents is reduced to one scored column.

Two frames, kept apart on purpose. The DICTIONARY frame is `[., L]`:
one column per decoder row, which is one per feature only while the map is 1-1. The SCORED
frame is `[., R]`: one column per feature that has a latent, after the readout reduction.
`synth_encode` returns the dictionary frame; everything the detectors see is the scored one.
Under a 1-1 map both are the identity gather, so the readout seam is inert there.

Two draws, same derivations as the benchmark: scoring = `held_out_sample_seed(seed)`, probe
fit = `probe_fit_sample_seed(seed)`. (The benchmark's third, in-sample MATCHING draw has no
purpose here and is not taken.) The synthetic "encoder" is a deterministic function of a draw:
planted support (true `A > 0` plus the damage's firing transform) -> per-token NNLS strengths
against the draw's real `h`.

`signed_normalized_decoder` runs on every synthetic path (parity with the trained read); on a
planted dictionary it must be a no-op, which the passthrough anchor proves bit-for-bit
against the real `oracle_read` (W7 in the plan).
"""

from __future__ import annotations

import dataclasses

import torch

from scoring.benchmark.reads import Read, assemble_gates, assemble_metrics
from scoring.benchmark.registry import probe_fit_sample_seed
from scoring.core.detectors import (DetectorInputs, compute_bundle, fit_probe_directions,
                                    s_res_cosine, s_res_from_directions)
from scoring.core.grid import held_out_sample_seed, pair_frame
from scoring.core.registry import CONSTANTS
from scoring.core.world import WorldBundle, regenerate_world, signed_normalized_decoder
from scoring.oracle.validate_metrics import RIDGE_LAMBDA, pure_inputs, reconstruction_fvu
from toygen import spec
from toygen.world import resolve_config

from synthdict.activations import nnls_acts, zeroed_rate
from synthdict.corruptions import (AbsorptionDials, Corruption, apply_hole, build_corruption,
                                   expand_support)
from synthdict.planted import READOUTS, resolve_map

_TINY = 1e-12
_NAN = float("nan")


def resolved_config(toy: str, seed: int, cfg_overrides: dict | None = None) -> dict:
    """The benchmark's own config idiom (`spec.replace` on the frozen dataclass), plus
    test-only shape overrides. Overrides land in the recorded `resolved_config`, so a
    shrunken world can never masquerade as the real toy."""
    cfg = spec.replace(resolve_config(toy), seed=int(seed), **(cfg_overrides or {}))
    return dataclasses.asdict(cfg)


ACTS_MODELS = ("nnls", "true_A")
NNLS_LAMBDA = RIDGE_LAMBDA           # the regularizer the encoder is run with and recorded as


def synth_encode(bundle: WorldBundle, corruption: Corruption | None, world_seed: int,
                 sample_seed: int, acts_mode: str = "nnls"
                 ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(acts [n, L], LATENT-space support, total holed tokens) for one draw.

    `acts_mode="true_A"` passes the oracle coefficients through untouched — the passthrough
    anchor only, and therefore refused when a corruption is present.
    """
    if acts_mode not in ACTS_MODELS:
        raise ValueError(f"unknown acts_mode {acts_mode!r}; the encoder is 'nnls' "
                         f"('true_A' is the passthrough anchor)")
    support = bundle.A > 0
    n_holed = 0
    # The hole is absorption's firing transform; no other damage carries an eta.
    if corruption is not None and isinstance(corruption.dials, AbsorptionDials):
        support, holed = apply_hole(support, corruption, world_seed, sample_seed)
        n_holed = sum(holed.values())
    if acts_mode == "true_A":
        if corruption is not None:
            raise ValueError("acts_mode='true_A' is the passthrough anchor; it cannot carry "
                             "a corruption (the damage would be silently ignored)")
        return bundle.A, support, 0
    W_raw = corruption.W_raw if corruption is not None else bundle.g.double()
    # Feature space -> LATENT space, once, immediately before the solve. Below this line every
    # array is indexed by decoder row.
    if corruption is not None:
        support = expand_support(support, corruption, world_seed, sample_seed)
    return nnls_acts(bundle.h, W_raw, support, lam=NNLS_LAMBDA), support, n_holed


def corrupted_pair_mask(corruption: Corruption | None, feats: list[int],
                        pairs: list[tuple[int, int]]) -> torch.Tensor:
    """`[n_pairs]` bool: which scored pairs this damage touched, under its declared rule.

    ordered_edge              (parent, child) is an absorbed edge
    either_endpoint_feature   either feature is damaged (a hedged parent)
    candidate_parent_feature  the candidate parent, the pair's FIRST feature, is damaged

    The rule travels with the corruption: the edge rule on a feature damage would return an
    all-False mask and make `intact` the whole target class in `export.py`.
    """
    if corruption is None:
        return torch.zeros(len(pairs), dtype=torch.bool)
    rule = corruption.corrupted_pair_rule
    if rule == "ordered_edge":
        corrupted = set(corruption.corrupted_edges)
        return torch.tensor(
            [(feats[a], feats[b]) in corrupted for a, b in pairs], dtype=torch.bool)
    if rule == "either_endpoint_feature":
        damaged = set(corruption.corrupted_features)
        return torch.tensor(
            [feats[a] in damaged or feats[b] in damaged for a, b in pairs], dtype=torch.bool)
    if rule == "candidate_parent_feature":
        damaged = set(corruption.corrupted_features)
        return torch.tensor([feats[a] in damaged for a, _ in pairs], dtype=torch.bool)
    raise ValueError(f"unknown corrupted_pair_rule {rule!r}")


def touched_pair_mask(corruption: Corruption | None, feats: list[int],
                      pairs: list[tuple[int, int]]) -> torch.Tensor:
    """`[n_pairs]` bool: pairs with either feature touched by the damage (row or firing changed).

    A superset of the corrupted mask. Classes labelled in both orderings (superparent,
    frequency, topical) put a pair whose SECOND feature was damaged outside the corrupted arm;
    counting it as intact would contaminate the control.
    """
    if corruption is None:
        return torch.zeros(len(pairs), dtype=torch.bool)
    touched = set(corruption.touched_features())
    return torch.tensor([feats[a] in touched or feats[b] in touched for a, b in pairs],
                        dtype=torch.bool)


def _damaged_latents(corruption: Corruption | None, pmap) -> list[int]:
    if corruption is None:
        return []
    return sorted({j for f in corruption.touched_features() for j in pmap.feature_to_latents[f]})


def synthetic_read(toy: str, seed: int, dials, readout: str,
                   n_tokens: int, with_probe: bool = True, acts_mode: str = "nnls",
                   cfg_overrides: dict | None = None,
                   probe_fit_seed: int | None = None) -> Read:
    """Build, encode, and score one synthetic dictionary as a `Read(read="synthetic")`.

    `dials` is any damage's dial dataclass; the damage is chosen by its type. `dials=None` is
    the uncorrupted dictionary (W = g), and with `acts_mode="true_A"` that is the passthrough
    anchor, which must reproduce `oracle_read` bit-for-bit (tested).

    NO MATCHER RUNS HERE. The feature->latent correspondence is the PLANTED map (see
    `synthdict.planted`), and `readout` declares how a multi-latent feature is reduced to one
    scored column. On a 1-1 map every readout is the same gather, bit-for-bit.
    """
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    rc = resolved_config(toy, seed, cfg_overrides)
    score_seed = held_out_sample_seed(int(seed))
    score = regenerate_world(rc, sample_seed=score_seed, n_tokens=n_tokens)
    F = int(score.g.shape[0])

    # The dictionary is a property of the WORLD: geometry and tree are identical across the
    # three draws (both come from cfg.seed), so building the corruption from the scoring
    # bundle's g/CONT is the same dictionary every draw sees.
    corruption = None
    if dials is not None:
        corruption = build_corruption(score, dials, world_seed=int(seed), readout=readout)
    W_raw = corruption.W_raw if corruption is not None else score.g.double()

    acts_ho, support_ho, holed_ho = synth_encode(score, corruption, seed, score_seed, acts_mode)
    oriented_ho = signed_normalized_decoder(W_raw, acts_ho, score.h)
    b_dec = torch.zeros(score.g.shape[1], dtype=score.g.dtype)

    fvu = {"scoring": reconstruction_fvu(score.h, acts_ho, W_raw)}
    zeroed = {"scoring": zeroed_rate(acts_ho, support_ho)}
    pmap = resolve_map(corruption, F, readout)
    dl = _damaged_latents(corruption, pmap)
    # The pooled rate hides zeroing that concentrates on the damaged columns (absorption's
    # carry pushes the parent's NNLS strength to 0 on co-firing tokens).
    zeroed_damaged = {"scoring": zeroed_rate(acts_ho[:, dl], support_ho[:, dl])}
    holed = {"scoring": holed_ho}

    # The planted correspondence, not an inferred one. The reductions take the [., L]
    # dictionary frame to the [., R] SCORED frame under the declared readout; for a one-to-one
    # map every one of them is a gather over `arange(F)` and the seam is inert.
    feats = pmap.feats()
    recovered = pmap.recovered()
    di = DetectorInputs(acts_rec=pmap.reduce_acts(acts_ho),
                        W_unit=pmap.reduce_unit(oriented_ho, acts_ho),
                        W_raw=pmap.reduce_raw(W_raw, acts_ho), h=score.h, b_dec=b_dec,
                        tokens=score.tokens, vocab=score.cfg.vocab)

    fit_seed = probe_fit_sample_seed(int(seed)) if probe_fit_seed is None else int(probe_fit_seed)
    probe, P, avail = None, None, None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens)
        acts_f, support_f, holed_f = synth_encode(fw, corruption, seed, fit_seed, acts_mode)
        # SELF-label: the synthetic SAE's own activations, restricted to the scored universe —
        # the deployed `probe_self_W` convention (trained_read does the same with L.encode).
        # Fitting labels come from the SAME planted columns the scoring frame reads, so the
        # fitted direction for position k belongs to the latent whose decoder row k carries.
        acts_f_rec = pmap.reduce_acts(acts_f)
        P, avail = fit_probe_directions(fw.h, acts_f_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, di.W_unit)
        fvu["probe_fit"] = reconstruction_fvu(fw.h, acts_f, W_raw)
        zeroed["probe_fit"] = zeroed_rate(acts_f, support_f)
        zeroed_damaged["probe_fit"] = zeroed_rate(acts_f[:, dl], support_f[:, dl])
        holed["probe_fit"] = holed_f

    # Detectors and gates from one pass, with the probe directions already frozen — the same
    # order `reads.oracle_read` uses, so the two paths cannot diverge on which draw the
    # rank gate saw.
    bnd = compute_bundle(di, CONSTANTS, s_res_mode="cosine",
                         probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]

    pairs, y = pair_frame(feats, score.pair_labels)

    corrupted_pair = corrupted_pair_mask(corruption, feats, pairs)
    touched_pair = touched_pair_mask(corruption, feats, pairs)

    # True-direction cosine on the same universe — the trained read's diagnostic control.
    g_unit = (score.g / score.g.norm(dim=1, keepdim=True).clamp_min(_TINY)).double()
    g_matched = s_res_cosine(g_unit[torch.tensor(feats, dtype=torch.long)])
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)

    sev = (corruption.realized_severity if corruption is not None
           else torch.zeros(0, dtype=torch.float64))
    extra = {
        "support": bnd["support"],
        "dials": (dataclasses.asdict(dials) if dials is not None else None),
        "corruption_kind": (corruption.kind if corruption is not None else "none"),
        "corrupted_edges": (tuple(corruption.corrupted_edges) if corruption is not None else ()),
        "corrupted_features": (tuple(corruption.corrupted_features)
                               if corruption is not None else ()),
        "realized_severity": sev,
        "realized_severity_median": (float(sev.median()) if sev.numel() else _NAN),
        # WHAT that severity measures, and WHICH pairs the mask marks. Both are per-damage.
        "severity_kind": (corruption.severity_kind if corruption is not None else "none"),
        "corrupted_pair_rule": (corruption.corrupted_pair_rule
                                if corruption is not None else "ordered_edge"),
        "corrupted_pair": corrupted_pair,
        "touched_pair": touched_pair,
        "hedged_children": (tuple(corruption.hedged_children) if corruption is not None else ()),
        "composition_pairs": (tuple(corruption.composition_pairs)
                              if corruption is not None else ()),
        "details": (dict(corruption.details) if corruption is not None else {}),
        "acts_model": acts_mode,
        "nnls_lambda": NNLS_LAMBDA,
        "readout": readout,
        "planted_map_sha256": pmap.sha256(),
        "feature_to_latents": pmap.feature_to_latents,
        "n_latents": pmap.n_latents,
        "n_lost_features": F - len(feats),
        "zeroed_rate": zeroed,
        "zeroed_rate_damaged": zeroed_damaged,
        "fvu": fvu,
        "fvu_true_A": reconstruction_fvu(score.h, score.A, score.g),
        "n_holed_total": holed,
        "W_raw": W_raw,
        "G_g_matched": g_matched[pa, pb].double(),
        "resolved_config": rc,
        "scoring_sample_seed": score_seed,
        "probe_fit_sample_seed": (fit_seed if with_probe else None),
        "probe_fit_labels": ("the synthetic SAE's own activations" if with_probe else None),
        "true_l0": float(score.A.gt(0).double().sum(dim=1).mean()),
        # Two L0s that coincide only while the map is 1-1. `latent_l0` counts columns of the
        # [n, L] DICTIONARY frame; `feature_l0` counts the [n, R] SCORED frame.
        "latent_l0": float(acts_ho.gt(0).double().sum(dim=1).mean()),
        "feature_l0": float(di.acts_rec.gt(0).double().sum(dim=1).mean()),
    }
    return Read(
        toy=toy, seed=int(seed), read="synthetic", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=F, feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, di.W_unit, probe, pairs),
        gate_vals=assemble_gates(bnd["gates"], pairs),
        pair_labels=score.pair_labels, W_unit=di.W_unit,
        recovered=recovered, detector_matrices=dets, extra=extra,
    )


def oracle_equivalent_read(toy: str, seed: int, n_tokens: int,
                           cfg_overrides: dict | None = None,
                           with_probe: bool = True) -> Read:
    """A transcription of `reads.oracle_read` that accepts shape overrides, used ONLY as the
    reference side of the small-world probe anchor.

    It deliberately shares NO synthdict code (no synth_encode, no orientation): the anchor
    would be circular otherwise. That it matches the real `oracle_read` bit-for-bit at the
    unshrunken config is itself asserted by the full-size anchor test.
    """
    rc = resolved_config(toy, seed, cfg_overrides)
    score_seed = held_out_sample_seed(int(seed))
    bundle = regenerate_world(rc, sample_seed=score_seed, n_tokens=n_tokens)
    F = int(bundle.g.shape[0])
    feats = list(range(F))
    idx = torch.tensor(feats, dtype=torch.long)
    inp = pure_inputs(bundle, feats, bundle.A[:, idx])
    probe, P, avail = None, None, None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=probe_fit_sample_seed(int(seed)),
                              n_tokens=n_tokens)
        fi = pure_inputs(fw, feats, fw.A[:, idx])
        P, avail = fit_probe_directions(fi.h, fi.acts_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, inp.W_unit)
    bnd = compute_bundle(inp, CONSTANTS, s_res_mode="cosine",
                         probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]
    pairs, y = pair_frame(feats, bundle.pair_labels)
    return Read(
        toy=toy, seed=int(seed), read="oracle", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=F, feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, inp.W_unit, probe, pairs),
        gate_vals=assemble_gates(bnd["gates"], pairs),
        pair_labels=bundle.pair_labels, W_unit=inp.W_unit,
        recovered=torch.ones(F, dtype=torch.bool),
        detector_matrices=dets,
        extra={"support": bnd["support"], "resolved_config": rc},
    )
