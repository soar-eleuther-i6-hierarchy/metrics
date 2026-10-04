"""A benchmark `Read` built from a synthetic dictionary: the oracle read's draws, with no matcher.

The planted map (`synthdict.planted`) is the feature->latent correspondence. `synth_encode`
returns the dictionary frame [., L], one column per decoder row; the detectors see the scored
frame [., R], one column per feature after the readout. Draws follow the benchmark: scoring on
`held_out_sample_seed(seed)`, probe fit on `probe_fit_sample_seed(seed)`. A split's sub-groups are
planted on every draw (`plant_subgroups`), so h is the world the split dictionary was built for.
"""

from __future__ import annotations

import dataclasses

import torch

from scoring.benchmark.reads import Read, assemble_gates, assemble_metrics
from scoring.config import BENCHMARK
from scoring.core.detectors import (DetectorInputs, compute_bundle, decoder_cosine,
                                    fit_probe_directions, s_res_from_directions)
from scoring.core.frame import held_out_sample_seed, pair_frame, probe_fit_sample_seed
from scoring.core.world import WorldBundle, regenerate_world, signed_normalized_decoder
from scoring.pathology_detection import ABSORPTION_SPLITTING
from toygen import spec
from toygen.world import resolve_config

from synthdict.activations import nnls_acts, reconstruction_fvu, zeroed_rate
from synthdict.corruptions import (AbsorptionDials, Corruption, apply_hole, build_corruption,
                                   expand_support)
from synthdict.detect import detect_pathologies, pathology_detection_applies
from synthdict.planted import READOUTS, resolve_map
from synthdict.subgroups import plant_subgroups

_TINY = 1e-12
_NAN = float("nan")


def resolved_config(toy: str, seed: int, cfg_overrides: dict | None = None) -> dict:
    """The benchmark's config idiom (`spec.replace`) plus test-only shape overrides, which land
    in the recorded config so a shrunken world cannot pass as the real toy."""
    cfg = spec.replace(resolve_config(toy), seed=int(seed), **(cfg_overrides or {}))
    return dataclasses.asdict(cfg)


ACTS_MODELS = ("nnls", "true_A")


def synth_encode(bundle: WorldBundle, corruption: Corruption | None, world_seed: int,
                 sample_seed: int, acts_mode: str = "nnls"
                 ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(acts [n, L], latent-space support, total holed tokens) for one draw.

    `acts_mode="true_A"` passes the true coefficients through: the passthrough anchor only, so
    it refuses a corruption.
    """
    if acts_mode not in ACTS_MODELS:
        raise ValueError(f"unknown acts_mode {acts_mode!r}; the encoder is 'nnls' "
                         f"('true_A' is the passthrough anchor)")
    support = bundle.A > 0
    n_holed = 0
    # The hole is absorption's firing transform; no other damage carries an eta.
    if corruption is not None and isinstance(corruption.dials, AbsorptionDials):
        support, holed = apply_hole(support, corruption)
        n_holed = sum(holed.values())
    if acts_mode == "true_A":
        if corruption is not None:
            raise ValueError("acts_mode='true_A' is the passthrough anchor; it cannot carry "
                             "a corruption (the damage would be silently ignored)")
        return bundle.A, support, 0
    W_raw = corruption.W_raw if corruption is not None else bundle.g.double()
    # feature space -> latent space, once; from here every array is indexed by decoder row
    if corruption is not None:
        support = expand_support(support, corruption, world_seed, sample_seed)
    return nnls_acts(bundle.h, W_raw, support, lam=BENCHMARK.ridge_lambda), support, n_holed


def corrupted_pair_mask(corruption: Corruption | None, feats: list[int],
                        pairs: list[tuple[int, int]]) -> torch.Tensor:
    """`[n_pairs]` bool: which scored pairs this damage touched, under its declared rule.

    ordered_edge              (parent, child) is an absorbed edge
    either_endpoint_feature   either feature is damaged (a hedged parent)
    candidate_parent_feature  the candidate parent, the pair's first feature, is damaged

    The rule travels with the corruption: the edge rule on a feature damage would mark nothing
    and make `intact` the whole target class in `export.py`.
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
    """`[n_pairs]` bool: pairs with either feature's row or firing changed by the damage.

    A superset of the corrupted mask; it keeps a pair whose second feature was damaged out of
    the intact control.
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
                   probe_fit_seed: int | None = None, with_pathology_detection: bool = False,
                   pathology_detection_settings=None) -> Read:
    """Build, encode, and score one synthetic dictionary as a `Read(read="synthetic")`.

    The dials type picks the damage; `dials=None` is the undamaged dictionary (W = g), and with
    `acts_mode="true_A"` it must reproduce the benchmark's oracle read bit-for-bit. With
    `with_pathology_detection`, the pathology detectors run where they apply
    (`pathology_detection_applies`).
    """
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    rc = resolved_config(toy, seed, cfg_overrides)
    score_seed = held_out_sample_seed(int(seed))
    score = regenerate_world(rc, sample_seed=score_seed, n_tokens=n_tokens)
    F = int(score.g.shape[0])

    # geometry and tree come from cfg.seed, so the corruption built on the scoring draw is the
    # dictionary every draw sees
    corruption = None
    if dials is not None:
        corruption = build_corruption(score, dials, world_seed=int(seed), readout=readout)
    score = plant_subgroups(score, corruption, int(seed), score_seed)
    kind = corruption.kind if corruption is not None else "none"
    run_detection = (bool(with_pathology_detection)
                     and pathology_detection_applies(kind, acts_mode))
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

    # reduce the [., L] dictionary frame to the [., R] scored frame; a plain gather under a 1-1 map
    feats = pmap.feats()
    recovered = pmap.recovered()
    di = DetectorInputs(acts_rec=pmap.reduce_acts(acts_ho),
                        W_unit=pmap.reduce_unit(oriented_ho, acts_ho),
                        W_raw=pmap.reduce_raw(W_raw, acts_ho), h=score.h, b_dec=b_dec,
                        tokens=score.tokens, vocab=score.cfg.vocab)

    fit_seed = probe_fit_sample_seed(int(seed)) if probe_fit_seed is None else int(probe_fit_seed)
    probe, P, avail = None, None, None
    if with_probe or run_detection:
        fw = plant_subgroups(regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens),
                             corruption, int(seed), fit_seed)
        acts_f, support_f, holed_f = synth_encode(fw, corruption, seed, fit_seed, acts_mode)
    if with_probe:
        # labels are the dictionary's own activations on the scored columns, as on a trained read,
        # so the probe at position k belongs to the latent scored at k
        acts_f_rec = pmap.reduce_acts(acts_f)
        P, avail = fit_probe_directions(fw.h, acts_f_rec)
        probe = s_res_from_directions(P, avail, di.W_unit)
        fvu["probe_fit"] = reconstruction_fvu(fw.h, acts_f, W_raw)
        zeroed["probe_fit"] = zeroed_rate(acts_f, support_f)
        zeroed_damaged["probe_fit"] = zeroed_rate(acts_f[:, dl], support_f[:, dl])
        holed["probe_fit"] = holed_f

    # probe directions are fixed before this pass, in the same order as the oracle read
    bnd = compute_bundle(di, probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]

    pairs, y = pair_frame(feats, score.pair_labels)

    corrupted_pair = corrupted_pair_mask(corruption, feats, pairs)
    touched_pair = touched_pair_mask(corruption, feats, pairs)

    # true-direction cosine on the same features, the trained read's diagnostic control
    g_unit = (score.g / score.g.norm(dim=1, keepdim=True).clamp_min(_TINY)).double()
    g_matched = decoder_cosine(g_unit[torch.tensor(feats, dtype=torch.long)])
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)

    sev = (corruption.realized_severity if corruption is not None
           else torch.zeros(0, dtype=torch.float64))
    detection = None
    if run_detection:
        settings = (ABSORPTION_SPLITTING if pathology_detection_settings is None
                    else pathology_detection_settings)
        detection = detect_pathologies(
            score, fw, acts_ho, acts_f, W_raw, corruption, pmap, settings,
            int(seed)) | {"eval_sample_seed": score_seed, "fit_sample_seed": fit_seed}
    extra = {
        "support": bnd["support"],
        "dials": (dataclasses.asdict(dials) if dials is not None else None),
        "corruption_kind": (corruption.kind if corruption is not None else "none"),
        "corrupted_edges": (tuple(corruption.corrupted_edges) if corruption is not None else ()),
        "corrupted_features": (tuple(corruption.corrupted_features)
                               if corruption is not None else ()),
        "realized_severity": sev,
        "realized_severity_median": (float(sev.median()) if sev.numel() else _NAN),
        # what the severity measures and which pairs the mask marks, both per damage
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
        "nnls_lambda": BENCHMARK.ridge_lambda,
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
        # latent_l0 on the [n, L] dictionary frame, feature_l0 on the [n, R] scored frame
        "latent_l0": float(acts_ho.gt(0).double().sum(dim=1).mean()),
        "feature_l0": float(di.acts_rec.gt(0).double().sum(dim=1).mean()),
        "pathology_detection": detection,
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
