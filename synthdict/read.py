"""synthetic_read — a benchmark `Read` built from a synthetic dictionary. No SAE anywhere.

Mirrors `scoring/benchmark/reads.py::oracle_read` at its seams, with one deliberate
difference: NO MATCHER. Matching solves an inverse problem — whose dictionary is this? —
and synthesis has none, because we built it. The feature->latent correspondence is the
PLANTED map (`synthdict.planted`), and `readout` declares how a feature carried by several
latents would be reduced to one scored column. Round-1 damages preserve one-latent-per-feature,
so the map is the identity and the gather is a no-op.

Two draws, same derivations as the benchmark: scoring = `held_out_sample_seed(seed)`, probe
fit = `probe_fit_sample_seed(seed)`. (The benchmark's third, in-sample MATCHING draw has no
purpose here and is not taken.) The synthetic "encoder" is a deterministic function of a draw:
planted support (true `A > 0` plus the corruption's eta-hole) -> per-token ridge magnitudes
against the draw's real `h`.

`signed_normalized_decoder` runs on every synthetic path (parity with the trained read); on a
planted dictionary it must be a no-op, which the passthrough anchor proves bit-for-bit
against the real `oracle_read` (W7 in the plan).
"""

from __future__ import annotations

import dataclasses

import torch

from scoring.benchmark.reads import Read, assemble_metrics
from scoring.benchmark.registry import probe_fit_sample_seed
from scoring.core.detectors import (DetectorInputs, compute_all, fit_probe_directions,
                                    s_res_cosine, s_res_from_directions)
from scoring.core.grid import held_out_sample_seed, pair_frame
from scoring.core.registry import CONSTANTS
from scoring.core.world import WorldBundle, regenerate_world, signed_normalized_decoder
from scoring.oracle.validate_metrics import pure_inputs, reconstruction_fvu
from toygen import spec
from toygen.world import resolve_config

from synthdict.activations import ridge_acts, support_flip_rate
from synthdict.corruptions import AbsorptionDials, Corruption, absorb, apply_hole
from synthdict.planted import READOUTS, resolve_map

_TINY = 1e-12
_NAN = float("nan")


def resolved_config(toy: str, seed: int, cfg_overrides: dict | None = None) -> dict:
    """The benchmark's own config idiom (`spec.replace` on the frozen dataclass), plus
    test-only shape overrides. Overrides land in the recorded `resolved_config`, so a
    shrunken world can never masquerade as the real toy."""
    cfg = spec.replace(resolve_config(toy), seed=int(seed), **(cfg_overrides or {}))
    return dataclasses.asdict(cfg)


def synth_encode(bundle: WorldBundle, corruption: Corruption | None, world_seed: int,
                 sample_seed: int, acts_mode: str = "ridge"
                 ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(acts [n, F], planted support after the hole, total holed tokens) for one draw.

    `acts_mode="true_A"` passes the oracle coefficients through untouched — the passthrough
    anchor only, and therefore refused when a corruption is present.
    """
    support = bundle.A > 0
    n_holed = 0
    if corruption is not None:
        support, holed = apply_hole(support, corruption, world_seed, sample_seed)
        n_holed = sum(holed.values())
    if acts_mode == "true_A":
        if corruption is not None:
            raise ValueError("acts_mode='true_A' is the passthrough anchor; it cannot carry "
                             "a corruption (the hole would be silently ignored)")
        return bundle.A, support, 0
    W_raw = corruption.W_raw if corruption is not None else bundle.g.double()
    if acts_mode == "ridge":
        # Deployed-encoder-like: joint least squares over the whole support. Beta carry lets
        # the corrupted child EXPLAIN AWAY the parent on co-firing tokens (nonpositive parent
        # coefficients = extra firing holes beyond eta) - absorption's own reconstruction
        # mechanism, but a measured LEAK of beta into the firing channel (review finding;
        # measured parent recall ~0.85 at beta=0.6, eta=0). Read against `support_flip_rate`.
        return ridge_acts(bundle.h, W_raw, support), support, n_holed
    if acts_mode == "clean":
        # Firing-preserving: the dial-independent construct SYNTH_PRECOMMIT promised.
        # Uncorrupted latents keep their TRUE magnitudes on the (holed) planted support, so
        # beta cannot move any UNCORRUPTED latent's firing decision (a corrupted child's own
        # coefficient is still ridge-determined and can rarely flip: measured 1.78e-7 at one
        # hole-only dial point - review LOW-1); only the corrupted children re-fit, against
        # the residual. Where the hole removed the parent, the parent's mass sits in that
        # residual and flows into the child's carry term (absorption's story); on un-holed
        # tokens the carried g_p component double-counts the parent, so reconstruction
        # degrades with beta (hedging's story). The FVU column records that price.
        A_masked = bundle.A.double() * support.double()
        if corruption is None:
            return A_masked, support, n_holed
        cc = sorted({c for _, c in corruption.corrupted_edges})
        acts = A_masked.clone()
        acts[:, cc] = 0.0
        residual = bundle.h.double() - acts @ bundle.g.double()   # uncorrupted rows ARE g rows
        idx = torch.tensor(cc, dtype=torch.long)
        acts[:, idx] = ridge_acts(residual, W_raw[idx], support[:, idx])
        return acts, support, n_holed
    raise ValueError(f"unknown acts_mode {acts_mode!r}")


def synthetic_read(toy: str, seed: int, dials: AbsorptionDials | None, readout: str,
                   n_tokens: int, with_probe: bool = True, acts_mode: str = "ridge",
                   cfg_overrides: dict | None = None,
                   probe_fit_seed: int | None = None) -> Read:
    """Build, encode, and score one synthetic dictionary as a `Read(read="synthetic")`.

    `dials=None` is the uncorrupted dictionary (W = g); with `acts_mode="true_A"` that is the
    passthrough anchor and must reproduce `oracle_read` bit-for-bit (tested).

    NO MATCHER RUNS HERE. The feature->latent correspondence is the PLANTED map (see
    `synthdict.planted`), and `readout` declares how a multi-latent feature would be reduced
    to one scored column. Round-1 damages all preserve one-latent-per-feature, so the map is
    the identity and `readout="identity"` is the only implemented policy.
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
        corruption = absorb(score.g, score.CONT, dials, world_seed=int(seed))
    W_raw = corruption.W_raw if corruption is not None else score.g.double()

    acts_ho, support_ho, holed_ho = synth_encode(score, corruption, seed, score_seed, acts_mode)
    oriented_ho = signed_normalized_decoder(W_raw, acts_ho, score.h)
    b_dec = torch.zeros(score.g.shape[1], dtype=score.g.dtype)

    fvu = {"scoring": reconstruction_fvu(score.h, acts_ho, W_raw)}
    flips = {"scoring": support_flip_rate(acts_ho, support_ho)}
    holed = {"scoring": holed_ho}

    # The planted correspondence, not an inferred one. `cols` gathers each scored feature's
    # latent column; for the identity map that is `arange(F)` and the gather is a no-op, which
    # is what makes the matcher-free read bit-identical to the matched one on every round-1
    # dial point (the regression gate checks exactly this).
    pmap = resolve_map(corruption, F, readout)
    feats = pmap.feats()
    recovered = pmap.recovered()
    cols = pmap.columns()
    di = DetectorInputs(acts_rec=acts_ho[:, cols], W_unit=oriented_ho[cols],
                        W_raw=W_raw[cols], h=score.h, b_dec=b_dec,
                        tokens=score.tokens, vocab=score.cfg.vocab)

    dets = compute_all(di, CONSTANTS, s_res_mode="cosine")

    fit_seed = probe_fit_sample_seed(int(seed)) if probe_fit_seed is None else int(probe_fit_seed)
    probe = None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens)
        acts_f, support_f, holed_f = synth_encode(fw, corruption, seed, fit_seed, acts_mode)
        # SELF-label: the synthetic SAE's own activations, restricted to the scored universe —
        # the deployed `probe_self_W` convention (trained_read does the same with L.encode).
        # Fitting labels come from the SAME planted columns the scoring frame reads, so the
        # fitted direction for position k belongs to the latent whose decoder row k carries.
        P, avail = fit_probe_directions(fw.h, acts_f[:, cols], CONSTANTS)
        probe = s_res_from_directions(P, avail, di.W_unit)
        fvu["probe_fit"] = reconstruction_fvu(fw.h, acts_f, W_raw)
        flips["probe_fit"] = support_flip_rate(acts_f, support_f)
        holed["probe_fit"] = holed_f

    pairs, y = pair_frame(feats, score.pair_labels)

    corrupted = set(corruption.corrupted_edges) if corruption is not None else set()
    corrupted_pair = torch.tensor(
        [(feats[a], feats[b]) in corrupted for a, b in pairs], dtype=torch.bool)

    # True-direction cosine on the same universe — the trained read's diagnostic control.
    g_unit = (score.g / score.g.norm(dim=1, keepdim=True).clamp_min(_TINY)).double()
    g_matched = s_res_cosine(g_unit[torch.tensor(feats, dtype=torch.long)])
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)

    sev = (corruption.realized_severity if corruption is not None
           else torch.zeros(0, dtype=torch.float64))
    extra = {
        "dials": (dataclasses.asdict(dials) if dials is not None else None),
        "corruption_kind": (corruption.kind if corruption is not None else "none"),
        "corrupted_edges": (tuple(corruption.corrupted_edges) if corruption is not None else ()),
        "realized_severity": sev,
        "realized_severity_median": (float(sev.median()) if sev.numel() else _NAN),
        "corrupted_pair": corrupted_pair,
        "acts_mode": acts_mode,
        "readout": readout,
        "planted_map_sha256": pmap.sha256(),
        "n_latents": pmap.n_latents,
        "support_flip_rate": flips,
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
        "realized_l0": float(acts_ho.gt(0).double().sum(dim=1).mean()),
    }
    return Read(
        toy=toy, seed=int(seed), read="synthetic", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=F, feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, di.W_unit, probe, pairs),
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
    dets = compute_all(inp, CONSTANTS, s_res_mode="cosine")
    probe = None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=probe_fit_sample_seed(int(seed)),
                              n_tokens=n_tokens)
        fi = pure_inputs(fw, feats, fw.A[:, idx])
        P, avail = fit_probe_directions(fi.h, fi.acts_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, inp.W_unit)
    pairs, y = pair_frame(feats, bundle.pair_labels)
    return Read(
        toy=toy, seed=int(seed), read="oracle", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=F, feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, inp.W_unit, probe, pairs),
        pair_labels=bundle.pair_labels, W_unit=inp.W_unit,
        recovered=torch.ones(F, dtype=torch.bool),
        detector_matrices=dets,
        extra={"resolved_config": rc},
    )
