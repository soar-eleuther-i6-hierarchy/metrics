"""The two reads, reduced to one shape.

An ORACLE read scores the true coefficients `A` against the true directions `g`: no SAE, no
recovery loss, every feature in the universe. A TRAINED read loads a checkpoint, matches
learned latents to true features with the Hungarian matcher at `rho_star`, reduces to the
recovered universe, and scores a HELD-OUT draw. The trained path mirrors
`training/score_trained.py` exactly, so its scores land on the identical universe as the saved
trained arrays and the harness gate in `run_benchmark.py` can hold it to them.

Both produce a `Read` with the same field names, so `evaluate` never branches on which one it
got. Two things differ and are recorded rather than hidden:

  `G`      cosine over unit decoders. On the oracle read those ARE the true `g`, so this is
           `G_g`; on the trained read they are the learned `W_dec`, so it is `G_W`.
  `S_res`  the Tree-SAE probe: `probe_true_g` on the oracle read (true firing labels, true
           directions) and `probe_self_W` on the trained read (the deployed detector).

PRECOMMIT s5 forbids the cosine-mode `s_res` key from populating the probe comparator rows, so
the two are computed separately and named separately; the internal `s_res` key never leaves
this module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from scoring.benchmark.registry import GATES, METRICS, probe_fit_sample_seed
from scoring.core.detectors import (compute_bundle, fit_probe_directions,
                                    s_res_cosine, s_res_from_directions)
from scoring.core.grid import held_out_sample_seed, pair_frame, reduce_to_recovered
from scoring.core.recovery import activation_corr, match_features, per_class_recovery
from scoring.core.registry import CONSTANTS, DETECTORS
from scoring.core.world import regenerate_world, signed_normalized_decoder
from scoring.oracle.validate_metrics import pure_inputs
from toygen import labels
from toygen import spec
from toygen.world import resolve_config

_NAN = float("nan")
_TINY = 1e-12


@dataclass(frozen=True)
class Read:
    """One (toy, seed, read) scored universe.

    feats     true feature id at each recovered POSITION; `pairs` are positions into it, so
              `feats[a]` is the true id the null population keys on.
    vals      per-ordered-pair score vectors, float64, one per name in `registry.METRICS`.
    gate_vals per-ordered-pair GATE vectors, float64 tristates over {1.0, 0.0, NaN}, one per
              name in `registry.GATES`. A separate field rather than more entries in `vals`,
              whose docstring promises one entry per METRICS name and which is what
              `detector_matrices` and the AUROC grid iterate blindly -- a tristate scored as a
              continuous detector would produce an AUROC over a three-valued variable.
    """

    toy: str
    seed: int
    read: str                       # "oracle" | "trained"
    n_tokens: int
    s_res_mode: str                 # "probe" | "absent"
    F: int
    feats: list[int]
    pairs: list[tuple[int, int]]
    y: torch.Tensor
    vals: dict[str, torch.Tensor]
    gate_vals: dict[str, torch.Tensor]
    pair_labels: torch.Tensor
    W_unit: torch.Tensor
    recovered: torch.Tensor         # [F] bool, the scored universe
    detector_matrices: dict[str, torch.Tensor]
    extra: dict

    @property
    def n_recovered(self) -> int:
        return len(self.feats)


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------
def wide_matrix(outdegree: torch.Tensor) -> torch.Tensor:
    """`min(outdegree[p,c], outdegree[c,p])`, undefined unless BOTH orderings are finite.

    `torch.minimum` propagates NaN, so a missing reverse ordering yields NaN; `torch.fmin` would
    return the finite side and manufacture a score for a pair that was never measured
    (PRECOMMIT s6). But `minimum` does NOT propagate an infinity -- `min(+inf, 5)` is 5 -- so
    non-finite entries are masked explicitly rather than relying on the reduction. `_orient`
    already converts inf to NaN before this sees `outdegree`, so today that mask is belt and
    braces; this function is public and the docstring has to be true of the function, not of
    one caller.
    """
    ok = torch.isfinite(outdegree) & torch.isfinite(outdegree.transpose(0, 1))
    w = torch.minimum(outdegree, outdegree.transpose(0, 1))
    return torch.where(ok, w, torch.full_like(w, _NAN))


def _scored(mat: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    return mat[pa, pb].double()


def add_derived(vals: dict[str, torch.Tensor], outdegree_matrix: torch.Tensor | None,
                pairs: list[tuple[int, int]] | None) -> dict[str, torch.Tensor]:
    """Add `abs_asymmetry_R` and (when the matrix is given) `wide`.

    `abs_asymmetry_R` was added as a registered metric so the deleted SYM predicate's Q99 could
    be fitted on the ABSOLUTE distribution rather than the signed one. SYM is gone with the
    quantile rule; the metric stays because the artifacts carry it and it is still reported.
    """
    out = dict(vals)
    if "asymmetry_R" in out:
        out["abs_asymmetry_R"] = out["asymmetry_R"].abs()
    if outdegree_matrix is not None and pairs is not None:
        out["wide"] = _scored(wide_matrix(outdegree_matrix), pairs)
    return out


def class_totals(pair_labels: torch.Tensor) -> dict[str, int]:
    """Generated ordered pairs per class over the FULL feature set: the `N_total` denominator.

    Taken from the answer key, never from the scored pairs -- on the trained read the latter
    would silently drop every unrecovered target and turn end-to-end recall into
    recall-given-recovery under a different name.
    """
    F = int(pair_labels.shape[0])
    eye = torch.eye(F, dtype=torch.bool)
    return {name: int(((pair_labels == labels._index(name)) & ~eye).sum())
            for name in labels.LABELS}


def class_recovered(pair_labels: torch.Tensor, in_universe: torch.Tensor) -> dict[str, int]:
    """Ordered pairs per class with BOTH endpoints in the scored universe (`N_recovered`)."""
    return {name: n for name, (n, _tot) in
            per_class_recovery(in_universe, pair_labels, list(labels.LABELS)).items()}


def assemble_metrics(dets: dict[str, torch.Tensor], W_unit: torch.Tensor,
                     probe: torch.Tensor | None, pairs: list[tuple[int, int]]
                     ) -> dict[str, torch.Tensor]:
    """Every non-`s_res` detector, plus `G` (cosine) and `S_res` (probe), plus derived.

    Public because it is the seam where PRECOMMIT s5's separation is enforced: `G` is cosine
    over unit decoders and `S_res` is the probe, and neither may be populated from the other.
    Both reads go through this one function so the two cannot diverge.
    """
    vals = {d: _scored(dets[d], pairs) for d in DETECTORS if d != "s_res"}
    vals["G"] = _scored(s_res_cosine(W_unit), pairs)
    vals["S_res"] = (_scored(probe, pairs) if probe is not None
                     else torch.full((len(pairs),), _NAN, dtype=torch.float64))
    vals = add_derived(vals, dets["outdegree"], pairs)
    missing = [m for m in METRICS if m not in vals]
    if missing:
        raise RuntimeError(f"the read did not produce {missing}; a registered expression "
                           f"would go untestable for a harness reason, not a metric one")
    return vals


def assemble_gates(gate_mats: dict[str, torch.Tensor],
                   pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Every registered gate, scored onto the ordered-pair frame.

    Mirrors `assemble_metrics` and raises for the same reason: a gate missing from the read
    would make every rule that names it UNSCORABLE, and an expression reported as untestable
    for a harness reason looks exactly like one untestable for a measurement reason.
    """
    missing = [g for g in GATES if g not in gate_mats]
    if missing:
        raise RuntimeError(f"the read did not produce {missing}; a registered expression "
                           f"would go untestable for a harness reason, not a metric one")
    return {g: _scored(gate_mats[g], pairs) for g in GATES}


# --------------------------------------------------------------------------
# the oracle read
# --------------------------------------------------------------------------
def oracle_read(toy: str, seed: int, n_tokens: int, with_probe: bool = True,
                probe_fit_seed: int | None = None) -> Read:
    """Score the true coefficients against the true directions. No SAE, no recovery loss.

    `with_probe=False` skips the per-feature probe training, which dominates the runtime. The
    resulting `s_res_mode` is recorded as `"absent"`, NOT as `"probe"` with an all-NaN column:
    the two probe comparators must come out INVALID MEASUREMENT rather than look like
    rejections, and `run_benchmark` refuses to write such a run as a full one.

    TWO SEEDS, AND THEY DO DIFFERENT THINGS.

      `cfg.seed`      the directions `g` (`geometry.build_directions`), and the tree when
                      `randomize_structure` is on -- which it is not here.
      `sample_seed`   the entire Monte-Carlo stream: tokens, doc topics, firing, strengths,
                      noise.

    `resolve_config` NEVER sets the seed -- every config factory leaves it at the `ToyConfig`
    default of 0 -- and it REJECTS a `seed=` override, so `spec.replace` on the frozen dataclass
    is the only typed route. Mutating the `asdict`ed dict would work but silently swallow a
    typo'd key, because `regenerate_world` filters to known fields. This mirrors
    `training/train_toy.py:106-107` exactly, which is what lets an oracle read share a world
    with a checkpoint WITHOUT loading one: the resolved configs come out byte-identical.

    Passing the seed only as `sample_seed` -- which this did until B2.1 -- gives a fresh DRAW
    over SEED-0 GEOMETRY at every seed. That reads as "a different world" and is not one, and no
    co-firing detector can see it, because the token draw really did change.

    The scoring draw is `held_out_sample_seed(seed)`, the same offset draw the trained read
    scores on, so the two reads at one seed are the same world AND the same draw.
    """
    cfg = spec.replace(resolve_config(toy), seed=int(seed))
    rc = dataclasses.asdict(cfg)
    score_seed = held_out_sample_seed(int(seed))
    bundle = regenerate_world(rc, sample_seed=score_seed, n_tokens=n_tokens)
    F = int(bundle.g.shape[0])
    feats = list(range(F))
    idx = torch.tensor(feats, dtype=torch.long)
    inp = pure_inputs(bundle, feats, bundle.A[:, idx])

    # The probe is fitted on its OWN draw and frozen before anything is scored (PRECOMMIT s6
    # step 2). `probe_fit_seed` overrides the derived draw; passing the SCORING draw is the
    # bridge that reproduces the unseparated pilot numbers, and is the only gate this change
    # gets, because `harness_gate` deliberately excludes `s_res`.
    #
    # It is fitted BEFORE the detectors now, because `gate_sres_rank` needs the frozen
    # directions and `compute_bundle` builds the gates alongside the detectors. Neither the
    # fit nor the detectors touch global RNG, so the reordering moves no number -- which the
    # phase gate asserted element-wise rather than assumed.
    fit_seed = probe_fit_sample_seed(int(seed)) if probe_fit_seed is None else int(probe_fit_seed)
    probe, P, avail = None, None, None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens)
        fi = pure_inputs(fw, feats, fw.A[:, idx])
        # TRUE firing as the fitting label on the oracle read; the trained read uses the SAE's.
        P, avail = fit_probe_directions(fi.h, fi.acts_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, inp.W_unit)

    # cosine mode: the internal `s_res` key holds G here and is dropped by `assemble_metrics`.
    bnd = compute_bundle(inp, CONSTANTS, s_res_mode="cosine",
                         probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]

    pairs, y = pair_frame(feats, bundle.pair_labels)
    return Read(
        toy=toy, seed=seed, read="oracle", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=F, feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, inp.W_unit, probe, pairs),
        gate_vals=assemble_gates(bnd["gates"], pairs),
        pair_labels=bundle.pair_labels, W_unit=inp.W_unit,
        recovered=torch.ones(F, dtype=torch.bool),
        detector_matrices=dets,
        extra={"support": bnd["support"],
               "true_l0": float(bundle.A.gt(0).double().sum(dim=1).mean()),
               "resolved_config": rc,
               # Recorded so an artifact says WHICH draw it was scored on, not just which
               # experiment seed it belongs to. The trained read records the same quantity.
               "scoring_sample_seed": score_seed,
               "matching_sample_seed": None,    # the oracle read has no matcher
               "probe_fit_sample_seed": (fit_seed if with_probe else None),
               "probe_fit_labels": ("true firing (A)" if with_probe else None)},
    )


# --------------------------------------------------------------------------
# the trained read
# --------------------------------------------------------------------------
def trained_read(ckpt: str, n_tokens: int, with_probe: bool = True,
                 probe_fit_seed: int | None = None) -> Read:
    """Load a checkpoint and score its recovered universe on a held-out draw.

    Mirrors `training/score_trained.py:44-56` step for step: the matcher runs on the
    TRAINING-seed draw, scoring runs on the offset held-out draw, decoders are the oriented
    normalized ones. That is what lets the harness gate in `run_benchmark` hold the nine
    non-`s_res` detectors to the checkpoint's own saved arrays element-wise.
    """
    from scoring.trained.loaders import load_sae

    L = load_sae(ckpt)
    W, rc = L.W_dec, L.meta["resolved_config"]
    train_seed = int(L.meta["train_seed"])

    # The checkpoint's own stamped config already carries the training seed, so the trained read
    # is correct for free -- the B2.1 bug lives only on config-building paths. Assert it anyway:
    # a checkpoint whose `resolved_config.seed` disagreed with its `train_seed` would mean the
    # trainer scored a different world than it trained on, which no downstream check would see.
    cfg_seed = rc.get("seed")
    if cfg_seed is not None and int(cfg_seed) != train_seed:
        raise RuntimeError(
            f"checkpoint {ckpt} has resolved_config.seed={cfg_seed} but train_seed={train_seed}. "
            f"Geometry comes from cfg.seed and the draw from sample_seed, so these disagreeing "
            f"means the trained world is not the world this checkpoint is labelled with.")

    inw = regenerate_world(rc, sample_seed=train_seed, n_tokens=n_tokens)
    ai = L.encode(inw.h)
    oi = signed_normalized_decoder(W, ai, inw.h)
    res = match_features(activation_corr(inw.A, ai), inw.g, oi, rho=CONSTANTS["rho_star"])

    ho = regenerate_world(rc, sample_seed=held_out_sample_seed(train_seed), n_tokens=n_tokens)
    ah = L.encode(ho.h)
    oh = signed_normalized_decoder(W, ah, ho.h)
    feats, di, _index_map = reduce_to_recovered(
        ah, oh, W, res.match, res.recovered,
        h=ho.h, b_dec=L.b_dec, tokens=ho.tokens, vocab=ho.cfg.vocab)
    # Same fitting draw as the oracle read at this seed (identical config, identical seed, so
    # identical observations), but the SAE's OWN activations as labels -- the deployed
    # `probe_self_W` detector. The self-LABEL circularity is untouched by this change and stays
    # recorded; only the shared sampling noise with the scoring draw is removed. Fitted before
    # the detectors because `gate_sres_rank` reads the frozen directions.
    fit_seed = (probe_fit_sample_seed(train_seed) if probe_fit_seed is None
                else int(probe_fit_seed))
    probe, P, avail = None, None, None
    if with_probe:
        fw = regenerate_world(rc, sample_seed=fit_seed, n_tokens=n_tokens)
        af = L.encode(fw.h)
        # Restrict to the SAME recovered positions the scoring frame uses, so the fitted
        # columns line up with `di.W_unit`'s rows.
        of = signed_normalized_decoder(W, af, fw.h)
        _f2, dfit, _im2 = reduce_to_recovered(
            af, of, W, res.match, res.recovered,
            h=fw.h, b_dec=L.b_dec, tokens=fw.tokens, vocab=fw.cfg.vocab)
        P, avail = fit_probe_directions(dfit.h, dfit.acts_rec, CONSTANTS)
        probe = s_res_from_directions(P, avail, di.W_unit)

    bnd = compute_bundle(di, CONSTANTS, s_res_mode="cosine",
                         probe_directions=P, probe_available=avail)
    dets = bnd["detectors"]

    pairs, y = pair_frame(feats, ho.pair_labels)

    # `reduce_to_recovered` keeps a feature only if it is recovered AND matched, so the scored
    # universe is that intersection -- not `res.recovered` alone.
    in_universe = torch.zeros(int(res.match.shape[0]), dtype=torch.bool)
    in_universe[torch.tensor(feats, dtype=torch.long)] = True

    # The true-direction cosine on the SAME recovered endpoints. A diagnostic control, never a
    # registered metric: it separates "training moved the geometry" from "recovery dropped the
    # hard endpoints", which a trained-only number cannot.
    g_unit = (ho.g / ho.g.norm(dim=1, keepdim=True).clamp_min(_TINY)).double()
    g_matched = s_res_cosine(g_unit[torch.tensor(feats, dtype=torch.long)])

    return Read(
        toy=str(rc.get("name", "unknown")), seed=train_seed, read="trained", n_tokens=n_tokens,
        s_res_mode="probe" if with_probe else "absent",
        F=int(ho.g.shape[0]), feats=feats, pairs=pairs, y=y,
        vals=assemble_metrics(dets, di.W_unit, probe, pairs),
        gate_vals=assemble_gates(bnd["gates"], pairs),
        pair_labels=ho.pair_labels, W_unit=di.W_unit,
        recovered=in_universe,
        detector_matrices=dets,
        extra={"support": bnd["support"], "ckpt": str(ckpt), "G_g_matched": _scored(g_matched, pairs),
               "match": res.match, "n_recovered": len(feats),
               "effective_alpha": effective_alpha(ho.tree),
               "alpha_designed_in_meta": rc.get("alpha"),
               "resolved_config": rc,
               # Same two provenance keys the oracle read records, so `verify_manifest` can
               # cross-check that both reads of a (seed, toy) used one world and one draw.
               "scoring_sample_seed": held_out_sample_seed(train_seed),
               "matching_sample_seed": train_seed,
               "probe_fit_sample_seed": (fit_seed if with_probe else None),
               "probe_fit_labels": ("the SAE's own activations" if with_probe else None),
               "sae_meta": {
                   k: L.meta.get(k) for k in ("train_seed", "variant", "k", "d_sae")}},
    )


def effective_alpha(tree) -> dict:
    """The alpha actually planted on containment edges, per edge.

    Saved caches record `alpha_designed` from the config's nominal `alpha` field, which reads
    0.48 even for `only_firing` where `alpha_zero_every=1` zeroes EVERY edge. That is a
    provenance error in an artifact about to be frozen, so the realized values are recorded.
    """
    a = [float(al) for c in range(tree.F) for (_p, _pe, al) in tree.parents.get(c, [])]
    if not a:
        return {"n_edges": 0, "median": _NAN, "min": _NAN, "max": _NAN, "frac_zero": _NAN}
    t = torch.tensor(a, dtype=torch.float64)
    return {"n_edges": int(t.numel()), "median": float(t.median()),
            "min": float(t.min()), "max": float(t.max()),
            "frac_zero": float((t == 0).double().mean())}
