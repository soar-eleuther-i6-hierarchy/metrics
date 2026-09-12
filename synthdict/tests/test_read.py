"""synthetic_read — the Phase-0 gates and the read-shape contract.

The passthrough anchor is the load-bearing test of the whole study: an uncorrupted synthetic
read must be indistinguishable from the REAL `oracle_read`, bit for bit, through the shared
pipeline. It doubles as the W7 orientation no-op proof, because the synthetic path runs
`signed_normalized_decoder` and the oracle path does not.
"""

from __future__ import annotations

import torch

from scoring.benchmark.registry import METRICS
from synthdict.corruptions import AbsorptionDials
from synthdict.read import synthetic_read

TOY = "only_isa"
SMALL = {"n_roots": 12}          # shrink via cfg_overrides for the expensive-probe tests
N_TOK = 3000


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality with NaN==NaN (torch.equal is False on identical NaN cells)."""
    if a.shape != b.shape:
        return False
    na, nb = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(na, nb) and torch.equal(a[~na], b[~nb]))

def _dials(beta=0.6, eta=0.6, f=1.0):
    return AbsorptionDials(beta=beta, eta=eta, edge_fraction=f)


# --------------------------------------------------------------------------
# gate (a): the passthrough anchor
# --------------------------------------------------------------------------

def test_passthrough_anchor_matches_oracle_read_bit_for_bit():
    from scoring.benchmark.reads import oracle_read

    orc = oracle_read(TOY, seed=0, n_tokens=N_TOK, with_probe=False)
    syn = synthetic_read(TOY, seed=0, dials=None, readout="identity",
                         n_tokens=N_TOK, with_probe=False, acts_mode="true_A")
    assert syn.read == "synthetic"
    assert syn.feats == orc.feats and syn.pairs == orc.pairs
    assert torch.equal(syn.y, orc.y)
    assert torch.equal(syn.W_unit, orc.W_unit)
    for m in METRICS:
        if m == "S_res":
            continue                      # no probe in this variant; both are all-NaN
        assert _bit_equal(syn.vals[m], orc.vals[m]), f"{m} diverged from oracle_read"


def test_passthrough_anchor_probe_matches_oracle_probe():
    # Small world so the per-child probes are cheap; the anchor here is the PROBE path:
    # self-label on true_A acts == oracle_read's true-firing label, same fit draw.
    from synthdict.read import oracle_equivalent_read

    orc = oracle_equivalent_read(TOY, seed=0, n_tokens=N_TOK, cfg_overrides=SMALL)
    syn = synthetic_read(TOY, seed=0, dials=None, readout="identity",
                         n_tokens=N_TOK, with_probe=True, acts_mode="true_A",
                         cfg_overrides=SMALL)
    assert _bit_equal(syn.vals["S_res"], orc.vals["S_res"])
    for m in METRICS:
        assert _bit_equal(syn.vals[m], orc.vals[m]), f"{m} diverged"


def test_orientation_is_noop_on_planted_dictionaries():
    # W7: if the ground-truth-free sign rule ever flips a planted row, that is a finding
    # about the orientation rule and this fails loudly.
    syn = synthetic_read(TOY, seed=0, dials=_dials(), readout="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    W = syn.extra["W_raw"]
    unit = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    assert torch.equal(syn.W_unit, unit)


# --------------------------------------------------------------------------
# gate (b): the ridge anchor at beta = 0
# --------------------------------------------------------------------------

def test_ridge_anchor_at_zero_dials():
    syn = synthetic_read(TOY, seed=0, dials=_dials(beta=0.0, eta=0.0), readout="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    ex = syn.extra
    assert ex["support_flip_rate"]["scoring"] < 0.005          # W1 gate
    assert ex["fvu"]["scoring"] < ex["fvu_true_A"] + 0.02      # ridge lands at the noise floor
    assert ex["dials"]["beta"] == 0.0 and ex["dials"]["eta"] == 0.0


# --------------------------------------------------------------------------
# the corrupted-pair mask and provenance
# --------------------------------------------------------------------------

def test_corrupted_pair_mask_marks_exactly_the_ordered_edges():
    syn = synthetic_read(TOY, seed=0, dials=_dials(f=0.5), readout="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    mask = syn.extra["corrupted_pair"]
    edges = set(syn.extra["corrupted_edges"])
    marked = {(syn.feats[a], syn.feats[b]) for i, (a, b) in enumerate(syn.pairs) if mask[i]}
    assert marked == edges                                     # (p, c) ordering only
    assert len(edges) == max(1, round(0.5 * 12))


def test_provenance_fields_present():
    syn = synthetic_read(TOY, seed=0, dials=_dials(), readout="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    ex = syn.extra
    for key in ("dials", "corruption_kind", "realized_severity", "realized_severity_median",
                "support_flip_rate", "fvu", "n_holed_total", "readout", "planted_map_sha256",
                "n_latents", "scoring_sample_seed", "resolved_config"):
        assert key in ex, key
    assert ex["readout"] == "identity"
    # the map itself, not just that a hash was stamped
    from synthdict.planted import PlantedMap
    assert ex["planted_map_sha256"] == PlantedMap.identity(syn.F).sha256()
    assert ex["n_latents"] == syn.F
    # the matcher's provenance is gone, not renamed: nothing infers a correspondence here
    assert "match" not in ex and "matched_corr" not in ex and "matching_sample_seed" not in ex
    assert ex["scoring_sample_seed"] == 10_000                 # held_out_sample_seed(0)


def test_resolved_config_carries_the_seed():
    # ToyConfig.seed defaults to 0, so dropping the override is invisible to every seed-0 test
    from synthdict.read import resolved_config

    assert resolved_config("only_isa", 3)["seed"] == 3


def test_eta_reaches_the_scored_activations():
    """The hole must show up in the SCORED draw's activations, not only where it was planted.

    Formerly the second half of `test_eta_starves_parent_matching` (whose matched_corr half
    went with the matcher). This assertion is the named killer for the "hole never applied to
    the scoring draw" anchor, so it outlives the test it came from.
    """
    lo = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=0.0), readout="identity",
                        n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    hi = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=1.0), readout="identity",
                        n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    assert hi.extra["realized_l0"] < lo.extra["realized_l0"] - 0.05


# --------------------------------------------------------------------------
# probe draw separation
# --------------------------------------------------------------------------

def test_probe_fit_draw_is_separate_and_recorded():
    from scoring.benchmark.registry import probe_fit_sample_seed
    from scoring.core.grid import held_out_sample_seed

    default = synthetic_read(TOY, seed=0, dials=_dials(), readout="identity",
                             n_tokens=N_TOK, with_probe=True, cfg_overrides=SMALL)
    assert default.extra["probe_fit_sample_seed"] == probe_fit_sample_seed(0)
    bridged = synthetic_read(TOY, seed=0, dials=_dials(), readout="identity",
                             n_tokens=N_TOK, with_probe=True, cfg_overrides=SMALL,
                             probe_fit_seed=held_out_sample_seed(0))
    assert not _bit_equal(default.vals["S_res"], bridged.vals["S_res"])


# --------------------------------------------------------------------------
# integration: the frozen evaluator accepts the read
# --------------------------------------------------------------------------

def test_run_read_evaluates_a_synthetic_read_end_to_end():
    from scoring.benchmark.registry import EXPRESSIONS
    from scoring.benchmark.run_benchmark import run_read

    syn = synthetic_read(TOY, seed=0, dials=_dials(), readout="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    report, arrays = run_read(syn)
    assert report["read"] == "synthetic"
    assert set(report["expressions"]) == set(EXPRESSIONS)
    for m in METRICS:
        assert m in arrays
    for name in EXPRESSIONS:
        assert f"pass__{name}" in arrays and f"scorable__{name}" in arrays
    # the report's extra must have dropped every tensor (JSON contract)
    assert not any(isinstance(v, torch.Tensor) for v in report["extra"].values())


def test_probe_labels_are_self_not_truth():
    # Deployed convention (probe_self_W): fit labels = the synthetic SAE's own activations.
    # At eta=1 the self labels lose the parent on every child token, so the fitted probe must
    # differ from a truth-labeled fit; fitting on fw.A instead is the mutation this anchors.
    import dataclasses as _dc

    import torch as _t

    from scoring.benchmark.registry import probe_fit_sample_seed
    from scoring.core.detectors import fit_probe_directions
    from scoring.core.world import regenerate_world
    from synthdict.read import resolved_config

    syn = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=1.0), readout="identity",
                         n_tokens=N_TOK, with_probe=True, cfg_overrides=SMALL)
    rc = resolved_config(TOY, 0, SMALL)
    fw = regenerate_world(rc, sample_seed=probe_fit_sample_seed(0), n_tokens=N_TOK)
    from scoring.core.registry import CONSTANTS
    from scoring.core.detectors import s_res_from_directions
    P_true, avail_true = fit_probe_directions(fw.h, fw.A, CONSTANTS)
    truth_probe = s_res_from_directions(P_true, avail_true, syn.W_unit)
    pa = _t.tensor([a for a, _ in syn.pairs]); pb = _t.tensor([b for _, b in syn.pairs])
    assert not _bit_equal(syn.vals["S_res"], truth_probe[pa, pb].double())


# --------------------------------------------------------------------------
# clean acts mode (the firing-preserving construct; user decision 2026-09-06)
# --------------------------------------------------------------------------

def test_clean_mode_preserves_planted_firing_exactly():
    # THE property the mode exists for: beta cannot move any firing decision.
    # Under the ridge mode this fails (parent recall ~0.85 at beta=0.6, eta=0).
    from scoring.core.world import regenerate_world
    from synthdict.corruptions import absorb
    from synthdict.read import resolved_config, synth_encode

    rc = resolved_config(TOY, 0, SMALL)
    w = regenerate_world(rc, sample_seed=7, n_tokens=N_TOK)
    d = _dials(beta=0.8, eta=0.6)
    corruption = absorb(w.g, w.CONT, d, world_seed=0)
    acts, support, _ = synth_encode(w, corruption, 0, 7, acts_mode="clean")
    # Exactness is guaranteed for the UNCORRUPTED channel (corrupted children stay
    # ridge-determined and can flip ~1e-7 of entries at 200k tokens - review LOW-1);
    # at this size the full equality holds, and the channel-scoped one is the contract.
    cc = sorted({c for _, c in corruption.corrupted_edges})
    others = torch.tensor([j for j in range(w.g.shape[0]) if j not in set(cc)])
    assert torch.equal(acts[:, others] > 0, support[:, others])
    assert torch.equal(acts > 0, support)                 # holds at this scale (see above)


def test_clean_mode_uncorrupted_latents_keep_true_magnitudes():
    from scoring.core.world import regenerate_world
    from synthdict.corruptions import absorb
    from synthdict.read import resolved_config, synth_encode

    rc = resolved_config(TOY, 0, SMALL)
    w = regenerate_world(rc, sample_seed=7, n_tokens=N_TOK)
    d = _dials(beta=0.8, eta=0.6, f=0.5)
    corruption = absorb(w.g, w.CONT, d, world_seed=0)
    acts, support, _ = synth_encode(w, corruption, 0, 7, acts_mode="clean")
    cc = {c for _, c in corruption.corrupted_edges}
    others = [j for j in range(w.g.shape[0]) if j not in cc]
    oi = torch.tensor(others, dtype=torch.long)
    assert torch.equal(acts[:, oi], w.A[:, oi].double() * support[:, oi].double())


def test_clean_mode_hole_still_applies():
    # eta must keep its bite in clean mode: the parent is zeroed exactly where holed
    from scoring.core.world import regenerate_world
    from synthdict.corruptions import absorb
    from synthdict.read import resolved_config, synth_encode

    rc = resolved_config(TOY, 0, SMALL)
    w = regenerate_world(rc, sample_seed=7, n_tokens=N_TOK)
    corruption = absorb(w.g, w.CONT, _dials(beta=0.0, eta=1.0), world_seed=0)
    acts, support, _ = synth_encode(w, corruption, 0, 7, acts_mode="clean")
    for (p, c) in corruption.corrupted_edges:
        child_fires = w.A[:, c] > 0
        assert float(acts[child_fires, p].abs().max()) == 0.0
    # Magnitude honesty is checked at eta=0 (NO holes): the residual is A_c*g_c + noise, so
    # the fit must return ~A_c; fitting against the FULL h instead adds ~alpha*A_p (0.48
    # here). At eta=1 the parent mass flows into the child BY DESIGN (absorption's story),
    # so that regime cannot discriminate - first version of this test asserted it there and
    # failed on the designed behavior, not a bug.
    c2 = absorb(w.g, w.CONT, _dials(beta=0.0, eta=0.0), world_seed=0)
    acts2, _s2, _h2 = synth_encode(w, c2, 0, 7, acts_mode="clean")
    cc = torch.tensor(sorted({c for _, c in c2.corrupted_edges}), dtype=torch.long)
    on = w.A[:, cc] > 0
    rel = (acts2[:, cc][on] - w.A[:, cc].double()[on]).abs() / w.A[:, cc].double()[on].clamp_min(1e-9)
    assert float(rel.median()) < 0.1


def test_clean_mode_no_corruption_is_true_A():
    from scoring.core.world import regenerate_world
    from synthdict.read import resolved_config, synth_encode

    rc = resolved_config(TOY, 0, SMALL)
    w = regenerate_world(rc, sample_seed=7, n_tokens=N_TOK)
    acts, _s, _h = synth_encode(w, None, 0, 7, acts_mode="clean")
    assert torch.equal(acts, w.A.double())


def test_clean_read_beta_only_keeps_cofiring_clean():
    # The read-level consequence: at (beta=0.6, eta=0) coverage on corrupted edges is
    # EXACTLY 1.0 under clean mode (it was ~0.85 under ridge - the review's CRITICAL).
    syn = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=0.0), readout="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL,
                         acts_mode="clean")
    mask = syn.extra["corrupted_pair"]
    cov = syn.vals["coverage_R"][mask]
    assert float(cov.min()) == 1.0
    assert syn.extra["support_flip_rate"]["scoring"] < 1e-5   # corrupted-child flips only (LOW-1)
    assert syn.extra["acts_mode"] == "clean"


def test_clean_mode_corrupted_magnitudes_use_the_corrupted_decoder():
    # The residual fit runs against d_c' (W_raw), NOT g_c: at eta=0 the residual is
    # A_c*g_c + noise, so a_c ~= A_c * cos(g_c, d_c') = A_c * (1 + beta*alpha)/||g_c + beta*g_p||.
    # Fitting against g_c instead returns a_c ~= A_c (ratio 1.0) - a surviving mutant the
    # review measured at +10% FVU and 13.5% magnitude shift (MED-3a); this pins the basis.
    import math

    from scoring.core.world import regenerate_world
    from synthdict.corruptions import absorb
    from synthdict.read import resolved_config, synth_encode

    beta, alpha = 0.8, 0.48
    rc = resolved_config(TOY, 0, SMALL)
    w = regenerate_world(rc, sample_seed=7, n_tokens=N_TOK)
    corruption = absorb(w.g, w.CONT, _dials(beta=beta, eta=0.0), world_seed=0)
    acts, _s, _h = synth_encode(w, corruption, 0, 7, acts_mode="clean")
    want = (1 + beta * alpha) / math.sqrt(1 + beta ** 2 + 2 * beta * alpha)
    cc = torch.tensor(sorted({c for _, c in corruption.corrupted_edges}), dtype=torch.long)
    on = w.A[:, cc] > 0
    ratio = (acts[:, cc][on] / w.A[:, cc].double()[on]).median()
    assert abs(float(ratio) - want) < 0.02
