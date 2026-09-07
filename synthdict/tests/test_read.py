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
    syn = synthetic_read(TOY, seed=0, dials=None, match_mode="identity",
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
    syn = synthetic_read(TOY, seed=0, dials=None, match_mode="identity",
                         n_tokens=N_TOK, with_probe=True, acts_mode="true_A",
                         cfg_overrides=SMALL)
    assert _bit_equal(syn.vals["S_res"], orc.vals["S_res"])
    for m in METRICS:
        assert _bit_equal(syn.vals[m], orc.vals[m]), f"{m} diverged"


def test_orientation_is_noop_on_planted_dictionaries():
    # W7: if the ground-truth-free sign rule ever flips a planted row, that is a finding
    # about the orientation rule and this fails loudly.
    syn = synthetic_read(TOY, seed=0, dials=_dials(), match_mode="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    W = syn.extra["W_raw"]
    unit = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    assert torch.equal(syn.W_unit, unit)


# --------------------------------------------------------------------------
# gate (b): the ridge anchor at beta = 0
# --------------------------------------------------------------------------
def test_ridge_anchor_at_zero_dials():
    syn = synthetic_read(TOY, seed=0, dials=_dials(beta=0.0, eta=0.0), match_mode="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    ex = syn.extra
    assert ex["support_flip_rate"]["scoring"] < 0.005          # W1 gate
    assert ex["fvu"]["scoring"] < ex["fvu_true_A"] + 0.02      # ridge lands at the noise floor
    assert ex["dials"]["beta"] == 0.0 and ex["dials"]["eta"] == 0.0


# --------------------------------------------------------------------------
# the corrupted-pair mask and provenance
# --------------------------------------------------------------------------
def test_corrupted_pair_mask_marks_exactly_the_ordered_edges():
    syn = synthetic_read(TOY, seed=0, dials=_dials(f=0.5), match_mode="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    mask = syn.extra["corrupted_pair"]
    edges = set(syn.extra["corrupted_edges"])
    marked = {(syn.feats[a], syn.feats[b]) for i, (a, b) in enumerate(syn.pairs) if mask[i]}
    assert marked == edges                                     # (p, c) ordering only
    assert len(edges) == max(1, round(0.5 * 12))


def test_provenance_fields_present():
    syn = synthetic_read(TOY, seed=0, dials=_dials(), match_mode="identity",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    ex = syn.extra
    for key in ("dials", "corruption_kind", "realized_severity", "realized_severity_median",
                "support_flip_rate", "fvu", "n_holed_total", "match_mode",
                "scoring_sample_seed", "matching_sample_seed", "resolved_config"):
        assert key in ex, key
    assert ex["match_mode"] == "identity"
    assert ex["scoring_sample_seed"] == 10_000                 # held_out_sample_seed(0)


# --------------------------------------------------------------------------
# hungarian mode
# --------------------------------------------------------------------------
def test_hungarian_beta0_recovers_everything_identically():
    syn = synthetic_read(TOY, seed=0, dials=_dials(beta=0.0, eta=0.0), match_mode="hungarian",
                         n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    F = syn.F
    assert syn.feats == list(range(F))
    assert bool(syn.recovered.all())
    assert torch.equal(syn.extra["match"], torch.arange(F))


def test_eta_starves_parent_matching():
    lo = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=0.0), match_mode="hungarian",
                        n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    hi = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=1.0), match_mode="hungarian",
                        n_tokens=N_TOK, with_probe=False, cfg_overrides=SMALL)
    parents = sorted({p for p, _ in lo.extra["corrupted_edges"]})
    c_lo = lo.extra["matched_corr"][parents].mean()
    c_hi = hi.extra["matched_corr"][parents].mean()
    assert float(c_hi) < float(c_lo)
    # the hole must reach the SCORED activations too (catches a hole applied only at matching)
    assert hi.extra["realized_l0"] < lo.extra["realized_l0"] - 0.05


# --------------------------------------------------------------------------
# probe draw separation
# --------------------------------------------------------------------------
def test_probe_fit_draw_is_separate_and_recorded():
    from scoring.benchmark.registry import probe_fit_sample_seed
    from scoring.core.grid import held_out_sample_seed

    default = synthetic_read(TOY, seed=0, dials=_dials(), match_mode="identity",
                             n_tokens=N_TOK, with_probe=True, cfg_overrides=SMALL)
    assert default.extra["probe_fit_sample_seed"] == probe_fit_sample_seed(0)
    bridged = synthetic_read(TOY, seed=0, dials=_dials(), match_mode="identity",
                             n_tokens=N_TOK, with_probe=True, cfg_overrides=SMALL,
                             probe_fit_seed=held_out_sample_seed(0))
    assert not _bit_equal(default.vals["S_res"], bridged.vals["S_res"])


# --------------------------------------------------------------------------
# integration: the frozen evaluator accepts the read
# --------------------------------------------------------------------------
def test_run_read_evaluates_a_synthetic_read_end_to_end():
    from scoring.benchmark.registry import EXPRESSIONS
    from scoring.benchmark.run_benchmark import run_read

    syn = synthetic_read(TOY, seed=0, dials=_dials(), match_mode="identity",
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

    syn = synthetic_read(TOY, seed=0, dials=_dials(beta=0.6, eta=1.0), match_mode="identity",
                         n_tokens=N_TOK, with_probe=True, cfg_overrides=SMALL)
    rc = resolved_config(TOY, 0, SMALL)
    fw = regenerate_world(rc, sample_seed=probe_fit_sample_seed(0), n_tokens=N_TOK)
    from scoring.core.registry import CONSTANTS
    from scoring.core.detectors import s_res_from_directions
    P_true, avail_true = fit_probe_directions(fw.h, fw.A, CONSTANTS)
    truth_probe = s_res_from_directions(P_true, avail_true, syn.W_unit)
    pa = _t.tensor([a for a, _ in syn.pairs]); pb = _t.tensor([b for _, b in syn.pairs])
    assert not _bit_equal(syn.vals["S_res"], truth_probe[pa, pb].double())
