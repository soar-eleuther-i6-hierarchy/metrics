#!/usr/bin/env python
"""Mutation harness for synthdict — every anchor must be KILLED by the test that names it.

Pattern follows `tests_local/mutation_benchmark.py`: patch one behavior in the source, run the
single named test, require it to FAIL, restore the source. A surviving mutation means the test
suite passes for a reason unrelated to the behavior it claims to pin (the 33-tautologies bug
class), and the harness exits nonzero.

Run:  .venv-exp0/bin/python synthdict/tests/mutation_synthdict.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


@dataclass(frozen=True)
class Mutation:
    name: str
    file: str                    # relative to experiment_0 root
    old: str
    new: str
    test: str                    # pytest node id expected to FAIL under the mutation


MUTATIONS = [
    # ---------------- corruptions.py ----------------
    Mutation("beta sign flip",
             "synthdict/corruptions.py",
             "d = g[c].double() + float(dials.beta)",
             "d = g[c].double() - float(dials.beta)",
             "synthdict/tests/test_corruptions.py::test_carry_direction_is_g_c_plus_beta_g_p"),
    Mutation("rbar dropped from the carry",
             "synthdict/corruptions.py",
             "float(dials.beta) * float(dials.rbar)",
             "float(dials.beta) * 1.0",
             "synthdict/tests/test_corruptions.py::test_rbar_scales_the_carry"),
    Mutation("unit-norm dropped on the corrupted row",
             "synthdict/corruptions.py",
             "W[c] = d / d.norm().clamp_min(_TINY)",
             "W[c] = d",
             "synthdict/tests/test_corruptions.py::test_corrupted_rows_are_unit_norm"),
    Mutation("severity measured against the child instead of the parent",
             "synthdict/corruptions.py",
             "gp = g[p].double()",
             "gp = g[c].double()",
             "synthdict/tests/test_corruptions.py::test_realized_severity_is_cos_with_parent_not_child"),
    Mutation("edge_fraction ignored (everything corrupted)",
             "synthdict/corruptions.py",
             "k = max(1, round(edge_fraction * len(edges)))",
             "k = len(edges)",
             "synthdict/tests/test_corruptions.py::test_edge_fraction_selects_the_right_count"),
    Mutation("edge selection insensitive to the world seed",
             "synthdict/corruptions.py",
             "manual_seed(int(world_seed) + EDGE_SEED_OFFSET)",
             "manual_seed(EDGE_SEED_OFFSET)",
             "synthdict/tests/test_corruptions.py::test_edge_selection_deterministic_and_seed_sensitive"),
    Mutation("eta ignored (full hole regardless)",
             "synthdict/corruptions.py",
             "want = round(eta * int(child_tok.numel()))",
             "want = int(child_tok.numel())",
             "synthdict/tests/test_corruptions.py::test_eta_zero_is_identity"),
    Mutation("hole applied to the child column instead of the parent",
             "synthdict/corruptions.py",
             "out[child_tok[perm[:want]], p] = False",
             "out[child_tok[perm[:want]], c] = False",
             "synthdict/tests/test_corruptions.py::test_hole_never_touches_the_child_column"),
    Mutation("hole tokens drawn from the parent's firing instead of the child's",
             "synthdict/corruptions.py",
             "child_tok = support[:, c].nonzero(as_tuple=True)[0]",
             "child_tok = support[:, p].nonzero(as_tuple=True)[0]",
             "synthdict/tests/test_corruptions.py::test_hole_removes_parent_on_eta_fraction_of_child_tokens"),
    Mutation("hole seed drops the world seed",
             "synthdict/corruptions.py",
             "+ 1_000_003 * int(world_seed)",
             "+ 0 * int(world_seed)",
             "synthdict/tests/test_corruptions.py::test_hole_seed_depends_on_world_seed"),
    Mutation("hole seed drops the draw seed (same hole for every draw)",
             "synthdict/corruptions.py",
             "+ 7_919 * int(sample_seed)",
             "+ 0 * int(sample_seed)",
             "synthdict/tests/test_corruptions.py::test_hole_deterministic_and_draw_dependent"),
    # ---------------- activations.py ----------------
    Mutation("ridge solves off the planted support (dense SAE)",
             "synthdict/activations.py",
             "idx = support[t].nonzero(as_tuple=True)[0]",
             "idx = torch.arange(S, device=Wd.device)",
             "synthdict/tests/test_activations.py::test_acts_zero_off_support"),
    Mutation("ridge term dropped",
             "synthdict/activations.py",
             "gram = gram + lam * torch.eye",
             "gram = gram + 0.0 * torch.eye",
             "synthdict/tests/test_activations.py::test_lambda_actually_regularizes"),
    Mutation("flip counts strict negatives only (planted-but-zero missed)",
             "synthdict/activations.py",
             "flipped = (acts[support] <= 0.0).sum()",
             "flipped = (acts[support] < 0.0).sum()",
             "synthdict/tests/test_activations.py::test_support_flip_rate_counts_nonpositive_on_support"),
    # ---------------- read.py ----------------
    Mutation("scoring draw not held out (scores on the matching draw)",
             "synthdict/read.py",
             "score_seed = held_out_sample_seed(int(seed))\n    score = regenerate_world(",
             "score_seed = int(seed)\n    score = regenerate_world(",
             "synthdict/tests/test_read.py::test_provenance_fields_present"),
    Mutation("corrupted-pair mask transposed to (child, parent)",
             "synthdict/read.py",
             "[(feats[a], feats[b]) in corrupted for a, b in pairs]",
             "[(feats[b], feats[a]) in corrupted for a, b in pairs]",
             "synthdict/tests/test_read.py::test_corrupted_pair_mask_marks_exactly_the_ordered_edges"),
    Mutation("probe fitted on TRUE firing instead of the self label",
             "synthdict/read.py",
             "P, avail = fit_probe_directions(fw.h, acts_f[:, lat], CONSTANTS)",
             "P, avail = fit_probe_directions(fw.h, fw.A[:, lat], CONSTANTS)",
             "synthdict/tests/test_read.py::test_probe_labels_are_self_not_truth"),
    Mutation("hole never applied to the scoring draw",
             "synthdict/read.py",
             "acts_ho, support_ho, holed_ho = synth_encode(score, corruption, seed, score_seed, acts_mode)",
             "acts_ho, support_ho, holed_ho = synth_encode(score, None, seed, score_seed, acts_mode)",
             "synthdict/tests/test_read.py::test_eta_starves_parent_matching"),
    # ------- audit round 2: the 11 proven survivors, now anchored -------
    Mutation("hungarian matcher fed the SCORING draw (M1)",
             "synthdict/read.py",
             "res = match_features(activation_corr(inw.A, acts_in), inw.g, oriented_in,",
             "res = match_features(activation_corr(score.A, acts_ho), score.g, oriented_ho,",
             "synthdict/tests/test_read.py::test_hungarian_matches_on_the_matching_draw"),
    Mutation("matching seed stamped/used as the scoring seed (M9)",
             "synthdict/read.py",
             "matching_seed = int(seed)\n        inw = regenerate_world(",
             "matching_seed = score_seed\n        inw = regenerate_world(",
             "synthdict/tests/test_read.py::test_eta_starves_parent_matching"),
    Mutation("dial_dirname swaps beta and eta (M2)",
             "synthdict/run_synth.py",
             'return f"beta{dials.beta:g}-eta{dials.eta:g}-f{dials.edge_fraction:g}"',
             'return f"beta{dials.eta:g}-eta{dials.beta:g}-f{dials.edge_fraction:g}"',
             "synthdict/tests/test_driver.py::test_dial_point_end_to_end"),
    Mutation("collector swaps beta and eta (M3)",
             "synthdict/report.py",
             '"beta": meta["dials"]["beta"], "eta": meta["dials"]["eta"],',
             '"beta": meta["dials"]["eta"], "eta": meta["dials"]["beta"],',
             "synthdict/tests/test_report.py::test_collect_and_write"),
    Mutation("collector reads f from eta (M10)",
             "synthdict/report.py",
             '"f": meta["dials"]["edge_fraction"],',
             '"f": meta["dials"]["eta"],',
             "synthdict/tests/test_report.py::test_collect_and_write"),
    Mutation("resolved_config drops the seed override (M4)",
             "synthdict/read.py",
             "cfg = spec.replace(resolve_config(toy), seed=int(seed), **(cfg_overrides or {}))",
             "cfg = spec.replace(resolve_config(toy), **(cfg_overrides or {}))",
             "synthdict/tests/test_read.py::test_resolved_config_carries_the_seed"),
    Mutation("census corruption rebuilt under the wrong world seed (M5)",
             "synthdict/run_synth.py",
             "corruption = absorb(world.g, world.CONT, dials, world_seed=int(seed))",
             "corruption = absorb(world.g, world.CONT, dials, world_seed=int(seed) + 1)",
             "synthdict/tests/test_driver.py::test_dial_point_end_to_end"),
    Mutation("report's null read from the calibration half (M6)",
             "synthdict/report.py",
             'null_eval = (y == labels._index("unrelated")) & (npz["split"] == 2)',
             'null_eval = (y == labels._index("unrelated")) & (npz["split"] == 1)',
             "synthdict/tests/test_report.py::test_collect_and_write"),
    Mutation("report swaps pass and scorable masks (M7)",
             "synthdict/report.py",
             'p, s = npz[f"pass__{name}"].astype(bool), npz[f"scorable__{name}"].astype(bool)',
             'p, s = npz[f"scorable__{name}"].astype(bool), npz[f"pass__{name}"].astype(bool)',
             "synthdict/tests/test_report.py::test_collect_and_write"),
    Mutation("TARGET_BY_TOY swapped (M8)",
             "synthdict/report.py",
             'TARGET_BY_TOY = {"only_isa": "is_a", "only_firing": "firing_only"}',
             'TARGET_BY_TOY = {"only_isa": "firing_only", "only_firing": "is_a"}',
             "synthdict/tests/test_report.py::test_collect_and_write"),
    # NOT anchored — an EQUIVALENT MUTANT in round 1: "lat = torch.tensor(feats)" instead of
    # "lat = match_t[...]" is behavior-identical whenever the match is the identity on the kept
    # features, which holds at every reachable round-1 state (verified at the grid extremes and
    # in the recovery-drop test, where drops occur but kept features still match themselves).
    # The routed form is kept for correctness the moment a future corruption permutes matches.
    Mutation("clean mode ignores the hole (masks by raw firing)",
             "synthdict/read.py",
             "A_masked = bundle.A.double() * support.double()",
             "A_masked = bundle.A.double() * (bundle.A > 0).double()",
             "synthdict/tests/test_read.py::test_clean_mode_preserves_planted_firing_exactly"),
    Mutation("clean mode fits against the TRUE g rows instead of the corrupted decoder (MED-3a)",
             "synthdict/read.py",
             "acts[:, idx] = ridge_acts(residual, W_raw[idx], support[:, idx])",
             "acts[:, idx] = ridge_acts(residual, bundle.g.double()[idx], support[:, idx])",
             "synthdict/tests/test_read.py::test_clean_mode_corrupted_magnitudes_use_the_corrupted_decoder"),
    # MED-3b NOT anchored - a NEAR-EQUIVALENT mutant: replacing the joint solve over the
    # corrupted-children set with a per-child loop shifts FVU by ~1.7% and nothing
    # claim-bearing; an exact-equality anchor would only restate the implementation.
    Mutation("clean mode fits corrupted children against h, not the residual (kills via the magnitude-honesty check)",
             "synthdict/read.py",
             "residual = bundle.h.double() - acts @ bundle.g.double()",
             "residual = bundle.h.double()",
             "synthdict/tests/test_read.py::test_clean_mode_hole_still_applies"),
    # ---------------- report.py ----------------
    Mutation("intact side leaks corrupted pairs (mask not excluded)",
             "synthdict/report.py",
             "intact = is_target & ~corr_mask",
             "intact = is_target",
             "synthdict/tests/test_report.py::test_collect_and_write"),
    Mutation("corrupted side computed on the whole target class",
             "synthdict/report.py",
             "corrupted = is_target & corr_mask",
             "corrupted = is_target",
             "synthdict/tests/test_report.py::test_collect_and_write"),
]


def run(mutations=MUTATIONS) -> int:
    survivors = []
    for m in mutations:
        path = ROOT / m.file
        src = path.read_text()
        if m.old not in src:
            print(f"STALE ANCHOR (old string not found): {m.name}")
            survivors.append(m.name)
            continue
        if src.count(m.old) != 1:
            print(f"AMBIGUOUS ANCHOR (old string not unique): {m.name}")
            survivors.append(m.name)
            continue
        path.write_text(src.replace(m.old, m.new))
        try:
            r = subprocess.run([PY, "-m", "pytest", m.test, "-q", "-x", "--no-header"],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            path.write_text(src)
        if r.returncode == 0:
            print(f"SURVIVED: {m.name}  ({m.test})")
            survivors.append(m.name)
        else:
            print(f"killed:   {m.name}")
    print(f"\n{len(mutations) - len(survivors)}/{len(mutations)} mutations killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(run())
