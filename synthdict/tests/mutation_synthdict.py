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
