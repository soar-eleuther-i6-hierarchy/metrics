"""The shared gates and rules, pinned on a 2 x 2 frame worked out by hand.

`metrics/rules/` is where both pipelines take their decisions from, and a decision that
changes silently changes every downstream number. This pins every gate and every rule on a
frame small enough to verify on paper: two candidate parents, two candidate children.

    python3 -m tests.test_rules

Frame (counts), N = 1,000 tokens:

    cofire = [[40, 5],      fire_p = [100, 50]      fire_c = [45, 30]
              [35, 0]]

    R      = [[0.889, 0.167],   R_rev = [[0.40, 0.05],
              [0.778, 0.000]]            [0.70, 0.00]]

Support needs >= 20 firings each and >= 30 co-firings: (0,0) and (1,0) only.
(0,0) is strict containment (R >= 0.5, R_rev < 0.5); (1,0) is mutual containment (both >= 0.5).
Both co-fire far above independence (expected 4.5 and 2.25). Survival is 0.9 and 0.2; column
1 is untestable. Reconstruction gains are all 0.05. No endpoint is dense.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.rules import GEMMA_MATRYOSHKA as C  # noqa: E402
from metrics.rules import RULESET_VERSION, PairStats, score_pairs  # noqa: E402

NAN = float("nan")


def _same(t: torch.Tensor, expect: list[list[float]]) -> bool:
    e = torch.tensor(expect, dtype=torch.float64)
    return bool(((t == e) | (torch.isnan(t) & torch.isnan(e))).all())


def _frame(**probe) -> PairStats:
    cofire = torch.tensor([[40.0, 5.0], [35.0, 0.0]])
    fire_p = torch.tensor([100.0, 50.0])
    fire_c = torch.tensor([45.0, 30.0])
    return PairStats(
        cofire=cofire, fire_p=fire_p, fire_c=fire_c,
        R=cofire / fire_c, R_rev=cofire / fire_p.unsqueeze(1),
        parent_gain=torch.full((2, 2), 0.05), child_gain=torch.tensor([0.05, 0.05]),
        survival=torch.tensor([[0.9, NAN], [0.2, NAN]]), survival_scale="raw",
        high_outdeg_p=torch.tensor([0.0, 0.0]), high_outdeg_c=torch.tensor([0.0, 0.0]),
        n_tokens=1000, **probe,
    )


def test_gates_without_probe() -> None:
    g = score_pairs(_frame(), C)["gates"]
    assert _same(g["gate_support"], [[1, 0], [1, 0]])
    assert _same(g["gate_contains"], [[1, NAN], [1, NAN]])
    assert _same(g["gate_strictly_contains"], [[1, NAN], [0, NAN]])
    assert _same(g["gate_mutually_contains"], [[0, NAN], [1, NAN]])
    assert _same(g["gate_recon"], [[1, 1], [1, 1]])            # defined everywhere, not gated by support
    assert _same(g["gate_high_outdegree"], [[0, 0], [0, 0]])
    assert _same(g["gate_freq_survives"], [[1, NAN], [0, NAN]])
    assert _same(g["gate_pmi_positive"], [[1, NAN], [1, NAN]])
    assert _same(g["gate_sres_rank"], [[NAN, NAN], [NAN, NAN]])  # no probe: unmeasurable, not failed


def test_rules_without_probe() -> None:
    r = score_pairs(_frame(), C)["rules"]
    mask, scorable = r["rule_containment"]
    assert _same(mask.double(), [[1, 0], [0, 0]]) and _same(scorable.double(), [[1, 0], [1, 0]])
    mask, scorable = r["rule_hierarchy"]                        # reads the probe: unscorable everywhere
    assert not mask.any() and not scorable.any()
    mask, scorable = r["rule_dense_endpoint"]
    assert not mask.any() and scorable.all()
    mask, scorable = r["rule_frequency_driven"]                 # (1,0) fails survival but is not strict
    assert not mask.any() and _same(scorable.double(), [[1, 0], [1, 0]])


def test_probe_rank_and_hierarchy() -> None:
    # pool of 8 decoders: parents at 0 and 1, children at 2 and 3. Child 0's probe ranks its
    # own decoder first, parent 0 second and parent 1 last; child 1 has no probe.
    corr = torch.tensor([[0.9, 0.1, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = score_pairs(_frame(probe_corr=corr, parent_ids=torch.tensor([0, 1]),
                             child_ids=torch.tensor([2, 3]),
                             probe_available=torch.tensor([True, False])), C)
    assert C.sres_rank_top_k == 5
    assert _same(out["gates"]["gate_sres_rank"], [[1, NAN], [0, NAN]])
    mask, scorable = out["rules"]["rule_hierarchy"]
    assert _same(mask.double(), [[1, 0], [0, 0]])               # strict + PMI + survival + rank
    assert _same(scorable.double(), [[1, 0], [1, 0]])


def test_either_endpoint_passes_when_one_side_is_undefined() -> None:
    """The review's first item: a dense parent with a rare child is a pass, not NaN."""
    s = _frame()
    s = PairStats(**{**s.__dict__, "high_outdeg_p": torch.tensor([1.0, 0.0]),
                     "high_outdeg_c": torch.tensor([NAN, 0.0])})
    g = score_pairs(s, C)["gates"]["gate_high_outdegree"]
    # parent 0 is dense: both its pairs pass, including (0,0) whose child is undefined.
    # parent 1 is not dense and defined: (1,0) reads 0 even though child 0 is undefined,
    # since one defined side decides. NaN would need both sides undefined.
    assert _same(g, [[1, 1], [0, 0]])


def test_ruleset_is_versioned() -> None:
    assert isinstance(RULESET_VERSION, int) and RULESET_VERSION >= 2


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"[rules] {len(tests)} checks pinned on the 2x2 frame, ruleset version {RULESET_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
