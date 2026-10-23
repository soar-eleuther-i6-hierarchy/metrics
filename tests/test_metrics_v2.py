"""metrics_v2 is a re-arrangement, not a re-implementation.

Every entry of `metrics_v2.OLD_TO_NEW` must resolve to the *identical* object as
its `metrics.*` original (`is`, not equality), so no formula and no default can
have drifted in the move. The three new functions are checked on tiny tensors
against their own definitions.

    python3 -m tests.test_metrics_v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import metrics  # noqa: E402
import metrics_v2  # noqa: E402
from metrics import coverage, in_block, joint_child, outdegree, reconstruction, sres, token_control  # noqa: E402

_OLD_MODULES = (metrics, coverage, in_block, joint_child, outdegree, reconstruction, sres, token_control)


def _old(name: str):
    for m in _OLD_MODULES:
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(f"{name!r} is not in metrics/")


def _new(dotted: str):
    pkg, fn = dotted.split(".")
    return getattr(getattr(metrics_v2, pkg), fn)


def test_every_old_name_is_the_same_object() -> None:
    for old, new in metrics_v2.OLD_TO_NEW.items():
        assert _old(old) is _new(new), f"{old} -> {new} is a different object"


def test_the_table_covers_the_old_public_api() -> None:
    missing = [n for n in metrics.__all__ if n not in metrics_v2.OLD_TO_NEW and n not in metrics_v2.DROPPED]
    assert not missing, f"metrics.__all__ names with no v2 name: {missing}"


def test_rules_are_not_moved() -> None:
    from metrics import rules
    assert metrics_v2.rules is rules


def test_decoder_cosine() -> None:
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    mid = torch.tensor([1.0, 1.0, 0.0])
    cos = metrics_v2.structure.decoder_cosine(torch.stack([e1, e2]), torch.stack([e1, mid, 3 * e2]))
    expect = torch.tensor([[1.0, 2 ** -0.5, 0.0], [0.0, 2 ** -0.5, 1.0]], dtype=torch.float64)
    assert torch.allclose(cos, expect), cos
    zero = metrics_v2.structure.decoder_cosine(torch.zeros(1, 3), torch.stack([e1]))
    assert float(zero[0, 0]) == 0.0, "a zero row gives 0, not NaN"


def test_absorption_signature() -> None:
    R = torch.tensor([[1.0, 0.2], [0.1, 0.9]])
    cos = torch.tensor([[0.1, 0.8], [0.8, 0.1]])
    out = metrics_v2.pathology.absorption_signature(R, cos, tau=0.5, cos_min=0.5)
    # aligned decoders + failed containment: (0,1) and (1,0) only
    assert out["flag"].tolist() == [[False, True], [True, False]], out["flag"]
    support = torch.tensor([[True, False], [True, True]])
    out = metrics_v2.pathology.absorption_signature(R, cos, tau=0.5, cos_min=0.5, support=support)
    assert out["flag"].tolist() == [[False, False], [True, False]]


def test_composition_signature() -> None:
    # parents 0,1 unrelated; parent 2 contains parent 3 (nested); parents 4,5 co-extensive
    fire_p = torch.tensor([100.0, 100.0, 100.0, 50.0, 60.0, 60.0])
    cof = torch.zeros(6, 6)
    for i in range(6):
        cof[i, i] = fire_p[i]
    cof[0, 1] = cof[1, 0] = 10.0     # P(0|1)=0.1, P(1|0)=0.1
    cof[2, 3] = cof[3, 2] = 50.0     # P(2|3)=1.0 : nested
    cof[4, 5] = cof[5, 4] = 60.0     # both 1.0 : duplicates
    edges = torch.zeros(6, 3, dtype=torch.bool)
    edges[0, 0] = edges[1, 0] = True   # child 0 under unrelated parents  -> flagged
    edges[2, 1] = edges[3, 1] = True   # child 1 under nested parents     -> not
    edges[4, 2] = edges[5, 2] = True   # child 2 under duplicate parents  -> not
    out = metrics_v2.pathology.composition_signature(edges, cof, fire_p, tau=0.5)
    assert out["flag"].tolist() == [True, False, False], out
    assert out["pairs"] == {0: [(0, 1)]}


def test_direction_duplicates() -> None:
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    W = torch.stack([e1, 2 * e1, -e1, e2, torch.tensor([1.0, 1.0, 0.0])])   # 0,1,2 share one direction
    out = metrics_v2.pathology.direction_duplicates(W, cos_min=0.9)
    assert out["pairs"] == [(0, 1), (0, 2), (1, 2)], out["pairs"]          # sign ignored, scale ignored
    assert torch.isnan(out["cosine"][0, 0])
    assert abs(float(out["cosine"][0, 4]) - 2 ** -0.5) < 1e-12


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    n = len(metrics_v2.OLD_TO_NEW)
    print(f"[v2] {n} old names resolve to their identical objects; "
          f"{len(metrics_v2.NEW)} new functions checked; rules not moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
