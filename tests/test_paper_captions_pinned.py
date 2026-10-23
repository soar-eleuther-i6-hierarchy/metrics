"""The paper captions cannot be checked for truth, so their INPUTS are pinned.

This test is for `reporting/make_report_figures.py`. It guards one direction: the
panel stays exactly where it was and the DATA underneath it changes.

That is not hypothetical. Switching the reference toy checkpoint left two captions
asserting a world that had stopped existing:

    "every miss is an edge whose child the SAE never learned"   -- nothing missed
    "The SAE conflated two concepts the grammar keeps apart"    -- it conflated none

Both figures drew the same two panels before and after, so a panel fingerprint would
have passed. Every number in those sentences was computed and updated itself; the
sentences around them were hand-written English and did not.

What this test does is narrow, and worth stating plainly: **it does not check that any
caption is correct.** It cannot. It fails when the data a caption is written against is
regenerated, which sends whoever regenerated it back to the caption. A tripwire, not a
proof.

Only inputs whose captions carry hand-written claims are pinned. Adding a file here
costs nothing; leaving one out costs a caption that quietly goes stale.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402


def _fingerprint(path: Path) -> str:
    """sha256[:16] of the file's canonical JSON.

    Canonicalised rather than hashed raw, so a reformat or a key reordering does not
    fire the tripwire: what matters is whether a caption's facts moved.
    """
    return hashlib.sha256(
        json.dumps(json.loads(path.read_text()), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()[:16]


# input file -> (fingerprint, the captions written against it)
#
# WHEN THIS FAILS: do not re-pin first. Open the captions listed beside the file,
# read every hand-typed number and every claim in them against the regenerated data,
# and only then re-pin. Re-pinning without reading is the same as deleting the test,
# and it is the easier of the two to do by accident.
EXPECTED: dict[str, tuple[str, tuple[str, ...]]] = {
    "trained_toy_calibration.json": (
        "4f44a42fc782bbae", ("calibration_trained_toy_recovery", "calibration_toy_tree_recovered"),
    ),
    "batch_topk_toy_calibration.json": (
        "50444bb187d218d7", ("calibration_toy_tree_recovered",),
    ),
    "synthetic_toy_calibration.json": (
        "8d9718f64200ea5c", ("calibration_synthetic_toy_scorecard",),
    ),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_caption_inputs_unchanged_since_captions_were_read(name):
    path = C.OUT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} absent; nothing to pin")
    expected, captions = EXPECTED[name]
    actual = _fingerprint(path)
    assert actual == expected, (
        f"\n{name} changed since its captions were last read against it.\n"
        f"  expected {expected}, got {actual}\n"
        f"  captions written against it: {', '.join(captions)}\n\n"
        "Open each one in reporting/make_report_figures.py::_captions and check every\n"
        "hand-written claim still holds. Then re-pin this hash -- not before."
    )


def test_every_pinned_input_names_a_real_caption():
    """A pin that guards no caption is a pin nobody will maintain."""
    sys.argv = ["make_report_figures"]
    from reporting import make_report_figures as F

    known = set(F.CLAIMS)
    for name, (_, captions) in EXPECTED.items():
        unknown = sorted(set(captions) - known)
        assert not unknown, f"{name} pins captions that no figure emits: {unknown}"
