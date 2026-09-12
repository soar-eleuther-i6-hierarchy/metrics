#!/usr/bin/env python
"""Bit-identity gate: does the matcher-free pipeline reproduce the matched one, exactly?

Round 1 measured that identity and hungarian columns agreed at all 46 dial points, so removing
the matcher should move NO number. This checks that claim instead of asserting it: re-run dial
points under the new code into a scratch tag, then compare every array against the saved
round-1 artifact.

  python synthdict/tests/regression_gate.py --old outputs_local/synthdict/SYNTH-R1 \
      --new outputs_local/synthdict/GATE-MF --expect 2

Expected differences, and ONLY these: the arrays `match` / `matched_corr` are gone (they were
the matcher's own output), and meta `match_mode` is now `readout`. Any metric array that moves
is a bug, not a correction.

RUN IT UNDER THE SAME BLAS THREAD COUNT AS THE ARTIFACT BEING COMPARED. Measured 2026-09-12:
the same code, same seed, same draw gives bit-identical arrays run-to-run, but differs at
~1e-15 on the magnitude detectors (`recon_2a`, `joint_child_mass`) between 8 threads and 96,
because the parallel reduction order changes. Round 1's timed-pilot point ran unbounded while
the grid points ran at 8, so a gate against the pilot point must also run unbounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Written by the matcher and expected to vanish; everything else must match bit-for-bit.
RETIRED_ARRAYS = ("match", "matched_corr")


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality with NaN == NaN (float `==` is False on identical NaN cells)."""
    if a.shape != b.shape:
        return False
    if a.dtype.kind not in "fc":
        return bool(np.array_equal(a, b))
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.array_equal(a[~na], b[~nb]))


def compare_artifact(old_dir: Path, new_dir: Path) -> dict:
    old = np.load(old_dir / "scores.npz", allow_pickle=True)
    new = np.load(new_dir / "scores.npz", allow_pickle=True)
    old_keys, new_keys = set(old.files) - {"__meta__"}, set(new.files) - {"__meta__"}

    moved, missing, added = [], [], []
    for k in sorted(old_keys & new_keys):
        if not bit_equal(old[k], new[k]):
            a, b = old[k].astype(np.float64), new[k].astype(np.float64)
            fin = np.isfinite(a) & np.isfinite(b)
            worst = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else float("nan")
            moved.append((k, worst))
    for k in sorted(old_keys - new_keys):
        if k not in RETIRED_ARRAYS:
            missing.append(k)
    added = sorted(new_keys - old_keys)

    om = json.loads(str(old["__meta__"]))
    nm = json.loads(str(new["__meta__"]))
    meta_note = (f"{om.get('match_mode')!r} -> readout={nm.get('readout')!r}, "
                 f"planted_map={nm.get('planted_map_sha256', '')[:12]}")
    # Provenance that must NOT drift: same world, same draws, same corruption.
    pins = {}
    for k in ("corrupted_edges_sha256", "n_corrupted_edges", "realized_severity_median",
              "scoring_sample_seed", "probe_fit_sample_seed", "acts_mode", "dials"):
        if om.get(k) != nm.get(k):
            pins[k] = (om.get(k), nm.get(k))
    return {"moved": moved, "missing": missing, "added": added,
            "meta_note": meta_note, "pins_drifted": pins,
            "n_compared": len(old_keys & new_keys)}


def compare_census(old_dir: Path, new_dir: Path) -> dict:
    """Census counts, compared separately - `report.py` lifts them into `dose_response.csv`.

    They are NOT in `scores.npz`, so the array comparison above cannot see them, and they ride
    on `scoring/trained/absorption.py`, which is outside this refactor and can move
    independently. The classifier hash is compared first so a moved count is attributable.
    """
    fo, fn = old_dir / "census.json", new_dir / "census.json"
    if not (fo.exists() and fn.exists()):
        return {}
    o, n = json.loads(fo.read_text()), json.loads(fn.read_text())
    drift = {}
    if o.get("absorption_classifier_sha256") != n.get("absorption_classifier_sha256"):
        drift["absorption_classifier_sha256"] = "CHANGED (classify_dictionary itself moved)"
    for k in ("counts", "n_census_absorbed", "n_census_absorbed_among_planted",
              "n_census_absorbed_outside_planted", "corrupted_edges_sha256"):
        if o.get(k) != n.get(k):
            drift[k] = (o.get(k), n.get(k))
    return drift


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True, help="round-1 tag dir (the saved artifacts)")
    ap.add_argument("--new", required=True, help="scratch tag dir written by the new code")
    ap.add_argument("--expect", type=int, required=True,
                    help="how many artifacts MUST be compared; a silently partial gate is "
                         "not a passed gate")
    args = ap.parse_args()

    old_root, new_root = Path(args.old), Path(args.new)
    pairs = []
    for npz in sorted(new_root.rglob("scores.npz")):
        rel = npz.parent.relative_to(new_root)
        counterpart = old_root / rel
        if (counterpart / "scores.npz").exists():
            pairs.append((counterpart, npz.parent))
    if len(pairs) != args.expect:
        raise SystemExit(
            f"expected {args.expect} comparable artifacts, found {len(pairs)} between "
            f"{old_root} and {new_root}. A gate that silently compares fewer points than "
            f"intended is not a passed gate.")

    failed = 0
    for old_dir, new_dir in pairs:
        r = compare_artifact(old_dir, new_dir)
        bad = r["moved"] or r["missing"] or r["pins_drifted"]
        status = "MOVED" if bad else "identical"
        print(f"[{status}] {new_dir.relative_to(new_root)}  "
              f"({r['n_compared']} arrays compared; {r['meta_note']})")
        if r["moved"]:
            print("   arrays that MOVED: " +
                  ", ".join(f"{k} (max|d|={d:.3e})" for k, d in r["moved"]))
        if r["missing"]:
            print(f"   arrays missing beyond the retired matcher ones: {r['missing']}")
        if r["pins_drifted"]:
            print(f"   provenance drifted: {r['pins_drifted']}")
        if r["added"]:
            print(f"   (new arrays, informational: {r['added']})")
        cen = compare_census(old_dir, new_dir)
        if cen:
            print(f"   CENSUS moved: {cen}")
            bad = True
        failed += bool(bad)

    print(f"\n{len(pairs) - failed}/{len(pairs)} artifacts bit-identical")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
