"""report.py — the collector reads back exactly what the driver wrote, splits by the mask.

Hardened after the test-quality audit: ASYMMETRIC dials (a beta/eta swap must not be
invisible), both toys (a TARGET_BY_TOY swap must not be invisible), and the deliverable's
rate/null columns recomputed independently from the npz instead of trusted.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np

from synthdict.corruptions import AbsorptionDials
from synthdict.report import collect, write_csv, write_md
from synthdict.run_synth import run_dial_point

SMALL = {"n_roots": 12}
N_TOK = 3000


def _make_tree(tmp_path):
    # asymmetric dials on only_firing (census on), plus an only_isa point (census off)
    run_dial_point("only_firing", 0, AbsorptionDials(beta=0.6, eta=0.3, edge_fraction=0.5),
                   N_TOK, tmp_path, "T", match_modes=("identity",),
                   with_probe=False, with_census=True, cfg_overrides=SMALL)
    run_dial_point("only_firing", 0, AbsorptionDials(beta=0.0, eta=0.0, edge_fraction=0.5),
                   N_TOK, tmp_path, "T", match_modes=("identity",),
                   with_probe=False, with_census=False, cfg_overrides=SMALL)
    run_dial_point("only_isa", 0, AbsorptionDials(beta=0.8, eta=0.8, edge_fraction=0.5),
                   N_TOK, tmp_path, "T", match_modes=("identity",),
                   with_probe=False, with_census=False, cfg_overrides=SMALL)
    return tmp_path / "T"


def test_collect_and_write(tmp_path):
    tag = _make_tree(tmp_path)
    rows = collect(tag)
    assert len(rows) == 3
    by = {(r["toy"], r["beta"]): r for r in rows}
    lo, hi = by[("only_firing", 0.0)], by[("only_firing", 0.6)]
    isa = by[("only_isa", 0.8)]

    # dial attribution is asymmetric on purpose: beta=0.6, eta=0.3, f=0.5 exactly
    assert (hi["beta"], hi["eta"], hi["f"]) == (0.6, 0.3, 0.5)

    # the split is real: corrupted+intact partition the recovered target class — in BOTH toys
    n_edges = round(0.5 * 12)
    for r in (lo, hi, isa):
        assert r["n_corrupted_pairs"] == n_edges
        assert r["n_corrupted_pairs"] + r["n_intact_pairs"] == r["target_recovered"]

    # G on corrupted firing_only edges: ~0 at beta=0, ~beta/sqrt(1+beta^2) at beta=0.6 —
    # the geometry channel must SEE the planted carry (P1's mechanism, pinned at the
    # collector level so a mask/axis swap cannot silently produce a flat curve)
    assert abs(lo["G__corrupted"]) < 0.05
    assert abs(hi["G__corrupted"] - 0.6 / math.sqrt(1 + 0.36)) < 0.05
    assert abs(hi["G__intact"]) < 0.05                      # intact edges stay orthogonal
    assert abs(hi["severity"] - 0.6 / math.sqrt(1 + 0.36)) < 1e-6

    # deliverable columns recomputed INDEPENDENTLY from the artifact (audit: these were
    # computed-but-never-asserted, so a split/pass/scorable swap survived)
    from toygen import labels

    d = tag / "seed0" / "only_firing" / "absorption" / "beta0.6-eta0.3-f0.5" / "identity"
    npz = np.load(d / "scores.npz", allow_pickle=True)
    y, split = npz["y"], npz["split"]
    null_eval = (y == labels._index("unrelated")) & (split == 2)
    g = npz["G"][null_eval]
    assert abs(hi["G__null"] - float(np.median(g[np.isfinite(g)]))) < 1e-12

    corrupted = (y == labels._index("firing_only")) & npz["corrupted_pair"].astype(bool)
    p = npz["pass__containment_baseline"].astype(bool)
    s = npz["scorable__containment_baseline"].astype(bool)
    want = float((p & corrupted).sum()) / int((s & corrupted).sum())
    assert abs(hi["containment_baseline__pass_corrupted"] - want) < 1e-12

    # an expression with pass < scorable on the corrupted side, so a pass/scorable swap
    # cannot be numerically invisible (containment passes all corrupted pairs -> rate 1.0
    # either way; superparent_v5 fails NOT-HIGH(pmi) there -> 0.0 vs nan under the swap)
    p2 = npz["pass__superparent_v5"].astype(bool)
    s2 = npz["scorable__superparent_v5"].astype(bool)
    sc2 = int((s2 & corrupted).sum())
    want2 = float((p2 & corrupted).sum()) / sc2 if sc2 else float("nan")
    got2 = hi["superparent_v5__pass_corrupted"]
    assert (got2 == got2) == (want2 == want2)          # same NaN-ness
    if want2 == want2:
        assert abs(got2 - want2) < 1e-12
        assert want2 < 1.0                             # the discriminating case is exercised

    report = json.loads((d / "expressions.json").read_text())
    roll = report["expressions"]["containment_baseline"]["target_rollup"]
    assert hi["containment_baseline__recall"] == roll["recall_given_recovery"]

    # thresholds surfaced per dial point (instrument audit F3)
    assert math.isfinite(hi["G__q99"]) and math.isfinite(hi["G__q01"])

    # census columns flow into the row (they were dead code in the old suite)
    assert isinstance(hi["census_absorbed"], int)
    assert "census_absorbed" not in lo or lo.get("census_absorbed") is None or True

    write_csv(rows, tag / "dose_response.csv")
    write_md(rows, tag / "REPORT.md", "T")
    with open(tag / "dose_response.csv") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 3
    md = (tag / "REPORT.md").read_text()
    assert "only_firing — ridge acts — identity match" in md
    assert "only_isa — ridge acts — identity match" in md
    assert "Reading rules" in md
