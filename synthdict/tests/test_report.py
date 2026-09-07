"""report.py — the collector reads back exactly what the driver wrote, splits by the mask."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from synthdict.corruptions import AbsorptionDials
from synthdict.report import collect, write_csv, write_md
from synthdict.run_synth import run_dial_point

SMALL = {"n_roots": 12}
N_TOK = 3000


def test_collect_and_write(tmp_path):
    for beta in (0.0, 0.8):
        run_dial_point("only_firing", 0, AbsorptionDials(beta=beta, eta=beta, edge_fraction=0.5),
                       N_TOK, tmp_path, "T", match_modes=("identity",),
                       with_probe=False, with_census=False, cfg_overrides=SMALL)
    rows = collect(tmp_path / "T")
    assert len(rows) == 2
    lo, hi = sorted(rows, key=lambda r: r["beta"])
    assert lo["beta"] == 0.0 and hi["beta"] == 0.8

    # the split is real: corrupted+intact partition the recovered target class
    n_edges = round(0.5 * 12)
    assert lo["n_corrupted_pairs"] == n_edges
    assert lo["n_corrupted_pairs"] + lo["n_intact_pairs"] == lo["target_recovered"]

    # G on corrupted firing_only edges: ~0 at beta=0, ~beta/sqrt(1+beta^2) at beta=0.8 —
    # the geometry channel must SEE the planted carry (this is P1's mechanism, pinned here
    # at the collector level so a mask/axis swap cannot silently produce a flat curve)
    assert abs(lo["G__corrupted"]) < 0.05
    assert abs(hi["G__corrupted"] - 0.8 / math.sqrt(1 + 0.64)) < 0.05
    assert abs(hi["G__intact"]) < 0.05                      # intact edges stay orthogonal
    assert abs(hi["severity"] - 0.8 / math.sqrt(1 + 0.64)) < 1e-6

    write_csv(rows, tmp_path / "T" / "dose_response.csv")
    write_md(rows, tmp_path / "T" / "REPORT.md", "T")
    with open(tmp_path / "T" / "dose_response.csv") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 2 and got[0]["toy"] == "only_firing"
    md = (tmp_path / "T" / "REPORT.md").read_text()
    assert "only_firing — identity match" in md
    assert "G__corrupted" in md
