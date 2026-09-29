"""
Survival funnel: of the edges coverage proposes, how many are confirmed by every gate?

    python3 -m reporting.make_survival_funnel \
        --toy outputs/toy-temporal/pipeline/matryoshka_toy \
        --tsae-toy outputs/toy-temporal/pipeline/tsaeip_s0 ... \
        [--tsae-stats PATH]     # exp0_stats.pt for the gemma T-SAE run, if available

Outputs, under outputs/paper_figuers/:
    survival_funnel.png    one line per setting: share of candidate edges still confirmed
                           after each gate, in the order of the hierarchy rule
    survival_funnel.json   the counts behind every point, plus how many edges were
                           unmeasurable at each gate
    survival_funnel.md     the same counts as a table

Stages, cumulative and strict: an edge that a gate cannot measure (survival with too few
rare-token firings, a child with too few probe positives) is not confirmed, so it leaves
the funnel at that gate; the JSON records how many did.

    1  candidate edges       reverse coverage >= tau, both endpoints and the pair supported
    2  above chance          PMI >= 0.5, the table's chance-level cut
    3  survives on rare      survival >= 0.5
    4  reconstruction        parent and child gains >= 0.01 on the child's tokens
    5  probe                 both decoders in the child probe's top k

Every mask is recomputed from the run's cached statistics with the same functions
run_metrics.py uses; the probe verdicts come from second_pass.json. Where the gemma T-SAE
statistics file is absent (it lives on the compute node), stages 1, 2 and 5 are exact from
the reports and stages 3 and 4 are drawn as the interval the reports bound, never as a value.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

import config as C
from metrics import (coverage_legs, edge_reconstruction_condition, frequency_controlled_coverage,
                     independence_scores, keep_edges)
from reporting.make_pure_metrics import PAPER_DIR, SOURCE_SHORT, SOURCES, _json
from run_metrics import block_selector

STAGES = ["1 candidates", "2 PMI ≥ 0.5", "3 survival ≥ 0.5", "4 reconstruction", "5 direction alignment"]
CHANCE_PMI = 0.5     # the table's "at chance level" cut


def _local_index(sel, n_total: int) -> torch.Tensor:
    """global id -> local column for one block, -1 outside it."""
    idx = torch.full((n_total,), -1, dtype=torch.long)
    if isinstance(sel, slice):
        lo, hi = sel.start, sel.stop
        idx[lo:hi] = torch.arange(hi - lo)
    else:
        idx[sel] = torch.arange(len(sel))
    return idx


def funnel_from_stats(stats, probe_edges, *, probe_all_pass=False, has_probe=True) -> dict:
    """Cumulative counts for block pair 0->1 from one cached-statistics file."""
    key = "0->1"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    sel = block_selector(stats)
    fire_p, fire_c = fire[sel(0)], fire[sel(1)]
    cofire = stats["cofire"][key].double()

    R, _ = coverage_legs(cofire, fire_p, fire_c)
    m1 = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT, cofire=cofire, min_joint=C.MIN_JOINT)
    pmi = independence_scores(cofire, fire_p, fire_c, total, C.MIN_JOINT)["pmi"]
    m2 = m1 & (pmi >= CHANCE_PMI)                       # NaN compares False: unmeasurable leaves
    surv = frequency_controlled_coverage(stats["cofire_by_bucket"][key].double(),
                                         stats["fire_c_by_bucket"][1].double(), m1,
                                         min_fire_low=C.FREQ_MIN_FIRE_LOW)["survival"]
    unm3 = int((m2 & torch.isnan(surv)).sum())
    m3 = m2 & (surv >= C.FREQ_SURVIVAL_MIN)
    recon = edge_reconstruction_condition(stats["err_sum_c"][1].double(), stats["g_parent_sum"][key].double(),
                                          stats["g_child_sum"][1].double(), C.RECON_REL_GAIN_MIN)
    m4 = m3 & recon["passes"]
    out = {"counts": [int(m.sum()) for m in (m1, m2, m3, m4)],
           "unmeasurable": {STAGES[2]: unm3}}
    if not has_probe:
        out["counts"].append(None)
        return out
    if probe_all_pass:
        m5, unm5 = m4, 0
    else:
        n_total = fire.numel()
        lp, lc = _local_index(sel(0), n_total), _local_index(sel(1), n_total)
        passed = torch.zeros_like(m4)
        scored = torch.zeros_like(m4)
        for e in probe_edges:
            i, j = int(lp[e["parent"]]), int(lc[e["child"]])
            if i >= 0 and j >= 0:
                scored[i, j] = True
                if e["pass"]:
                    passed[i, j] = True
        unm5 = int((m4 & ~scored).sum())
        m5 = m4 & passed
    out["counts"].append(int(m5.sum()))
    out["unmeasurable"][STAGES[4]] = unm5
    return out


def funnel_from_reports(report: dict, second: dict) -> dict:
    """Exact stages 1, 2, 5 and an interval for 3 and 4, from the report files alone."""
    p = next(q for q in report["pairs"] if q["pair"] == "0->1")
    s = second["0->1"]["sres"]
    n1 = p["n_candidate_edges"]
    n2 = n1 - p["independence_null"]["n_chance_level"]
    fq, rc = p["freq_control"], p["reconstruction"]
    # strict: untestable and frequency-driven edges leave at stage 3; where they sit among
    # the stage-2 survivors is unknown, so stage 3 is an interval
    n3_hi = min(n2, n1 - fq["n_freq_driven"] - (n1 - fq["n_testable"]))
    n3_lo = max(0, n2 - fq["n_freq_driven"] - (n1 - fq["n_testable"]))
    n4_hi, n4_lo = min(n3_hi, rc["n_pass"]), max(0, n3_lo - (n1 - rc["n_pass"]))
    # the probe passes are counted over the shortlist; how many of them sit among the
    # stage-4 survivors is unknown, so the last stage is an interval as well
    n5_hi, n5_lo = min(s["n_pass"], n4_hi), max(0, s["n_pass"] - (n1 - n4_lo))
    return {"counts": [n1, n2, None, None, None],
            "interval": {STAGES[2]: [n3_lo, n3_hi], STAGES[3]: [n4_lo, n4_hi],
                         STAGES[4]: [n5_lo, n5_hi]},
            "unmeasurable": {STAGES[2]: n1 - fq["n_testable"],
                             STAGES[4]: s["n_shortlist_edges"] - s["n_edges_scored"]},
            "note": "stages 3 and 4 bounded from the report; statistics file not on disk"}


def _shares(f: dict) -> list:
    n1 = f["counts"][0]
    return [None if c is None else c / n1 for c in f["counts"]]


def collect(toy_dir, tsae_toy_dirs, tsae_stats) -> dict:
    out = {}
    g = C.OUT_DIR / "gemma-2-2b" / "layer_12"
    st = torch.load(g / "exp0_stats.pt", weights_only=False)
    out["matryoshka_gemma"] = funnel_from_stats(st, _json(g / "second_pass.json")["0->1"]["sres"]["edges"])

    t = C.OUT_DIR / "gemma-2-2b-tsae" / "layer_12"
    if tsae_stats and Path(tsae_stats).exists():
        st = torch.load(tsae_stats, weights_only=False)
        out["tsae_gemma"] = funnel_from_stats(st, _json(t / "second_pass.json")["0->1"]["sres"]["edges"])
    else:
        out["tsae_gemma"] = funnel_from_reports(_json(t / "metrics_report.json"), _json(t / "second_pass.json"))

    if toy_dir:
        st = torch.load(Path(toy_dir) / "toy_stats.pt", weights_only=False)
        # the toy's probe pass comes from the Tier-2 calibration: 9 of 9 testable edges pass
        cal = (_json(C.OUT_DIR / "trained_toy_calibration.json") or {}).get("per_token") or {}
        all_pass = bool(cal) and cal.get("n_pass") == cal.get("n_testable")
        out["matryoshka_toy"] = funnel_from_stats(st, None, probe_all_pass=all_pass)
    if tsae_toy_dirs:
        runs = [funnel_from_stats(torch.load(Path(d) / "toy_stats.pt", weights_only=False), None, has_probe=False)
                for d in tsae_toy_dirs]
        out["tsae_toy"] = {"runs": runs}
    runs = []
    for d in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("fmt_*")):
        if (d / "exp0_stats.pt").exists() and (d / "second_pass.json").exists():
            st = torch.load(d / "exp0_stats.pt", weights_only=False)
            runs.append(funnel_from_stats(st, _json(d / "second_pass.json")["0->1"]["sres"]["edges"]))
    if runs:
        out["matryoshka_pcfg"] = {"runs": runs}
    return out


def _series(entry: dict):
    """(mean shares, sd shares or None, interval shares or None) for one source."""
    if "runs" in entry:
        cols = list(zip(*[_shares(r) for r in entry["runs"]]))
        mean = [None if any(v is None for v in c) else statistics.mean(c) for c in cols]
        sd = [None if any(v is None for v in c) else (statistics.stdev(c) if len(c) > 1 else 0.0) for c in cols]
        return mean, sd, None
    sh = _shares(entry)
    iv = None
    if "interval" in entry:
        n1 = entry["counts"][0]
        iv = {STAGES.index(k): (lo / n1, hi / n1) for k, (lo, hi) in entry["interval"].items()}
        for i, (lo, hi) in iv.items():           # the dashed path runs through the midpoints
            sh[i] = (lo + hi) / 2
    return sh, None, iv


def write_figure(data: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#2B2B33", "#5A6B7B", "#E6E8EB"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#D8DBE0", "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = list(range(len(STAGES)))
    for key, lab, col in SOURCES:
        if key not in data:
            continue
        mean, sd, iv = _series(data[key])
        pts = [(i, v) for i, v in enumerate(mean) if v is not None]
        # a bounded stage is drawn as its interval, and the line is dashed across it
        if iv:
            xs = [i for i, _ in pts]
            for i, (lo, hi) in iv.items():
                ax.plot([i, i], [lo, hi], color=col, lw=5, alpha=0.3, solid_capstyle="butt", zorder=2)
            exact = [p for p in pts if p[0] not in iv]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", color=col, lw=1.6, zorder=3)
            ax.scatter([p[0] for p in exact], [p[1] for p in exact], s=34, color=col, edgecolor="white",
                       linewidth=1.0, zorder=4)
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if "runs" in data[key]:               # one thin line per run behind the mean
                for r in data[key]["runs"]:
                    rs = _shares(r)
                    rp = [(i, v) for i, v in enumerate(rs) if v is not None]
                    ax.plot([p[0] for p in rp], [p[1] for p in rp], "-", color=col, lw=0.8,
                            alpha=0.25, zorder=2)
            ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=3)
            ax.scatter(xs, ys, s=34, color=col, edgecolor="white", linewidth=1.0, zorder=4)
            if len(pts) < len(STAGES):       # stops early: no probe for this setting
                ax.scatter([xs[-1]], [ys[-1]], s=60, facecolor="white", edgecolor=col, linewidth=1.6, zorder=5)
        # end label: the final share, one number per line; an interval shows its bounds
        xe, ye = pts[-1]
        label = f"{ye:.2f}" if not (iv and xe in iv) else f"{iv[xe][0]:.2f}–{iv[xe][1]:.2f}"
        dy = 9 if len(pts) < len(STAGES) else 0    # a line that stops early labels above the others
        ax.annotate(label, (xe, ye), textcoords="offset points", xytext=(7, dy),
                    va="center", fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n", 1) for s in STAGES], fontsize=8.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.3, len(STAGES) - 0.4)
    ax.set_ylabel("share of candidate edges still confirmed", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=col, lw=1.8, marker="o", ms=5, label=SOURCE_SHORT.get(k, lab))
               for k, lab, col in SOURCES if k in data]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path = out / "survival_funnel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_tables(data: dict, out: Path):
    lines = ["| Setting | " + " | ".join(STAGES) + " | final share |", "| --- | " + " | ".join("---:" for _ in STAGES) + " | ---: |"]
    for key, lab, _ in SOURCES:
        if key not in data:
            continue
        e = data[key]
        if "runs" in e:
            cols = list(zip(*[r["counts"] for r in e["runs"]]))
            cells = ["n/a" if any(v is None for v in c) else f"{statistics.mean(c):.1f} ± {statistics.stdev(c) if len(c) > 1 else 0:.1f}" for c in cols]
        else:
            cells = []
            for i, c in enumerate(e["counts"]):
                if c is not None:
                    cells.append(f"{c:,}")
                else:
                    lo, hi = e["interval"][STAGES[i]]
                    cells.append(f"[{lo:,}, {hi:,}]")
        mean, _, iv = _series(e)
        last = max(i for i, v in enumerate(mean) if v is not None)
        final = f"{iv[last][0]:.3f}–{iv[last][1]:.3f}" if iv and last in iv else f"{mean[last]:.3f}"
        lines.append(f"| {lab} | " + " | ".join(cells) + f" | {final} |")
    md = ("# Survival funnel, block pair 0->1\n\nCumulative counts: edges confirmed by every gate up to that "
          "stage, in the order of the hierarchy rule. Unmeasurable edges leave the funnel at the gate that "
          "cannot measure them (counts in survival_funnel.json). A bracketed cell is an interval the reports "
          "bound when the statistics file is not on disk.\n\n" + "\n".join(lines) + "\n")
    (out / "survival_funnel.md").write_text(md)
    (out / "survival_funnel.json").write_text(json.dumps({"stages": STAGES, "chance_pmi": CHANCE_PMI, "data": data}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", type=Path, default=None, help="pipeline dir holding toy_stats.pt (Matryoshka toy)")
    ap.add_argument("--tsae-toy", type=Path, nargs="+", default=None, help="pipeline dirs, one per T-SAE toy seed")
    ap.add_argument("--tsae-stats", type=Path, default=None, help="exp0_stats.pt of the gemma T-SAE run")
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    a = ap.parse_args()
    data = collect(a.toy, a.tsae_toy, a.tsae_stats)
    a.out.mkdir(parents=True, exist_ok=True)
    write_tables(data, a.out)
    png = write_figure(data, a.out)
    print((a.out / "survival_funnel.md").read_text())
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
