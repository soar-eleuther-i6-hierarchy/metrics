"""
Pure metrics, one table and one figure, no pathology mapping.

    python3 -m reporting.make_pure_metrics                 # write table + figure
    python3 -m reporting.make_pure_metrics --toy DIR       # a toy report made by run_metrics
    python3 -m reporting.make_pure_metrics --list          # show inputs, write nothing

Outputs, under outputs/paper_figuers/:
    pure_metrics.md     the table as Markdown
    pure_metrics.csv    the same numbers, one row per (source, metric)
    pure_metrics.tex    the same table as LaTeX
    pure_metrics.png    small multiples, one panel per rate metric
    pure_metrics_pcfg_by_density.md   the PCFG column broken out by formatting density

Every number is read from a metrics_report.json (stage 02) or a second_pass.json
(stage 03), block pair 0->1. Nothing is typed in. A source whose report is missing
is left out and named on stderr, never filled with a placeholder.

The four sources are in a fixed order with a fixed colour each, from the same
Okabe-Ito slots make_report_figures uses, so this figure and the others agree.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

import config as C

PAPER_DIR = C.OUT_DIR / "paper_figuers"
PAIR = "0->1"

# Fixed order, fixed colour. Same slots as make_report_figures.CAT.
SOURCES = [
    ("matryoshka_gemma", "Matryoshka, gemma-2-2b L12", "#0072B2"),
    ("tsae_gemma", "Temporal SAE, gemma-2-2b L12", "#E69F00"),
    ("matryoshka_toy", "Matryoshka, Bussmann toy", "#009E73"),
    ("tsae_toy", "Temporal SAE, Bussmann toy (3 seeds)", "#CC79A7"),
    ("matryoshka_pcfg", "Matryoshka, PCFG L2 (12 runs)", "#D55E00"),
]

# (key, metric, quantity, kind). `metric` is the paper's Section 3 name, numbered as in its
# Table 1; `quantity` is which number of that metric is reported. kind: "count", "value",
# "rate_small" or "rate". Rates are drawn; the rest are table-only. Order follows Section 3.
METRICS = [
    ("n_edges", "1 Containment", "candidate edges", "count"),
    ("density", "1 Containment", "edge density (edges / P x C)", "rate_small"),
    ("chance", "2 Co-firing above chance", "share at chance level (PMI < 0.5)", "rate"),
    ("mean_pmi", "2 Co-firing above chance", "mean PMI over edges", "value"),
    ("freq_driven", "3 Survival on rare tokens", "frequency-driven share (survival < 0.5)", "rate"),
    ("survival", "3 Survival on rare tokens", "mean survival on rare tokens", "rate"),
    ("recon_pass", "4 Reconstruction contribution", "pass share", "rate"),
    ("probe_pass", "5 Direction alignment", "pass share", "rate"),
    ("superparents", "6 Dense parents, multi-parenting", "dense parents", "count"),
    ("poly", "6 Dense parents, multi-parenting", "multi-parenting share", "rate"),
    ("gini", "6 Dense parents, multi-parenting", "out-degree Gini", "rate"),
    ("sibling", "7 Sibling overlap", "mean pairwise Jaccard", "rate"),
    ("joint_cov", "9 Parent coverage by children", "mean support coverage", "rate"),
]


# Figure keys. The figure carries no explanatory text: each panel is "(metric number) short
# quantity" and each row a short source name; the caption in the paper spells both out.
PANEL = {
    "chance": "(2) at chance level",
    "freq_driven": "(3) frequency-driven",
    "survival": "(3) survival on rare tokens",
    "recon_pass": "(4) reconstruction pass",
    "probe_pass": "(5) direction alignment pass",
    "poly": "(6) multi-parenting",
    "gini": "(6) out-degree Gini",
    "sibling": "(7) sibling overlap",
    "joint_cov": "(9) parent coverage",
}
SOURCE_SHORT = {
    "matryoshka_gemma": "Matryoshka, gemma",
    "tsae_gemma": "T-SAE, gemma",
    "matryoshka_toy": "Matryoshka, toy",
    "tsae_toy": "T-SAE, toy",
    "matryoshka_pcfg": "Matryoshka, PCFG",
}


def _title(key: str, metric: str, quantity: str) -> str:
    """Panel title: the short key when one is defined, else number and quantity."""
    return PANEL.get(key, f"({metric.split(' ', 1)[0]}) {quantity}")


def _metric_cells(rows):
    """The metric name printed once per group, blank on the group's later rows."""
    last = None
    for key, metric, quantity, kind in rows:
        yield key, (metric if metric != last else ""), quantity, kind
        last = metric


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pair(report, name=PAIR):
    return next((p for p in report["pairs"] if p["pair"] == name), None)


def _shape(report):
    br = report.get("block_ranges") or []
    if len(br) < 2:
        return None
    (a0, a1), (b0, b1) = br[0], br[1]
    return (a1 - a0), (b1 - b0)


def read_one(report_path: Path, second_path: Path | None, probe_override=None,
             shape_override=None) -> dict | None:
    """Every metric for one run's block pair 0->1, or None when the report is missing."""
    rep = _json(report_path)
    if rep is None:
        print(f"[pure_metrics] missing {report_path}", file=sys.stderr)
        return None
    p = _pair(rep)
    if p is None:
        print(f"[pure_metrics] no pair {PAIR} in {report_path}", file=sys.stderr)
        return None
    shape = shape_override or _shape(rep)
    n = p["n_candidate_edges"]
    out = {
        "n_edges": n,
        "density": (n / (shape[0] * shape[1])) if shape else float("nan"),
        "recon_pass": p["reconstruction"]["frac_pass"],
        "freq_driven": p["freq_control"]["frac_freq_driven"],
        "survival": p["freq_control"]["mean_survival"],
        "chance": p["independence_null"]["frac_chance_level"],
        "mean_pmi": p["independence_null"]["mean_edge_pmi"],
        "superparents": p["n_superparents"],
        "poly": p["degree"]["poly_frac"],
        "gini": p["degree"]["outdeg_gini"],
        "sibling": p["sibling_redundancy"]["mean_redundancy"],
        "joint_cov": p["joint_child_cov_mean"],
        "probe_pass": float("nan"),
        "tokens": rep.get("total_tokens"),
        "shape": shape,
    }
    if probe_override is not None:
        out["probe_pass"] = probe_override
    elif second_path is not None:
        sp = _json(second_path)
        s = ((sp or {}).get(PAIR) or {}).get("sres") or {}
        if s.get("n_edges_scored"):
            out["probe_pass"] = s["frac_pass"]
    return out


def _agg(rs: list[dict]) -> dict:
    """Mean, sd and the per-run values of every metric over several runs. A source with
    `n_runs` set is drawn as mean ± sd with the runs as small dots."""
    m = {}
    for k, _, _, _ in METRICS:
        vals = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        m[k] = statistics.mean(vals) if vals else float("nan")
        m[k + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        m[k + "_runs"] = vals
    m["n_runs"] = len(rs)
    m["tokens"] = rs[0].get("tokens") if rs else None
    m["shape"] = rs[0].get("shape") if rs else None
    return m


def read_pcfg() -> tuple[dict | None, dict]:
    """Mean and sd over the formatting-sweep runs, plus a per-density breakdown."""
    runs = {}
    for d in sorted(glob.glob(str(C.OUT_DIR / "pcfg-matryoshka" / "fmt_*"))):
        r = read_one(Path(d) / "metrics_report.json", Path(d) / "second_pass.json")
        if r is not None:
            runs[os.path.basename(d)] = r
    if not runs:
        return None, {}
    by_density = {}
    for name, r in runs.items():
        dens = name.split("_")[1]                          # fmt_2400_s0 -> 2400
        by_density.setdefault(f"0.{dens}", []).append(r)
    return _agg(list(runs.values())), {k: _agg(v) for k, v in sorted(by_density.items())}


def _toy_shape(toy_dir: Path):
    """The toy's blocks are index lists (which latents recovered parents, which recovered
    children), not ranges, so the report's block_ranges are gemma's and the density
    denominator has to come from the stats file the report was made from."""
    stats = toy_dir.parent / "toy_stats.pt"
    if stats.exists():
        import torch
        bi = (torch.load(stats, weights_only=False).get("config") or {}).get("block_indices")
        if bi and len(bi) >= 2:
            return (len(bi[0]), len(bi[1]))
    return None


def collect(toy_dir: Path | None, tsae_toy_dirs: list[Path] | None = None) -> dict[str, dict]:
    g = C.OUT_DIR / "gemma-2-2b" / "layer_12"
    t = C.OUT_DIR / "gemma-2-2b-tsae" / "layer_12"
    data = {}
    r = read_one(g / "metrics_report.json", g / "second_pass.json")
    if r:
        data["matryoshka_gemma"] = r
    r = read_one(t / "metrics_report.json", t / "second_pass.json")
    if r:
        data["tsae_gemma"] = r
    if toy_dir is not None:
        # The toy's probe pass lives in the Tier-2 calibration file, not in a second pass.
        cal = _json(C.OUT_DIR / "trained_toy_calibration.json")
        pt = (cal or {}).get("per_token") or {}
        probe = (pt["n_pass"] / pt["n_testable"]) if pt.get("n_testable") else None
        r = read_one(toy_dir / "metrics_report.json", None, probe_override=probe,
                     shape_override=_toy_shape(toy_dir))
        if r:
            data["matryoshka_toy"] = r
    if tsae_toy_dirs:
        # The Temporal SAE toy checkpoints (one per seed) go through the same adapter and
        # run_metrics as the Matryoshka toy. They have no probe pass: the toy adapter
        # caches no residuals, and the Tier-2 calibration file is the Matryoshka toy's.
        rs = []
        for d in tsae_toy_dirs:
            r = read_one(d / "metrics_report.json", None, shape_override=_toy_shape(d))
            if r:
                rs.append(r)
        if rs:
            data["tsae_toy"] = _agg(rs)
    pc, by_density = read_pcfg()
    if pc:
        data["matryoshka_pcfg"] = pc
        data["_pcfg_by_density"] = by_density
    return data


# --- table -------------------------------------------------------------------
def _fmt(v, kind, sd=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if kind == "count":
        s = f"{v:,.0f}" if sd is None else f"{v:,.0f}"
    elif kind == "rate_small":
        s = f"{v:.1e}"
    else:
        s = f"{v:.3f}"
    if sd is not None and kind != "rate_small":
        s += f" ± {sd:.3f}" if kind != "count" else f" ± {sd:,.0f}"
    return s


def write_tables(data: dict, out: Path):
    cols = [(k, lab) for k, lab, _ in SOURCES if k in data]
    lines = ["| Metric | Quantity | " + " | ".join(lab for _, lab in cols) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in cols) + " |"]
    rows_csv = []
    for key, metric, quantity, kind in _metric_cells(METRICS):
        cells = []
        for src, _ in cols:
            d = data[src]
            sd = d.get(key + "_sd") if d.get("n_runs") else None
            cells.append(_fmt(d.get(key), kind, sd))
            rows_csv.append({"source": src, "metric": key, "value": d.get(key),
                             "sd": sd if sd is not None else ""})
        lines.append(f"| {metric} | {quantity} | " + " | ".join(cells) + " |")
    tok = ", ".join(f"{lab}: {data[src].get('tokens'):,} tokens"
                    for src, lab in cols if data[src].get("tokens"))
    shapes = ", ".join(f"{lab}: {data[src]['shape'][0]} x {data[src]['shape'][1]}"
                       for src, lab in cols if data[src].get("shape"))
    md = ("# Pure metrics, block pair 0->1\n\n"
          "Every number read from the run's metrics_report.json (stage 02) and second_pass.json "
          "(stage 03). No pathology mapping. PCFG cells are mean ± sd over the formatting-sweep "
          "runs; Temporal SAE toy cells are mean ± sd over its seeds. The Matryoshka toy's probe "
          "pass comes from trained_toy_calibration.json; the Temporal SAE toy has no probe pass "
          "(the toy adapter caches no residuals), shown as n/a.\n\n"
          + "\n".join(lines) + "\n\n"
          f"Tokens: {tok}.\n\nBlock shapes (parents x children): {shapes}.\n")
    (out / "pure_metrics.md").write_text(md)
    with (out / "pure_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "metric", "value", "sd"])
        w.writeheader()
        w.writerows(rows_csv)
    # LaTeX
    tex = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule",
           "Metric & Quantity & " + " & ".join(lab.replace("&", "\\&") for _, lab in cols) + " \\\\",
           "\\midrule"]
    for key, metric, quantity, kind in _metric_cells(METRICS):
        cells = []
        for src, _ in cols:
            d = data[src]
            sd = d.get(key + "_sd") if d.get("n_runs") else None
            cells.append(_fmt(d.get(key), kind, sd).replace("±", "$\\pm$"))
        tex.append(f"{metric} & {quantity.replace('<', '$<$')} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "pure_metrics.tex").write_text("\n".join(tex) + "\n")

    if "_pcfg_by_density" in data:
        bd = data["_pcfg_by_density"]
        dens = list(bd.keys())
        lines = ["| Metric | Quantity | " + " | ".join(f"density {d} (n={bd[d]['n_runs']})" for d in dens) + " |",
                 "| --- | --- | " + " | ".join("---:" for _ in dens) + " |"]
        for key, metric, quantity, kind in _metric_cells(METRICS):
            lines.append(f"| {metric} | {quantity} | " + " | ".join(
                _fmt(bd[d][key], kind, bd[d][key + "_sd"]) for d in dens) + " |")
        (out / "pure_metrics_pcfg_by_density.md").write_text(
            "# PCFG formatting sweep, block pair 0->1, mean ± sd over seeds\n\n" + "\n".join(lines) + "\n")


# --- figure ------------------------------------------------------------------
def write_figure(data: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#2B2B33", "#5A6B7B", "#E6E8EB"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#D8DBE0", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    srcs = [(k, SOURCE_SHORT.get(k, lab), col) for k, lab, col in SOURCES if k in data]
    rates = [(k, _title(k, m, q)) for k, m, q, kind in METRICS if kind == "rate"]
    ncol = 4
    nrow = math.ceil(len(rates) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.1 * nrow + 0.3), sharey=True)
    axes = axes.ravel()
    ys = list(range(len(srcs)))[::-1]
    for ax, (key, label) in zip(axes, rates):
        xmax = 1.0
        for y, (src, lab, col) in zip(ys, srcs):
            d = data[src]
            v = d.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                ax.text(0.01, y, "n/a", va="center", ha="left", color=MUTED, fontsize=8)
                continue
            xmax = max(xmax, v)
            multi = bool(d.get("n_runs"))
            if multi:
                runs = d.get(key + "_runs") or []
                ax.scatter(runs, [y] * len(runs), s=12, color=col, alpha=0.35, zorder=2,
                           linewidths=0)
                ax.errorbar([v], [y], xerr=[[d[key + "_sd"]]], fmt="none", ecolor=col,
                            elinewidth=1.2, capsize=2, zorder=3)
            ax.scatter([v], [y], s=42, color=col, edgecolor="white", linewidth=1.2, zorder=4)
            # no value labels: the table carries the numbers, the figure the shape
        ax.set_xlim(0, max(1.0, xmax * 1.05))
        ax.set_ylim(-0.7, len(srcs) - 0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab, _ in srcs])
        ax.set_title(label, fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes[len(rates):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        for t, (_, _, col) in zip(ax.get_yticklabels(), srcs):
            t.set_color(col)
            t.set_fontweight("bold")
    fig.tight_layout()
    path = out / "pure_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_grid(data: dict, out: Path):
    """One grid: a row per setting, a column per rate metric, one blue ramp light-to-dark
    with the value, and the number printed in the cell (± sd where there are several runs).
    Same numbers as the dot panels; this form lets a reader compare a column at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    INK, MUTED, GRID = "#2B2B33", "#5A6B7B", "#E6E8EB"
    plt.rcParams.update({"font.size": 9, "text.color": INK, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    srcs = [(k, SOURCE_SHORT.get(k, lab)) for k, lab, _ in SOURCES if k in data]
    rates = [(k, _title(k, m, q)) for k, m, q, kind in METRICS if kind == "rate"]
    ramp = LinearSegmentedColormap.from_list("blue", ["#F4F7FC", "#1F4E8C"])

    fig, ax = plt.subplots(figsize=(1.05 * len(rates) + 2.2, 0.55 * len(srcs) + 1.4))
    for i, (src, _) in enumerate(srcs):
        d = data[src]
        for j, (key, _) in enumerate(rates):
            v = d.get(key)
            missing = v is None or (isinstance(v, float) and math.isnan(v))
            face = "#FFFFFF" if missing else ramp(min(max(v, 0.0), 1.0))
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=face,
                                       edgecolor=GRID if missing else "none", lw=0.8))
            if missing:
                txt, col = "n/a", MUTED
            else:
                sd = d.get(key + "_sd") if d.get("n_runs") else None
                txt = f"{v:.2f}" if sd is None else f"{v:.2f}\n±{sd:.2f}"
                col = "white" if v >= 0.55 else INK
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8.5, color=col)
    ax.set_xlim(0, len(rates))
    ax.set_ylim(len(srcs), 0)
    ax.set_xticks([j + 0.5 for j in range(len(rates))])
    def _wrap(t: str) -> str:
        num, rest = t.split(") ", 1)
        words = rest.split()
        if len(words) <= 1:
            return f"{num})\n{rest}"
        cut = max(1, len(words) // 2)                      # balanced two-line wrap
        return f"{num})\n{' '.join(words[:cut])}\n{' '.join(words[cut:])}"
    ax.set_xticklabels([_wrap(t) for _, t in rates], fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([i + 0.5 for i in range(len(srcs))])
    ax.set_yticklabels([lab for _, lab in srcs], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # one ramp legend, 0 to 1
    sm = plt.cm.ScalarMappable(cmap=ramp, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_ticks([0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, colors=MUTED)
    fig.tight_layout()
    path = out / "pure_metrics_grid.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", type=Path, default=None,
                    help="directory holding a metrics_report.json for the trained toy (from run_metrics)")
    ap.add_argument("--tsae-toy", type=Path, nargs="+", default=None,
                    help="one report directory per Temporal SAE toy seed (from run_metrics)")
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    data = collect(a.toy, a.tsae_toy)
    if a.list:
        for k, lab, _ in SOURCES:
            print(f"{'ok ' if k in data else '-- '} {lab}")
        return
    a.out.mkdir(parents=True, exist_ok=True)
    write_tables(data, a.out)
    png = write_figure(data, a.out)
    grid = write_grid(data, a.out)
    print((a.out / "pure_metrics.md").read_text())
    print(f"wrote {png}\nwrote {grid}")


if __name__ == "__main__":
    main()
