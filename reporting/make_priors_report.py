#!/usr/bin/env python3
"""Tables and figures for the Priors in Time (Temporal Feature Analysis) results.

    python3 -m reporting.make_priors_report          # write both files
    python3 -m reporting.make_priors_report --list   # show the plan, write nothing

Output: outputs/paper_figuers/priors_saes-tables.tex and priors_saes-figures.tex

This method does not produce parent-child feature edges, so none of the tables below report
any. Its hierarchy is a clustering of token positions into events, and its released code
contains no clustering at all. What is reported here is what it does claim: a split of the
representation into a predicted and a novel part.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reporting.temporal_common import (DEFAULT_OUT, PIT, TOY, emit, figure, ms, read, table, toy_pit)

GEN = "make_priors_report"
TODO = "TODO(caption, human-written). {}"


def _g():
    """The frozen gemma measurements. Written by pit_report.py on the node.

    An earlier version parsed these out of a printed log with a regular expression. That
    breaks when someone edits a print statement, and breaks silently when the pattern still
    matches something else.
    """
    return read(PIT / "results.json")


def t_gemma_quality():
    """What the released checkpoint looks like at the depth it was fitted to."""
    d = _g()
    if not d:
        return None
    q = d["quality_at_best"]
    rows = [
        ["dictionary width", f"{d['width']:,}"],
        ["tokens scored", f"{q['tokens']:,}"],
        ["FVE", f"{q['fve']:.4f}"],
        ["novel $L_0$", rf"{q['novel_l0']:.2f} \footnotesize(TopK setting, $k={d['k']}$)"],
        ["dead features", f"{q['dead_features']:,} / {d['width']:,}"],
        ["avg max cosine", f"{q['avg_max_cos']:.4f}"],
        [r"\textbf{predictive share of reconstruction energy}",
         rf"\textbf{{{q['predictive_energy_share']:.3f}}}"],
    ]
    return table(
        "pit-gemma-quality",
        TODO.format("Released Priors in Time checkpoint at \\texttt{gemma-2-2b}, output of "
                    "block 12, on 200 documents of \\texttt{pile-10k}. Measured with the "
                    "authors' own forward pass."),
        ["quantity", "value"], rows, align="lr",
        note="Nine parts in ten of what reconstructs a token comes from its context. No "
             "hierarchy metric is reported: this method produces no feature edges, so the "
             "metric set has no matching input.")


def t_gemma_layer():
    """The depth, settled by reconstruction rather than by reading the config."""
    d = _g()
    if not d:
        return None
    rows = []
    for r in d["layer_sweep"]:
        v = f"{r['fve']:.4f}"
        best = r["hidden_states"] == d["best_depth"]
        rows.append([rf"\texttt{{hidden\_states[{r['hidden_states']}]}}",
                     f"block {r['block_output']}",
                     (r"\textbf{" + v + "}") if best else v])
    return table(
        "pit-gemma-layer",
        TODO.format("Fraction of variance explained at seven depths. An SAE reconstructs the "
                    "layer it was fitted to best. The checkpoint's \\texttt{conf.yaml} records "
                    rf"\texttt{{block\_id: {d['conf_block_id']}}} while the repository README "
                    rf"lists layer {d['readme_layer']}."),
        ["activation", "block output", "FVE"], rows, align="llr",
        note="Negative values mean the reconstruction is worse than predicting the mean. "
             "The peak is the output of block 12, matching the README.")


def t_toy_event():
    """Novel and predictive code at planted parent-state changes."""
    d = toy_pit()
    if not d:
        return None
    rows = []
    for key, name in [("pit_temporal", "temporal, original tree"),
                      ("pit_decorr", "temporal, decorrelated tree"),
                      ("pit_null", "i.i.d. (no events)")]:
        r = d.get(key)
        if not r:
            continue
        rows.append([name, len(r), ms([x["fve"] for x in r], 3), ms([x["l0"] for x in r], 2)])
    return table(
        "pit-toy",
        TODO.format("Priors in Time trained on the temporal toy, three seeds per row. "
                    "The null row uses \\texttt{--persistence 0}, which removes the temporal "
                    "correlation while leaving the per-token distribution unchanged."),
        ["condition", "$n$", "FVE", "$L_0$"], rows, align="lcrr",
        note="$L_0$ is fixed by TopK at $k=2$ and is a setting rather than a measurement. "
             "The event-boundary measurements are in Table~\\ref{tab:pit-boundary}.")


def _boundary():
    return read(TOY / "priors_in_time_boundary.json")


def t_toy_boundary():
    """Read from the frozen measurements, not transcribed. Written by scripts/event_test.py."""
    d = _boundary()
    if not d:
        return None
    names = [("pit_temporal", "temporal, original"),
             ("pit_decorr", "temporal, decorrelated"),
             ("pit_null", r"\textbf{i.i.d. (no events)}")]
    rows = []
    for key, name in names:
        got = d["cells"].get(key)
        if not got:
            continue
        m = lambda k: st.mean(g[k] for g in got)
        at, wi = m("novel_mag_at"), m("novel_mag_within")
        pa, pw = m("pred_cos_at"), m("pred_cos_within")
        bold = key == "pit_null"
        f = lambda v, p=3: (r"\textbf{" + f"{v:.{p}f}" + "}") if bold else f"{v:.{p}f}"
        rows.append([name, f"{at:.3f}", f"{wi:.3f}", f"{at/wi:.2f}", f(pa, 4), f"{pw:.4f}"])
    return table(
        "pit-boundary",
        TODO.format("Novel-code magnitude and predictive-code similarity at planted "
                    "parent-state changes, three seeds per row. Two predictions were stated "
                    "before the run: novel activity higher at changes, and the difference "
                    "absent in the null."),
        ["condition", "novel at", "novel between", "ratio",
         "pred.\\ cos at", "pred.\\ cos between"],
        rows, align="lrrrrr",
        note="The novel-code ratio exceeds one in the null as well, where the data carries no "
             "events. The predictive column separates the null from the other two. That "
             "measure was recorded alongside but not stated in advance.")


def f_boundary(out_dir: Path):
    d = _boundary()
    if not d:
        return None
    keys = [("pit_temporal", "temporal\noriginal"),
            ("pit_decorr", "temporal\ndecorrelated"),
            ("pit_null", "i.i.d.\n(no events)")]
    keys = [(k, n) for k, n in keys if d["cells"].get(k)]
    cells = [n for _, n in keys]
    m = lambda k, f: st.mean(g[f] for g in d["cells"][k])
    nov = [(m(k, "novel_mag_at"), m(k, "novel_mag_within")) for k, _ in keys]
    prd = [(m(k, "pred_cos_at"), m(k, "pred_cos_within")) for k, _ in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, data, ylab, ylim in ((axes[0], nov, "novel-code magnitude", None),
                                 (axes[1], prd, "predictive-code cosine", (0, 1.05))):
        x = range(len(cells)); w = 0.36
        ax.bar([i - w/2 for i in x], [d[0] for d in data], w, label="at a change")
        ax.bar([i + w/2 for i in x], [d[1] for d in data], w, label="between changes")
        ax.set_xticks(list(x)); ax.set_xticklabels(cells, fontsize=8)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=.25, lw=.5, axis="y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = out_dir / "pit_boundary.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return figure("pit-boundary",
                  TODO.format("Novel and predictive code at planted parent-state changes. "
                              "The null condition contains no events."),
                  "figuers/pit_boundary")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    figdir = (args.out / "figuers") if not args.list else Path("/tmp")
    figdir.mkdir(parents=True, exist_ok=True)

    tabs = [("gemma-layer", "gemma-2-2b layer 12", t_gemma_layer()),
            ("gemma-quality", None, t_gemma_quality()),
            ("toy", "Toy model", t_toy_event()),
            ("toy-boundary", None, t_toy_boundary())]
    figs = [("toy-boundary", "Toy model", f_boundary(figdir))]
    tabs = [(s, sec, t) for s, sec, t in tabs if t]
    figs = [(s, sec, t) for s, sec, t in figs if t]

    if args.list:
        print("tables: ", [s for s, _, _ in tabs])
        print("figures:", [s for s, _, _ in figs])
        return 0
    a = emit(args.out, "priors_saes-tables.tex", GEN, tabs)
    b = emit(args.out, "priors_saes-figures.tex", GEN, figs)
    print(f"  {a.name}   {len(tabs)} tables")
    print(f"  {b.name}  {len(figs)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
