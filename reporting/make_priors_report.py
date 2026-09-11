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
        TODO.format("The released Priors in Time model, at layer 12 of "
                    "\\texttt{gemma-2-2b}, on 200 documents. We ran it through the "
                    "authors' own code rather than rebuilding it ourselves. The last row "
                    "is the point: this method splits each token into what its context "
                    "already suggests and what is new. Nine parts in ten come from the "
                    "context."),
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
        TODO.format("Which layer this model was trained on. We had to check, because "
                    "the model's own settings file and its README disagree. An SAE "
                    "rebuilds the layer it was trained on better than any other, so the "
                    "best row identifies the layer. A negative value means the "
                    "reconstruction is worse than simply guessing the average. The "
                    "settings file records "
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
        TODO.format("Priors in Time trained on the small world, each row run three "
                    "times. The last row is our control. In it we removed the link "
                    "between one token and the next, and changed nothing else. Any "
                    "difference between the control and the rows above it therefore "
                    "comes from time, and not from how often features fire."),
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
        # The null row is emphasised on the ratio, because the ratio is the result: it
        # is above one where the data has no events at all. The two predictive columns
        # were emphasised here while they appeared to separate the null on three seeds.
        # On eight they do not, so the emphasis is gone and so is the claim.
        bold = key == "pit_null"
        rat = f"{at/wi:.2f}"
        rows.append([name, f"{at:.3f}", f"{wi:.3f}",
                     (r"\textbf{" + rat + "}") if bold else rat,
                     f"{pa:.4f}", f"{pw:.4f}"])
    return table(
        "pit-boundary",
        TODO.format("What the two parts do at points where we changed the hidden "
                    "state on purpose. We wrote down two predictions before running "
                    "this. The new part would be more active at those points. That extra "
                    "activity would disappear in the control, which has no such points "
                    "to find. The first held. The second did not. The ratio is above one "
                    "in the control at every one of its eight runs, and the control is "
                    "the steadiest of the three rows. The last two columns were once "
                    "read as separating the control, on three runs. With eight they do "
                    "not, and we no longer report them as a result."),
        ["condition", "novel at", "novel between", "ratio",
         "pred.\\ cos at", "pred.\\ cos between"],
        rows, align="lrrrrr",
        note="Eight runs per row. The ratio is above one in the null as well, where the data "
             "carries no events at all, and the null is the steadiest of the three rows "
             "(spread 0.139 against 0.359 and 0.408). The two predictive columns are shown "
             "for completeness. They were once read as separating the null, on three runs; "
             "on eight they do not, and they are no longer offered as a result.")


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
                  TODO.format("The same measurements, drawn. The control row has no "
                              "planted changes in it at all, so it shows what these "
                              "measures do when there is nothing to find."),
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
