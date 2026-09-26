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

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from reporting.temporal_common import (DEFAULT_OUT, INK, MUTED, PIT, TOY, emit, figure,
                                       ms, read, table, toy_pit)

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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, data, ylab, ylim in ((axes[0], nov, "novel-code magnitude", None),
                                 (axes[1], prd, "predictive-code cosine", (0, 1.12))):
        x = range(len(cells)); w = 0.36
        # Same two hues as the before/after figure next to it, so one reader's
        # "at a change" is the same colour on both.
        ax.bar([i - w/2 for i in x], [d[0] for d in data], w, color=TEMPORAL,
               label="at a change")
        ax.bar([i + w/2 for i in x], [d[1] for d in data], w, color=CONTROL,
               label="between changes")
        ax.set_xticks(list(x)); ax.set_xticklabels(cells, fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_ylim(*(ylim or (0, max(max(d) for d in data) * 1.18)))
        ax.grid(alpha=.25, lw=.5, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    # Below both panels rather than inside one: in the axes it covered the third
    # pair of bars, and the left panel has no free corner at any y-limit.
    fig.legend(handles=axes[0].containers, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    p = out_dir / "pit_boundary.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return figure("pit-boundary",
                  TODO.format("The same measurements, drawn. The control row has no "
                              "planted changes in it at all, so it shows what these "
                              "measures do when there is nothing to find."),
                  "figuers/pit_boundary")


# --------------------------------------------------------------------------- before/after
# Two hues, not three: the axis is cells, but the distinction that matters is
# "data with planted events" against "the control that has none". Okabe-Ito,
# checked with the palette validator rather than by eye -- both clear 3:1 on white
# and the pair clears CVD dE 8, and the control is named in text as well.
TEMPORAL, CONTROL = "#0072B2", "#D55E00"

# The two predictions, as written into the test script before any number was read.
# Read from the frozen file rather than retyped, so the panel cannot drift from
# what was actually claimed.
CELLS = [("pit_temporal", "temporal\noriginal tree", False),
         ("pit_decorr", "temporal\ndecorrelated tree", False),
         ("pit_null", "i.i.d. control\n(no events)", True)]


def f_before_after(out_dir: Path):
    """What was predicted about the novel code, and what the novel code did.

    The before/after pair the calibration world gets, rotated onto this
    architecture. Priors in Time has no nested blocks and no parent-child edges, so
    there is no graph and no block occupancy to draw twice. Its claim is about one
    number at one kind of position: novel-code magnitude at a change in parent
    state, over its magnitude between changes. So that number is the geometry, and
    it is drawn twice on one axis -- predicted above, measured below.
    """
    d = _boundary()
    if not d:
        return None
    cells = [(k, lab, ctl) for k, lab, ctl in CELLS if d["cells"].get(k)]
    if not cells:
        return None
    preds = d.get("predictions_stated_before_the_run", [])
    n_seeds = max(len(d["cells"][k]) for k, _, _ in cells)

    def ratios(k):
        return [g["novel_mag_at"] / g["novel_mag_within"] for g in d["cells"][k]]

    def share(k):
        g = d["cells"][k]
        return st.mean(x["n_boundary"] / (x["n_boundary"] + x["n_within"]) for x in g)

    TOP = max(max(ratios(k)) for k, _, _ in cells) + 0.35
    fig, (before, after) = plt.subplots(
        2, 1, figsize=(7.8, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6], "hspace": 0.17})

    # ---- before: the prediction, drawn as an open band so it cannot be read as a
    # measurement. "Higher" names a region, not a value, and a region is what is drawn.
    for i, (k, _, ctl) in enumerate(cells):
        col = CONTROL if ctl else TEMPORAL
        if ctl:
            before.plot([i - 0.30, i + 0.30], [1.0, 1.0], lw=3.5, color=col,
                        solid_capstyle="butt", zorder=4)
            before.text(i, 1.0 + 0.14, "no difference", ha="center", va="bottom",
                        fontsize=8.5, color=col)
        else:
            before.add_patch(Rectangle((i - 0.30, 1.0), 0.60, TOP - 1.0,
                                       facecolor=col, alpha=0.13, lw=0, zorder=2))
            before.plot([i - 0.30, i + 0.30], [1.0, 1.0], lw=1.4, color=col,
                        ls=(0, (3, 2)), zorder=4)
            before.annotate("", xy=(i, TOP - 0.04), xytext=(i, 1.30),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
            before.text(i, 1.16, "higher", ha="center", va="bottom",
                        fontsize=8.5, color=col, zorder=5)

    # ---- after: every seed, the spread, and the mean
    for i, (k, _, ctl) in enumerate(cells):
        col = CONTROL if ctl else TEMPORAL
        v = ratios(k)
        after.plot([i, i], [min(v), max(v)], lw=1.4, color=col, alpha=0.45, zorder=2)
        after.scatter([i + (j - (len(v) - 1) / 2) * 0.045 for j in range(len(v))], v,
                      s=26, facecolor="white", edgecolor=col, linewidths=1.2, zorder=3)
        after.plot([i - 0.26, i + 0.26], [st.mean(v)] * 2, lw=3.0, color=col,
                   solid_capstyle="butt", zorder=4)
        after.text(i + 0.32, st.mean(v), f"{st.mean(v):.2f}", ha="left", va="center",
                   fontsize=8.5, fontweight="bold", color=INK)
        after.text(i + 0.32, st.mean(v) - 0.13, f"spread {st.pstdev(v):.2f}",
                   ha="left", va="center", fontsize=7, color=MUTED)

    for ax, ttl in ((before, "before"), (after, "after")):
        ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(0.88, TOP + 0.10)
        ax.set_xlim(-0.62, len(cells) - 0.38 + 0.58)
        ax.grid(alpha=.22, lw=.5, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, 1.015, ttl, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)
    # One axis label for both panels: the scale is shared, and repeating it twice
    # was two-thirds of the left margin.
    fig.supylabel("novel-code magnitude at a change $\\div$ between changes",
                  fontsize=9, color=INK, x=0.022)

    # The share of positions that count as a change. This is the thing the control
    # was built to remove, so it belongs under the cells it explains.
    after.set_xticks(range(len(cells)))
    after.set_xticklabels(
        [f"{lab}\na change at {share(k):.0%}\nof positions" for k, lab, _ in cells],
        fontsize=8.5)
    after.tick_params(axis="x", length=0, pad=6)

    handles = [Line2D([], [], color=TEMPORAL, lw=3, label="data with planted events"),
               Line2D([], [], color=CONTROL, lw=3, label="i.i.d. control, no events"),
               Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                      label="no difference (ratio 1)"),
               Line2D([], [], ls="none", marker="o", markersize=6,
                      markerfacecolor="white", markeredgecolor=INK, label="one seed")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.95, bottom=0.165)

    fig.savefig(out_dir / "pit_before_after.png", dpi=300)
    plt.close(fig)
    # The predictions are quoted, not retyped, so the note cannot drift from the
    # file. They contain a command-line flag, and LaTeX turns a bare `--` into an
    # en-dash, so the flag is set in \texttt with the ligature broken.
    def _tex(t):
        return t.replace("--persistence 0", r"\texttt{-{}-persistence 0}")
    note = None
    if len(preds) == 2:
        note = "Stated before the run: (1) {}; (2) {}.".format(*(_tex(x) for x in preds))
    return figure("pit-before-after",
                  TODO.format("What we predicted about the novel code, and what we "
                              "measured. This method splits each token's activation "
                              "into a predictive part and a novel part. It has no "
                              "nested blocks and no parent-child edges, so there is no "
                              "recovered graph to show. We test it on one number "
                              "instead. That number is the novel code's magnitude at a "
                              "planted change in parent state, divided by its magnitude "
                              "between changes. The top panel shows the two predictions "
                              "we wrote down before the run. The bottom panel shows all "
                              f"{n_seeds} runs of each condition. The first prediction "
                              "holds. The second does not. The control also sits above "
                              "1, and its spread is the smallest of the three."),
                  "figuers/pit_before_after", note=note)


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
    figs = [("toy-before-after", "Toy model", f_before_after(figdir)),
            ("toy-boundary", None, f_boundary(figdir))]
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
