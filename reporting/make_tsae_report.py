#!/usr/bin/env python3
"""Tables and figures for the Temporal SAE results.

    python3 -m reporting.make_tsae_report            # write both files
    python3 -m reporting.make_tsae_report --list     # show the plan, write nothing

Output: outputs/paper_figuers/tsaes-tables.tex and tsaes-figures.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from reporting.temporal_common import (DEFAULT_OUT, INK, MSAE, MUTED, TSAE, emit, figure,
                                       ms, read, table, text_on, toy_tsae, tree_rates)

GEN = "make_tsae_report"
TODO = "TODO(caption, human-written). {}"


# --------------------------------------------------------------------------- tables
def t_toy_conditions():
    """The four cells at the faithful 20/80 split, three seeds each."""
    d = toy_tsae()["original_tree"]
    order = [("tsae_temporal", "temporal, cosine"),
             ("tsaeip_temporal", "temporal, inner product"),
             ("msae_temporal", "temporal, no term"),
             ("tsae_null", "i.i.d., cosine"),
             ("tsaeip_null", "i.i.d., inner product")]
    rows = []
    for key, name in order:
        r = d.get(key)
        if not r:
            continue
        rows.append([name, len(r),
                     ms([x["fve"] for x in r], 3), ms([x["l0"] for x in r], 2),
                     ms([float(x["parents_in_b0"]) for x in r], 2),
                     ms([x["nce_cosine"] for x in r], 3),
                     ms([x["nce_inner"] for x in r], 3)])
    return table(
        "tsae-toy-conditions",
        TODO.format("The Temporal SAE on the small made-up world. Each row is one "
                    "setting, run three times. The last column is the time rule. It has "
                    "a value it cannot go above, $5.545$ here, which is what it shows "
                    "when it has learned nothing at all. A row sitting at that value "
                    "means the rule did nothing in that setting."),
        ["condition", "$n$", "FVE", "$L_0$", "parents in $B_0$",
         "InfoNCE (cos)", "InfoNCE (inner)"],
        rows, align="lcrrrrr",
        note="Each model is scored under both similarity functions; a model is comparable "
             "only to a baseline measured the same way. The tree has three parents.")


def t_decorrelated():
    """Frequency against depth: the same conditions on a tree where they disagree."""
    a, b = toy_tsae()["original_tree"], toy_tsae()["decorrelated_tree"]
    rows = []
    for key, name in [("tsaeip_temporal", "inner product"),
                      ("tsae_temporal", "cosine"),
                      ("msae_temporal", "no term")]:
        ra, rb = a.get(key), b.get(key)
        if not (ra and rb):
            continue
        rows.append([name,
                     ms([float(x["parents_in_b0"]) for x in ra], 2),
                     ms([float(x["parents_in_b0"]) for x in rb], 2),
                     ms([float(x["distractors_in_b0"]) for x in rb], 2)])
    return table(
        "tsae-decorrelated",
        TODO.format("How many of the three parents end up in the first block, on two "
                    "worlds. In the first world the parents are also the features that "
                    "fire most often. In the second we made them rare instead. "
                    "Everything else is the same, including how many features fire at "
                    "once. The parents are learned in both worlds. Only the first block "
                    "changes."),
        ["condition", "parents, original", "parents, decorrelated",
         "distractors, decorrelated"],
        rows, align="lrrr",
        note="Out of three parents. All three are still learned somewhere in the dictionary "
             "in every run, so this is placement rather than a failure to learn them.")


def t_gemma_pipeline():
    """T-SAE against Matryoshka through the same metric code, same layer, same corpus."""
    t, m = read(TSAE / "metrics_report.json"), read(MSAE / "metrics_report.json")
    if not (t and m):
        return None
    def pick(rep, key):
        return next((p for p in rep["pairs"] if p.get("pair") == key), {})
    def dims(rep, key):
        p, c = (int(x) for x in key.split("->"))
        br = rep["block_ranges"]
        return br[p][1]-br[p][0], br[c][1]-br[c][0]
    rows = []
    for name, rep, key in [("Temporal SAE", t, "0->1"),
                           ("Matryoshka", m, "0->1"),
                           ("Matryoshka", m, "1->2")]:
        b = pick(rep, key); P, C = dims(rep, key)
        n = b.get("n_candidate_edges", 0)
        rows.append([f"{name} {key.replace('->', '$\\rightarrow$')}",
                     f"{P}$\\times${C}", f"{n:,}", f"{n/(P*C):.2e}",
                     b.get("n_superparents", "--"),
                     f"{b.get('joint_child_cov_mean', float('nan')):.3f}",
                     f"{b.get('n_dropped_min_joint', '--'):,}"
                     if isinstance(b.get("n_dropped_min_joint"), int) else "--"])
    return table(
        "tsae-gemma-pipeline",
        TODO.format("Two SAEs at layer 12 of \\texttt{gemma-2-2b}, measured the same "
                    "way on the same text. The blocks differ a lot in size, so the "
                    "number of links found cannot be compared directly. The density "
                    "column can: it is the share of all possible pairs that became a "
                    "link. The shape column gives the block sizes behind each row."),
        ["block pair", "shape", "edges", "density", "superparents",
         "joint-child cov.", "dropped by $\\MinJoint$"],
        rows, align="llrrrrr",
        note="Raw counts do not compare across pairs of different shape; density does. "
             "The Matryoshka $1\\rightarrow2$ row is the closest in shape, not a matched "
             "control.")


def t_gemma_sres():
    """The probe test: how much of each candidate set holds up."""
    sp = read(TSAE / "second_pass.json")
    base = Path(__file__).resolve().parent.parent / "outputs" / "gemma-2-2b"
    rows = []
    if sp:
        s = (sp.get("0->1") or {}).get("sres", {})
        rows.append(["Temporal SAE $0\\rightarrow1$", f"{s.get('n_edges_scored', 0):,}",
                     f"{s.get('n_pass', 0):,}", f"{s.get('frac_pass', 0):.3f}"])
    import json as _j
    f = base / "layer_12" / "second_pass.json"
    if f.is_file():
        d = _j.loads(f.read_text())
        # Read the pairs present rather than naming three. The probe stage runs on
        # whatever the report graded, and B3->B4 is off by default only because of
        # memory, so a fourth pair can appear without this file being touched.
        graded = sorted((k for k in d if "->" in k),
                        key=lambda k: tuple(int(x) for x in k.split("->")))
        for pair in graded:
            s = (d.get(pair) or {}).get("sres", {})
            if s.get("n_edges_scored"):
                rows.append([f"Matryoshka ${pair.replace('->', chr(92)+'rightarrow')}$",
                             f"{s['n_edges_scored']:,}", f"{s['n_pass']:,}",
                             f"{s['frac_pass']:.3f}"])
    if not rows:
        return None
    return table(
        "tsae-gemma-sres",
        TODO.format("Our strictest test, at layer 12 of \\texttt{gemma-2-2b}. It does "
                    "not ask whether the two features fire together. It asks whether the "
                    "parent tells us something about the child that the child does not "
                    "already tell us by itself. The last column is the share of tested "
                    "links that passed."),
        ["block pair", "edges scored", "pass", "fraction"],
        rows, align="lrrr",
        note="No child was untestable in either run, so neither fraction is thinned by "
             "missing negatives.")


def t_threshold_reversal():
    """The same threshold sweep on two sources, moving in opposite directions."""
    g_t, g_m = read(TSAE / "threshold_sweep.json"), read(MSAE / "threshold_sweep.json")
    if not (g_t and g_m):
        return None
    def at(js, tau):
        r = [x for x in js["rows"]
             if x["knob"] == "tau" and x["value"] == tau and x["pair"] == "0->1"]
        return (r[0]["n_accepted"], r[0]["chance_after"]) if r else (None, None)
    rows = []
    for tau in (0.1, 0.5, 0.6, 0.7, 0.9):
        nm, cm = at(g_m, tau); nt, ct = at(g_t, tau)
        if nm is None:
            continue
        mark = r" $\leftarrow$ operating" if tau == 0.5 else ""
        rows.append([f"{tau}{mark}", f"{nm:,}", "--" if cm is None else f"{cm:.2f}",
                     f"{nt:,}", "--" if ct is None else f"{ct:.2f}"])
    return table(
        "tsae-threshold-reversal",
        TODO.format("What happens when we raise our main cut-off, at layer 12 of "
                    "\\texttt{gemma-2-2b}. Raising a cut-off should keep only the "
                    "stronger links. The column on the right is the share of links that "
                    "could be explained by the two features simply being common. For "
                    "Matryoshka that share goes up, which is the opposite of what a "
                    "stricter cut-off should do."),
        ["$\\EdgeTau$", "MSAE edges", "chance share", "T-SAE edges", "chance share"],
        rows, align="lrrrr",
        note="Chance share is the fraction of accepted edges with PMI below 0.5. The two "
             "columns differ in first-block width, 128 against 3{,}276, which is confounded "
             "with the architecture here.")


# --------------------------------------------------------------------------- figures
def f_decorrelated(out_dir: Path):
    a, b = toy_tsae()["original_tree"], toy_tsae()["decorrelated_tree"]
    cells = [("tsaeip_temporal", "inner product"), ("tsae_temporal", "cosine"),
             ("msae_temporal", "no term")]
    have = [(k, n) for k, n in cells if a.get(k) and b.get(k)]
    if not have:
        return None
    import statistics as _st
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    xs = range(len(have))
    for src, lbl, mk in ((a, "parents frequent", "o"), (b, "parents rare", "s")):
        y = [_st.mean(x["parents_in_b0"] for x in src[k]) for k, _ in have]
        e = [_st.pstdev([x["parents_in_b0"] for x in src[k]]) for k, _ in have]
        ax.errorbar(xs, y, yerr=e, marker=mk, capsize=3, label=lbl, lw=1.6)
    ax.set_xticks(list(xs)); ax.set_xticklabels([n for _, n in have])
    ax.set_ylabel("parents in first block (of 3)"); ax.set_ylim(-0.2, 3.4)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    p = out_dir / "tsae_decorrelated_parents.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return figure("tsae-decorrelated-parents",
                  TODO.format("How many parents land in the first block, on the two "
                              "worlds. The bars show the average of three runs. The "
                              "lines through them show how much the three runs differed "
                              "from each other."),
                  "figuers/tsae_decorrelated_parents", width="0.62\\linewidth")


def f_threshold_reversal(out_dir: Path):
    g_t, g_m = read(TSAE / "threshold_sweep.json"), read(MSAE / "threshold_sweep.json")
    if not (g_t and g_m):
        return None
    def series(js):
        r = sorted((x for x in js["rows"]
                    if x["knob"] == "tau" and x["pair"] == "0->1"), key=lambda z: z["value"])
        return [x["value"] for x in r], [x["chance_after"] for x in r]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for js, lbl, mk in ((g_m, "Matryoshka (128 wide)", "o"),
                        (g_t, "Temporal SAE (3,276 wide)", "s")):
        x, y = series(js)
        ax.plot(x, [v if v is not None else float("nan") for v in y],
                marker=mk, lw=1.6, label=lbl)
    ax.axvline(0.5, color="0.6", ls="--", lw=1)
    ax.set_xlabel(r"edge threshold $\tau$")
    ax.set_ylabel("share of accepted edges at chance")
    ax.set_ylim(-0.05, 1.05); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    p = out_dir / "tsae_threshold_reversal.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return figure("tsae-threshold-reversal",
                  TODO.format("The same cut-off, drawn. Going right means a stricter "
                              "cut-off. Going up means more of the surviving links could "
                              "be explained by the two features simply being common. The "
                              "dashed line is the setting we use everywhere else in this "
                              "paper."),
                  "figuers/tsae_threshold_reversal", width="0.62\\linewidth")


# --------------------------------------------------------------------------- block figure
# Fills for the block-occupancy figure. Okabe-Ito, the same hues the calibration
# figures use, so the two papers read as one system. Checked with the palette
# validator rather than by eye: every adjacent pair clears CVD dE 8 and every fill
# clears 3:1 against white, so the letter inside each square is a second encoding
# rather than the only relief.
SLOT = {"parent":     ("#009E73", "parent", "P"),
        "child":      ("#0072B2", "child", "C"),
        "distractor": ("#D55E00", "distractor", "D"),
        "other":      ("#9AA3AD", "other recovered feature", ""),
        "empty":      ("#E4E7EA", "unused slot", "")}

B0, B1 = 4, 20                       # latent_sizes [4, 20] -- the faithful 20/80 split
N_PAR, N_CHI, N_DIS = 3, 9, 8        # the tree, by role


def _row(ax, dx, y, b0_roles, b1_roles, h=0.52):
    """One dictionary at column offset `dx`: four wide B0 slots, then twenty narrow B1 ones.

    The two blocks are drawn at different widths on purpose. B0 is where the claim
    lives and its four slots are read one at a time; B1 is context and is read as a
    bar. Same geometry in both panels, so before and after line up by eye -- the
    reason the calibration twin draws its world twice.
    """
    for i, r in enumerate(b0_roles):
        col, _, letter = SLOT[r]
        ax.add_patch(Rectangle((dx + i * 1.0, y), 0.88, h, facecolor=col,
                               edgecolor="white", lw=0.9, zorder=2))
        if letter:
            ax.text(dx + i * 1.0 + 0.44, y + h / 2, letter, ha="center", va="center",
                    fontsize=6.2, color=text_on(col), zorder=3)
    for i, r in enumerate(b1_roles):
        ax.add_patch(Rectangle((dx + 4.8 + i * 0.44, y), 0.38, h,
                               facecolor=SLOT[r][0], edgecolor="white", lw=0.7,
                               zorder=2))


def _b0_after(run):
    """B0 slot by slot, from the run's own counters. B0 is always full at 4."""
    b0 = (["parent"] * run["parents_in_b0"]
          + ["distractor"] * run["distractors_in_b0"])
    # Whatever is left of the four is a child: the counters name parents and
    # distractors, and the tree has no third kind of feature.
    return b0 + ["child"] * (B0 - len(b0))


def _b1_after(run):
    """What B1 holds, from the run's own counters -- no role is invented.

    Only B0's composition is recorded per role, so B1 names the one role it can:
    every parent is learned in every seed (3.000 +- 0.000), so the parents missing
    from B0 are in B1. The rest of what was recovered goes in as `other`, because
    which of them are children and which distractors is not in the file.
    """
    par_out = N_PAR - run["parents_in_b0"]
    rest = max(run["features_recovered"] - B0 - par_out, 0)
    return (["parent"] * par_out + ["other"] * rest
            + ["empty"] * (B1 - par_out - rest))


def f_block_before_after(out_dir: Path):
    """Where the nesting says the parents go, and where training put them.

    The before/after pair the calibration world gets, rotated onto this
    architecture: a Temporal SAE has no parent-child edges to recover, so there is
    no graph to redraw. What it has is a nested dictionary with a four-slot coarse
    block and a claim about what belongs there. So the thing drawn twice is the
    dictionary, and the two tree columns carry the test: the declared destination is
    identical in both, and only the firing rates differ.
    """
    cells = toy_tsae()
    conds = [(k, n) for k, n in (("tsaeip_temporal", "inner product"),
                                 ("tsae_temporal", "cosine"),
                                 ("msae_temporal", "no term"))
             if cells["original_tree"].get(k) and cells["decorrelated_tree"].get(k)]
    if not conds:
        return None
    rates = tree_rates()
    trees = [("original_tree", "original", "original tree"),
             ("decorrelated_tree", "decorrelated", "decorrelated tree")]
    WIDE = 4.8 + B1 * 0.44                  # one dictionary, end to end
    COL = WIDE + 2.4                        # x offset between the two columns
    LEFT = -4.8                             # room for the two text columns
    n_seeds = max(len(cells[t][k]) for t, _, _ in trees for k, _ in conds)

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 6.0),
                             gridspec_kw={"height_ratios": [1.0, 3.4], "hspace": 0.04})
    before, after = axes

    # ---- before: the same declared placement in both columns, only the rates differ
    declared_b0 = ["parent"] * N_PAR + ["empty"] * (B0 - N_PAR)
    declared_b1 = (["child"] * N_CHI + ["distractor"] * N_DIS
                   + ["empty"] * (B1 - N_CHI - N_DIS))
    for ti, (_, rkey, label) in enumerate(trees):
        dx = ti * COL
        _row(before, dx, 0.0, declared_b0, declared_b1)
        for x, lab in ((0.0, rf"$B_0$ -- {B0} slots"), (4.8, rf"$B_1$ -- {B1} slots")):
            before.text(dx + x, 0.60, lab, ha="left", va="bottom", fontsize=7.5,
                        color=MUTED)
        before.text(dx, 1.14, label, ha="left", va="bottom", fontsize=9.5,
                    fontweight="bold", color=INK)
        r = rates.get(rkey)
        if r:
            before.text(dx, 0.97,
                        f"parents fire at {r['parent'][1]:g}, "
                        f"distractors at {r['distractor'][1]:g}",
                        ha="left", va="bottom", fontsize=7.5, color=MUTED)
    before.text(LEFT, 1.14, "before", ha="left", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)
    before.set_ylim(-0.14, 1.46)

    # ---- after: one row per seed, conditions grouped, both trees on the same rows
    ys = {}
    for ci, (key, name) in enumerate(conds):
        for si in range(n_seeds):
            ys[(ci, si)] = -(ci * (n_seeds * 0.62 + 0.40) + si * 0.62)
        mid = (ys[(ci, 0)] + ys[(ci, n_seeds - 1)]) / 2 + 0.26
        after.text(-1.8, mid, name, ha="right", va="center", fontsize=8.5, color=INK)
    for ti, (tkey, _, label) in enumerate(trees):
        dx = ti * COL
        after.text(dx, 0.80, label, ha="left", va="bottom", fontsize=9.5,
                   fontweight="bold", color=INK)
        for ci, (key, _) in enumerate(conds):
            for si, run in enumerate(sorted(cells[tkey][key], key=lambda r: r["seed"])):
                _row(after, dx, ys[(ci, si)], _b0_after(run), _b1_after(run))
                if ti == 0:              # seed labels once, the rows are shared
                    after.text(-0.30, ys[(ci, si)] + 0.26, f"seed {run['seed']}",
                               ha="right", va="center", fontsize=6.4, color=MUTED)
    lo = min(ys.values())
    for ti in range(len(trees)):         # B0 is the claim: box it in both columns
        after.add_patch(Rectangle((ti * COL - 0.16, lo - 0.16), 4.20,
                                  0.84 - lo, fill=False, edgecolor=MUTED,
                                  lw=0.9, ls=(0, (3, 2)), zorder=4))
    after.text(LEFT, 0.80, "after", ha="left", va="bottom",
               fontsize=9.5, fontweight="bold", color=INK)
    after.set_ylim(lo - 0.40, 1.30)

    for ax in axes:
        ax.set_xlim(LEFT, COL + WIDE + 0.4)
        ax.axis("off")

    handles = [Line2D([], [], ls="none", marker="s", markersize=8,
                      markerfacecolor=SLOT[r][0], markeredgecolor="white",
                      label=SLOT[r][1])
               for r in ("parent", "child", "distractor", "other", "empty")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=5,
                     bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8,
                     handlelength=1.4, columnspacing=1.9)
    leg._legend_box.align = "left"
    fig.subplots_adjust(left=0.012, right=0.994, top=0.975, bottom=0.085)

    fig.savefig(out_dir / "tsae_block_before_after.png", dpi=300)
    plt.close(fig)
    return figure("tsae-block-before-after",
                  TODO.format("Where the three parents should sit in the "
                              "dictionary, and where training put them. The top row is "
                              "the same in both columns. The first block has four "
                              "slots, and the three parents are what should fill them. "
                              "The two columns differ only in how often each feature "
                              "fires. The bottom half shows every run on both trees. "
                              "Only $B_0$ is recorded feature by feature. In $B_1$ we "
                              "can name the parents, because all three are learned in "
                              "every run. The rest of that block is shown as a count."),
                  "figuers/tsae_block_before_after")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # The figure directory must exist before any figure is drawn, not after: matplotlib
    # writes straight to the path and fails if the parent is missing.
    figdir = (args.out / "figuers") if not args.list else Path("/tmp")
    figdir.mkdir(parents=True, exist_ok=True)

    tabs = [("toy", "Toy model", t_toy_conditions()),
            ("toy-decorr", None, t_decorrelated()),
            ("gemma", "gemma-2-2b layer 12", t_gemma_pipeline()),
            ("gemma-sres", None, t_gemma_sres()),
            ("thresholds", None, t_threshold_reversal())]
    figs = [("toy-before-after", "Toy model", f_block_before_after(figdir)),
            ("toy-decorr", None, f_decorrelated(figdir)),
            ("thresholds", "gemma-2-2b layer 12", f_threshold_reversal(figdir))]

    tabs = [(s, sec, t) for s, sec, t in tabs if t]
    figs = [(s, sec, t) for s, sec, t in figs if t]
    if args.list:
        print("tables: ", [s for s, _, _ in tabs])
        print("figures:", [s for s, _, _ in figs])
        return 0
    a = emit(args.out, "tsaes-tables.tex", GEN, tabs)
    b = emit(args.out, "tsaes-figures.tex", GEN, figs)
    print(f"  {a.name}   {len(tabs)} tables")
    print(f"  {b.name}  {len(figs)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
