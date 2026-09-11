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

from reporting.temporal_common import (DEFAULT_OUT, MSAE, TSAE, emit, figure, ms, read, table, toy_tsae)

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
        TODO.format("Temporal SAE on the toy, 20/80 split, k=2, three seeds per row. "
                    "Chance InfoNCE is $\\ln 256 = 5.545$."),
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
        TODO.format("Parent features recovered into the first block, on two trees. "
                    "In the original, parents are also the most frequent features; in the "
                    "decorrelated tree they are rare and the distractors frequent. "
                    "Expected $L_0$ is held at 1.12 and 1.11."),
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
        TODO.format("\\texttt{gemma-2-2b} layer 12, both architectures through the same "
                    "metric code on the same corpus."),
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
        for pair in ("0->1", "1->2", "2->3"):
            s = (d.get(pair) or {}).get("sres", {})
            if s.get("n_edges_scored"):
                rows.append([f"Matryoshka ${pair.replace('->', chr(92)+'rightarrow')}$",
                             f"{s['n_edges_scored']:,}", f"{s['n_pass']:,}",
                             f"{s['frac_pass']:.3f}"])
    if not rows:
        return None
    return table(
        "tsae-gemma-sres",
        TODO.format("Probe test at \\texttt{gemma-2-2b} layer 12. The probe asks whether the "
                    "parent adds information beyond the child, rather than whether the two "
                    "co-fire."),
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
        TODO.format("$\\EdgeTau$ swept on \\texttt{gemma-2-2b} layer 12, one run per "
                    "architecture. The chance share rises with $\\EdgeTau$ for Matryoshka and "
                    "stays at zero for the Temporal SAE."),
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
                  TODO.format("Parent recovery into the first block on the two trees. "
                              "Error bars are the spread over three seeds."),
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
                  TODO.format("Chance share against $\\tau$ on \\texttt{gemma-2-2b} layer 12. "
                              "The dashed line marks the operating value."),
                  "figuers/tsae_threshold_reversal", width="0.62\\linewidth")


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
    figs = [("toy-decorr", "Toy model", f_decorrelated(figdir)),
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
