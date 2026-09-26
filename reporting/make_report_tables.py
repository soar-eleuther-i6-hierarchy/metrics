"""
Paper tables — LaTeX, ordered as the paper reads, every number read from data.

    python3 -m reporting.make_report_tables              # write tables.tex
    python3 -m reporting.make_report_tables --list       # the plan and its inputs
    python3 -m reporting.make_report_tables --out DIR    # somewhere else

Output: outputs/paper_figuers/tables.tex

The companion to `make_report_figures.py`, and it follows the same two rules for
the same reasons.

**Nothing is hardcoded that can be derived.** Thresholds come from `config`, block
structure from `config`, and every count from the JSON report it describes. The
figure generator had to be rebuilt once because its titles quoted layer-6 numbers
that had been withdrawn; a table of numbers is that failure mode with more places
to hide.

**Nothing is skipped silently.** A table whose input is missing prints the path it
wanted; `--list` shows the whole plan without writing.

What is *not* derived, and cannot be: the properties matrix. Its cells are a
reading of what each metric is able to separate, argued in the module docstrings
and the root README, not measured on any run. It is a literal here, and the
docstring says so where the table is defined rather than leaving a reader to
assume the ticks were computed.

Ordering mirrors the figures: the instrument is defined before any number it
produces is quoted, then the results, then the appendix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import config as C
# The gemma/PCFG comparison's pairing rules and its six measures. Shared with
# `make_report_figures`, which draws the same claim: two copies of the alignment
# logic would eventually print different numbers for it.
from reporting import cross_source as X

DEFAULT_OUT = C.OUT_DIR / "paper_figuers"


# --- data access ------------------------------------------------------------
def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pair(report, name="0->1"):
    return next((p for p in report["pairs"] if p["pair"] == name), None)


def gemma_layers(merge: bool = True):
    """The graded layers, with any variant run of the same layer folded in.

    Pass ``merge=False`` for any table whose columns are whole-run totals. B3->B4 was
    graded at layer 12 alone, so a merged layer 12 pools four pairs where every other
    gemma layer pools three. That is not a measurement moving: the reconstruction column
    fell from 30% to 19% purely because a fourth pair with a 5% pass rate joined the pool
    at one row and nowhere else. Per-pair tables want the merge; whole-run totals must not
    have it, or the rows stop being comparable with each other.

    This file kept its own loader, and when B3->B4 was graded in a separate run the
    figures picked it up and the tables did not. One document then said the pair was
    never computed while the figure beside it printed its numbers. The merge is imported
    rather than copied so the two cannot drift apart again; its agreement check is
    documented there.
    """
    from .make_report_figures import _merge_variant_pairs
    G = C.OUT_DIR / C.SOURCE_NAME
    out = []
    for L in C.NAV_LAYERS:
        d = G / f"layer_{L:02d}"
        r = _json(d / "metrics_report.json")
        if r:
            out.append((L, _merge_variant_pairs(d, r) if merge else r))
    return out


def _layer_name(dirname: str) -> str:
    """``layer_00`` -> ``layer 0``. The zero pad is a sort key, not a number.

    Run directories are padded so ``ls`` orders them, and reading that pad
    straight into a label printed the PCFG runs as L00--L03 beside gemma's
    L1--L24, which invites a reader to take the width for a difference in what
    is being counted. The pad is dropped for display only; nothing on disk or
    in any path is renamed.
    """
    m = re.fullmatch(r"layer_0*(\d+)", dirname)
    return f"layer {int(m.group(1))}" if m else dirname.replace("_", " ")


def pcfg_runs():
    base = C.OUT_DIR / "pcfg-matryoshka"
    return [(d.name, _json(d / "metrics_report.json"), _json(d / "second_pass.json"))
            for d in sorted(base.glob("layer_*")) if (d / "metrics_report.json").exists()]


# --- LaTeX helpers ----------------------------------------------------------
def esc(t) -> str:
    t = str(t)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    return t


TWOCOLUMN = False       # set from --twocolumn; the ICLR template is onecolumn


def wrap(text, frac: float) -> str:
    r"""A table cell that wraps, using only the LaTeX kernel.

    The obvious way to wrap a cell is a `p{}` column, and the obvious way to stop
    it justifying is `>{\raggedright\arraybackslash}`. Both need the `array`
    package, and neither fails politely when it is absent: `\arraybackslash`
    reports "control sequence never \def'ed" while leaving `\\` redefined, so
    rows stop terminating and the last column runs off the page; `>` reports
    "Illegal character in array arg". This generator cannot see the preamble it
    will be pasted into, so it must not depend on a package it cannot check --
    and a hand-off .tex that needs a preamble edit to compile is a .tex that
    arrives broken.

    `\parbox` is in the kernel. `[t]` puts the cell's first line on the row's
    baseline, which is what a `p{}` column does too, and `\raggedright` inside
    the box is scoped to the box, so it never touches the tabular's `\\`. The
    trailing `\strut` keeps single-line and multi-line rows the same height.
    """
    return rf"\parbox[t]{{{frac}\linewidth}}{{\raggedright {text}\strut}}"


def table(label, caption, header, rows, align=None, star=False, note=None,
          size="small"):
    r"""One booktabs table, caption ABOVE the rules.

    Caption above is the convention for tables and below for figures, and it is
    what `iclr2026_conference.tex` does in its own example.

    `star` asks for a full-width float, which only exists in a twocolumn class.
    The ICLR template is `\documentclass{article}` -- onecolumn -- where `table*`
    is legal but pointless and interacts badly with float placement, so it
    degrades to `table` unless --twocolumn says otherwise.
    """
    env = "table*" if (star and TWOCOLUMN) else "table"
    align = align or ("l" + "r" * (len(header) - 1))
    # `label` may be several names: a table that grew to cover more than its
    # original scope keeps its old label as an alias, so a \ref written against
    # the narrower table still resolves instead of printing "??".
    labels = [label] if isinstance(label, str) else list(label)
    L = [rf"\begin{{{env}}}[tbp]", r"  \centering", rf"  \{size}",
         rf"  \caption{{{caption}}}"]
    L += [rf"  \label{{tab:{n}}}" for n in labels]
    L += [rf"  \begin{{tabular}}{{{align}}}", r"    \toprule",
          "    " + " & ".join(header) + r" \\", r"    \midrule"]
    # a plain string is emitted verbatim: that is how a grouped table gets its
    # \addlinespace and its section headings without a second table builder
    L += ["    " + (r if isinstance(r, str)
                    else " & ".join(str(c) for c in r) + r" \\") for r in rows]
    L += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        L.append(rf"  \\[2pt] \footnotesize {note}")
    L.append(rf"\end{{{env}}}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 1. Setup. Everything a reader needs to reproduce a number, from config.
# ---------------------------------------------------------------------------
def t_setup(layers, has_bos=True):
    b = C.BLOCK_RANGES
    tok = layers[0][1]["total_tokens"] if layers else None
    rows = [
        ("Base model", rf"\texttt{{{esc(C.MODEL_NAME)}}}, residual stream"),
        ("Layers graded", ", ".join(str(L) for L, _ in layers) or "---"),
        ("SAE", rf"\texttt{{{esc(C.SAE_RELEASE)}}}, Matryoshka, $D = {C.D_SAE:,}$"),
        ("Nested blocks", ", ".join(f"B{i} $[{s},{e})$" for i, (s, e) in enumerate(b))),
        ("Corpus", rf"\texttt{{{esc(C.DATASET)}}}, {C.N_DOCS} docs, context {C.CONTEXT_SIZE}"),
        ("Tokens per layer", f"{tok:,} (BOS excluded)" if tok else "---"),
        ("Firing threshold", rf"$a_f > {C.FIRE_THRESHOLD:g}$ post-JumpReLU"),
        ("Edge criterion", rf"$R \geq {C.EDGE_TAU}$, fire $\geq {C.MIN_FIRE_COUNT}$, "
                           rf"co-fire $\geq {C.MIN_JOINT}$"),
        ("Reconstruction gate", rf"both relative gains $\geq {C.RECON_REL_GAIN_MIN}$"),
        (r"$S_\mathrm{res}$ rule", rf"both decoders in the top $k = {C.SRES_RANK_TOP_K}$"),
        ("Sibling flag", rf"mean pairwise Jaccard $\geq {C.SIBLING_REDUNDANCY_FLAG}$"),
        ("Frequency control", rf"survival $\geq {C.FREQ_SURVIVAL_MIN}$; buckets split at "
                              rf"{C.FREQ_HIGH_MASS:g}/{C.FREQ_MID_MASS:g} cumulative token mass"),
        ("Superparent gate", rf"fan-out $\geq {100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}\%$ "
                             "of the child block"),
    ]
    return table(
        "setup", r"\textbf{Setup.} Every threshold is global and identical for every SAE "
        "source: holding them fixed is what makes a cross-source comparison mean anything, and "
        "no threshold is tuned per source. The block structure is read from the cached statistics "
        "of the run being graded, not from this table, so a dictionary of a different shape is "
        "sliced correctly. Token counts exclude the beginning-of-sequence position"
        + (r" (Table~\ref{tab:bos})." if has_bos else "."),
        ["", ""], rows, align="ll")


# ---------------------------------------------------------------------------
# 2. The battery. Thresholds from config; the honest-name note is the point.
# ---------------------------------------------------------------------------
def t_battery():
    rows = [
        ("1a", "Reverse coverage $R$", r"$P(\text{parent} \mid \text{child})$",
         rf"$\geq {C.EDGE_TAU}$", r"\texttt{coverage}"),
        ("1b", "Forward coverage $F$", r"$P(\text{child} \mid \text{parent})$",
         "reported", r"\texttt{coverage}"),
        ("2", "Independence null", "PMI against independent firing",
         r"$> 0$", r"\texttt{independence\_null}"),
        ("3", "Token-frequency control", "does the edge survive on rare tokens",
         rf"$\geq {C.FREQ_SURVIVAL_MIN}$", r"\texttt{token\_control}"),
        ("4", "Reconstruction ablation", "both features carry mass on the child's tokens",
         rf"$\geq {C.RECON_REL_GAIN_MIN}$", r"\texttt{reconstruction}"),
        ("5", r"Probe $S_\mathrm{res}$", "both decoders near the child-concept probe",
         rf"top-{C.SRES_RANK_TOP_K} rank", r"\texttt{sres}"),
        ("6", "Out-degree", "one parent covering most of the next block",
         rf"$\geq {100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}\%$", r"\texttt{outdegree}"),
        ("7", "Sibling redundancy", "co-activation among one parent's children",
         rf"$\geq {C.SIBLING_REDUNDANCY_FLAG}$", r"\texttt{sibling\_redundancy}"),
        ("8", "Joint-child coverage", r"$R_\mathrm{supp}$, $R_\mathrm{mass}$, energy share",
         "reported", r"\texttt{joint\_child}"),
        ("---", "In-block directed coverage", "containment within one block, and duplicates",
         "asymmetry", r"\texttt{in\_block\_edges}"),
    ]
    return table(
        "battery", r"\textbf{The metric set.} Eight questions about the same edge, plus an "
        "unnumbered in-block measurement that grades same-level pairs rather than candidate "
        "edges. Metric~1 defines the candidate set and 2--8 grade it: 2--5 weigh a single edge "
        "with increasingly demanding evidence, from chance to frequency to reconstruction to "
        "meaning, and 6--8 widen the view from one edge to the parent's whole neighbourhood. "
        r"Metric~4 is named honestly as a \emph{contribution filter} rather than as Tree SAE's "
        r"$S_\mathrm{res}$: two strong but unrelated co-firing features pass both of its legs, "
        "and its pass rate scales with block energy. The probe-based $S_\\mathrm{res}$ is "
        "metric~5, and its target is the self-label "
        r"$\mathbf{1}[f_c > 0]$, so a corrupted latent yields a probe that validates its own "
        "corruption; we report it as self-labeled and as a self-consistency check, not as ground "
        "truth.",
        ["", "Metric", "What it asks", "Threshold", "Module"], rows,
        align="llllr", star=True)


# ---------------------------------------------------------------------------
# 3. The properties matrix. NOT derived -- see the module docstring.
# ---------------------------------------------------------------------------
# Column order matches Sec. "Edge properties": target first, then the eight
# pathologies -- Splitting, Absorption, Composition, Superparent,
# Multi-parenting, Siblings, Frequency, Topic. Multi-parenting is separate from
# Superparent on purpose: a superparent is an out-degree pathology (one parent
# swallowing the block) and multi-parenting an in-degree one (one child claimed
# by several parents); only `outdegree` reads in-degree at all. Composition (one
# latent encoding a conjunction of co-occurring concepts) is an open column like
# Topic: nothing in the battery tests atomicity. Out-degree gets partial credit
# there only because a composed child surfaces as multi-parented, a symptom the
# metric reports without naming its cause.
#
# Three cells re-graded against the Tier-1 scorecard (2026-09-01):
# Reconstruction x Superparent is Y, not x — rejecting superparent edges is that
# metric's designated calibration job and it rejects 33/33 with a wide margin.
# S_res x Absorption drops to x: an absorbed edge never reaches the candidate
# set (R = 0), and the self-labeled probe cannot grade what it never sees.
# S_res x Frequency drops Y -> P: the mechanism is plausible (a coincidental
# parent's decoder should rank low against the child probe) but no calibration
# case measures it; restore Y only once one does.
MATRIX = [
    ("1 Reverse coverage",        "P", "x", "x", "x", "x", "x", "x", "x", "x"),
    ("1 Forward coverage",        "P", "x", "x", "x", "P", "x", "x", "x", "x"),
    ("2 Independence null",       "P", "x", "x", "x", "Y", "P", "x", "Y", "x"),
    ("3 Frequency control",       "P", "x", "x", "x", "P", "x", "x", "Y", "x"),
    ("4 Reconstruction",          "P", "x", "x", "x", "Y", "x", "x", "P", "x"),
    (r"5 Probe $S_\mathrm{res}$", "Y", "x", "x", "x", "P", "P", "P", "P", "x"),
    ("6 Out-degree",              "x", "x", "x", "P", "Y", "Y", "x", "P", "x"),
    ("7 Sibling redundancy",      "P", "Y", "x", "x", "x", "x", "Y", "x", "x"),
    ("8 Joint-child",             "P", "Y", "x", "x", "Y", "x", "x", "x", "x"),
    ("9 In-block coverage",       "Y", "Y", "x", "x", "x", "x", "Y", "x", "x"),
]
GLYPH = {"Y": r"$\bullet$", "P": r"$\circ$", "x": r"--"}


def t_matrix():
    cols = ["Parent$\\rightarrow$child", "Splitting", "Absorption", "Composition",
            "Superparent", "Multi-parenting", "Siblings", "Frequency", "Topic"]
    rows = [[r[0]] + [GLYPH[c] for c in r[1:]] for r in MATRIX]
    # HUMAN-WRITTEN CAPTION. Transcribed from the manuscript (rewritten by hand,
    # 2026-08-31, with the target/pathologies grouping row and the new column
    # order). The Sec.~\ref{sec:pathologies} reference resolves only once the
    # pathologies subsection is in the paper -- it is, since the same revision.
    # Edit the paper and this string together, or neither.
    out = table(
        "matrix", r"Mapping of evaluation metrics to the specific failure cases they are "
        r"designed to identify, defined in Sec.~\ref{sec:pathologies}. Detection capability "
        r"is measured by linear separability, where ``$\bullet$'' indicates successful "
        r"detection, ``$\circ$'' indicates partial detection, and ``--'' indicates that the "
        r"metric is blind to the failure.",
        [""] + cols, rows, align="l" + "c" * len(cols), star=True)
    # a second header row (target vs pathologies) that the shared table() helper
    # has no slot for: injected between \toprule and the column names
    group = ("     & \\multicolumn{1}{c}{target} & \\multicolumn{8}{c}{pathologies} \\\\\n"
             "    \\cmidrule(lr){2-2}\\cmidrule(lr){3-10}")
    out = out.replace("    \\toprule", "    \\toprule\n" + group, 1)
    # ten columns overflow the onecolumn ICLR text width and LaTeX will not
    # break a tabular -- shrink to \linewidth (needs graphicx, which the ICLR
    # template loads)
    out = out.replace("  \\begin{tabular}",
                      "  \\setlength{\\tabcolsep}{4pt}\n"
                      "  \\resizebox{\\linewidth}{!}{%\n  \\begin{tabular}", 1)
    return out.replace("  \\end{tabular}", "  \\end{tabular}}", 1)


# ---------------------------------------------------------------------------
# 4. The tiers.
# ---------------------------------------------------------------------------
def t_tiers(toy, tt, align_json):
    r"""The ladder: three rungs, ground truth traded against realism.

    **The PCFG SAE is not a rung here.** It used to be Tier 3, and it was the one
    row that could not do what the column headings promise: the ladder's argument
    is that each rung licenses the one above it by scoring the battery against a
    known answer, and the PCFG run has no such score. Its grammar is known, but
    nothing maps a latent to a grammar symbol, so its result cell reported the same
    battery outputs as the released-SAE rung -- a row that consumed a quarter of
    the table to say "same as the rung below, without the ground truth".

    Two claims it made before that had already been withdrawn, kept here because a
    reader of an older draft has them: it did NOT "keep the known tree" (Tier 2's
    tree is a clean 20-feature, 9-edge hierarchy; the PCFG's is
    document/section/paragraph/sentence over S-V-O roles -- different structures,
    with the word "tree" doing double duty), and it did NOT "isolate base-model
    dependence" (grammar, base model, corpus and dictionary size all move together,
    so it bounds that contribution rather than isolating it).

    The PCFG runs have not left the paper: Table~\ref{tab:sources} and
    Table~\ref{tab:layers} carry them, which is where a source without a ground
    truth belongs -- beside the other source the battery is run on, not on a ladder
    of things scored against known answers. The helper that would give it a score
    exists (`pcfg_bridge.grammar.vocab.role_of` in the PCFG repo), so this is a
    rung that could be built, not one that is ruled out.
    """
    n_pass = sum(r["pass"] for r in toy) if toy else None
    # Counted from the qualitative reports, not written down: it was "40" from
    # when five layers had been read, and a sixth was published under it.
    n_surv = 0
    for q in sorted((C.OUT_DIR / C.SOURCE_NAME).glob("layer_*/qualitative_check.json")):
        rep = _json(q) or {}
        n_surv += sum(1 for rows in rep.values() for r in rows
                      if r.get("category") == "survivor")
    # The PCFG runs are still gathered by the caller and still reported -- in
    # Table~\ref{tab:sources} and Table~\ref{tab:layers} -- but they no longer build
    # a row here, so nothing is computed for one.
    # Columns 2-4 hold sentences, so each is a \parbox rather than an l column
    # that would run off the page. Fractions sum to 0.65, leaving 0.35 for the
    # Tier column and the eight \tabcolsep gaps.
    W = (0.22, 0.18, 0.25)
    rows = [
        ("1 Synthetic toy", "hand-built statistics", "by construction",
         rf"{n_pass}/{len(toy)} rows, 21/21 functions" if toy else "---"),
        ("2 Trained toy", "an SAE trained on a known tree", "the tree is known",
         rf"precision {tt['precision']:.2f}, recall {tt['recall']:.2f}" if tt else "---"),
        # "Released", not "Real": every rung here is a real SAE, and calling only
        # this one real would demote the others. What sets it apart is that we did
        # not train it. The release id lives in the note, not the cell: in \texttt
        # it is a single unbreakable 29-character token, wider than any sensible
        # column, and TeX does not shrink an overfull box -- it prints it straight
        # across the next column. Which is what it did.
        ("3 Released SAE", rf"\texttt{{{esc(C.MODEL_NAME)}}}", "none --- human reading",
         f"{n_surv} survivors read against autointerp labels" if n_surv else "---"),
    ]
    rows = [[r[0]] + [wrap(c, w) for c, w in zip(r[1:], W)] for r in rows]
    note = " ".join(t for t in [
        rf"Tier~3's dictionary is the released \texttt{{{esc(C.SAE_RELEASE)}}}.",
        (rf"Lateral control (not a tier): {align_json['n_respected']}/"
         rf"{align_json['n_testable']} testable true edges run early block "
         r"$\rightarrow$ late on the trained toy." if align_json else None),
    ] if t) or None
    # Tier 2's own size, from its calibration file. The two toys are NOT the same
    # world in two conditions, which "closes exactly that gap" used to imply:
    # Tier 1 grades a pathology-injected world and Tier 2 a clean, smaller tree,
    # so the world changes along with the statistics. Same shape of overclaim as
    # Tier 3's withdrawn "isolating", one rung down -- softer, because it was
    # carried by one word rather than asserted.
    tier2 = ("on a simpler toy"
             if not tt else
             rf"on a clean {tt['n_features']}-feature tree with "
             rf"{len(tt['true_edges'])} edges and no injected pathologies")
    return table(
        "tiers", r"\textbf{Three tiers, trading ground truth against realism.} Each rung "
        "licenses the one above it: Tier~1 proves the arithmetic and nothing about whether "
        rf"an SAE would learn such a structure. Tier~2 attacks that gap, but {tier2} rather "
        "than on the pathology-injected world Tier~1 grades, so it changes the world as well "
        "as the statistics and does not isolate training as the single moving part either. "
        "What it does isolate cleanly is blame: only what the SAE actually learned reaches the "
        "metrics, so a missed edge counts against a metric only if the SAE learned both "
        r"endpoints. Tier~3 is named \emph{released} rather than \emph{real}: the rungs below "
        "are real SAEs too, really trained, and what sets this one apart is that it is a "
        "published checkpoint we did not train. That is a constraint and not only a label --- "
        r"we do not control its dictionary, which is why the $S_\mathrm{res}$ column of "
        r"Table~\ref{tab:gemma} exists for one layer only. It has no known answer at all. "
        "A fourth source, a Matryoshka SAE over a transformer trained on a PCFG corpus, runs "
        "through the same metrics but is not a rung: its grammar is known and nothing yet maps "
        "a latent to a grammar symbol, so it has no recovery score to license anything. It is "
        r"reported beside gemma in Table~\ref{tab:sources} and Table~\ref{tab:layers}.",
        ["Tier", "Runs on", "Ground truth", "Result"], rows,
        # p{} rather than l: Tier 3's cells are sentences, and in an l column a
        # sentence does not wrap -- it runs off the page edge, and LaTeX only
        # warns about the overfull box, so nobody notices until the PDF.
        # Fractions of \linewidth rather than centimetres, because the widths
        # have to hold under whatever \textwidth the class sets and under
        # --twocolumn, where the float becomes table* and \linewidth changes.
        # 0.70 for the three, leaving the l column and eight \tabcolsep gaps
        # inside the remaining 0.30.
        # Plain l columns: the wrapping lives in each cell's \parbox, so this
        # table needs booktabs and nothing else. See wrap().
        align="llll", star=True, note=note)


# ---------------------------------------------------------------------------
# 5. Tier-1 scorecard.
# ---------------------------------------------------------------------------
def t_tier1(toy):
    order = sorted(toy, key=lambda r: (r.get("margin_kind") == "categorical", -r["margin"]))
    rows = []
    for r in order:
        cat = r.get("margin_kind") == "categorical"
        m = "---" if cat else (r">1000\times$" if r["margin"] >= 1000 else
                               rf"{r['margin']:.1f}$\times$")
        m = r"$>1000\times$" if (not cat and r["margin"] >= 1000) else m
        rows.append([esc(r["metric"]), esc(r["job"]),
                     r"\checkmark" if r["pass"] else r"$\times$", m])
    n_pass = sum(r["pass"] for r in toy)
    return table(
        "tier1", rf"\textbf{{Tier 1: every metric against a known tree.}} {n_pass} of {len(toy)} "
        "rows pass. Margin is how decisively the metric separated the "
        "class it must keep from the class it must reject; rows scored categorically have no "
        "margin, because their answer is right or it is not and a value of $1.0$ on a ratio scale "
        "would read as no separation. The last two rows are negative controls that pass when the "
        "metrics do \\emph{not} act.",
        ["Row", "The job it claims", "", "Margin"], rows, align="llcr", star=True)


# ---------------------------------------------------------------------------
# 6. The gemma result, per layer.
# ---------------------------------------------------------------------------
def t_gemma(layers, second, has_bos=True, pcfg=()):
    """Every graded block pair, at every graded layer, on every source.

    It reported B0->B1 on gemma alone, which is one cell of the grid the battery
    actually produced: three more adjacent pairs per gemma layer, seven per PCFG
    layer, and the S_res column filled for every one of them rather than for the
    outermost pair. A table that shows one pair cannot be read against the claim
    the paper makes, which is about WHERE in the block structure the pathology
    sits -- one row can neither support that claim nor contradict it.

    Rows are grouped (source, block pair, layer) and the pair is printed once per
    group, the same grouping the matrix figure uses; the two sources are sections
    of one table rather than two tables because every threshold is identical
    across them and it is that identity the table is evidence for.
    """
    _sp = [L for L, _ in layers if second.get(L)]

    def block_pairs(report):
        """Adjacent pairs the run's own block structure declares, not the ones it filled.

        A pair no layer here graded is a row carrying the reason. Dropping it would let
        the table read as if the block structure ended where the runs did.
        """
        return list(range(len(report.get("block_ranges") or []) - 1))

    def cells(q, sp):
        s = "---"
        if sp and sp.get("n_edges_scored"):
            s = rf"{sp['n_pass']}/{sp['n_edges_scored']:,}"
        if q["n_candidate_edges"] == 0:
            # nothing proposed is a measurement; the columns downstream of it
            # have nothing to report and say so rather than printing 0\%
            return ["0"] + ["---"] * 5
        return [f"{q['n_candidate_edges']:,}",
                rf"{100 * q['reconstruction']['frac_pass']:.0f}\%",
                rf"{100 * q['independence_null']['frac_chance_level']:.0f}\%",
                rf"{100 * q['freq_control']['frac_freq_driven']:.1f}\%"
                if q.get("freq_control") else "---",
                rf"{100 * q['degree']['poly_frac']:.0f}\%",
                s]

    NC = 8

    def section(title, runs):
        """One source's block: a heading row, then pair-major (pair, layer) rows."""
        out = [rf"\multicolumn{{{NC}}}{{l}}{{\textbf{{{title}}}}} \\", r"\addlinespace[1pt]"]
        ranges = next((r.get("block_ranges") for _, r in runs if r.get("block_ranges")), None)
        for k in block_pairs(runs[0][1]) if ranges else []:
            pr, name = f"{k}->{k + 1}", rf"B{k}$\rightarrow$B{k + 1}"
            if k:
                out.append(r"\addlinespace[2pt]")
            filled = [(lab, r) for lab, r in runs if _pair(r, pr)]
            if not filled:
                P = ranges[k][1] - ranges[k][0]
                Cn = ranges[k + 1][1] - ranges[k + 1][0]
                out.append([name, "---",
                            rf"\multicolumn{{{NC - 2}}}{{c}}{{\emph{{not graded at any "
                            rf"layer shown: {P:,}$\times${Cn:,}}}}}"])
                continue
            for i, (lab, r) in enumerate(filled):
                q = _pair(r, pr)
                sp = ((r.get("second_pass") or {}).get(pr) or {}).get("sres")
                out.append([name if i == 0 else "", lab] + cells(q, sp))
        return out

    rows = section(rf"\texttt{{gemma-2-2b}} --- $D = {C.D_SAE:,}$, "
                   rf"{len(C.BLOCK_RANGES)} nested blocks",
                   [(f"L{L}", r) for L, r in layers])
    if pcfg:
        d_pcfg = pcfg[0][1].get("block_ranges", [[0, 0]])[-1][1]
        nb = len(pcfg[0][1].get("block_ranges") or [])
        rows.append(r"\addlinespace[3pt] \midrule \addlinespace[2pt]")
        rows += section(rf"PCFG Matryoshka --- $D = {d_pcfg:,}$, {nb} nested blocks",
                        [(_layer_name(n).replace("layer ", "L"), r) for n, r, _ in pcfg])

    if len(_sp) == len(layers):
        sres_note = (r" The $S_\mathrm{res}$ pass has been run on every graded layer of both "
                     r"sources and on every pair, not on the outermost one alone; its rates "
                     r"are only comparable against the shared null of Table~\ref{tab:null}, "
                     "because a top-$k$ rank rule is only as strict as the dictionary is "
                     "large.")
    elif _sp:
        sres_note = (r" The $S_\mathrm{res}$ column is filled for layer"
                     + ("s~" if len(_sp) > 1 else "~")
                     + ", ".join(str(L) for L in _sp)
                     + " alone: stage~03 needs a cached token stream that the released "
                     "statistics do not carry, so a blank cell means the pass has not been "
                     "run there, not that nothing passed.")
    else:
        sres_note = r" No layer has had the $S_\mathrm{res}$ pass (stage~03) yet."

    n_gem = sum(1 for k in range(len(C.BLOCK_RANGES) - 1)
                for _, r in layers if _pair(r, f"{k}->{k + 1}"))
    n_pcf = sum(1 for _, r, _ in pcfg for q in r["pairs"])
    return table(
        ["results", "gemma"],
        r"\textbf{Every graded block pair, at every graded layer, on both SAE sources.} "
        rf"{n_gem} graded pairs on \texttt{{gemma-2-2b}}"
        + (rf" and {n_pcf} on the PCFG Matryoshka SAEs" if pcfg else "") + ", under one "
        "threshold set that is tuned to neither (Table~\\ref{tab:setup}). Read left to right: "
        "coverage proposes thousands of edges, the reconstruction filter keeps most of them, "
        "and the independence null rejects most of them --- the over-connection is what the "
        "parent's own firing rate already forces, not capture of frequent tokens, which the "
        "frequency control puts at 1--2\\% in the outermost pair. Read down instead and the "
        "claim is locational: multi-parenting sits at the coarsest boundary "
        "B0$\\rightarrow$B1 at every layer of both sources, while the deeper pairs fail the "
        "frequency control instead. Multi-parenting is the measure that does not depend on "
        "the candidate set and the only one of the project's original four claims to survive "
        r"BOS exclusion"
        + (r" (Table~\ref{tab:bos})" if has_bos else "") + ". A pair no layer here "
        "graded carries its reason rather than being dropped, a layer that did not run a "
        "pair others ran shows a dash, and a pair that proposed nothing shows a candidate "
        "count of zero with the columns downstream of it blank: none of the three is the "
        "same statement as a clean pair." + sres_note,
        ["Pair", "Layer", "Candidates", "Recon.", "At chance", "Freq.-driven",
         r"$\geq 2$ parents", r"$S_\mathrm{res}$"],
        rows, align="ll" + "r" * 6, star=True, size="footnotesize")


# ---------------------------------------------------------------------------
# 7. Cross-source (appendix).
# ---------------------------------------------------------------------------
def t_sources(layers, second, pcfg):
    # layers is replaced below with the unmerged set: see gemma_layers(merge=...).
    """One row per graded RUN, summed over every block pair that run graded.

    It carried gemma's layer 6 against the PCFG runs, at B0->B1 only, which made
    a cross-source table out of two cells. Two things were wrong with that: five
    of gemma's six graded layers were absent, so the row that represented gemma
    was chosen rather than measured; and reading only the outermost pair left the
    deeper ones -- where the two sources differ most -- out of the comparison
    entirely.

    Every column is now a whole-run total over the pairs that run graded, which
    is this table's own altitude. The per-pair breakdown behind these totals is
    Table~\\ref{tab:results}: this one answers "is it the same battery on both
    sources", that one answers "where in the block structure".
    """
    def run_row(label, r, sp_json=None):
        cfg = r.get("config") or {}
        d = cfg.get("d_sae") or (r.get("block_ranges") or [[0, 0]])[-1][-1]
        nb = len(r.get("block_ranges") or [])
        sp = sp_json or r.get("second_pass") or {}
        cand = recon = sres_pass = sres_n = 0
        for q in r["pairs"]:
            cand += q["n_candidate_edges"]
            recon += q["reconstruction"]["n_pass"]
            s = (sp.get(q["pair"]) or {}).get("sres") or {}
            sres_pass += s.get("n_pass", 0)
            sres_n += s.get("n_edges_scored", 0)
        return [label, f"{d:,}", str(nb), f"{r['total_tokens']:,}",
                str(len(r["pairs"])), f"{cand:,}",
                rf"{100 * recon / cand:.0f}\%" if cand else "---",
                rf"{sres_pass}/{sres_n:,}" if sres_n else "---"]

    # Whole-run totals only pool pairs every row could have. A variant run graded one
    # extra pair at one layer; including it here would move that row's percentages
    # without any measurement having changed.
    layers = gemma_layers(merge=False)
    rows = [run_row(rf"\texttt{{gemma-2-2b}} L{L}", r, second.get(L)) for L, r in layers]
    if layers and pcfg:
        rows.append(r"\addlinespace[2pt]")
    rows += [run_row(f"PCFG {esc(_layer_name(name))}", r, sp) for name, r, sp in pcfg]

    n_pairs = sum(len(r["pairs"]) for _, r in layers) + sum(len(r["pairs"]) for _, r, _ in pcfg)
    return table(
        "sources", r"\textbf{The same metric set across SAE sources, every graded run.} "
        f"All {len(layers) + len(pcfg)} graded runs, and within each run every block pair it "
        f"graded --- {n_pairs} pairs in total, not the outermost one alone. Identical metric "
        "code and identical global thresholds; only the block structure follows the source, "
        "read from the statistics file being graded. Candidates, the reconstruction rate and "
        r"the $S_\mathrm{res}$ column are whole-run totals over those pairs, so a run is "
        "summarised rather than represented by one of its pairs; the per-pair breakdown is "
        r"Table~\ref{tab:results}. The reconstruction column should be read with care on "
        "PCFG: the weakest candidate edge there sits $3.5\\times$ above the threshold, so the "
        "filter is inert on that source and its surviving edges have passed coverage alone. "
        r"$S_\mathrm{res}$ pass rates are not comparable between rows of different $D$ --- see "
        r"Table~\ref{tab:null}.",
        ["Run", "$D$", "Blocks", "Tokens", "Pairs", "Candidates", "Recon.",
         r"$S_\mathrm{res}$"],
        rows, star=True)


# ---------------------------------------------------------------------------
# 7a. gemma against PCFG, layer by layer (appendix). The companion to
#     `cross_source_layer_response`; both read `reporting/cross_source.py`, so
#     the figure's gaps and this table's rows cannot disagree.
# ---------------------------------------------------------------------------
def t_layers(rs):
    """Every graded run of both sources on one axis, ordered by relative depth.

    Sorted by (L+1)/N rather than grouped by source, so the two interleave and a
    reader can see for themselves that the alignment is a choice: PCFG's layer 1
    is the toy's half-way point and lands beside gemma's layer 12, while it is
    also literally layer 1 and gemma has one of those too. Both keys are printed
    for that reason.

    Nothing here is a mean over layers. The figure reduces this table to six gap
    numbers, and a table whose job is to let someone check that reduction has to
    carry the rows it was computed from.
    """
    by_layer = X.matched(rs, "layer")
    shared = sorted({o["layer"] for o, _, _ in by_layer})
    rows = []
    for r in rs:
        rows.append([
            ("PCFG" if not r["is_ref"] else r"\texttt{gemma-2-2b}") + f" L{r['layer']}",
            f"{r['layer']}/{r['n_layers']}",
            f"{r['depth']:.2f}",
            f"{r['tokens']:,}",
            f"{r['n_cand']:,}",
            rf"{X.density(r):.2f}\%",
            rf"{100 * r['poly']:.0f}\%",
            f"{r['sib']:.3f}",
            f"{r['gini']:.3f}",
            rf"{100 * r['recon']:.0f}\%",
            rf"{100 * r['chance']:.0f}\%",
        ])
    # Both facts a reader needs to audit the restriction and the denominator,
    # computed rather than asserted.
    beyond = "; ".join(f"{r['label']} {r['beyond_b0b1']:,}" for r in rs if not r["is_ref"])
    dens_pairs = sorted({r["n_pairs"] for r in rs})
    note = (rf"Restricted to block pair B0$\rightarrow$B1. Candidate edges in \emph{{all}} "
            rf"deeper pairs combined, on the PCFG runs: {beyond}---so no deeper pair supports "
            rf"a comparison. Both PCFG rows are one grammar configuration "
            rf"({X.grammar_line(rs)}), a single point of the three-axis sweep Exp 2 specifies, "
            rf"so this is gemma against one PCFG corpus rather than against PCFG. "
            rf"Density is a share of every ordered B0$\times$B1 feature pair, "
            rf"{dens_pairs[0]:,} on one source and {dens_pairs[-1]:,} on the other "
            rf"({100 * (dens_pairs[-1] / dens_pairs[0] - 1):.0f}\% apart), which is coincidence "
            r"and not design: it is what makes the two densities directly comparable rather "
            r"than merely each normalised by itself.")
    return table(
        "layers", r"\textbf{The same metric set on two base models, layer by layer.} Rows are "
        r"ordered by position in their own network, $(L{+}1)/N$, so the sources interleave; "
        rf"layer{'s' if len(shared) > 1 else ''} "
        + " and ".join(str(s) for s in shared) + " carry an SAE on both, which is where the "
        "comparison needs no matching argument at all. The token budgets differ by "
        rf"{max(r['tokens'] for r in rs) / min(r['tokens'] for r in rs):.0f}$\times$; that cuts "
        r"\emph{against} the density gap rather than explaining it, because the joint-support "
        r"guard is an absolute count and more tokens make it easier to clear, yet the source "
        r"with more tokens is the sparser one. What the two agree on is the \emph{shape} of the "
        r"relation---multi-parenting, sibling redundancy and the frequency control---and what "
        r"they do not is its \emph{strength}: density, the reconstruction filter and the "
        r"base-rate null. That split should be read with Table~\ref{tab:align}, which shows the "
        r"three agreeing measures are also the three sitting against a boundary on both sources, "
        r"where agreement is close to free.",
        ["Run", "$L/N$", r"$\frac{L+1}{N}$", "Tokens", "Cand.", "Density",
         r"$\geq 2$ par.", "Sib.", "Gini", "Recon.", "At chance"],
        rows, align="lrrrrrrrrrr", star=True, note=note)


# ---------------------------------------------------------------------------
# 7b. What the pairing in 7a rests on (appendix).
# ---------------------------------------------------------------------------
def t_align(rs):
    """Same block index or same fraction of the network -- measured, not assumed.

    The two rules pair PCFG's layers with opposite ends of gemma, so the choice
    changes what the comparison says and is worth a table of its own rather than
    a sentence in someone else's caption.
    """
    by_layer, by_depth = X.matched(rs, "layer"), X.matched(rs, "depth")
    rows = []
    for title, key, g_layer in X.ranked(by_layer):
        rows.append([X.tex(title), f"{g_layer:.1f}", f"{X.gap(by_depth, key):.1f}",
                     f"{X.spread(rs, key):.1f}", f"{X.gap_ratio(rs, by_layer, key):.2f}"])
    ml = sum(g for _, _, g in X.ranked(by_layer)) / len(X.PANELS)
    md = sum(X.gap(by_depth, k) for _, k in X.PANELS) / len(X.PANELS)
    rows.append([r"\textbf{mean}", rf"\textbf{{{ml:.1f}}}", rf"\textbf{{{md:.1f}}}", "", ""])
    # The measure the two readings disagree about most sharply, named from the
    # data rather than from what the last run happened to show.
    by_pts = [t for t, _, _ in X.ranked(by_layer)]
    by_ratio = sorted(X.PANELS, key=lambda p: X.gap_ratio(rs, by_layer, p[1]))
    flip = X.tex(max(X.PANELS, key=lambda p: by_pts.index(p[0]) - by_ratio.index(p))[0])
    return table(
        "align", r"\textbf{What the cross-source comparison rests on, and how far to trust it.} "
        "Two questions, both of which are usually settled silently. \\emph{First}, comparing base "
        "models of different depth needs a rule for pairing their layers, and the two defensible "
        "rules pick opposite ends of gemma: by block index PCFG's layers pair with gemma's "
        + " and ".join(f"L{q['layer']}" for _, q, _ in by_layer) + ", by relative depth "
        r"$(L{+}1)/N$ with gemma's "
        + " and ".join(f"L{q['layer']}" for _, q, _ in by_depth) + rf". Block index wins on the "
        rf"mean ({ml:.1f} points against {md:.1f}), which is why it is what "
        r"Figure~\ref{fig:cross-source-layer-response} draws---though the margin rests largely "
        r"on one measure, and the reconstruction filter is inert on PCFG in any case. "
        r"\emph{Second}, a small gap is only impressive if the measure could have been far "
        r"apart. The last two columns ask that: \emph{range} is how far each measure moves "
        "across all "
        rf"{len(rs)} graded runs, and the ratio is the gap against it. The two readings do not "
        rf"agree---{flip} is near the bottom on raw points and near the top on the ratio---so "
        "both are printed and neither is called correct. The honest summary is that the three "
        r"measures with the smallest raw gaps are also the three pinned against a boundary "
        r"($\geq 2$ parents at 89--100\%, sibling redundancy at 3--6\%, the frequency control at "
        r"0--2\%), where agreement costs the two sources nothing. \textbf{On "
        rf"{len(by_layer)} shared layers none of this is a result." + "}",
        ["Measure", "Gap by layer index", "Gap by rel.\\ depth", "Range over all runs",
         "Gap / range"],
        rows, align="lrrrr", star=True)


# ---------------------------------------------------------------------------
# 8. The k/D null (appendix).
# ---------------------------------------------------------------------------
def t_null(layers, second, pcfg):
    k = C.SRES_RANK_TOP_K
    rows = [["Synthetic toy (Tier 1)", "42", rf"{100 * k / 42:.1f}\%", "1", "---",
             "---", "---"],
            ["Trained toy (Tier 2)", "20", rf"{100 * k / 20:.1f}\%", "1", "---",
             "5/5 true edges", "---"]]
    def whole_run(r, sp_json):
        """Passes and edges scored over EVERY pair the run probed, not the outermost.

        The observed rate used to be read off B0->B1 alone, which made a table
        about a null that depends on the dictionary depend on a choice of block
        pair as well. Every pair is probed against the same dictionary, so every
        pair belongs in the same rate.
        """
        sp = sp_json or r.get("second_pass") or {}
        n_pass = n_scored = n_pairs = 0
        for q in r["pairs"]:
            sr = (sp.get(q["pair"]) or {}).get("sres") or {}
            if sr.get("n_edges_scored"):
                n_pairs += 1
            n_pass += sr.get("n_pass", 0)
            n_scored += sr.get("n_edges_scored", 0)
        return n_pass, n_scored, n_pairs

    # Whole-run rates, like t_sources: pool only pairs every row could have. The
    # variant run that added B3->B4 at layer 12 carries no second_pass today, so this
    # table does not move either way; it takes the unmerged set so it still will not
    # once the probe stage is run there.
    obs = []
    for L, r in gemma_layers(merge=False):
        n_pass, n_scored, n_pairs = whole_run(r, second.get(L))
        if n_scored:
            obs.append((rf"\texttt{{gemma-2-2b}} L{L}", C.D_SAE,
                        100 * n_pass / n_scored, n_pairs, n_scored))
    for name, r, sp in pcfg:
        n_pass, n_scored, n_pairs = whole_run(r, sp)
        if n_scored:
            d = (r.get("config") or {}).get("d_sae") or 1792
            obs.append((f"PCFG {esc(_layer_name(name))}", d,
                        100 * n_pass / n_scored, n_pairs, n_scored))
    for name, d, o, n_pairs, n_scored in obs:
        null = 100 * k / d
        rows.append([name, f"{d:,}", rf"{null:.3f}\%", str(n_pairs), f"{n_scored:,}",
                     rf"{o:.2f}\%",
                     rf"{o / null:.0f}$\times$" if o > 0 else "below chance"])
    return table(
        "null", rf"\textbf{{The rank rule's null depends on dictionary size.}} $S_\mathrm{{res}}$ "
        rf"passes an edge when both decoders fall in the top $k = {k}$ of the probe's "
        "correlations over the whole dictionary. It is a geometry test, so an unrelated parent "
        "passes whenever chance puts it there, at a rate of $k/D$. The same $k$ is therefore two "
        "orders of magnitude stricter on gemma than on a 42-feature toy, and a measured pass rate "
        "is only interpretable against its own null: two runs on the \\emph{same} 1{,}792-latent "
        "dictionary land on opposite sides of theirs. Reporting $S_\\mathrm{res}$ shares across "
        "sources without this correction compares nothing. The observed rate is over "
        r"\emph{every} block pair the run probed, not its outermost pair alone: each pair "
        "is scored against the same dictionary, so each belongs in the same rate, and the "
        "pair and edge counts behind it are printed so a rate resting on a handful of edges "
        r"is visible as one. The per-pair breakdown is Table~\ref{tab:results}.",
        ["Source", "$D$", "Null $k/D$", "Pairs", "Edges scored", "Observed",
         "vs.\\ null"], rows, star=True)


# ---------------------------------------------------------------------------
# 9. BOS before/after (appendix).
# ---------------------------------------------------------------------------
def t_bos(matched):
    """`matched` is [(current_report, archived_v1_report)] per shared layer."""
    def mean(reports, fn):
        return sum(fn(_pair(r)) for r in reports) / len(reports)

    cur, arch = [c for c, _ in matched], [a for _, a in matched]
    spec = [
        ("Candidate edges", lambda p: p["n_candidate_edges"], "{:,.0f}", True),
        (r"Improve reconstruction, \%", lambda p: 100 * p["reconstruction"]["frac_pass"],
         "{:.0f}", True),
        (r"Frequency-driven, \%", lambda p: 100 * p["freq_control"]["frac_freq_driven"],
         "{:.1f}", True),
        ("Mean frequency survival", lambda p: p["freq_control"]["mean_survival"], "{:.2f}", True),
        ("Sibling redundancy", lambda p: p["sibling_redundancy"]["mean_redundancy"],
         "{:.2f}", True),
        ("Superparents flagged", lambda p: p["n_superparents"], "{:.1f}", True),
        (r"Children with $\geq 2$ parents, \%",
         lambda p: 100 * (p["degree"].get("poly_frac") if p["degree"].get("poly_frac") is not None
                          else p["degree"]["n_multi_parented"]
                          / max(p["degree"]["n_children_with_parent"], 1)), "{:.0f}", False),
    ]
    rows = []
    for name, fn, fmt, shared in spec:
        a, b = mean(arch, fn), mean(cur, fn)
        rows.append([name, r"\checkmark" if shared else "--",
                     fmt.format(a), fmt.format(b), rf"{b / a:.2f}$\times$" if a else "---"])
    return table(
        "bos", r"\textbf{One contaminating token position moved five of six metrics.} Means over "
        rf"the {len(matched)} layers graded both before and after the fix, "
        "block pair B0$\\rightarrow$B1, before and after excluding the "
        "beginning-of-sequence position. BOS is an attention sink on which effectively every "
        "feature fires; with 400 documents it gave every pair in the dictionary 400 joint firings "
        r"against a support guard of $\mathrm{MIN\_JOINT} = 30$, so the guard admitted pairs that "
        "never co-occur anywhere else. Every quantity computed from that one co-firing matrix "
        "moved, and the one that is not --- a ratio over children that already have a parent --- "
        "did not. \\textbf{The \\emph{before} column is withdrawn} and is reproduced only to size "
        "the error. Agreement among detectors that share an input is much weaker evidence than "
        "their number implies.",
        ["Quantity", "Reads co-fire", "Before", "After", "Change"], rows, star=True)


# Tables the manuscript no longer carries, on the mentor's reading: their content
# belongs in prose, where a reader gets the reasoning rather than a grid to decode.
# Section 2.5 now states the battery and its thresholds in text, Section 3.1 names
# the three tiers in three lines, and Section 3.1.1 walks the Tier-1 scorecard as a
# numbered procedure. They are still buildable with --internal, because they remain
# useful for reading a run at a glance; what they are not is paper material, and a
# regeneration that silently put them back would undo an editorial decision.
WITHDRAWN = {
    "setup":   "withdrawn from the paper — Section 2.5 states the setup in prose (--internal to build)",
    "battery": "withdrawn from the paper — Section 2.5 states the battery in prose (--internal to build)",
    "tiers":   "withdrawn from the paper — Section 3.1 names the three tiers in text (--internal to build)",
    "tier1":   "withdrawn from the paper — Section 3.1.1 walks the scorecard as a procedure (--internal to build)",
}


# ---------------------------------------------------------------------------
def build(dry: bool, internal: bool = False):
    written, skipped, parts = [], [], []
    G = C.OUT_DIR / C.SOURCE_NAME
    layers = gemma_layers()
    # layer -> second_pass.json, for every layer that has one
    second = {L: _json(G / f"layer_{L:02d}" / "second_pass.json") for L, _ in gemma_layers()}
    second = {L: v for L, v in second.items() if v}
    toy = _json(C.OUT_DIR / "synthetic_toy_calibration.json")
    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    align = _json(C.OUT_DIR / "block_tree_alignment.json")
    pcfg = pcfg_runs()
    # Archived v1 reports matched BY LAYER to the current ones: a layer graded
    # only after the BOS fix has no counterpart and must not veto the pairs
    # that do exist. `matched` carries (current, archived) per shared layer.
    arch_by_L = {}
    for p in sorted((C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json")):
        arch_by_L.setdefault(int(p.parent.name.split("__")[0].split("_")[1]), _json(p))
    matched = [(r, arch_by_L[L]) for L, r in layers if L in arch_by_L]
    # Two tables cite Table~\ref{tab:bos}. It is only emitted when the archived
    # pre-BOS reports are present, so the citation has to be conditional too --
    # otherwise a clone without outputs_archive/ compiles with a dangling ref
    # printing '??', which is exactly the kind of quiet staleness this file
    # exists to prevent.
    has_bos = len(matched) >= 2

    def add(slot, name, ok, why, fn, section=None):
        if name in WITHDRAWN and not internal:
            skipped.append((name, WITHDRAWN[name]))
            return
        if not ok:
            skipped.append((name, why))
            return
        written.append(f"{slot:<7} {name:<10} <- {why}" if dry else name)
        if not dry:
            parts.append((slot, section, fn()))

    add("INT 1", "setup", bool(layers), "config + a graded layer's token count",
        lambda: t_setup(layers, has_bos), "Internal (withdrawn from the paper)")
    add("INT 2", "battery", True, "config thresholds", t_battery)
    add("MAIN 1", "matrix", True, "argued, not measured -- see the module docstring",
        t_matrix, "The instrument")
    add("INT 3", "tiers", bool(toy and tt), "toy + trained-toy calibration JSON"
        if (toy and tt) else "needs both calibration JSONs",
         lambda: t_tiers(toy, tt, align), "Internal (withdrawn from the paper)")
    add("INT 4", "tier1", bool(toy), f"{len(toy) if toy else 0} scorecard rows"
        if toy else "needs synthetic_toy_calibration.json", lambda: t_tier1(toy))
    add("MAIN 2", "gemma", bool(layers), f"{len(layers)} gemma layer reports"
        if layers else "needs gemma metrics_report.json",
        lambda: t_gemma(layers, second, has_bos, pcfg), "The result")
    add("APP 1", "sources", bool(layers and pcfg), f"gemma + {len(pcfg)} PCFG runs"
        if (layers and pcfg) else "needs a gemma report and a PCFG report",
        lambda: t_sources(layers, second, pcfg), "Appendix")
    # The layer-by-layer cross-source pair. `t_layers` needs one layer index
    # graded on both sources; `t_align` additionally needs the two pairing rules
    # to actually disagree, which they only do once the reference source has
    # layers the toy cannot reach — otherwise the table would be two identical
    # columns.
    xs = X.rows()
    x_shared = sorted({o["layer"] for o, _, _ in X.matched(xs, "layer")})
    add("APP 2", "layers", bool(x_shared),
        f"{len(xs)} graded runs, layer{'s' if len(x_shared) > 1 else ''} "
        f"{', '.join(str(s) for s in x_shared)} on both sources"
        if x_shared else "needs one layer index graded on both gemma and PCFG",
        lambda: t_layers(xs))
    add("APP 3", "align", X.alignment_gaps(xs) is not None,
        "the same runs under both pairing rules" if X.alignment_gaps(xs) is not None
        else "the two pairing rules select the same runs here, so the table is empty",
        lambda: t_align(xs))
    add("APP 4", "null", bool(second or pcfg), "measured pass rates + config"
        if (second or pcfg) else "needs at least one second_pass.json",
        lambda: t_null(layers, second, pcfg))
    add("APP 5", "bos", has_bos,
        f"{len(matched)} layers graded both before and after BOS exclusion"
        if arch_by_L else "needs the pre-BOS reports in outputs_archive/",
        lambda: t_bos(matched))
    n_mag = len(list((C.OUT_DIR / C.SOURCE_NAME).glob(
        "layer_*/edge_activation_magnitudes.json")))
    add("APP 6", "magnitudes", n_mag > 0,
        f"{n_mag} per-layer edge_activation_magnitudes.json reports"
        if n_mag else "needs edge_activation_magnitudes.json (validation/edge_activation_magnitudes.py)",
        magnitudes_table)
    return written, skipped, parts


HEAD = r"""% ===========================================================================
% Tables.
%
% Generated by reporting/make_report_tables.py. Every number was read
% from the JSON it describes at generation time; do not edit one by hand.
%
% Ordered as the paper reads: the instrument is defined before any number it
% produces is quoted. MAIN 1-6 are the main text, APP 1-3 the appendix.
%
% Requires: booktabs. Nothing else -- deliberately. Wrapping cells use
% \parbox, a kernel command, rather than array's >{...}p{} columns: this
% file cannot see the preamble it is pasted into, and a missing array
% package does not fail politely ("Illegal character in array arg", or
% rows that stop terminating and a last column off the page).
% Emitted for a ONECOLUMN class, which is what the ICLR
% template uses; pass --twocolumn for full-width table* floats.
% Captions sit above the rules, as tables take them and as the ICLR template's
% own example does.
%
% To compile on its own, prepend
%     \documentclass{article}\usepackage{booktabs}\begin{document}
% and append \end{document}.
% ===========================================================================
"""


def threshold_macros() -> str:
    r"""Three LaTeX \newcommand per tunable threshold, generated from config.

    For each constant NAME the paper gets \Name (its math symbol, e.g. \tau —
    what equations and prose use everywhere), \NameName (the constant's
    identifier in typewriter font — stated once at the symbol's first use, so
    a reader can find the exact knob in the released configuration) and
    \NameVal (its current value). Re-choosing a threshold in config.py
    re-numbers the paper on the next build instead of requiring prose edits.
    Copy the file beside the paper sources and \input it from the methodology
    section.
    """
    def num(x):
        if 0 < abs(x) < 1e-2:           # 1e-3 reads as a power of ten in math mode
            m, e = f"{x:e}".split("e")
            m = m.rstrip("0").rstrip(".")
            return (("" if m in ("1", "-1") else m + r" \times ")
                    + f"10^{{{int(e)}}}")
        return f"{x:g}"

    SPEC = [  # (config constant, math symbol)
        ("FIRE_THRESHOLD",          r"\theta"),
        ("EDGE_TAU",                r"\tau"),
        ("MIN_FIRE_COUNT",          r"n_{\mathrm{fire}}"),
        ("MIN_JOINT",               r"n_{\mathrm{co}}"),
        ("FREQ_HIGH_MASS",          r"\mu"),
        ("FREQ_SURVIVAL_MIN",       r"\sigma"),
        ("RECON_REL_GAIN_MIN",      r"\delta"),
        ("SRES_RANK_TOP_K",         r"k"),
        ("SUPERPARENT_OUTDEG_FRAC", r"\phi"),
        ("SIBLING_REDUNDANCY_FLAG", r"\rho"),
        ("SHARE_ENERGY_SPLIT",      r"\eta"),
    ]
    L = ["% thresholds_macros.tex — GENERATED from config.py by reporting/make_report_tables.py.",
         "% Do not edit by hand: re-run `python3 -m reporting.make_report_tables` after",
         "% changing a threshold, then copy this file beside the paper sources.",
         "% \\<Name> is the math symbol (use everywhere); \\<Name>Name is the constant's",
         "% identifier (state once at first use); \\<Name>Val is its current value."]
    for const, sym in SPEC:
        camel = "".join(w.capitalize() for w in const.split("_"))
        shown = const.replace("_", r"\_")
        L.append(f"\\newcommand{{\\{camel}}}{{\\ensuremath{{{sym}}}}}")
        L.append(f"\\newcommand{{\\{camel}Name}}{{\\texttt{{{shown}}}}}")
        L.append(f"\\newcommand{{\\{camel}Val}}{{{num(getattr(C, const))}}}")
    return "\n".join(L) + "\n"


def magnitudes_table():
    r"""Endpoint activation magnitudes on kept edges, one row per (layer, pair).

    The numbers behind two review questions: whether children activate more
    strongly than their parents, and whether a parent can be active yet too
    weak on its child's tokens for the reconstruction contribution filter to
    see it. Read from edge_activation_magnitudes.json (written by
    validation/edge_activation_magnitudes.py); emitted to its own file so the
    appendix can \input it directly. Returns None when no layer has the
    report yet.
    """
    paths = sorted((C.OUT_DIR / C.SOURCE_NAME).glob(
        "layer_*/edge_activation_magnitudes.json"))
    if not paths:
        return None
    rows = []
    for i, p in enumerate(paths):
        rep = json.loads(p.read_text())
        layer = int(p.parent.name.split("_")[1])
        if i:
            rows.append(r"\addlinespace")
        first = True
        for key, v in rep["pairs"].items():
            s = v["summary"]
            if not s["n_edges"]:
                continue
            rows.append((str(layer) if first else "",
                         key.replace("->", r"$\to$"),
                         f"{s['n_edges']:,}",
                         f"{s['child_over_parent']['p50']:.2f}",
                         rf"{100 * s['frac_child_stronger']:.0f}\%",
                         f"{s['parent_shared_over_all']['p50']:.2f}",
                         rf"{100 * s['frac_parent_halved_on_child']:.0f}\%"))
            first = False
    # TODO(caption, by hand): the definitions below are checked against
    # validation/edge_activation_magnitudes.py and can stand. What a reader
    # should take from the table is still yours to write and is not here.
    caption = (r"Activation strength at the two ends of each edge the metrics "
               r"kept, by layer and block pair. Shared tokens are the positions "
               r"where both features fire. Child/parent is the median over "
               r"edges of the child's mean activation on shared tokens divided "
               r"by the parent's. A value below $1$ means the child is the "
               r"weaker of the two. Child stronger is the share of edges where "
               r"this ratio is above $1$. Parent shared/all is the median over "
               r"edges of the parent's mean activation on shared tokens divided "
               r"by its mean over all positions where it fires. A value below "
               r"$1$ means the parent is weaker on its child's tokens than it "
               r"is in general. Parents halved is the share of edges where this "
               r"second ratio falls below $0.5$.")
    return table("magnitudes", caption,
                 ["Layer", "Pair", "Edges", "Child/parent", "Child stronger",
                  "Parent shared/all", "Parents halved"],
                 rows, align="llrrrrr")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the plan, write nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for tables.tex")
    ap.add_argument("--twocolumn", action="store_true",
                    help="emit table* full-width floats (needs a twocolumn class)")
    ap.add_argument("--internal", action="store_true",
                    help="also build the tables the paper withdrew (setup, battery, "
                         "tiers, tier1) — useful for reading a run, not paper material")
    args = ap.parse_args()
    globals()["TWOCOLUMN"] = args.twocolumn

    written, skipped, parts = build(args.list, internal=args.internal)
    if not args.list:
        args.out.mkdir(parents=True, exist_ok=True)
        L, seen = [HEAD], set()
        for slot, section, tex in parts:
            if section and section not in seen:
                seen.add(section)
                L += ["", r"% " + "-" * 73, f"% {section}", r"% " + "-" * 73]
            L += ["", f"% [{slot}]", tex]
        (args.out / "tables.tex").write_text("\n".join(L) + "\n")
        (args.out / "thresholds_macros.tex").write_text(threshold_macros())
        print(f"[tab] wrote {args.out / 'thresholds_macros.tex'}")
        mag = magnitudes_table()
        if mag is not None:
            (args.out / "magnitudes_table.tex").write_text(
                "% magnitudes_table.tex — GENERATED by reporting/make_report_tables.py\n"
                "% from outputs/gemma-2-2b/layer_*/edge_activation_magnitudes.json.\n"
                "% Do not edit by hand; the caption is a placeholder — reword it by hand\n"
                "% in this generator, then regenerate.\n" + mag + "\n")
            print(f"[tab] wrote {args.out / 'magnitudes_table.tex'}")
        else:
            print("[tab] SKIP magnitudes_table.tex — no edge_activation_magnitudes.json found")

    print(f"[tab] {'plan for' if args.list else 'wrote'} {args.out / 'tables.tex'}")
    for w in written:
        print(f"  ok    {w}")
    for name, why in skipped:
        print(f"  SKIP  {name}\n          {why}")
    if skipped and not args.list:
        print(f"\n[tab] {len(written)} written, {len(skipped)} skipped — the set above is not "
              "complete, and the reasons are printed rather than left to be noticed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
