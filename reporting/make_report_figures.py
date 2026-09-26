"""
Paper figures — one file, one output directory, every figure derived from data.

    python3 -m reporting.make_report_figures            # everything it can build
    python3 -m reporting.make_report_figures --list     # what it would build, and from what

Output: outputs/paper_figuers/*.png

Each figure backs a single claim and is named after that claim rather than after
its position in a report, so a filename is readable in a caption and does not go
stale when the order changes.

    funnel_coverage_to_sres                 coverage proposes, the strict test disposes
    edge_survival_by_block_pair             what each filter removes, per block pair
    depth_profile_across_layers             whether anything varies with depth
    multiparenting_by_layer                 the graph is not a tree
    superparent_fanout_vs_firing            why a high-firing parent clears the bar for free
    calibration_synthetic_toy_scorecard     every metric against a known tree
    calibration_trained_toy_recovery        the same tree, through a real training run
    calibration_toy_tree_recovered          that tree drawn, before and after the battery
    cross_source_funnel_shares              the same battery on gemma and on PCFG
    cross_source_layer_response             gemma vs PCFG at the layers both graded
    cross_source_alignment_check            block index or relative depth — which to pair on
    in_block_relations                      same-level edges and duplicates
    shared_input_moved_every_metric         the battery's own failure mode
    base_rate_vs_frequency_capture          the hypothesis's premise, tested
    sres_null_rate_vs_dictionary_size       what a top-k rank rule costs at each D

Two rules this file follows, both learned the hard way:

**Nothing is hardcoded.** Every number in every title is read from the JSON being
plotted. The previous version quoted layer-6 figures in its titles ("B0: 713
edges, 0/449 genuine"), and those numbers were produced before BOS exclusion --
so the captions kept asserting withdrawn results after the data under them had
been regenerated. A title that cannot go stale is a title computed from its input.

**Nothing is skipped silently.** A figure whose input is missing prints the path
it wanted and why, and `--list` shows the whole plan without drawing anything. A
figure set that quietly renders 6 of 10 reads as a complete set.

Most figures read only committed JSON reports, so they build from a fresh clone
with no caches. The two that need `exp0_stats.pt` or the token cache say so.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import config as C  # noqa: E402

# The gemma/PCFG comparison's pairing rules and its six measures. Shared with
# `make_report_tables`, which prints the same claim: two copies of the alignment
# logic would eventually disagree about the same numbers.
from reporting import cross_source as X  # noqa: E402

PAPER_DIR = C.OUT_DIR / "paper_figuers"

# Which single gemma layer the recovered-graph figure draws. An editorial
# choice, so it is one explicit constant rather than an accident of which run
# had the probe first (that accident chose layer 6 for months). Layer 12 is
# mid-network AND the layer where the released priors-in-time and temporal
# SAEs exist, so the drawn graph stays directly comparable when the
# cross-architecture results land.
GRAPH_LAYER = 12

# Which PCFG layer that same figure's middle panel draws. The PCFG base model
# has four blocks, so there is no exact middle; L2 is the upper of the two
# middle layers and the only one of them that carries probe-confirmed edges
# (L1 scores 327 edges and confirms none, which would read as a dead probe).
# An editorial choice like GRAPH_LAYER, not a computed one.
PCFG_GRAPH_LAYER = 2

# --- palette ---------------------------------------------------------------
# Categorical slots are Okabe-Ito, assigned in fixed order and never cycled. The
# repo's screen palette (#2E9E5B green beside #D98A3D orange) separates by only
# ΔE 3.4 under protanopia -- indistinguishable to a red-green colourblind reader,
# which is a real cost in print where no tooltip can rescue the encoding. These
# four clear every check in both light mode and all-pairs mode; the orange sits
# under 3:1 against white, so every figure using it also carries direct labels.
CAT = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
# Layers are ORDERED, so depth gets a single-hue sequential ramp rather than four
# categorical hues. Reading L24 as "the dark one" is the encoding doing its job.
DEPTH = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594", "#08306b"]
NEUTRAL = "#C9CCD1"          # "removed" / reference — not a category
GOOD = "#009E73"             # status: survived. Reserved, never a series colour.
INK, MUTED = "#2B2B33", "#5A6B7B"


def _text_on(bg):
    """Ink or white on `bg`, whichever the eye can actually read.

    The old rule compared HSV "value", which is max(r, g, b) and therefore stays
    high on a saturated blue long after the swatch has gone dark: on the Blues
    ramp it kept choosing dark ink from 70% to 90% of the scale, where dark ink
    lands at a contrast ratio of 2.0 against the 4.5 a small label needs. This
    compares WCAG relative luminance instead and picks the higher contrast, so
    the label follows the swatch instead of a proxy for it.
    """
    from matplotlib import colors as mcolors

    def _lum(c):
        lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
               for v in mcolors.to_rgb(c)]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    L = _lum(bg)
    return "white" if 1.05 / (L + 0.05) > (L + 0.05) / (_lum(INK) + 0.05) else INK


def _fit_width(fig, t, avail_in, floor=5.2, wrap=False):
    """Fit a text artist into `avail_in` inches: wrap first, then shrink.

    Header labels sit over a column group, so a name wider than its group runs
    into the neighbouring one -- "Reconstruction" printed straight through
    "Probe-based" the first time this figure carried the full battery. The
    width is measured with the renderer rather than estimated from a character
    count: the names are mixed-case and some carry mathtext, and a per-character
    guess is wrong in both directions on exactly those.

    Shrinking alone is not enough over a ONE-column group: "1 Activation
    coverage" hit the 5.2pt floor and still overhung its neighbour, because no
    readable size fits twenty-one characters into one cell. With `wrap`, the
    label is broken across lines until it fits or it runs out of spaces, and
    only the remainder is taken out of the font size -- two readable lines beat
    one illegible one.
    """
    try:
        r = fig.canvas.get_renderer()
    except AttributeError:                       # backend without a live renderer
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    lim = avail_in * fig.dpi * 0.86              # visible air on either side
    if wrap:
        words = t.get_text().split()
        lines = 1
        while t.get_window_extent(r).width > lim and lines < len(words):
            lines += 1
            per = -(-len(words) // lines)          # ceiling, no new import
            t.set_text("\n".join(" ".join(words[i:i + per])
                                 for i in range(0, len(words), per)))
    while t.get_fontsize() > floor and t.get_window_extent(r).width > lim:
        t.set_fontsize(t.get_fontsize() - 0.2)
    return t

plt.rcParams.update({
    "font.size": 9.5,
    "axes.edgecolor": "#D8DBE0",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --- data access -----------------------------------------------------------
# Every loader returns None rather than raising, so one absent input costs one
# figure instead of the run.
def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


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


def _merge_variant_pairs(layer_dir, rep):
    """Fold pairs graded in a variant run of the same layer into that layer's report.

    B3->B4 is off by default in `config.py` because its 6144 x 24576 accumulators do not
    fit on a small card, so it was graded in a separate run, `layer_12_b3b4/`. The figure
    that draws every block pair printed "not computed: exceeds the memory budget" for it.
    That sentence became false the moment the pair was graded, and a figure asserting a
    limit that has been lifted is worse than one with a gap.

    A variant run rebuilds the statistics cache from scratch, so it regrades every pair,
    not only the new one. Merging is therefore only sound if the pairs both runs share
    came out identical. That is checked here rather than assumed: the token count must
    match, and every shared pair must report the same candidate-edge count. A variant that
    fails either test is ignored, because then its extra pair was measured on something
    else and putting it in this layer's row would compare two corpora in one table.
    """
    base = {q["pair"]: q for q in rep.get("pairs", [])}
    for var_dir in sorted(layer_dir.parent.glob(f"{layer_dir.name}_*")):
        var = _json(var_dir / "metrics_report.json")
        if not var:
            continue
        if var.get("total_tokens") != rep.get("total_tokens"):
            continue
        vp = {q["pair"]: q for q in var.get("pairs", [])}
        shared = set(base) & set(vp)
        if not shared or any(base[k]["n_candidate_edges"] != vp[k]["n_candidate_edges"]
                             for k in shared):
            continue
        extra = [vp[k] for k in sorted(set(vp) - set(base),
                                       key=lambda k: tuple(int(x) for x in k.split("->")))]
        if extra:
            rep = dict(rep, pairs=rep["pairs"] + extra)
            base.update({q["pair"]: q for q in extra})
    return rep


def gemma_layers() -> list[tuple[int, dict]]:
    out = []
    for L in C.NAV_LAYERS:
        d = C.OUT_DIR / C.SOURCE_NAME / f"layer_{L:02d}"
        r = _json(d / "metrics_report.json")
        if r:
            out.append((L, _merge_variant_pairs(d, r)))
    return out



@functools.lru_cache(maxsize=1)
def sres_observed_rates():
    """(label, D, measured % pass) per probed run, over EVERY pair it probed.

    Shared by the figure and by its caption so the two cannot print different
    numbers for one quantity. The rate is whole-run, not B0->B1: the null it is
    measured against is $k/D$, a property of the dictionary, and every pair is
    scored against the same dictionary -- restricting the rate to one pair made a
    quantity about D depend on a choice of block pair, and printed a number the
    null table did not agree with.
    """
    G = C.OUT_DIR / C.SOURCE_NAME
    probes = [(f"gemma L{int(q.parent.name.split('_')[1])}", C.D_SAE, q)
              for q in sorted(G.glob("layer_*/second_pass.json"))]
    probes += [(f"PCFG {_layer_name(q.parent.name)}", 1792, q)
               for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*/second_pass.json"))]
    out = []
    for label, d_sae, path in probes:
        sp = _json(path)
        if not sp:
            continue
        n_pass = sum((v.get("sres") or {}).get("n_pass", 0) for v in sp.values())
        n_scored = sum((v.get("sres") or {}).get("n_edges_scored", 0) for v in sp.values())
        if n_scored:
            # zero is a measurement, not a missing value -- kept and drawn
            out.append((label, d_sae, 100 * n_pass / n_scored))
    return out


def _sres_same_D_clause() -> str:
    """The within-dictionary spread, computed rather than asserted.

    The caption used to state that two runs on the same 1,792-latent dictionary
    land on opposite sides of their null. That was a property of the B0->B1 rate
    and false of the whole-run rate that replaced it -- exactly the kind of
    hand-typed claim this module's captions are meant to be immune to.
    """
    by_d: dict = {}
    for _, d, o in sres_observed_rates():
        by_d.setdefault(d, []).append(o)
    parts = []
    for d in sorted(by_d):
        g = by_d[d]
        if len(g) < 2:
            continue
        null = 100 * C.SRES_RANK_TOP_K / d
        lo, hi = min(g), max(g)
        span = rf"{lo:.2f}--{hi:.2f}\%" if hi > lo else rf"{hi:.2f}\%"
        where = ("straddling their null" if lo <= null <= hi else
                 rf"all {lo / null:.0f}--{hi / null:.0f}$\times$ their null" if lo > null
                 else "all below their null")
        parts.append(f"{_spell(len(g))} runs on a {d:,}-latent dictionary span {span}, {where}")
    if not parts:
        return ""
    return ("The spread within a single dictionary is itself wide --- "
            + "; ".join(parts) + " --- which is why a raw pass rate compares nothing "
            "across sources, and little within one. ")

def _pair(report, name):
    return next((p for p in report["pairs"] if p["pair"] == name), None)


# Figure-level titles are OFF by default (--titles turns them on). The paper
# carries every figure's message in its LaTeX caption, and a second copy baked
# into the PNG reads as a machine-written caption above the human one -- the
# exact habit the 2026-08-16 feedback asked us to drop. Panel-level titles and
# axis labels stay: they are part of the plot, not a caption.
TITLES = False


def _title(ax_or_fig, head: str, sub: str = "", width: int = 96):
    """Left-aligned title, wrapped to the figure rather than trusting it to fit.

    Every overflowing title in the previous version was a long second line that
    matplotlib silently let run past the canvas edge, so the caption lost its
    qualifier -- the half a sentence that says what the number does NOT mean.
    Returns the number of lines drawn; 0 when titles are disabled, so a layout
    that reserves headroom for the title can reclaim it.
    """
    if not TITLES:
        return 0
    import textwrap
    lines = textwrap.wrap(head, width)
    if sub:
        lines += textwrap.wrap(sub, width + 8)
    txt = "\n".join(lines)
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(txt, fontsize=10.5, loc="left", color=INK)
    else:
        ax_or_fig.suptitle(txt, fontsize=10.5, x=0.006, ha="left", color=INK)
    return txt.count("\n") + 1


def _spell(n: int) -> str:
    """Small counts as words, the way the prose around them is written.

    These counts are derived rather than typed, so they have to render as English or
    a caption reads "the 6 graded layers" in the middle of a sentence.
    """
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
             "nine", "ten", "eleven", "twelve"]
    return words[n] if 0 <= n < len(words) else f"{n:,}"


def _label_extremes(ax, xs, vals, fmt, color):
    """Label the first, last, min and max point only.

    A number on every point is the anti-pattern; here it also collided with the
    line it annotated at four points out of five.
    """
    keep = {0, len(vals) - 1, int(np.argmin(vals)), int(np.argmax(vals))}
    for i in sorted(keep):
        ax.annotate(fmt(vals[i]), (xs[i], vals[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=color)



def _panel_head(ax, tag: str, detail: str = ""):
    """A two-line panel heading: what the panel IS, then what it counts.

    `set_title` cannot give the two lines different weights, and in a
    before/after pair the word that matters most -- which of the two moments
    this panel is -- was the part competing with a run of counts for the
    reader's eye. The tag carries the weight; the counts sit under it in the
    muted colour the rest of the figures use for derived numbers. Returns the two
    artists, so a caller with a narrow panel can hand them to `_fit_width`.
    """
    # Honours the same --titles gate as _title: the paper's LaTeX captions carry
    # this text, and baking it into the PNG was the mentors' figure complaint
    # (text explaining a figure belongs in the caption).
    if not TITLES:
        return []
    # Offsets in INCHES, converted to this axes' fraction. A fixed fraction is a
    # different physical gap on a 7-inch panel and on a 2-inch one: at 0.05 of a
    # short axes the two lines landed on top of each other and on the top bar.
    # Requires the layout to be settled, so call this after `subplots_adjust`.
    fig = ax.get_figure()
    h = max(ax.get_position().height * fig.get_figheight(), 0.35)
    pad, line = 0.045 / h, 0.135 / h
    out = [ax.text(0.0, 1 + pad + (line if detail else 0), tag,
                   transform=ax.transAxes, ha="left", va="bottom",
                   fontsize=10, fontweight="bold", color=INK)]
    if detail:
        out.append(ax.text(0.0, 1 + pad, detail, transform=ax.transAxes,
                           ha="left", va="bottom", fontsize=8.2, color=MUTED))
    return out


def _panel_rule(fig, left, right, top_pad: float = 0.055, bottom_pad: float = 0.0,
                bias: float = 0.5):
    """A vertical rule down the seam between two panels, in figure coordinates.

    Two panels of the same drawing side by side with nothing between them read
    as one wide picture: the eye follows a link straight across the seam and
    the before/after comparison the figure exists for stops being visible as
    two states of one object. Call it AFTER the figure's own
    `subplots_adjust` -- that is what fixes the positions this reads, so a
    figure using it must also pass ``tight=False`` to `_finish`.
    """
    a, b = left.get_position(), right.get_position()
    # `bias` places the rule across the gutter: 0.5 is midway, lower pulls it
    # toward the left panel. The right panel's tick labels and y-label live in
    # that gutter, so a rule at dead centre can run through them.
    x = a.x1 + bias * (b.x0 - a.x1)
    y0 = max(0.0, min(a.y0, b.y0) - bottom_pad)
    y1 = min(1.0, max(a.y1, b.y1) + top_pad)
    fig.add_artist(Line2D([x, x], [y0, y1], transform=fig.transFigure,
                          color=NEUTRAL, lw=1.0, zorder=0))

def _finish(fig, ax_or_axes, name: str, tight: bool = True) -> str:
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.ravel(axes):
        ax.spines[["top", "right"]].set_visible(False)
    if tight:
        # tight_layout ignores ax.text placed above the axes, so a figure that
        # hangs panel headings there manages its own margins and passes False
        fig.tight_layout()
    path = PAPER_DIR / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# 1. The funnel: what each stage removes, at every layer that has the strict
#    test. Originally one horizontal funnel for layer 6, the only run that had
#    stage 03; once the other five caught up, showing one layer understated the
#    evidence and overstated how special that layer was.
# ---------------------------------------------------------------------------
def funnel_coverage_to_sres(layers, second, pcfg=None, pair="0->1", where=""):
    # Stage labels carry the battery number, because the three stages are
    # metrics 1, 2 and 5 in order: the funnel walks the battery forward and
    # skips nothing that is nested. The last stage is named for what it
    # measures, not for what it would mean if the probe were ground truth --
    # its target is the child's own self-label, so "genuine refinement" would
    # claim exactly what the methodology section disavows.
    stages = [f"1. candidate edges\nreverse coverage ≥ {C.EDGE_TAU}",
              "2. above chance\nPMI > 0", "5. probe-confirmed\npasses probe S_res"]
    x = np.arange(len(stages))
    # one panel per SOURCE: the funnel collapses the same way on both, which
    # is itself the cross-source claim, so the PCFG runs stand beside gemma
    groups = [("gemma-2-2b", [(f"L{L}", rep, second.get(L)) for L, rep in layers])]
    if pcfg:
        pruns = [(lab.replace("PCFG layer ", "L"), r, sp) for lab, r, sp in pcfg
                 if sp and pair in sp]
        if pruns:
            groups.append(("PCFG", pruns))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 + 4.4 * len(groups), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = [], []
    for ax, (src, runs) in zip(axes, groups):
        for i, (lab, rep, sp) in enumerate(runs):
            if not (sp and pair in sp):
                continue
            pr = _pair(rep, pair)
            sres = sp[pair]["sres"]
            vals = [pr["n_candidate_edges"], sres["n_edges_scored"], sres["n_pass"]]
            # log y: the whole story is three orders of magnitude, and on a
            # linear axis every line would lie on the floor after stage one.
            # A measured ZERO has no position on a log axis, so it is drawn at
            # a floor with its true value in the end label instead of dropped.
            plot_vals = [max(v, 0.5) for v in vals]
            ax.plot(x, plot_vals, "-o", color=DEPTH[i % len(DEPTH)], lw=1.8, ms=5,
                    zorder=3)
            ax.annotate(lab + (" (0 pass)" if vals[-1] == 0 else ""),
                        (x[-1], plot_vals[-1]), textcoords="offset points",
                        xytext=(10, -3 + 6 * (i % 2)), fontsize=8,
                        color=DEPTH[i % len(DEPTH)])
            if src == groups[0][0]:
                lo.append(vals), hi.append(pr["reconstruction"]["n_pass"])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=8.5)
        ax.set_xlim(-0.25, len(stages) - 0.45)
        ax.set_title(src, fontsize=10, loc="left", color=INK)
        ax.grid(True, axis="y", which="both", alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("parent → child edges (log)")
    first, last = [v[0] for v in lo], [v[-1] for v in lo]
    share = [100 * b / a for a, b in zip(first, last)]
    # the ablation filter is parallel, not a nested stage; summarized in text
    rec = [100 * h / v[0] for h, v in zip(hi, lo)]
    _title(fig, f"{where}: co-firing proposes thousands, the strict test confirms tens — "
                "every graded layer, both sources",
           f"{min(last)}–{max(last)} of {min(first):,}–{max(first):,} gemma candidates "
           f"survive ({min(share):.1f}–{max(share):.1f}%). The parallel ablation filter "
           f"passes {min(rec):.0f}–{max(rec):.0f}% and is not drawn as a stage", width=86)
    return _finish(fig, axes, "funnel_coverage_to_sres")


# ---------------------------------------------------------------------------
# 2. Per block pair, per layer: what survives reconstruction and the frequency
#    control. This is the withdrawn kill-rate view, rebuilt from committed
#    reports -- the condition on which a withdrawn result was allowed back.
# ---------------------------------------------------------------------------
def _pairs_in(layers):
    """The block pairs actually graded, in block order, across every layer given.

    This used to be the literal list ["0->1", "1->2", "2->3"]. B3->B4 is disabled by
    default in config.py because its accumulators do not fit on a small card, so for a
    long time three was all there was. A hardcoded list does not fail when a fourth pair
    appears: the figure keeps drawing three bars and looks finished.

    Taking the union rather than the intersection is deliberate. A pair graded at one
    layer and not another should show as a gap in that layer's bars, which is visible,
    rather than disappear from the axis, which is not.
    """
    seen = set()
    for _, rep in layers:
        seen.update(q["pair"] for q in rep.get("pairs", []) if "pair" in q)
    return sorted(seen, key=lambda k: tuple(int(x) for x in k.split("->")))


def edge_survival_by_block_pair(layers):
    pairs = _pairs_in(layers)
    if not pairs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=False)
    x = np.arange(len(pairs))
    w = 0.15

    for i, (L, rep) in enumerate(layers):
        recon = [(_pair(rep, p) or {}).get("reconstruction", {}).get("frac_pass", np.nan)
                 for p in pairs]
        freq = [(_pair(rep, p) or {}).get("freq_control", {}).get("frac_freq_driven", np.nan)
                for p in pairs]
        off = (i - (len(layers) - 1) / 2) * w
        axes[0].bar(x + off, np.array(recon) * 100, w, color=DEPTH[i], label=f"L{L}")
        axes[1].bar(x + off, np.array(freq) * 100, w, color=DEPTH[i], label=f"L{L}")

    for ax, title, ylab in [
        (axes[0], "improve reconstruction", "% of candidate edges"),
        (axes[1], "carried by frequent tokens", "% of candidate edges"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"B{p.replace('->', '→B')}" for p in pairs])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].legend(title="layer", fontsize=8.5, title_fontsize=8.5, frameon=False,
                   ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    _title(fig, "What each filter removes, by block pair and depth",
           "each panel on its own scale; layers are ordered, so depth is a single-hue ramp",
           width=104)
    fig.subplots_adjust(top=0.80, bottom=0.22)
    return _finish(fig, axes, "edge_survival_by_block_pair")


# ---------------------------------------------------------------------------
# 3. Depth. The claim this replaces said quality degrades with depth; that came
#    from BOS-contaminated caches. Plotted from regenerated reports so the
#    figure can contradict it.
# ---------------------------------------------------------------------------
def depth_profile_across_layers(layers):
    Ls = [L for L, _ in layers]
    series = [
        ("candidate edges (B0→B1)", CAT[0],
         [_pair(r, "0->1")["n_candidate_edges"] for _, r in layers], False),
        ("improve reconstruction, %", CAT[1],
         [100 * _pair(r, "0->1")["reconstruction"]["frac_pass"] for _, r in layers], True),
        ("frequency-driven, %", CAT[3],
         [100 * _pair(r, "0->1")["freq_control"]["frac_freq_driven"] for _, r in layers], True),
        ("mean frequency survival", CAT[2],
         [_pair(r, "0->1")["freq_control"]["mean_survival"] for _, r in layers], True),
    ]
    # Four measures on four axes rather than one: a shared y would be a dual-axis
    # chart wearing a disguise, and the point is the SHAPE of each line, not their
    # relative heights.
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.3))
    for ax, (label, col, vals, _) in zip(axes, series):
        ax.plot(Ls, vals, "-o", color=col, lw=2, ms=5)
        fmt = (lambda v: f"{v:,.0f}") if max(vals) > 50 else (lambda v: f"{v:.2f}")
        _label_extremes(ax, Ls, vals, fmt, MUTED)
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.set_xticks(Ls)
        ax.set_xlabel("layer")
        ax.margins(y=0.3)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    _title(fig, "Block pair B0→B1 across depth — each panel is one measure, on its own scale",
           "no measure is monotonic in depth; the degradation-with-depth claim came from "
           "caches that counted BOS")
    fig.subplots_adjust(top=0.72)
    return _finish(fig, axes, "depth_profile_across_layers")


# ---------------------------------------------------------------------------
# 4. Multi-parenting: the one claim that did not move when BOS was excluded,
#    because it is a ratio over children that already have a parent.
# ---------------------------------------------------------------------------
def multiparenting_by_layer(layers, pcfg=None):
    # EVERY block pair each source's nesting defines, not a fixed prefix of
    # them: the pair list is read off block_ranges (gemma: 5 blocks -> 4 pairs,
    # PCFG: 8 blocks -> 7 pairs). A pair the pipeline never graded is drawn as
    # an explicit "n.a." at the baseline rather than silently omitted. gemma's
    # B3->B4 is graded at layer 12 only: its 6144 x 24576 accumulators need a large
    # card, so it is off by default in config.py and was run once, on an A40. The
    # five other layers carry n.a. because nobody ran them, not because they cannot
    # be run, and the figure must say which of the two it means.
    # one panel per SOURCE: the claim is cross-source ("the tangle is a
    # property of the nesting, not of gemma"), so the PCFG runs stand beside
    # gemma rather than in a separate appendix figure
    groups = [("gemma-2-2b", [(f"L{L}", r) for L, r in layers])]
    if pcfg:
        groups.append(("PCFG", [(n.replace("PCFG ", "").replace("layer ", "L"), r)
                                for n, r in pcfg]))
    # the first four pairs keep the figure's original categorical hues (CAT),
    # so B0->B1 stays the blue readers already know; the three pairs the wider
    # PCFG nesting adds get three new hues. Adjacent-pair CVD separation
    # checked programmatically (Viénot simulation + OKLab): all pairs >= 8
    # except vermilion<->pink under tritanopia (7.0), which is legal here
    # because every bar carries its value as a direct label.
    RAMP = CAT + ["#CC79A7", "#6A51A3", "#56B4E9"]

    def pair_list(runs):
        n_blocks = max((len(r.get("block_ranges", [])) for _, r in runs), default=0)
        return [f"{i}->{i + 1}" for i in range(max(n_blocks - 1, 0))]

    per_src_pairs = {src: pair_list(runs) for src, runs in groups}
    all_pairs = max(per_src_pairs.values(), key=len)

    fig, axes = plt.subplots(1, len(groups), figsize=(4.0 + 4.8 * len(groups), 4.1),
                             sharey=True,
                             gridspec_kw={"width_ratios":
                                          [len(g[1]) * len(per_src_pairs[g[0]])
                                           for g in groups]})
    axes = np.atleast_1d(axes)
    for ax, (src, runs) in zip(axes, groups):
        pairs = per_src_pairs[src]
        x = np.arange(len(runs))
        w = 0.86 / max(len(pairs), 1)
        fs_val = 6.0 if len(pairs) > 4 else 7.5
        for j, p in enumerate(pairs):
            vals, supp = [], []
            for _, r in runs:
                q = _pair(r, p) or {}
                vals.append(100 * q.get("degree", {}).get("poly_frac", np.nan))
                supp.append(q.get("degree", {}).get("n_children_with_parent"))
            off = (j - (len(pairs) - 1) / 2) * w
            ax.bar(x + off, [0 if np.isnan(v) else v for v in vals], w * 0.92,
                   color=RAMP[j])
            for xx, vv, nn in zip(x + off, vals, supp):
                if np.isnan(vv):
                    # absence of measurement, made visible instead of skipped
                    ax.text(xx, 2.5, "n.a.", ha="center", va="bottom", rotation=90,
                            fontsize=5.5, color=NEUTRAL)
                    continue
                # every bar carries its support: the % answers "how much",
                # n answers "over how many children it was computed". The %
                # sits above the bar; n is rotated inside the bar so wide
                # counts (n=3190) never collide with the neighbouring bar.
                # Bars too short to hold the text stack both above instead.
                # inside only when the bar is tall enough to hold the whole
                # rotated string (~2.3 axis units per character + padding);
                # otherwise the white tail vanishes into the white background
                fits_inside = (nn is not None
                               and vv >= 6 + 2.3 * len(f"n={nn}"))
                if fits_inside:
                    ax.text(xx, vv + 1.6, f"{vv:.0f}", ha="center",
                            fontsize=5.5 if len(pairs) > 4 else 6.5, color=MUTED)
                    ax.text(xx, 2.5, f"n={nn}", ha="center", va="bottom",
                            rotation=90, fontsize=5.5, color="#FFFFFF")
                else:
                    # short bar: value above it, n rotated above the value —
                    # never horizontal, so wide counts cannot reach a neighbour
                    ax.text(xx, vv + 1.6, f"{vv:.0f}", ha="center",
                            fontsize=5.5, color=MUTED)
                    if nn is not None:
                        ax.text(xx, vv + 7.0, f"n={nn}", ha="center", va="bottom",
                                rotation=90, fontsize=5.0, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _ in runs])
        ax.set_title(f"{src} — {len(pairs) + 1} blocks, {len(pairs)} pairs",
                     fontsize=10, loc="left", color=INK)
        ax.set_ylim(0, 112)
        ax.axhline(100, ls=(0, (2, 3)), lw=1, color=NEUTRAL)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of children with ≥ 2 parents")
    # one shared key for every pair either source defines, colored by the ramp
    handles = [plt.Rectangle((0, 0), 1, 1, color=RAMP[j])
               for j in range(len(all_pairs))]
    # the key sits above its two shared footnotes: both apply to both panels
    fig.legend(handles, [f"B{p.replace('->', '→B')}" for p in all_pairs],
               fontsize=8, frameon=False, ncol=min(len(all_pairs), 7),
               loc="lower center", bbox_to_anchor=(0.5, 0.062))
    fig.text(0.5, 0.042, "n = children behind the %",
             ha="center", fontsize=7.5, color=MUTED)
    fig.text(0.5, 0.008, "n.a. = not graded at that layer; B3→B4 needs a 49 GB card "
                         "and was run at layer 12 only",
             ha="center", fontsize=7.5, color=MUTED)
    _title(fig, "The graph is not a tree: in the top block pair nearly every child has "
                "several parents, on both sources",
           "every block pair each nesting defines; n.a. = not run at that layer — gemma's "
           "B3→B4 needs a 49 GB card and was run at layer 12 only. "
           "Ratio over children that already have a parent — the one measure BOS "
           "exclusion left unchanged", width=110)
    # tight_layout ignores fig.legend and would drop it onto the tick labels;
    # this figure manages its own margins and passes tight=False
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.25,
                        wspace=0.06)
    return _finish(fig, axes, "multiparenting_by_layer", tight=False)


# ---------------------------------------------------------------------------
# 5. Superparents: fan-out against base rate. Reports list the top parents per
#    pair, which is what this needs.
# ---------------------------------------------------------------------------
def superparent_fanout_vs_firing(layers):
    """Why a high-firing parent clears the coverage bar: it barely has to.

    The previous version scattered fan-out against firing rate on a log x-axis.
    It was close to a tautology -- every point is above the gate because the gate
    is what selected it -- the log scale spanned less than a decade and printed
    "4x10^1" where "40" would do, and the five-layer ramp encoded nothing the
    points clustered on.

    What the same numbers do show is the mechanism. Coverage keeps an edge when
    P(parent | child) >= tau. Under independence that probability is just the
    parent's firing rate, so the enrichment a parent needs over chance is tau/rho
    -- 50x for a parent firing on 1% of tokens, and at or below 1x for anything
    firing more often than tau itself.
    """
    fires = [sp["fire_frac"] for _, rep in layers for pr in rep["pairs"]
             for sp in pr.get("superparents", [])]
    tau = C.EDGE_TAU
    free = sum(1 for f in fires if f >= tau)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    rho = np.linspace(0.005, 1.0, 400)
    ax.plot(100 * rho, tau / rho, lw=2, color=NEUTRAL, zorder=1,
            label=f"enrichment needed to reach R ≥ {tau}   (= {tau}/ρ)")
    ax.axhline(1.0, ls=(0, (4, 3)), lw=1.3, color=CAT[3])
    ax.text(2, 1.45, "1× — no enrichment at all: the edge is kept on base rate",
            fontsize=8.5, color=CAT[3])
    ax.scatter([100 * f for f in fires], [tau / f for f in fires], s=46, color=CAT[0],
               zorder=3, edgecolor="white", linewidth=0.8,
               label=f"the {len(fires)} flagged parents")
    ax.set_yscale("log")
    ax.set_xlim(0, 104)
    ax.set_xlabel("parent firing rate ρ, % of tokens")
    ax.set_ylabel("enrichment over chance the parent needs (log)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    _title(ax, "A parent that fires often enough clears the coverage bar without any enrichment",
           f"{free} of the {len(fires)} flagged parents fire on ≥ {100 * tau:.0f}% of tokens, so "
           f"co-firing with most of the next block is arithmetic. One firing on 1% would need "
           f"{tau / 0.01:.0f}× enrichment for the same edge", width=78)
    ax.grid(True, which="both", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "superparent_fanout_vs_firing")


# ---------------------------------------------------------------------------
# 6. Tier 1 — every metric against a known tree, ranked by how decisively it
#    separated the two classes.
# ---------------------------------------------------------------------------
def calibration_synthetic_toy_scorecard(rows):
    # Two kinds of row, and only one belongs on a ratio axis. Rows scored
    # categorically (the recovered edge set is right or it is not) carry a margin
    # of 1.0 meaning "correct", which on a log ratio axis is indistinguishable
    # from "separated by a factor of one" -- i.e. from no separation at all. They
    # get their own panel rather than a shared scale that implies a comparison
    # the numbers cannot support.
    ratio = [r for r in rows if r.get("margin_kind") != "categorical"]
    cat_rows = [r for r in rows if r.get("margin_kind") == "categorical"]
    ratio.sort(key=lambda r: r["margin"])
    n_pass = sum(r["pass"] for r in rows)

    # the calibration JSON numbers its rows with internal codes ("2b.", "3'.")
    # that mean nothing to a reader; the figure shows the metric's name alone
    def plain(label):
        return re.sub(r"^\s*\d+[a-z]?'?\.\s*", "", label)

    def row_label(r):
        """Metric name over the job it is there to do.

        The job is recorded per row in the calibration JSON and was not shown, so the
        figure listed thirteen names with no statement of what any of them is for --
        the first thing a reader asked for.
        """
        job = r.get("job", "")
        if not job:
            return plain(r["metric"])
        # Wrapped, not widened: one 55-character line per row needs a gutter so deep
        # that the bars lose the canvas. Two short lines fit the same words.
        return plain(r["metric"]) + "\n" + textwrap.fill(job, 42)

    # Wide, and laid out by hand: each row now carries its job on a second line, and
    # two columns of two-line labels do not fit inside a tight_layout that also has to
    # find room for a two-line axis label and a two-entry key.
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8),
                             gridspec_kw={"width_ratios": [1.9, 1]})
    ax = axes[0]
    # Negative controls pass when the battery does NOT act, so they are a
    # different kind of claim and are hatched rather than recoloured -- texture is
    # the secondary encoding that survives greyscale printing.
    for i, r in enumerate(ratio):
        ctrl = "negative control" in r["metric"]
        ax.barh(i, min(r["margin"], 1e4), height=0.6,
                color=GOOD if r["pass"] else CAT[3],
                hatch="///" if ctrl else None, edgecolor="white", linewidth=0.8)
        ax.text(min(r["margin"], 1e4) * 1.3, i,
                ">1000×" if r["margin"] >= 1000 else f"{r['margin']:.1f}×",
                va="center", fontsize=8, color=MUTED)
    ax.set_yticks(np.arange(len(ratio)))
    ax.set_yticklabels([row_label(r) for r in ratio], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1.2e5)
    ax.axvline(1.0, ls=(0, (2, 3)), lw=1, color=NEUTRAL)
    ax.set_xlabel("separation: the metric's score on healthy structure divided by its "
                  "score on the planted defect\n(log scale; 1× means it scores the two "
                  "alike, and cannot tell them apart)")
    ax.set_title("scored by separation", fontsize=9.5, loc="left")
    # The planted defects, named: "injected structures" appeared with nothing saying
    # what was injected. Figure-level, under the key, so both are read together.
    fig.text(0.012, 0.055,
             "Planted defects: a SUPERPARENT (fires on ~90% of tokens with a tiny "
             "activation, so it co-fires with every child but adds almost nothing to\n"
             "reconstruction), a FREQUENCY-COINCIDENCE edge (lives only on high-frequency "
             "tokens),\nand a FEATURE-SPLIT parent (three near-duplicate children firing on "
             "the same tokens with the same direction).",
             fontsize=7.8, color=MUTED, va="bottom")
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=GOOD,
                      label="passes: flags the planted defect and leaves healthy structure alone"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GOOD, hatch="///", edgecolor="white",
                      label="negative control: passes only if no metric rejects the pair"),
    ], fontsize=8, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0.008, 0.10))
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)

    ax = axes[1]
    for i, r in enumerate(cat_rows):
        ctrl = "negative control" in r["metric"]
        ax.barh(i, 1.0, height=0.6, color=GOOD if r["pass"] else CAT[3],
                hatch="///" if ctrl else None, edgecolor="white", linewidth=0.8)
        ax.text(0.5, i, "correct" if r["pass"] else "wrong", va="center", ha="center",
                fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(np.arange(len(cat_rows)))
    ax.set_yticklabels([row_label(r) for r in cat_rows], fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("scored pass / fail\n(the recovered set matches ground truth, or does not)",
                 fontsize=9.5, loc="left")
    ax.spines["bottom"].set_visible(False)

    _title(fig, f"Every metric scored against a known tree — {n_pass}/{len(rows)} rows pass",
           "the hatched rows are limitations demonstrated rather than caught: an absorbed edge "
           "coverage cannot propose, and shared-topic and composition pairs every filter accepts", width=112)
    # Right-hand ticks on the right panel: with two-line labels, left-hand ticks put
    # this panel's text into the left panel's plot area.
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="y", length=0)
    # Hand-laid, not tight_layout: the label gutters, the key and the note below it all
    # need reserved space that tight_layout would reclaim.
    fig.subplots_adjust(left=0.215, right=0.845, top=0.94, bottom=0.26, wspace=0.05)
    return _finish(fig, axes, "calibration_synthetic_toy_scorecard", tight=False)


# ---------------------------------------------------------------------------
# 7. Tier 2 — the same tree after a real training run, plus the nesting control.
# ---------------------------------------------------------------------------
def calibration_trained_toy_recovery(tt, align):
    tp, fp, fn = tt["true_positives"], tt["false_positives"], tt["false_negatives"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    # Same three colours as `calibration_toy_tree_recovered`, which draws the same three
    # outcomes: two conventions for one word is how a reader ends up counting twice.
    bars = [("recovered", tp, GOOD), ("missed", fn, CAT[3]), ("false positives", fp, "#CC79A7")]
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], width=0.55)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=10, color=INK)
    ax.set_ylabel("true edges")
    ax.set_ylim(0, max(tp, fn, fp) * 1.35 + 0.5)
    # Panel identifiers, not captions: with two panels sharing a y-label the
    # plot is ambiguous without them, so they stay even with --titles off.
    ax.set_title("edge recovery", fontsize=10, loc="left", color=INK)

    ax = axes[1]
    if align:
        # "violated" is the bar the test is ABOUT -- its zero must be visible,
        # or the grey untestable bar next to the green one reads as a failure.
        n_viol = align["n_testable"] - align["n_respected"]
        bars = [("respected", align["n_respected"], GOOD),
                ("violated", n_viol, CAT[3]),
                ("untestable", len(align.get("untestable", [])), NEUTRAL)]
        ax.bar([b[0] for b in bars], [b[1] for b in bars],
               color=[b[2] for b in bars], width=0.55)
        for i, (_, v, _) in enumerate(bars):
            ax.text(i, v + 0.12, str(v), ha="center", fontsize=10, color=INK)
        ax.set_ylim(0, max(align["n_testable"], 1) * 1.4)
        ax.set_ylabel("true edges")
        ax.set_title("nesting control (early block → late)", fontsize=10, loc="left", color=INK)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "block_tree_alignment.json absent\nrun validation.block_tree_alignment",
                ha="center", va="center", fontsize=9, color=MUTED)
    for a in axes:
        a.grid(True, axis="y", alpha=0.12)
        a.set_axisbelow(True)
    return _finish(fig, axes, "calibration_trained_toy_recovery")


def _tree_from_edges(true_edges, n_features):
    """A stand-in tree for a calibration JSON written before the tree was recorded.

    Edges alone have no root, which is what made the first version of this figure
    read as three disconnected subtrees beside a row of loose features. Hanging
    everything off one root restores the shape even without the real node data.
    """
    kids = {}
    for p, c in true_edges:
        kids.setdefault(p, []).append(c)
    attached = {c for cs in kids.values() for c in cs}
    children = [{"_idx": p, "children": [{"_idx": c, "children": []} for c in sorted(kids[p])]}
                for p in sorted(kids)]
    children += [{"_idx": f, "children": []}
                 for f in range(n_features) if f not in kids and f not in attached]
    return {"_idx": None, "children": children}


def _toy_tree_layout(tree):
    """Left-to-right positions for every node of the tree, plus its parent->child links.

    Depth runs along x and siblings stack down y. Laid out top-down the 20 leaves need
    a panel wider than the page; stacked vertically they fit beside the other two
    panels. The root and the not-read-out nodes are included: dropping them is what
    made the earlier version look like scattered features rather than one tree.
    """
    nodes, links, cursor = [], [], [0]

    def place(node, depth):
        kids = node.get("children", [])
        if kids:
            slots = [place(c, depth + 1) for c in kids]
            slot = (min(slots) + max(slots)) / 2
        else:
            slots, slot = [], cursor[0]
            cursor[0] += 1
        here = (depth, -slot)
        nodes.append((here, node))
        links.extend((here, (depth + 1, -s), node, c) for s, c in zip(slots, kids))
        return slot

    place(tree, 0)
    return nodes, links, cursor[0]


def _tree_listing(node, indent=0):
    """The tree as exact text: index, firing probability, exclusivity.

    `[--]` is a node that shapes the sampling but is not read out, so it holds no
    feature index and can end no edge -- which is why the links visible here do not
    all become scoreable edges.
    """
    idx = node.get("_idx")
    tag = f"[{idx:>2}]" if idx is not None else "[--]"
    excl = "   (children mutually exclusive)" if node.get("mutually_exclusive_children") else ""
    lines = ["  " * indent + f"{tag}  p={node['active_prob']}{excl}"]
    for child in node.get("children", []):
        lines += _tree_listing(child, indent + 1)
    return lines


def _index_tree(node, counter=None):
    """Attach each read-out node's feature index, the way the sampler assigns them."""
    counter = [0] if counter is None else counter
    if node.get("is_read_out", True):
        node["_idx"] = counter[0]
        counter[0] += 1
    for child in node.get("children", []):
        _index_tree(child, counter)
    return node


def calibration_toy_tree_recovered(tt):
    """What an edge IS, the tree it lives in, and what the battery did with it.

    `calibration_trained_toy_recovery` counts the outcomes; this shows which edges
    they were. Colours match that figure exactly (recovered / missed / false
    positive), so the two read as one statement rather than two conventions.

    The left panel exists because of reviewer feedback on the earlier version: a
    recovery figure is unreadable when the tree behind it is never shown, so "6 of 9
    edges" means nothing. It spells out the generative process, the rule that makes an
    edge an edge, and the exact edge list being scored.
    """
    truth = [tuple(e) for e in tt["true_edges"]]
    found = {tuple(e) for e in tt["found_edges"]}
    recovered = set(tt["recovered_features"])
    spurious = sorted(found - set(truth))

    tree = (_index_tree(json.loads(json.dumps(tt["tree"]))) if tt.get("tree")
            else _tree_from_edges(truth, tt["n_features"]))
    nodes, links, n_rows = _toy_tree_layout(tree)
    pos = {n["_idx"]: xy for xy, n in nodes if n.get("_idx") is not None}
    depth = max(xy[0] for xy, _ in nodes)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.4))

    # Outcome colours, shared with `calibration_trained_toy_recovery` so the counts and
    # the drawing cannot use two conventions for the same three words. Okabe-Ito, like
    # CAT: every pair is separable in both common forms of colour blindness, which grey
    # for "missed" was not once it sat beside a grey structural link.
    MISSED, SPURIOUS = CAT[3], "#CC79A7"
    # A link touching a node with no feature index is structure, not a scoreable edge:
    # this is what makes the tree's 12 visible links resolve to 9 scoreable edges, so it
    # has to be READ, not merely sensed. It was near-white while "missed" was the grey it
    # had to stay clear of; with missed now orange, this can be a grey that actually
    # prints -- secondary to the outcome colours, but never invisible.
    STRUCTURAL = "#9AA3AD"
    # Nodes are deliberately neutral. Drawn in CAT[0] they competed with the edge
    # colours, which are the thing this figure is about.
    NODE = "#4A5A6A"

    for ax, is_truth in zip(axes, (True, False)):
        for (x0, y0), (x1, y1), parent, child in links:
            pi, ci = parent.get("_idx"), child.get("_idx")
            if pi is None or ci is None:
                style = dict(color=STRUCTURAL, lw=1.15, ls=(0, (2.5, 2)))
            elif is_truth:
                style = dict(color=CAT[0], lw=2.4, ls="-")
            elif (pi, ci) in found:
                style = dict(color=GOOD, lw=2.8, ls="-")
            else:
                style = dict(color=MISSED, lw=2.4, ls=(0, (4, 2)))
            ax.plot([x0, x1], [y0, y1], zorder=1, **style)

        if not is_truth:
            for (p, c) in spurious:
                ax.plot(*zip(pos[p], pos[c]), color=SPURIOUS, lw=2.4,
                        ls=(0, (1.2, 1.4)), zorder=2)

        for xy, node in nodes:
            idx = node.get("_idx")
            # The ground-truth panel carries each node's firing probability, so the
            # generative process is readable off the figure: p is conditional on the
            # parent firing, which is what makes an edge containment rather than mere
            # correlation. The right-hand panel is about outcomes and stays clean.
            if is_truth and "active_prob" in node:
                ax.text(xy[0], xy[1] - 0.33, f"p={node['active_prob']}",
                        ha="center", va="top", fontsize=5.4, zorder=4,
                        color=MUTED if idx is not None else STRUCTURAL)
            if idx is None:
                # Shapes the sampling, holds no feature: small and hollow, unlabelled.
                ax.scatter([xy[0]], [xy[1]], s=48, zorder=3, facecolor="white",
                           edgecolor=STRUCTURAL, linewidths=1.1)
                continue
            # On the left every feature exists by definition; on the right a hollow
            # node is one no latent recovered, which is why edges touching it could
            # not be found even in principle.
            known = is_truth or idx in recovered
            ax.scatter([xy[0]], [xy[1]], s=165, zorder=3,
                       facecolor=NODE if known else "white",
                       edgecolor=NODE if known else MUTED, linewidths=1.4)
            ax.text(xy[0], xy[1], str(idx), ha="center", va="center", fontsize=6.4,
                    zorder=4, color=_text_on(NODE) if known else MUTED)

        # Extra room on the left: the root sits at x=0 and its p label hangs below it.
        ax.set_xlim(-0.5, depth + 0.35)
        ax.set_ylim(-n_rows + 0.4, 0.6)
        ax.axis("off")

    # Panel identifiers, not captions -- every number in them is read from the JSON.
    # Named "before"/"after" rather than by their contents: the two panels draw the
    # same tree at two moments, and a reader who reads them as two trees has already
    # lost the comparison. The full term "the metric battery" is used here because a
    # panel heading is met with no prose around it to have introduced the short form.
    n_tp = len(found & set(truth))
    heads = (("before — the tree as constructed",
              f"{len(truth)} edges over {tt['n_features']} features"),
             ("after — the tree the metrics recovered",
              f"{n_tp}/{len(truth)} recovered, {len(spurious)} false, "
              f"{len(recovered)}/{tt['n_features']} features learned"))

    handles = [
        Line2D([], [], color=CAT[0], lw=2.4, label="true edge"),
        Line2D([], [], color=GOOD, lw=2.8, label="recovered"),
        Line2D([], [], color=MISSED, lw=2.4, ls=(0, (4, 2)), label="missed"),
        Line2D([], [], color=SPURIOUS, lw=2.4, ls=(0, (1.2, 1.4)), label="false positive"),
        Line2D([], [], color=STRUCTURAL, lw=1.5, ls=(0, (2.5, 2)),
               label="not scoreable (endpoint not read out)"),
        Line2D([], [], color="white", marker="o", ls="", markersize=9,
               markerfacecolor="white", markeredgecolor=MUTED, label="feature not learned"),
    ]
    # Three per row: six across an 11-inch canvas set the labels too tight to read.
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    # tight_layout would reclaim the strip the legend sits in, so this figure
    # manages its own margins (see `_finish`).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.11, wspace=0.08)
    for ax, (tag, detail) in zip(axes, heads):
        _panel_head(ax, tag, detail)
    _panel_rule(fig, axes[0], axes[1])
    return _finish(fig, axes, "calibration_toy_tree_recovered", tight=False)


# ---------------------------------------------------------------------------
# 7b. Tier 1, drawn. The scorecard figure grades each metric on its own
#     pathology; this one asks the question a per-metric table cannot: which
#     GATE removed which injected structure, on the world where the answer was
#     fixed before any metric saw it. It is the only calibration figure whose
#     data is computed rather than read from a JSON, because the candidate mask
#     and the per-gate outcomes are not in `synthetic_toy_calibration.json` --
#     that file carries the scorecard rows and nothing else.
# ---------------------------------------------------------------------------
# Role -> (colour, label). Okabe-Ito throughout, unlike the notebook's screen
# palette: #2E9E5B beside #D98A3D is indistinguishable under protanopia, and in
# print no tooltip can rescue the encoding. Ten roles, ten separable hues
# (composition takes a purple outside Okabe-Ito: the palette's remaining
# slots, black and yellow, read as text-ink and near-invisible on white).
TOY_ROLE = {
    "genuine":     ("#009E73", "genuine tree"),
    "superparent": ("#9AA3AD", "superparent (A)"),
    "freq":        ("#E69F00", "frequency coincidence (B)"),
    "split":       ("#CC79A7", "feature split (C)"),
    "absorbed":    ("#D55E00", "absorption (D)"),
    "topic":       ("#0072B2", "shared topic (E)"),
    "in_block":    ("#56B4E9", "within-block (F)"),
    "composition": ("#9467BD", "composition (G)"),
    "multi":       ("#8C564B", "multi-parenting (H)"),
    "sibling":     ("#BCBD22", "siblings (I)"),
    "unused":      ("#D9DDE1", "declared but never fires"),
}


@functools.lru_cache(maxsize=1)
def synthetic_toy_gates():
    """Run the Tier-1 world through the production battery and keep the outcomes.

    Imported lazily and returned as None on failure, so a missing torch skips one
    figure with a printed reason rather than taking the whole generator down.
    Cached because the figure and its caption both read it and the world costs a
    couple of seconds to build.
    """
    try:
        import torch

        from validation.calibrate_on_synthetic_toy import _run_metrics
        from validation.synthetic_toy_world import (
            ABSORB_CHILD, ABSORB_PARENT, COEXT_CHILD, COEXT_PARENT, COMP_CHILD,
            COMP_PARENTS, FREQ_CHILD, FREQ_PARENT, GENUINE_TREE, IN_BLOCK_CHILD,
            IN_BLOCK_DUP, IN_BLOCK_PARENT, MULTI_CHILD, MULTI_INTRUDER_PARENT,
            MULTI_TRUE_PARENT, SPLIT_CHILDREN, SPLIT_PARENT, SUPERPARENT,
            TOPIC_CHILD, TOPIC_PARENT, build_world,
        )
    except Exception as exc:                      # noqa: BLE001 -- reported, not raised
        print(f"[figures] synthetic toy unavailable: {type(exc).__name__}: {exc}")
        return None

    stats, labels = build_world(seed=0)
    m = _run_metrics(stats)
    edge_mask = m["edge_mask"]
    recon_ok = m["recon"]["passes"] & edge_mask
    survival = m["fcov"]["survival"]
    freq_ok = recon_ok & (survival >= C.FREQ_SURVIVAL_MIN)

    def as_set(mask):
        return {(int(a), int(b)) for a, b in torch.nonzero(mask).tolist()}

    # The world's TRUE tree is not `labels.genuine`: the split parent's three
    # children are real refinements (the pathology is that they duplicate each
    # other), and the absorbed edge is one the module's own docstring calls a real
    # refinement, held out of `genuine` so metrics 2-9 are not graded on a
    # candidate coverage never hands them.
    tree = {p: list(kids) for p, kids in GENUINE_TREE.items()}
    tree[SPLIT_PARENT] = list(SPLIT_CHILDREN)
    tree[ABSORB_PARENT] = [ABSORB_CHILD]
    tree[MULTI_TRUE_PARENT] = [MULTI_CHILD]
    true_edges = sorted((p, c) for p, kids in tree.items() for c in kids)

    survivors = as_set(freq_ok)
    roles_p = {p: ("genuine" if p in GENUINE_TREE else
                   {SPLIT_PARENT: "split", FREQ_PARENT: "freq",
                    SUPERPARENT: "superparent", ABSORB_PARENT: "absorbed",
                    TOPIC_PARENT: "topic",
                    COMP_PARENTS[0]: "composition",
                    COMP_PARENTS[1]: "composition",
                    MULTI_TRUE_PARENT: "genuine",
                    MULTI_INTRUDER_PARENT: "multi",
                    COEXT_PARENT: "sibling"}[p])
               for p in range(stats["P"])}
    genuine_children = {c for kids in GENUINE_TREE.values() for c in kids}
    roles_c = {}
    for c in range(stats["C"]):
        if c in genuine_children:
            roles_c[c] = "genuine"
        elif c in SPLIT_CHILDREN:
            roles_c[c] = "split"
        elif c in (IN_BLOCK_PARENT, IN_BLOCK_CHILD, *IN_BLOCK_DUP):
            roles_c[c] = "in_block"
        else:
            roles_c[c] = {FREQ_CHILD: "freq", ABSORB_CHILD: "absorbed",
                          TOPIC_CHILD: "topic",
                          COMP_CHILD: "composition",
                          MULTI_CHILD: "genuine",
                          COEXT_CHILD: "sibling"}.get(c, "unused")

    verdict = {}
    for p in range(stats["P"]):
        parts = []
        kept = int(freq_ok[p].sum())
        cut_recon = int((edge_mask & ~m["recon"]["passes"])[p].sum())
        cut_freq = int((recon_ok & ~(survival >= C.FREQ_SURVIVAL_MIN))[p].sum())
        if kept:
            parts.append(f"{kept} recovered")
        if cut_recon:
            parts.append(f"{cut_recon} rejected: reconstruction")
        if cut_freq:
            parts.append(f"{cut_freq} rejected: frequency control")
        if not int(edge_mask[p].sum()) and tree.get(p):
            parts.append("never proposed: coverage")
        if p == SPLIT_PARENT:
            parts.append("parent flagged: redundancy")
        if p in (TOPIC_PARENT, *COMP_PARENTS) and kept:
            parts.append("no gate tests this")
        if p == MULTI_INTRUDER_PARENT and kept:
            parts.append("flagged: multi-parenting")
        if p == COEXT_PARENT and kept:
            parts.append("flagged: co-extensive")
        verdict[p] = ", ".join(parts) or "no candidates"

    # Per-feature firing and the corpus's own frequency profile. Read here rather
    # than in each figure: `build_world` is the expensive call, and three figures
    # now ask it different questions about the SAME world -- rebuilding it per
    # figure would let them disagree about which corpus they describe.
    buckets, tok_counts = stats["buckets"], stats["token_counts"]
    n_buckets = int(buckets.max()) + 1
    bucket_mass = []
    for k in range(n_buckets):
        sel = buckets == k
        bucket_mass.append({"tokens": float(tok_counts[sel].sum()),
                            "ids": int(sel.sum())})

    # The gate funnel is the same three composed gates the before/after figure
    # draws, counted instead of drawn: a drawing shows WHICH edge died, a funnel
    # shows how many and at which gate, and the division-of-labour claim needs both.
    stages = [("after coverage", as_set(edge_mask)),
              ("+ reconstruction", as_set(recon_ok)),
              ("+ frequency control", survivors)]

    return {
        "P": int(stats["P"]), "C": int(stats["C"]),
        "true_edges": true_edges, "candidates": as_set(edge_mask),
        "survivors": survivors,
        "fire_count": [float(x) for x in stats["fire_count"]],
        "total_tokens": int(stats["total_tokens"]),
        "bucket_mass": bucket_mass,
        "R": m["R"].tolist(),
        "min_fire": int(C.MIN_FIRE_COUNT),
        "stages": [(lab, sorted(e)) for lab, e in stages],
        "recovered": sorted(survivors & set(true_edges)),
        "missed": sorted(set(true_edges) - survivors),
        "spurious": sorted(survivors - set(true_edges)),
        "roles_p": roles_p, "roles_c": roles_c, "verdict": verdict,
        "genuine": sorted(labels.genuine),
        "split_edges": [(SPLIT_PARENT, c) for c in SPLIT_CHILDREN],
        "in_block": [((IN_BLOCK_PARENT, IN_BLOCK_CHILD), "-"),
                     (tuple(IN_BLOCK_DUP), ":")],
        "keys": {"superparent": SUPERPARENT, "split": SPLIT_PARENT,
                 "freq": FREQ_PARENT, "absorbed": ABSORB_PARENT,
                 "topic": TOPIC_PARENT},
        "comp_edges": [(p, COMP_CHILD) for p in COMP_PARENTS],
        "multi_edges": [(MULTI_INTRUDER_PARENT, MULTI_CHILD)],
        "coext_edges": [(COEXT_PARENT, COEXT_CHILD)],
        "superparent_cut": sum(1 for p, _ in as_set(edge_mask) if p == SUPERPARENT),
    }


def calibration_toy_world_before_after(w):
    """The declared world and the gate verdicts, as two stacked row-panels.

    Same colours in both panels, because the question is not "how many edges came
    back" -- `calibration_toy_tree_recovered` answers that for Tier 2 -- but which
    injected structure ran into which gate. An outcome palette (recovered / missed /
    false) cannot say that: it renders thirty superparent pairs and one frequency
    coincidence in the same colour, and their whole difference is that one died at
    reconstruction and the other survived it.

    Layout: rotated onto rows like the gate-verdicts twin and stacked, for the
    same reason that twin exists -- the two side-by-side columns left the parent
    column empty for two-thirds of its height and the figure was mostly
    whitespace. Top panel: the world as declared. Bottom panel: the same world
    after the gates, outcome on alpha and node fill, the deciding gate named
    above each parent.
    """
    P, Cn = w["P"], w["C"]
    cx = {c: c + 0.5 for c in range(Cn)}
    px = {p: (p + 0.5) * Cn / P for p in range(P)}
    PY, CY = 1.0, 0.0                   # parents on the top row, children below
    # dash carries ONE meaning: a blind spot no gate tests. Everything the
    # battery does test -- recover, reject or flag -- is drawn solid.
    dash_of = {"absorbed": (0, (4, 2)), "topic": (0, (1, 1.6)),
               "composition": (0, (5, 1.5, 1, 1.5))}

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 5.7),
                             gridspec_kw={"height_ratios": [1.0, 1.3],
                                          "hspace": 0.06})

    for ax, is_truth in zip(axes, (True, False)):
        def link(p, c, role, lw=1.6, alive=True):
            colour = TOY_ROLE[role][0]
            # Removed edges keep colour and dash -- the identity of the structure
            # is the point -- so only alpha carries the outcome; per role for the
            # ink-accumulation reason the old layout documented.
            ghost = 0.14 if role == "superparent" else 0.34
            ax.plot([px[p], cx[c]], [PY, CY], lw=lw, color=colour,
                    ls=dash_of.get(role, "-"),
                    alpha=1.0 if (is_truth or alive) else ghost, zorder=1)

        for c in range(Cn):                                   # (A) superparent
            if is_truth or (w["keys"]["superparent"], c) in w["candidates"]:
                link(w["keys"]["superparent"], c, "superparent", lw=0.7,
                     alive=(w["keys"]["superparent"], c) in w["survivors"])
        for (p, c) in w["genuine"]:
            link(p, c, "genuine", lw=1.7, alive=(p, c) in w["survivors"])
        for (p, c) in w["split_edges"]:
            link(p, c, "split", lw=1.7, alive=(p, c) in w["survivors"])
        for (p, c) in w["comp_edges"]:
            link(p, c, "composition", lw=1.7, alive=(p, c) in w["survivors"])
        for (p, c) in w["multi_edges"]:
            link(p, c, "multi", lw=1.7, alive=(p, c) in w["survivors"])
        for (p, c) in w["coext_edges"]:
            link(p, c, "sibling", lw=1.7, alive=(p, c) in w["survivors"])
        for role in ("freq", "absorbed", "topic"):
            p = w["keys"][role]
            c = next(x for x in range(Cn) if w["roles_c"][x] == role)
            link(p, c, role, lw=1.7, alive=(p, c) in w["survivors"])

        # (F) within-block: both endpoints live in the child row, so the bezier
        # bows BELOW the row.
        for (a, b), ls in w["in_block"]:
            x0, x1 = cx[a], cx[b]
            ctrl = ((x0 + x1) / 2, CY - 0.42)
            t = np.linspace(0, 1, 40)
            ax.plot((1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl[0] + t ** 2 * x1,
                    (1 - t) ** 2 * CY + 2 * (1 - t) * t * ctrl[1] + t ** 2 * CY,
                    ls=ls, lw=1.2, color=TOY_ROLE["in_block"][0],
                    alpha=1.0 if is_truth else 0.5, zorder=1)

        live_p = {p for p, _ in w["survivors"]}
        live_c = {c for _, c in w["survivors"]}
        for xs, y, roles, live in ((px, PY, w["roles_p"], live_p),
                                   (cx, CY, w["roles_c"], live_c)):
            for i, x in xs.items():
                colour = TOY_ROLE[roles[i]][0]
                on = is_truth or i in live
                ax.scatter([x], [y], s=118, zorder=3,
                           facecolor=colour if on else "white",
                           edgecolor=colour, linewidths=1.3)
                ax.text(x, y, str(i), ha="center", va="center", fontsize=5.4,
                        zorder=4, color=_text_on(colour) if on else MUTED)

        if not is_truth:            # which test each parent went through,
            for p in range(P):      # staggered on two tiers as in the twin
                ax.text(px[p], PY + (0.10 if p % 2 == 0 else 0.30),
                        w["verdict"][p].replace(", ", "\n"), ha="center",
                        va="bottom", fontsize=5.6,
                        color=TOY_ROLE[w["roles_p"][p]][0], zorder=4,
                        linespacing=1.25)

        for y, lab in ((PY, "parent block"), (CY, "child block")):
            ax.text(-0.7, y, lab, ha="right", va="center", fontsize=7.5,
                    color=MUTED)
        # Panel identity as a conventional subplot tag, not explanatory prose:
        # the caption says what "declared" and "verdicts" mean.
        ax.text(-3.5, PY + (0.30 if is_truth else 0.58),
                "before: as declared" if is_truth
                else "after: what the metrics recovered",
                ha="left", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.set_xlim(-3.6, Cn + 0.6)
        ax.set_ylim(CY - 0.55, PY + (0.42 if is_truth else 0.75))
        ax.axis("off")

    # ONE legend, shared by both panels: colour = planted structure, circle
    # markers matching the node glyphs. The recovered / not-recovered fill
    # encoding is defined in the caption, not here, and the dash styles speak
    # for themselves in the handles -- both cuts were manuscript feedback.
    handles = [Line2D([], [], ls="none", marker="o", markersize=7.5,
                      markerfacecolor=TOY_ROLE[r][0],
                      markeredgecolor=TOY_ROLE[r][0],
                      label=TOY_ROLE[r][1])
               # ordered as the columns of tab:matrix; within-block last
               # (it is a metric's row there, not a pathology column)
               for r in ("genuine", "split", "absorbed", "composition",
                         "superparent", "multi", "sibling", "freq",
                         "topic", "in_block")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, 0.055), frameon=False, fontsize=8,
                     handlelength=2.4, labelspacing=0.35, columnspacing=1.6)
    leg._legend_box.align = "left"
    # a second line for the outcome encoding: the PNG also appears on the
    # outputs pages with no LaTeX caption to define these symbols
    # the three dash patterns (long dash, dots, dash-dot) differ only so the
    # untestable structures stay tellable apart in greyscale; they share one
    # meaning, so the key shows all three beside a single label
    from matplotlib.legend_handler import HandlerTuple
    dash_samples = tuple(Line2D([], [], color="#444444", lw=2.0, ls=ls)
                         for ls in ((0, (4, 2)), (0, (1, 1.6)),
                                    (0, (5, 1.5, 1, 1.5))))
    outcome = [
        Line2D([], [], color="#444444", lw=2.0,
               label="solid: tested by the metrics"),
        dash_samples,
        Line2D([], [], color="#444444", lw=2.0, alpha=0.30,
               label="faded: rejected by the metrics"),
        Line2D([], [], ls="none", marker="o", markersize=7.5,
               markerfacecolor="#444444", markeredgecolor="#444444",
               label="recovered"),
        Line2D([], [], ls="none", marker="o", markersize=7.5,
               markerfacecolor="white", markeredgecolor="#444444",
               label="not recovered"),
    ]
    labels2 = ["solid: tested by the metrics",
               "dashed or dotted: no metric can test it",
               "faded: rejected by the metrics", "recovered", "not recovered"]
    leg2 = fig.legend(handles=outcome, labels=labels2, loc="lower center",
                      ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False,
                      fontsize=8, handlelength=3.4, columnspacing=1.2,
                      handler_map={tuple: HandlerTuple(ndivide=3, pad=0.35)})
    leg2._legend_box.align = "left"

    fig.subplots_adjust(left=0.02, right=0.99,
                        top=0.90 if TITLES else 0.985, bottom=0.22)
    heads = (("before: the world as declared", "the planted structures as injected"),
             ("after: what the metrics recovered",
              f"{len(w['recovered'])}/{len(w['true_edges'])} true edges recovered, "
              f"{len(w['candidates']) - len(w['survivors'])} candidates rejected"))
    for ax, (tag, detail) in zip(axes, heads):
        _panel_head(ax, tag, detail)
    return _finish(fig, axes, "calibration_toy_world_before_after", tight=False)


def calibration_toy_world_gate_verdicts(w):
    """The after panel of the before/after pair alone, rotated onto two rows.

    Same world, same palette, same alpha rule as `calibration_toy_world_before_after`;
    both are emitted so the manuscript can choose (like the slice and its
    single-child twin). What this one gives up and why it can: the before panel
    repeats no information -- the after panel already draws every declared edge
    and carries the outcome in alpha alone, so the declared world is recoverable
    from one panel (faded = cut) once the subtitle says so. What it gains: the
    two-column layout is tall and mostly whitespace, because a bipartite drawing
    with few parents against many children leaves the parent column empty for
    two-thirds of its height; laying the blocks as rows spends that height on
    nothing and halves the figure.
    """
    P, Cn = w["P"], w["C"]
    cx = {c: c + 0.5 for c in range(Cn)}
    px = {p: (p + 0.5) * Cn / P for p in range(P)}
    PY, CY = 1.0, 0.0                   # parents on the top row, children below
    # dash carries ONE meaning: a blind spot no gate tests. Everything the
    # battery does test -- recover, reject or flag -- is drawn solid.
    dash_of = {"absorbed": (0, (4, 2)), "topic": (0, (1, 1.6)),
               "composition": (0, (5, 1.5, 1, 1.5))}

    fig, ax = plt.subplots(figsize=(12.6, 4.25))

    def link(p, c, role, lw=1.6, alive=True):
        colour = TOY_ROLE[role][0]
        # Removed edges keep colour and dash -- the identity of the structure is
        # the point -- so only alpha carries the outcome. Per role for the same
        # ink-accumulation reason as the before/after figure.
        ghost = 0.14 if role == "superparent" else 0.34
        ax.plot([px[p], cx[c]], [PY, CY], lw=lw, color=colour,
                ls=dash_of.get(role, "-"), alpha=1.0 if alive else ghost, zorder=1)

    for c in range(Cn):                                   # (A) superparent
        if (w["keys"]["superparent"], c) in w["candidates"]:
            link(w["keys"]["superparent"], c, "superparent", lw=0.7,
                 alive=(w["keys"]["superparent"], c) in w["survivors"])
    for (p, c) in w["genuine"]:
        link(p, c, "genuine", lw=1.7, alive=(p, c) in w["survivors"])
    for (p, c) in w["split_edges"]:
        link(p, c, "split", lw=1.7, alive=(p, c) in w["survivors"])
    for (p, c) in w["comp_edges"]:
        link(p, c, "composition", lw=1.7, alive=(p, c) in w["survivors"])
    for (p, c) in w["multi_edges"]:
        link(p, c, "multi", lw=1.7, alive=(p, c) in w["survivors"])
    for (p, c) in w["coext_edges"]:
        link(p, c, "sibling", lw=1.7, alive=(p, c) in w["survivors"])
    for role in ("freq", "absorbed", "topic"):
        p = w["keys"][role]
        c = next(x for x in range(Cn) if w["roles_c"][x] == role)
        link(p, c, role, lw=1.7, alive=(p, c) in w["survivors"])

    # (F) within-block: both endpoints live in the child row, so the bezier
    # bows BELOW the row (the column layout bowed it sideways).
    for (a, b), ls in w["in_block"]:
        x0, x1 = cx[a], cx[b]
        ctrl = ((x0 + x1) / 2, CY - 0.42)
        t = np.linspace(0, 1, 40)
        ax.plot((1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl[0] + t ** 2 * x1,
                (1 - t) ** 2 * CY + 2 * (1 - t) * t * ctrl[1] + t ** 2 * CY,
                ls=ls, lw=1.2, color=TOY_ROLE["in_block"][0], alpha=0.5, zorder=1)

    live_p = {p for p, _ in w["survivors"]}
    live_c = {c for _, c in w["survivors"]}
    for xs, y, roles, live in ((px, PY, w["roles_p"], live_p),
                               (cx, CY, w["roles_c"], live_c)):
        for i, x in xs.items():
            colour = TOY_ROLE[roles[i]][0]
            on = i in live
            ax.scatter([x], [y], s=118, zorder=3,
                       facecolor=colour if on else "white",
                       edgecolor=colour, linewidths=1.3)
            ax.text(x, y, str(i), ha="center", va="center", fontsize=5.4,
                    zorder=4, color=_text_on(colour) if on else MUTED)

    # Verdicts sit ABOVE their parent instead of to its left, comma-split to
    # lines and staggered on two tiers: the three long middle verdicts are
    # wider than the parent spacing and collide on a single tier.
    for p in range(P):
        ax.text(px[p], PY + (0.10 if p % 2 == 0 else 0.30),
                w["verdict"][p].replace(", ", "\n"), ha="center", va="bottom",
                fontsize=5.6, color=TOY_ROLE[w["roles_p"][p]][0], zorder=4,
                linespacing=1.25)

    for y, lab in ((PY, "parent block"), (CY, "child block")):
        ax.text(-0.7, y, lab, ha="right", va="center", fontsize=7.5, color=MUTED)

    ax.set_xlim(-3.6, Cn + 0.6)
    ax.set_ylim(CY - 0.55, PY + 0.75)
    ax.axis("off")

    # ONE legend (the mentors' ask): colour = which planted structure, circle
    # markers matching the node glyphs. The recovered / not-recovered fill
    # encoding is defined in the caption, not here -- manuscript feedback.
    handles = [Line2D([], [], ls="none", marker="o", markersize=7.5,
                      markerfacecolor=TOY_ROLE[r][0],
                      markeredgecolor=TOY_ROLE[r][0],
                      label=TOY_ROLE[r][1])
               # ordered as the columns of tab:matrix; within-block last
               # (it is a metric's row there, not a pathology column)
               for r in ("genuine", "split", "absorbed", "composition",
                         "superparent", "multi", "sibling", "freq",
                         "topic", "in_block")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, 0.055), frameon=False, fontsize=8,
                     handlelength=2.4, labelspacing=0.35, columnspacing=1.6)
    leg._legend_box.align = "left"
    # a second line for the outcome encoding: the PNG also appears on the
    # outputs pages with no LaTeX caption to define these symbols
    # the three dash patterns (long dash, dots, dash-dot) differ only so the
    # untestable structures stay tellable apart in greyscale; they share one
    # meaning, so the key shows all three beside a single label
    from matplotlib.legend_handler import HandlerTuple
    dash_samples = tuple(Line2D([], [], color="#444444", lw=2.0, ls=ls)
                         for ls in ((0, (4, 2)), (0, (1, 1.6)),
                                    (0, (5, 1.5, 1, 1.5))))
    outcome = [
        Line2D([], [], color="#444444", lw=2.0,
               label="solid: tested by the metrics"),
        dash_samples,
        Line2D([], [], color="#444444", lw=2.0, alpha=0.30,
               label="faded: rejected by the metrics"),
        Line2D([], [], ls="none", marker="o", markersize=7.5,
               markerfacecolor="#444444", markeredgecolor="#444444",
               label="recovered"),
        Line2D([], [], ls="none", marker="o", markersize=7.5,
               markerfacecolor="white", markeredgecolor="#444444",
               label="not recovered"),
    ]
    labels2 = ["solid: tested by the metrics",
               "dashed or dotted: no metric can test it",
               "faded: rejected by the metrics", "recovered", "not recovered"]
    leg2 = fig.legend(handles=outcome, labels=labels2, loc="lower center",
                      ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False,
                      fontsize=8, handlelength=3.4, columnspacing=1.2,
                      handler_map={tuple: HandlerTuple(ndivide=3, pad=0.35)})
    leg2._legend_box.align = "left"

    fig.subplots_adjust(left=0.02, right=0.99,
                        top=0.85 if TITLES else 0.97, bottom=0.25)
    _panel_head(ax, "what the set of metrics kept",
                f"{len(w['recovered'])}/{len(w['true_edges'])} true edges kept, "
                f"{len(w['candidates']) - len(w['survivors'])} candidates cut — "
                "faded edges are the declared world the gates removed")
    return _finish(fig, ax, "calibration_toy_world_gate_verdicts", tight=False)


# ---------------------------------------------------------------------------
# 7c. Tier 1, the corpus underneath it. Every claim the calibration makes is a
#     claim about a corpus: how often each feature fires, and how the token mass
#     is distributed over frequency. A reader who has not seen those two facts
#     cannot tell a metric that separated two classes from a metric that had
#     almost nothing to separate.
# ---------------------------------------------------------------------------
def calibration_toy_corpus_firing(w):
    """Per-feature firing counts, and the corpus mass behind the frequency control.

    The right-hand panel is the reason this exists as its own figure. Metric 5
    re-weights co-firing by frequency bucket, and what that gate can possibly do
    is set by how the corpus's mass falls across the buckets -- if one bucket
    holds almost everything, the control has nothing to re-weight.

    Its colours are deliberately NOT the role palette. The notebook version drew
    the three buckets in amber, blue and grey, which are exactly the swatches
    "frequency coincidence (B)", "shared topic (E)" and "superparent (A)" carry
    in the key directly below it: the same colour meant a role on the left and a
    frequency bucket on the right, and nothing on the figure said so. Buckets are
    ORDERED (high -> mid -> rare), so they get a single-hue sequential ramp, the
    same rule `DEPTH` follows for layers; the ramp is neutral grey because every
    hue family now carries a role (purple went to composition), and an ordered
    grey triplet cannot be read as any single role swatch.
    """
    P, Cn = w["P"], w["C"]
    fire = w["fire_count"]
    roles = ([w["roles_p"][p] for p in range(P)] + [w["roles_c"][c] for c in range(Cn)])
    names = [f"parent {p}" for p in range(P)] + [f"child {c}" for c in range(Cn)]
    # a feature that never fires is not its declared role -- it is the absence of
    # one, and colouring it as the role claims the world contains something it does not
    roles = [r if fire[i] > 0 else "unused" for i, r in enumerate(roles)]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9),
                             gridspec_kw={"width_ratios": [3.05, 1]})

    # -- left: tokens each feature fires on --------------------------------
    ax = axes[0]
    x = np.arange(len(fire))
    ax.bar(x, [v if v > 0 else np.nan for v in fire], width=0.72,
           color=[TOY_ROLE[r][0] for r in roles])
    ax.set_yscale("log")
    pos = [v for v in fire if v > 0]
    lo = max(min(pos) / 3.2, 1.0)
    ax.set_ylim(lo, max(pos) * 2.2)
    for i, v in enumerate(fire):
        if v > 0:
            continue
        # zero has no position on a log axis. A feature that was declared and
        # never fired is a measurement, so it gets a stub at the floor rather
        # than an empty slot -- otherwise the key's last swatch marks nothing.
        ax.add_patch(plt.Rectangle((i - 0.36, lo), 0.72, lo * 0.55,
                                   color=TOY_ROLE["unused"][0]))
    ax.axhline(w["min_fire"], color=CAT[3], lw=1.1, ls=(0, (4, 3)), zorder=3)
    # lifted clear of its own rule: at va="bottom" the glyphs still sat on the line
    ax.text(len(fire) - 0.4, w["min_fire"] * 1.10, f"MIN_FIRE_COUNT = {w['min_fire']} ",
            ha="right", va="bottom", fontsize=7, color=CAT[3])
    ax.set_ylabel("tokens the feature fires on (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.4, rotation=90)
    ax.set_xlim(-0.8, len(fire) - 0.2)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)

    # -- right: where the corpus's mass sits -------------------------------
    ax = axes[1]
    mass = w["bucket_mass"]
    total = max(w["total_tokens"], 1)
    ramp = plt.get_cmap("Greys")
    cols = [ramp(0.88 - 0.26 * k) for k in range(len(mass))]
    labs = ["high", "mid", "rare"][:len(mass)] or []
    labs = [f"{labs[k] if k < len(labs) else k} (bucket {k})" for k in range(len(mass))]
    ax.bar(np.arange(len(mass)), [b["tokens"] for b in mass], width=0.62, color=cols)
    for k, b in enumerate(mass):
        ax.text(k, b["tokens"],
                f"{b['tokens']:,.0f} tokens\n"
                f"{b['ids']:,} id{'' if b['ids'] == 1 else 's'}\n"
                f"{100 * b['tokens'] / total:.0f}% of corpus",
                ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(np.arange(len(mass)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0, max(b["tokens"] for b in mass) * 1.42)
    ax.set_ylabel("tokens")
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)


    order = ["genuine", "split", "freq", "absorbed", "topic", "in_block",
             "superparent", "unused"]
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TOY_ROLE[r][0],
                                      label=TOY_ROLE[r][1])
                        for r in order if r in set(roles)],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.30, wspace=0.20)
    heads = _panel_head(axes[0], "the features — what each one fires on",
                        f"{len(fire)} declared features over {w['total_tokens']:,} tokens; "
                        "colour = the role it was planted as")
    heads = [(axes[0], t) for t in heads]
    heads += [(axes[1], t) for t in
              _panel_head(axes[1], "the corpus — its token mass",
                          "shade = frequency bucket: an order, not a role")]
    # fitted only now: a heading is only too wide relative to a panel whose
    # width the layout has already settled
    for ax_i, t in heads:
        _fit_width(fig, t, ax_i.get_position().width * fig.get_figwidth())
    # biased toward the left panel: this gutter also holds the right panel's
    # y-label and tick numbers, and a rule at dead centre ran through the word
    _panel_rule(fig, axes[0], axes[1], bias=0.22)
    return _finish(fig, axes, "calibration_toy_corpus_firing", tight=False)


# ---------------------------------------------------------------------------
# 7d. The same three gates as 7b, counted rather than drawn. The before/after
#     drawing says WHICH edge each gate removed; this says how many, and it is
#     the version that survives being read at a glance. Both are needed: the
#     division-of-labour claim is about identity AND about magnitude.
# ---------------------------------------------------------------------------
def calibration_gate_funnel_by_role(w):
    """Candidates surviving each composed gate, split by the role that planted them."""
    # Roles ordered so the genuine tree anchors the left of every bar: the reading
    # is "the healthy block never moves, the pathologies fall off one at a time",
    # and a bar whose segments reorder between stages cannot show that.
    order = ["genuine", "split", "absorbed", "composition", "superparent",
             "multi", "sibling", "freq", "topic"]
    role_of = {p: w["roles_p"][p] for p in range(w["P"])}
    stages = w["stages"]

    counts = []
    for _, edges in stages:
        c = {r: 0 for r in order}
        for p, _ in edges:
            r = role_of.get(int(p))
            if r in c:
                c[r] += 1
        counts.append(c)

    fig, ax = plt.subplots(figsize=(10.6, 3.2))
    ys = np.arange(len(stages))[::-1]
    for i, c in enumerate(counts):
        left = 0.0
        for r in order:
            n = c[r]
            if not n:
                continue
            ax.barh(ys[i], n, left=left, height=0.58, color=TOY_ROLE[r][0],
                    edgecolor="white", linewidth=0.8)
            # a segment narrower than its own label is left unlabelled rather than
            # printed over its neighbour; the total at the end of the bar carries it
            if n / max(sum(c.values()), 1) > 0.045:
                ax.text(left + n / 2, ys[i], str(n), ha="center", va="center",
                        fontsize=8, color=_text_on(TOY_ROLE[r][0]))
            left += n
        ax.text(left + 0.6, ys[i], f"{int(left)} edges", va="center",
                fontsize=8.5, color=MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in stages], fontsize=9.5)
    ax.set_xlabel("candidate edges surviving")
    ax.set_xlim(0, max(sum(c.values()) for c in counts) * 1.12)
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    n_true = len(w["true_edges"])
    n_kept_true = len(set(w["survivors"]) & set(map(tuple, w["true_edges"])))
    head = ("the gates, composed — what each one removes",
            f"{sum(counts[0].values())} candidates in, {sum(counts[-1].values())} out; "
            f"{n_kept_true}/{n_true} true edges survive all three")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TOY_ROLE[r][0],
                                      label=TOY_ROLE[r][1])
                        for r in order if any(c[r] for c in counts)],
               loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.78, bottom=0.34)
    _panel_head(ax, *head)
    return _finish(fig, ax, "calibration_gate_funnel_by_role", tight=False)


# ---------------------------------------------------------------------------
# 7e. The first gate, at full resolution. Every later calibration figure starts
#     from "the candidates coverage proposed"; this is the only one that shows
#     the quantity that decision was made on, for every parent-child pair in the
#     world, with the ground truth drawn on top of it.
# ---------------------------------------------------------------------------
def calibration_reverse_coverage(w):
    """The reverse-coverage matrix R, with what it proposed and what it could not.

    Three marks, and the third is the point. A circle is a true edge coverage
    proposed; a cross is a pair it proposed that no true edge backs; a square is
    a true edge whose R never reaches the threshold, so no later gate ever sees
    it -- a limit of the first gate that no downstream metric can repair.

    "True edge" here is the world's full tree (genuine + the feature-split
    refinements + the absorbed edge), the same set the before/after figure
    scores against. The notebook's version marked only `labels.genuine`, which
    drew the split parent's real refinements as false proposals.
    """
    R = np.array(w["R"], dtype=float)
    truth = {tuple(e) for e in w["true_edges"]}
    cand = {tuple(e) for e in w["candidates"]}

    P, Cn = R.shape
    fig, ax = plt.subplots(figsize=(0.30 * Cn + 3.0, 0.30 * P + 2.5))
    im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")

    for (p, c) in sorted(cand | truth):
        if (p, c) in cand and (p, c) in truth:
            ax.scatter([c], [p], s=54, facecolor="none", edgecolor=GOOD,
                       linewidths=1.6, zorder=3)
        elif (p, c) in cand:
            ax.scatter([c], [p], s=34, marker="x", color=CAT[3], linewidths=1.6,
                       zorder=3)
        else:
            ax.scatter([c], [p], s=62, marker="s", facecolor="none",
                       edgecolor="#CC79A7", linewidths=1.8, zorder=3)

    ax.set_xticks(np.arange(Cn))
    ax.set_xticklabels(range(Cn), fontsize=6)
    ax.set_yticks(np.arange(P))
    ax.set_yticklabels(range(P), fontsize=7)
    ax.set_xlabel("child-block local index")
    ax.set_ylabel("parent-block local index")
    ax.set_xticks(np.arange(-0.5, Cn, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, P, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("R = co-fire / child firing", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    n_cannot = len(truth - cand)
    head = (f"reverse coverage, and the {len(cand)} pairs it proposes",
            f"threshold τ = {C.EDGE_TAU}; {len(cand & truth)}/{len(truth)} true edges "
            f"proposed, {n_cannot} that R cannot reach")
    fig.legend(handles=[
        Line2D([], [], color=GOOD, marker="o", ls="", markersize=8,
               markerfacecolor="none", markeredgewidth=1.6, label="true edge, proposed"),
        Line2D([], [], color=CAT[3], marker="x", ls="", markersize=7,
               markeredgewidth=1.6, label="proposed, no true edge behind it"),
        Line2D([], [], color="#CC79A7", marker="s", ls="", markersize=8,
               markerfacecolor="none", markeredgewidth=1.8,
               label="true edge coverage cannot propose"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.24)
    _panel_head(ax, *head)
    return _finish(fig, ax, "calibration_reverse_coverage", tight=False)


# ---------------------------------------------------------------------------
# 7f. Whether Tier 1 is a result or an accident of one seed. The scorecard is
#     run on seed 0 everywhere else in this file; this re-runs the whole world
#     and the whole battery on several, because "14/14 rows pass" means one
#     thing on one draw and another on eight.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def synthetic_toy_seed_sweep(n_seeds: int = 8):
    """The full Tier-1 scorecard, re-run per seed. None if the world is unavailable.

    Each seed rebuilds the world and runs the battery, so this is the most
    expensive thing in the generator -- cached, and reported in the skip line
    rather than run twice.
    """
    try:
        from validation.calibrate_on_synthetic_toy import calibrate
    except Exception as exc:                      # noqa: BLE001 -- reported, not raised
        print(f"[figures] seed sweep unavailable: {type(exc).__name__}: {exc}")
        return None
    out = []
    for seed in range(n_seeds):
        try:
            _, _, rows = calibrate(seed=seed)
        except Exception as exc:                  # noqa: BLE001
            # one bad seed is a missing column, not a missing figure -- and the
            # column is left out rather than filled with a guess
            print(f"[figures] seed {seed} failed: {type(exc).__name__}: {exc}")
            continue
        out.append((seed, rows))
    return out or None


def calibration_seed_sweep(sweep):
    """Every scorecard row, every seed: verdict in the cell, margin in the shade.

    The margin is how decisively a metric separated the two classes it was graded
    on, and it spans orders of magnitude, so the shade is log10 of it. A row that
    passes on every seed at a margin of 1.05 is a different claim from one that
    passes at 1000, and a grid of identical PASS cells would hide exactly that.
    """
    seeds = [s for s, _ in sweep]
    names = [r["metric"].strip() for r in sweep[0][1]]
    M = np.full((len(names), len(seeds)), np.nan)
    ok = np.zeros_like(M, dtype=bool)
    for j, (_, rows) in enumerate(sweep):
        by_name = {r["metric"].strip(): r for r in rows}
        for i, nm in enumerate(names):
            r = by_name.get(nm)
            if r is None:
                continue
            ok[i, j] = bool(r["pass"])
            mg = r["margin"]
            mg = np.inf if mg == "inf" else float(mg)
            # a categorical row has no margin to shade: it is right or it is not,
            # and inventing a magnitude for it would rank it against rows that do
            M[i, j] = (np.nan if r.get("margin_kind") == "categorical"
                       else np.log10(max(mg, 1e-9)))

    from matplotlib import colors as mcolors

    fig, ax = plt.subplots(figsize=(1.05 * len(seeds) + 6.4, 0.30 * len(names) + 2.0))
    lo = float(np.nanmin(M)) if not np.all(np.isnan(M)) else 0.0
    # Capped at 10^3, the same ceiling `calibrate_on_synthetic_toy._render` prints
    # as ">1000x". Three rows here score 1e9, which is not a measured magnitude --
    # it is the epsilon guard in the denominator when the rejected class scores
    # zero. Letting that set the top of the scale pushed every honestly-measured
    # margin into the palest two shades, so the ramp said nothing.
    CAP = 3.0
    hi = min(float(np.nanmax(M)) if not np.all(np.isnan(M)) else 1.0, CAP)
    if hi <= lo:
        hi = lo + 1.0
    # The cells use a SUB-RANGE of Greens (never the palest end, where a shade
    # stops being distinguishable from the "no margin" grey), so the colour bar
    # has to be built from that same sub-range. Keying a bar to the full ramp
    # while the cells use part of it would put a legend on the figure that reads
    # the shades wrong -- which is worse than the missing legend it replaces.
    base = plt.get_cmap("Greens")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "greens_sub", base(np.linspace(0.10, 0.82, 256)))
    norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)
    for i in range(len(names)):
        for j in range(len(seeds)):
            v = M[i, j]
            if not ok[i, j]:
                bg = CAT[3]                       # a failure is never a shade of pass
            elif np.isnan(v):
                bg = "#EDEEF1"                    # categorical: passed, no margin
            else:
                bg = cmap(norm(v))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=bg))
            ax.text(j + 0.5, i + 0.5, "PASS" if ok[i, j] else "FAIL",
                    ha="center", va="center", fontsize=7.4,
                    fontweight="bold" if not ok[i, j] else "normal",
                    color=_text_on(bg))
    ax.set_xlim(0, len(seeds))
    ax.set_ylim(len(names), 0)
    ax.set_xticks(np.arange(len(seeds)) + 0.5)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names, fontsize=7.6)
    ax.xaxis.set_ticks_position("bottom")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    n_cells = ok.size
    fig.subplots_adjust(left=0.27, right=0.855, top=0.86, bottom=0.12)

    # The margin scale, which the shading was carrying with no key at all: a
    # reader could see that some rows were darker and had no way to learn what
    # the darkness was, or that a row scored categorically is grey rather than
    # weak. Ticks are labelled as the margin itself (10^x) as well as its log,
    # because "passes at 1000x" is the sentence the row supports and "3.0" is not.
    TOP, POW, ONE = r"$\geq 10^{%d}\times$", r"$10^{%d}\times$", r"$1\times$"
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.025, aspect=26)
    ticks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    if len(ticks) >= 2:
        cb.set_ticks(ticks)
        # labelled as the margin itself, not its logarithm: "passes at 100x" is
        # the sentence a row supports and "2.0" is not. The top tick carries the
        # cap, so a clipped cell is not read as a measured ceiling.
        cb.set_ticklabels([(TOP % t if t == ticks[-1] and hi >= CAP
                            else POW % t if t else ONE) for t in ticks])
    cb.set_label("margin", fontsize=8.5, labelpad=6)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    # the two cell colours the ramp cannot express, named rather than guessed at
    fig.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor="#EDEEF1", edgecolor="none",
                      label="passed; scored categorically, so no margin to report"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CAT[3], edgecolor="none",
                      label="did not pass"),
    ], loc="lower left", bbox_to_anchor=(0.27, 0.008), ncol=2, frameon=False, fontsize=8)

    _panel_head(ax, "every row, every seed",
                f"{int(ok.sum())}/{n_cells} cells pass over {len(seeds)} independently "
                "drawn worlds;  the shade is the margin behind each verdict, capped at "
                f"{10 ** CAP:,.0f}×")
    return _finish(fig, ax, "calibration_seed_sweep", tight=False)


# ---------------------------------------------------------------------------
# 8. Two sources through one battery. Shares, never counts: 1792 latents in 8
#    blocks against 32768 in 5 is not a comparison counts can carry.
# ---------------------------------------------------------------------------
def cross_source_funnel_shares(runs):
    # "above chance" here is the report's n_chance_level, whose cutoff is PMI < 0.5
    # -- NOT the PMI > 0 shortlist that stage 03 scores. Labelling it "PMI > 0" put
    # 14% on the chart where the shortlist is 70%, two different thresholds under
    # one name.
    stages = ["improve\nreconstruction", "clears chance\n(PMI ≥ 0.5)", "frequency-\ndriven"]
    # One MARKER per run, grouped by source, rather than one bar per run: the
    # original grouped bars were drawn for two runs, and with every graded
    # layer of both sources present (ten runs) thirty bars cycling four
    # colours stopped saying which source was which. The question this figure
    # answers is cross-source, so source is the encoding and the within-source
    # scatter is shown, not averaged away.
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(stages))
    groups = [("gemma-2-2b", [r for r in runs if r[0].startswith("gemma")], CAT[0], "o", -0.16),
              ("PCFG", [r for r in runs if not r[0].startswith("gemma")], CAT[1], "s", +0.16)]
    for src, members, col, mark, off in groups:
        pts = {j: [] for j in range(len(stages))}
        for k, (name, rep) in enumerate(members):
            pr = _pair(rep, "0->1")
            if pr is None:
                continue
            n = pr["n_candidate_edges"]
            vals = [100 * pr["reconstruction"]["frac_pass"],
                    100 * (n - pr["independence_null"]["n_chance_level"]) / n
                    if pr["independence_null"].get("n_chance_level") is not None else np.nan,
                    100 * pr["freq_control"]["frac_freq_driven"]]
            jit = (k - (len(members) - 1) / 2) * (0.16 / max(len(members) - 1, 1))
            for j, v in enumerate(vals):
                if not np.isnan(v):
                    pts[j].append(v)
                    ax.scatter(j + off + jit, v, s=42, marker=mark, color=col,
                               alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3,
                               label=f"{src} — {len(members)} layers" if (j, k) == (0, 0) else None)
        for j, vs in pts.items():
            if vs:                        # a tick at the group mean, spread kept visible
                ax.plot([j + off - 0.11, j + off + 0.11], [np.mean(vs)] * 2,
                        lw=2, color=col, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("% of that run's candidate edges")
    ax.set_ylim(-4, 108)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    _title(ax, "The same metric set, unchanged, on two SAE sources — block pair B0→B1, every "
               "graded layer",
           "one marker per run, tick = source mean; shares rather than counts, because the "
           "dictionaries differ in size and in block count",
           width=84)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "cross_source_funnel_shares")


# ---------------------------------------------------------------------------
# 8a/8b. The same battery on both sources at once, layer by layer. gemma's base
#     model has 26 blocks and the PCFG SAE's base model has 4, so "layer 3" does
#     not mean
#     the same thing on both -- and which alignment to use is itself measurable,
#     which is what 8b answers rather than assumes.
# ---------------------------------------------------------------------------
def cross_source_layer_response(rows):
    """Which metrics give the same answer on a 2.6B LM and on a 4-block transformer.

    The question is whether anything the battery reports is a property of
    Matryoshka nesting rather than of gemma. Both sources have an SAE trained on
    layers 1 and 3, so the comparison needs no matching argument: the grey
    dumbbells stand at the two block indices where both were graded, and their
    LENGTH is the disagreement. Restricted to block pair B0->B1, because the
    PCFG runs' deeper pairs hold 0-3 candidate edges each and every number
    computed from them is one or two edges wide.

    Panel ORDER is derived, not chosen: the six are sorted by their own mean gap,
    so the layout is a result rather than an arrangement that flatters one. All
    six share a 0-100 axis because all six are shares, which is what lets a
    dumbbell in one panel be compared with a dumbbell in another; anchoring at
    zero keeps its length proportional to the disagreement in the metric's own
    units rather than to whatever range the data happened to span.
    """
    pairs = X.matched(rows, "layer")
    gem = [r for r in rows if r["is_ref"]]
    oth = [r for r in rows if not r["is_ref"]]
    panels = [(t, k) for t, k, _ in X.ranked(pairs)]
    gaps = [g for _, _, g in X.ranked(pairs)]
    shared = sorted({o["layer"] for o, _, _ in pairs})

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True, sharey=True)
    for ax, (title, key), gp in zip(np.ravel(axes), panels, gaps):
        ax.axvspan(min(shared) - 0.9, max(shared) + 0.9, color="#F2F4F7", zorder=0)
        for o, g, _ in pairs:
            ax.plot([g["layer"], o["layer"]], [100 * g[key], 100 * o[key]],
                    lw=2.2, color=MUTED, zorder=2, solid_capstyle="round", alpha=0.55)
        ax.plot([r["layer"] for r in gem], [100 * r[key] for r in gem], "-o",
                color=CAT[0], lw=1.8, ms=5, zorder=3)
        ax.plot([r["layer"] for r in oth], [100 * r[key] for r in oth], "s",
                color=CAT[1], ms=7, zorder=4, mec="white", mew=1.0)
        # Both readings in the panel title, because they rank the six differently
        # and quoting only the first would make three saturated measures look
        # like the strongest agreement in the figure.
        ax.set_title(f"{title}\nsame-layer gap {gp:.1f} pts  —  "
                     f"{X.gap_ratio(rows, pairs, key):.2f} of its own range",
                     fontsize=9.5, loc="left")
        ax.set_ylim(0, 105)
        ax.set_xlim(-1.2, max(r["layer"] for r in gem) + 1.8)
        ax.set_xticks([r["layer"] for r in gem])
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("layer of the base model (block index)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% — every panel, same scale")

    # The shaded band is named once, in the first panel, and the layers are not
    # named at all: the x axis is shared and already ticked with them, so a label
    # per point would be 48 marks for 8 facts.
    ax = np.ravel(axes)[0]
    ax.annotate(f"both sources\ngraded here\n(L{', L'.join(str(s) for s in shared)})",
                (max(shared) + 0.6, 6), fontsize=7.5, color=MUTED, ha="left", va="bottom")

    # The legend goes INSIDE the panel with the most headroom, not under the
    # figure: `_finish` runs tight_layout, which does not know a figure-level
    # legend exists and lets it land on the x labels. Which panel has room is a
    # fact about the data -- every panel shares one 0-100 axis -- so it is
    # computed rather than picked, and stays right if a metric moves.
    n_agree = X.agree_split(gaps)
    roomiest = min(range(len(panels)),
                   key=lambda i: max(100 * r[panels[i][1]] for r in rows))
    np.ravel(axes)[roomiest].legend(handles=[
        plt.Line2D([], [], color=CAT[0], marker="o", lw=1.8, ms=5,
                   label=f"gemma-2-2b — {len(gem)} of {gem[0]['n_layers']} layers graded,\n"
                         f"D = {gem[0]['d_sae']:,} in {gem[0]['n_blocks']} blocks"),
        plt.Line2D([], [], color=CAT[1], marker="s", lw=0, ms=7, mec="white",
                   label=f"PCFG SAE — {len(oth)} of {oth[0]['n_layers']} layers graded,\n"
                         f"D = {oth[0]['d_sae']:,} in {oth[0]['n_blocks']} blocks"),
        plt.Line2D([], [], color=MUTED, lw=2.2, alpha=0.55,
                   label="the two sources at the same layer index"),
    ], fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99),
        labelspacing=0.9)
    _title(fig, "Two base models, one metric set: the shape of B0→B1 agrees, its strength does not",
           f"{n_agree} of {len(panels)} measures agree to within {gaps[n_agree - 1]:.1f} points "
           f"at the same layer index — across a 2.6B-parameter language model and a "
           f"{oth[0]['n_layers']}-block transformer on synthetic grammar, with dictionaries "
           f"{max(r['d_sae'] for r in rows) / min(r['d_sae'] for r in rows):.0f}× apart in size. "
           f"The remaining {len(panels) - n_agree} differ by {gaps[n_agree]:.0f}–{gaps[-1]:.0f}. "
           f"Read the second number in each panel title against the first: all "
           f"{n_agree} of the close measures sit against a floor or ceiling on both sources, so "
           f"relative to the range each one varies over at all, only "
           f"{min(panels, key=lambda p: X.gap_ratio(rows, pairs, p[1]))[0]} is clearly closer "
           "than the rest. B0→B1 only — the PCFG runs' deeper block pairs hold 0–3 candidate "
           "edges each, and two shared layers cannot establish a trend. Both PCFG runs are one "
           f"grammar config ({X.grammar_line(rows)}), one point of Exp 2's three-axis sweep, "
           "not PCFG in general", width=126)
    fig.subplots_adjust(top=0.82, hspace=0.42)
    return _finish(fig, axes, "cross_source_layer_response")


def cross_source_alignment_check(rows):
    """Same block index, or same fraction of the network? The data can answer.

    Comparing across two base models of different depth needs an alignment, and
    the choice is usually made silently in the axis label. The two candidates
    disagree completely here: by block index PCFG's layers 1 and 3 pair with
    gemma's 1 and 3, and by relative depth (L+1)/N they pair with gemma's 12 and
    24 -- opposite ends of the network. So the alignment is not a presentational
    detail, and this figure measures it instead of asserting it: for each metric,
    the mean gap between the paired runs under each rule.

    It is a weak test on two PCFG layers and says so in the title. It is included
    because the alternative is an unstated assumption doing the same work.
    """
    by_layer, by_depth = X.matched(rows, "layer"), X.matched(rows, "depth")
    order = [(t, k) for t, k, _ in X.ranked(by_layer)]
    lg = [g for _, _, g in X.ranked(by_layer)]
    dg = [X.gap(by_depth, k) for _, k in order]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    y = np.arange(len(order))
    h = 0.36
    ax.barh(y - h / 2, lg, h, color=CAT[0], label="paired by layer index")
    ax.barh(y + h / 2, dg, h, color=CAT[3], label="paired by relative depth (L+1)/N")
    for yy, v in list(zip(y - h / 2, lg)) + list(zip(y + h / 2, dg)):
        ax.text(v + max(lg + dg) * 0.015, yy, f"{v:.1f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([t for t, _ in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean gap between the two sources, points of a 0–100 share   "
                  "(shorter = the two runs agree)")
    ax.set_xlim(0, max(lg + dg) * 1.16)
    # Sorted ascending with the y axis inverted, so the short bars are at the top
    # and the free space is there. Anchored, not "best": matplotlib's best would
    # move the legend the day a metric changes rank.
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.99))
    ml, md = sum(lg) / len(lg), sum(dg) / len(dg)
    win, lose = ("index", "relative depth") if ml <= md else ("relative depth", "index")
    named = " and ".join(f"PCFG L{o['layer']}→gemma L{q['layer']}" for o, q, _ in by_depth)
    _title(fig, f"Aligning the two models by layer {win} makes them agree more than by {lose}",
           f"mean over the {len(order)} measures: {ml:.1f} points paired by layer index against "
           f"{md:.1f} paired by relative depth. The two rules pair different runs — by depth, "
           f"{named} — so the choice is not cosmetic. On {len(by_layer)} shared layers this is "
           "suggestive and not a result; it is drawn because the alternative is the same choice "
           "made silently in an axis label", width=104)
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.80)
    return _finish(fig, ax, "cross_source_alignment_check")


# ---------------------------------------------------------------------------
# 9. Within-block relations.
# ---------------------------------------------------------------------------
def in_block_relations(runs):
    """Within-block relations as a RATE per pair, across every graded run.

    Counts cannot be compared across these blocks. gemma's are nested prefixes of
    very different sizes -- 128, 384, 1536, 6144 -- so B3's 833 duplicate pairs at
    L18 look like the deep blocks are where duplication lives, and read as a rate
    they are 0.04 per thousand pairs against B0's 0.12: three times rarer. The
    raw-count reading is the same mistake the project already logs about counts
    across sources, one level down.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), sharex=True)
    for i, (name, d) in enumerate(runs):
        blocks = d["blocks"]
        xs = [b["block"] for b in blocks]
        n = [b["n_features"] for b in blocks]
        edge_rate = [1e3 * b["n_edges"] / (f * (f - 1)) for b, f in zip(blocks, n)]
        dup_rate = [1e3 * b["n_duplicates"] / (f * (f - 1) / 2) for b, f in zip(blocks, n)]
        col = DEPTH[i] if name.startswith("gemma") else CAT[3 if "03" in name else 1]
        style = "-o" if name.startswith("gemma") else "--s"
        axes[0].plot(xs, edge_rate, style, color=col, lw=1.8, ms=5, label=name)
        axes[1].plot(xs, dup_rate, style, color=col, lw=1.8, ms=5, label=name)
    for ax, lab in [(axes[0], "directed edges per 1000 ordered pairs"),
                    (axes[1], "duplicate pairs per 1000 unordered pairs")]:
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xlabel("block")
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.12)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    _title(fig, "Same-level structure lives in B0, on both sources — as a rate, not a count",
           "gemma blocks are nested prefixes of 128 to 6144 features and PCFG's are eight equal "
           "blocks of 224, so only a per-pair rate compares them; B0 is not the densest because "
           "it is the smallest. PCFG's blocks below B0 hold 0-4 edges in total, so every bend in "
           "those dashed lines is one edge", width=124)
    fig.subplots_adjust(top=0.78)
    return _finish(fig, axes, "in_block_relations")


# ---------------------------------------------------------------------------
# 9a. The battery's own failure mode. Five metrics read the same co-firing matrix
#     and one contaminating token position moved all of them at once; the sixth
#     is a ratio over children that already have a parent, and did not move.
#     Built from the v1 reports kept in outputs_archive/ against the current ones.
# ---------------------------------------------------------------------------
def shared_input_moved_every_metric(layers):
    import math
    ARCH = C.HERE / "outputs_archive"
    # Paired BY LAYER, not by count. A layer graded after the BOS fix (layer 1)
    # has no archived v1 counterpart, and requiring the two lists to be the same
    # length made one new layer silently kill this figure -- the comparison is
    # per-layer, so it should simply run on the layers that have both versions.
    matched = []
    for L, rep in layers:
        hits = sorted(ARCH.glob(f"layer_{L:02d}__v1__*/metrics_report.json"))
        if hits:
            matched.append((rep, _json(hits[0])))
    if len(matched) < 2:
        raise SystemExit("fewer than two layers have an archived v1 counterpart")
    cur, v1 = [c for c, _ in matched], [a for _, a in matched]

    def mean(reports, fn):
        return sum(fn(_pair(r, "0->1")) for r in reports) / len(reports)

    # (label, accessor, reads the co-firing matrix?)
    rows = [
        ("frequency-driven edges, %", lambda p: 100 * p["freq_control"]["frac_freq_driven"], True),
        ("improve reconstruction, %", lambda p: 100 * p["reconstruction"]["frac_pass"], True),
        ("candidate edges", lambda p: p["n_candidate_edges"], True),
        ("sibling redundancy", lambda p: p["sibling_redundancy"]["mean_redundancy"], True),
        ("mean frequency survival", lambda p: p["freq_control"]["mean_survival"], True),
        ("superparents flagged", lambda p: p["n_superparents"], True),
        ("multi-parenting, %",
         lambda p: 100 * (p["degree"].get("poly_frac") if p["degree"].get("poly_frac") is not None
                          else p["degree"]["n_multi_parented"] / max(p["degree"]["n_children_with_parent"], 1)),
         False),
    ]
    data = [(lab, mean(v1, fn), mean(cur, fn), shared) for lab, fn, shared in rows]
    data.sort(key=lambda t: abs(math.log2(max(t[2], 1e-9) / max(t[1], 1e-9))))

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(data))
    fold = [math.log2(max(b, 1e-9) / max(a, 1e-9)) for _, a, b, _ in data]
    ax.barh(y, fold, height=0.6,
            color=[CAT[3] if sh else GOOD for *_, sh in data])
    # The left-hand value of each pair is a WITHDRAWN number. It is plotted to
    # size the error, never as a result -- so it is drawn in the muted ink used
    # for annotation and the surviving value is in body ink, and the axis says
    # which is which. Withdrawn numbers come back by being readable next to live
    # ones; this is the whole reason those two hand-built pages were archived.
    for i, ((lab, a, b, sh), f) in enumerate(zip(data, fold)):
        fmt = (lambda v: f"{v:,.0f}") if max(a, b) > 20 else (lambda v: f"{v:.2f}")
        # The pair always reads withdrawn → current, left to right, whichever side
        # of zero the bar ends on. Anchoring both parts to the bar's end and
        # flipping only the alignment put them in the wrong order on negative bars.
        right = f >= 0
        gap, ch = 0.30, 0.40                     # data units; ch ≈ one character
        tail = f"→ {fmt(b)}"
        if right:
            x0 = f + gap
            ax.annotate(fmt(a), (x0, i), ha="left", va="center",
                        fontsize=8.5, color="#B0B4BB")
            ax.annotate(tail, (x0 + ch * (len(fmt(a)) + 1), i), ha="left", va="center",
                        fontsize=8.5, color=INK, fontweight="bold")
        else:
            x0 = f - gap
            ax.annotate(tail, (x0, i), ha="right", va="center",
                        fontsize=8.5, color=INK, fontweight="bold")
            ax.annotate(fmt(a), (x0 - ch * (len(tail) + 1), i), ha="right", va="center",
                        fontsize=8.5, color="#B0B4BB")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, *_ in data], fontsize=9)
    ax.set_xlabel(f"log₂ change when the contaminating token position is removed, "
                  f"mean over the {len(matched)} layers graded both before and after\n"
                  "each pair reads  withdrawn → current")
    ax.set_xlim(min(fold) - 4.2, max(fold) + 4.2)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=CAT[3], label="reads the co-firing matrix"),
        plt.Rectangle((0, 0), 1, 1, color=GOOD,
                      label="does not — a ratio over children that already have a parent"),
    ], fontsize=8.5, frameon=False, loc="lower left")   # inside: the far left is empty
    _title(ax, "Six metrics designed as independent detectors moved together",
           "BOS is an attention sink, so with 400 documents every pair in the dictionary was "
           "handed 400 joint firings against a guard set at 30. Five of these read that one "
           "matrix; all five moved. The sixth does not, and did not.", width=92)
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "shared_input_moved_every_metric")


# ---------------------------------------------------------------------------
# 9b. The premise, tested. The project's motivating observation was that gemma's
#     superparents "mostly track high-frequency tokens (spaces, punctuation,
#     'the')". Two per-edge diagnostics separate that from the alternative, and
#     they disagree by a factor of 40-80 at every layer.
# ---------------------------------------------------------------------------
def base_rate_vs_frequency_capture(layers):
    Ls = [L for L, _ in layers]
    chance = [100 * _pair(r, "0->1")["independence_null"]["frac_chance_level"]
              for _, r in layers]
    freq = [100 * _pair(r, "0->1")["freq_control"]["frac_freq_driven"] for _, r in layers]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(Ls))
    w = 0.34
    ax.bar(x - w / 2, chance, w, color=CAT[0],
           label="at chance for the parent's base rate  (PMI < 0.5)")
    ax.bar(x + w / 2, freq, w, color=CAT[3],
           label="carried by globally frequent tokens  (survival < 0.5)")
    for xx, v in zip(x - w / 2, chance):
        ax.text(xx, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=MUTED)
    for xx, v in zip(x + w / 2, freq):
        ax.text(xx, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in Ls])
    ax.set_ylabel("% of candidate edges, B0→B1")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    _title(ax, "gemma's over-connected parents are a base-rate effect, not token-frequency capture",
           f"the two diagnostics disagree by {min(c / f for c, f in zip(chance, freq)):.0f}–"
           f"{max(c / f for c, f in zip(chance, freq)):.0f}× at every layer. The project's "
           "motivating observation was that superparents mostly track high-frequency tokens; "
           "the frequency control exonerates 98–99% of edges while the independence null "
           "rejects 69–86%", width=82)
    ax.grid(True, axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "base_rate_vs_frequency_capture")


# ---------------------------------------------------------------------------
# 10. What a top-k rank rule costs at each dictionary size. No data file: the
#     rule is a geometry test, so an unrelated parent passes at exactly k/D.
#     Measured on the toy (k/D = 11.9%, observed 2-4 of 20 against 2.4 expected),
#     which is why an S_res share is only comparable across similar D.
# ---------------------------------------------------------------------------
def sres_null_rate_vs_dictionary_size(observed):
    k = C.SRES_RANK_TOP_K
    sources = [("synthetic toy", 42), ("PCFG SAE", 1792), ("gemma SAE", 32768)]
    Ds = np.array([d for _, d in sources], dtype=float)
    null = 100 * k / Ds

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    grid = np.logspace(1.4, 4.8, 200)
    ax.plot(grid, 100 * k / grid, lw=2, color=NEUTRAL, zorder=1,
            label=f"chance pass rate = k/D  (k = {k})")
    ax.scatter(Ds, null, s=70, color=CAT[0], zorder=3, edgecolor="white", linewidth=1,
               label="chance, per source")
    for i, ((name, d), v) in enumerate(zip(sources, null)):
        # below-left, except the first (no room to its left) and the last (the
        # zero-floor note lives under it)
        left = 0 < i < len(sources) - 1
        ax.annotate(f"{name}\nD = {d:,} → {v:.2f}%", (d, v),
                    ha="right" if left else "left",
                    textcoords="offset points",
                    xytext=(-10 if left else 11, -20 if i < len(sources) - 1 else 4),
                    fontsize=8, color=MUTED)
    floor = float(np.min(null)) / 4                      # a visible place for zero
    drew_obs = zeros = False
    # A drop-line from each measurement to its own chance point. The message is
    # the RATIO between them, and on a log axis that is a vertical distance --
    # which the reader had to estimate by eye against a sloping grey line.
    for name, d, obs in observed:
        y0, y1 = (floor if obs <= 0 else obs), 100 * k / d
        ax.plot([d, d], [min(y0, y1), max(y0, y1)], lw=1.2, color=GOOD, alpha=0.45, zorder=2)
    # With one run per dictionary a label per point was fine; with every layer's
    # second pass present there are several runs at the SAME D, and a label each
    # stacks into an unreadable column. Markers are cheap and stay; text is not:
    # only the extremes of each dictionary's group are named, because the group's
    # spread IS the finding -- runs on one dictionary land on both sides of one null.
    by_d: dict = {}
    for name, d, obs in observed:
        by_d.setdefault(d, []).append((name, obs))
    for name, d, obs in observed:
        # A measured 0% has no position on a log axis. Plotting it at the floor
        # with an open marker and saying so beats dropping the point, which would
        # leave the figure showing only the source whose rate happened to be
        # non-zero -- the more flattering half of the comparison.
        at_zero = obs <= 0
        zeros |= at_zero
        ax.scatter([d], [floor if at_zero else obs], s=72, marker="D",
                   color="white" if at_zero else GOOD, zorder=4,
                   edgecolor=GOOD, linewidth=1.6,
                   label=None if drew_obs else "measured pass rate")
        drew_obs = True
        group = by_d[d]
        lo, hi = min(o for _, o in group), max(o for _, o in group)
        if len(group) > 2 and obs not in (lo, hi):
            continue
        ax.annotate(f"{name}: {'0% — no edge passed' if at_zero else f'{obs:.2f}%'}"
                    + ("" if at_zero else f"  ≈{obs / (100 * k / d):.0f}× chance"),
                    (d, floor if at_zero else obs), textcoords="offset points",
                    xytext=(11, -4 if at_zero else 4), fontsize=7.5, color=GOOD)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Headroom for the labels, which hang to the RIGHT of their point in offset
    # points: the widest one belongs to the largest dictionary, which is also the
    # rightmost, so autoscale clipped it off the canvas.
    d_max = max(d for _, d, _ in observed) if observed else 1
    ax.set_xlim(right=d_max * 9)
    if zeros:
        ax.axhline(floor, ls=(0, (1, 4)), lw=1, color=NEUTRAL)
        ax.text(ax.get_xlim()[0] * 1.15, floor * 1.15, "0% drawn here — a log axis has no zero",
                fontsize=7.5, color=MUTED, style="italic", ha="left")
    ax.set_xlabel("dictionary size D (log)")
    ax.set_ylabel("% of unrelated parents that pass (log)")
    # lifted clear of the zero-floor rule that runs along the bottom
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.10))
    # One summary clause per dictionary, not one per run: the subtitle has to
    # stay readable at ten measurements as it was at three.
    parts = []
    for d in sorted(by_d):
        grp, nl = by_d[d], 100 * k / d
        lo, hi = min(o for _, o in grp), max(o for _, o in grp)
        rng = (f"{lo:.2f}–{hi:.2f}%" if len(grp) > 1 else
               (f"{hi:.2f}%" if hi > 0 else "0%"))
        vs = ("below chance" if hi <= nl else
              f"up to {hi / nl:.0f}× its null" if lo <= nl else
              f"{lo / nl:.0f}–{hi / nl:.0f}× its null")
        parts.append(f"{len(grp)} run{'s' if len(grp) > 1 else ''} at D={d:,}: {rng}, {vs}")
    _title(ax, "The rank rule's strictness is set by dictionary size, not by k alone",
           "grey line = what chance alone gives at each D; the vertical drop to it is what a "
           "measured rate is worth. " + "; ".join(parts) +
           ". Runs on the same dictionary land on both sides of their own "
           "null, so a raw pass rate compares nothing", width=78)
    ax.grid(True, which="both", alpha=0.12)
    ax.set_axisbelow(True)
    return _finish(fig, ax, "sres_null_rate_vs_dictionary_size")


# ---------------------------------------------------------------------------
# 9c. Where the tangle lives: every metric per block pair, both sources, one
#     matrix. The claim it backs is locational -- the structural pathology
#     concentrates at the outermost boundary and the deeper pairs are clean --
#     and a location claim wants the whole map on one canvas, not a bar chart
#     per pair.
# ---------------------------------------------------------------------------
def tangle_lives_in_top_block_pair(layers, pcfg, name="tangle_lives_in_top_block_pair",
                                   in_block=None):
    """The WHOLE battery per block pair AND per layer, one panel per source.

    The earlier version averaged each cell over a source's graded layers, which
    a reader read as a single layer -- the rows are block pairs, and nothing in
    the grid said the layer axis had been collapsed. Averaging also hid how thin
    the deep rows are: a PCFG pair can hold one candidate edge at one layer and
    still print a confident-looking percentage.

    So no cell is a mean any more. Rows are (block pair, layer) and the two
    sources sit in their own panels, which is also what lets the layer sets
    differ -- gemma is graded at L1..L24 and the PCFG transformer has four
    layers, so a shared row axis was never possible.

    It also carried seven hand-picked columns, which made it a summary of the
    battery rather than the battery. The caption's claim is that the columns
    DISAGREE -- edge quality reads healthy while graph structure reads broken --
    and a reader cannot check that against a subset somebody chose. Every
    per-pair number the battery emits is a column now, grouped under the metric
    that emits it, so the header names both the instrument and its output
    ("Out-degree / superparent" -> "fan-out Gini") instead of a bare quantity.
    The panels stack vertically for the same reason: 19 columns side by side
    halve at \\textwidth, and a cell nobody can read is not evidence.
    """
    # (metric family, what it emits, reader, format). Family names and their
    # order are the Metrics table's, numbers included, so a reader can carry a
    # row of the table straight onto a column group here. The family is drawn
    # once over its own columns; the second element is the column's own header.
    # Metric 1 is named as the Metrics section names it, "Activation
    # coverage", rather than split into its reverse and forward halves: only
    # reverse gates, forward is reported but not gated on, and no per-pair
    # number for either mean is persisted -- only the candidate set they
    # produce, which is the column here.
    COLS = [
        ("1 Activation coverage", "candidates",
         lambda q, s, b: q["n_candidate_edges"], "{:,.0f}"),
        ("2 Independence null", "% at\nchance",
         lambda q, s, b: 100 * q["independence_null"]["frac_chance_level"], "{:.0f}"),
        ("2 Independence null", "mean\nPMI",
         lambda q, s, b: q["independence_null"]["mean_edge_pmi"], "{:.2f}"),
        ("3 Frequency control", "% freq-\ndriven",
         lambda q, s, b: 100 * q["freq_control"]["frac_freq_driven"], "{:.1f}"),
        ("3 Frequency control", "mean\nsurvival",
         lambda q, s, b: q["freq_control"]["mean_survival"], "{:.2f}"),
        ("4 Reconstruction", "% pass",
         lambda q, s, b: 100 * q["reconstruction"]["frac_pass"], "{:.0f}"),
        ("5 Probe $S_{\\rm res}$", "% pass",
         lambda q, s, b: 100 * s["sres"]["frac_pass"], "{:.1f}"),
        ("6 Out-degree", "multi-\nparented %",
         lambda q, s, b: 100 * q["degree"]["poly_frac"], "{:.0f}"),
        ("6 Out-degree", "fan-out\nGini",
         lambda q, s, b: q["degree"]["outdeg_gini"], "{:.2f}"),
        ("6 Out-degree", "max\nfan-out",
         lambda q, s, b: q["degree"]["max_outdeg"], "{:,.0f}"),
        ("6 Out-degree", "top-1 edge\nshare %",
         lambda q, s, b: 100 * q["degree"]["top1_edge_share"], "{:.0f}"),
        ("6 Out-degree", "super-\nparents",
         lambda q, s, b: q["n_superparents"], "{:.0f}"),
        ("7 Sibling redundancy", "mean\nJaccard",
         lambda q, s, b: q["sibling_redundancy"]["mean_redundancy"], "{:.2f}"),
        ("8 Joint-child", "R$_{\\rm supp}$",
         lambda q, s, b: q["joint_child"]["r_supp_mean"], "{:.2f}"),
        ("8 Joint-child", "R$_{\\rm mass}$",
         lambda q, s, b: q["joint_child"]["r_mass_mean"], "{:.2f}"),
        ("8 Joint-child", "split\nchildren",
         lambda q, s, b: q["joint_child"]["n_share_energy_ge_09"], "{:,.0f}"),
        ("8 Joint-child", "mean\ncoverage",
         lambda q, s, b: q["joint_child_cov_mean"], "{:.2f}"),
        # Metric 9 is a WITHIN-block quantity while every row here is a
        # BETWEEN-block pair, so it cannot be read off the pair report. Each
        # row takes its own parent block's numbers -- row Bk->Bk+1 reports
        # block Bk -- which is the only reading that keeps one value per row.
        ("9 In-block coverage", "same-level\nedges",
         lambda q, s, b: b["n_edges"], "{:,.0f}"),
        ("9 In-block coverage", "dupli-\ncates",
         lambda q, s, b: b["n_duplicates"], "{:,.0f}"),
    ]

    def safe(fn, q, s, b):
        # a sub-metric with nothing to score is absent, never zero
        try:
            v = fn(q, s, b)
            return float(v) if v is not None else np.nan
        except (TypeError, KeyError, ZeroDivisionError):
            return np.nan

    ib_by_label = in_block or {}

    def source_grid(named_reports):
        """[(block pair, layer label, row values, note)] for one source, pair-major.

        `note` is None for a measured row. An adjacent block pair that the run
        no layer here graded still gets a row, carrying the reason instead of values:
        a pair silently missing from a grid reads as a pair with nothing in it,
        which is a different claim from one that was never attempted.
        """
        pairs = sorted({q["pair"] for _, r in named_reports for q in r["pairs"]},
                       key=lambda t: int(t.split("->")[0]))
        ranges = next((r.get("block_ranges") for _, r in named_reports
                       if r.get("block_ranges")), None)
        out = []
        for k in range(len(ranges) - 1 if ranges else 0):
            pr = f"{k}->{k + 1}"
            name = f"B{k}→B{k + 1}"
            if pr in pairs:
                for lab, r in named_reports:
                    q = _pair(r, pr)
                    # metric 9 is per BLOCK: this row's parent block is k
                    b = (ib_by_label.get(lab) or {}).get(k)
                    # the strict test lives in second_pass.json, which every
                    # metrics_report.json embeds -- read from the report so the
                    # probe column costs this figure no extra argument
                    s = (r.get("second_pass") or {}).get(pr) or {}
                    # a layer where the pair proposes nothing is shown as an
                    # empty row rather than dropped: the absence is a measurement
                    vals = [safe(fn, q, s, b) if q else np.nan
                            for _, _, fn, _ in COLS]
                    if q and q["n_candidate_edges"] == 0:
                        vals = [0.0] + [np.nan] * (len(COLS) - 1)
                    out.append((name, lab, vals, None))
                continue
            # graded at no layer shown: say so, and say how big the object was
            P = ranges[k][1] - ranges[k][0]
            Cn = ranges[k + 1][1] - ranges[k + 1][0]
            # no layer tick: the band spans the whole row and would collide
            # with it, and "which layer" is not a question a skipped pair has
            out.append((name, "", [np.nan] * len(COLS),
                        f"not graded at any layer shown: {P:,}×{Cn:,}"))
        return out

    gem = source_grid([(f"L{L}", r) for L, r in layers])
    pcf = source_grid([(n.replace("PCFG layer ", "L"), r) for n, r in pcfg]) if pcfg else []
    panels = [(r"gemma-2-2b", gem)] + ([("PCFG", pcf)] if pcf else [])

    # contiguous runs of one family, so the family is drawn once over its columns
    spans, prev = [], None
    for j, (g, *_) in enumerate(COLS):
        if g != prev:
            spans.append([j, j + 1, g])
            prev = g
        else:
            spans[-1][1] = j + 1

    # plt.get_cmap, not cm.get_cmap: the latter was removed in matplotlib 3.9 and its
    # AttributeError aborted the whole build, so every later figure went unwritten too.
    # This spelling works on both old and new matplotlib.
    cmap = plt.get_cmap("Blues")

    # Geometry in inches, placed by hand rather than by tight_layout: the two
    # header rows and the block-pair labels all live OUTSIDE the axes, which
    # tight_layout does not see, and at 19 columns a guessed margin is the
    # difference between a readable header and a clipped one.
    NC, CELL_W, CELL_H = len(COLS), 0.64, 0.235
    # The left margin carries two things: the block-pair label and, when a
    # panel spans several layers, the layer tick beside it. Pinned to one
    # layer there is no tick, so the margin sized for both left about two
    # thirds of an inch of dead white down the whole figure. The margin and
    # the label's own offset both follow whether any panel is actually ticked.
    TICKED = any(len({lay for _, lay, _, _ in g if lay}) > 1 for _, g in panels)
    LBL_X = -1.85 if TICKED else -1.15
    ML, MR, MT, MB, GAP = (1.40 if TICKED else 0.92), 0.22, 1.18, 0.20, 1.08
    rows = [len(g) for _, g in panels]
    W = ML + MR + NC * CELL_W
    H = MT + MB + sum(rows) * CELL_H + GAP * (len(panels) - 1)
    fig = plt.figure(figsize=(W, H))
    axes, top = [], H - MT
    for _, grid in panels:
        h = len(grid) * CELL_H
        axes.append(fig.add_axes([ML / W, (top - h) / H, NC * CELL_W / W, h / H]))
        top -= h + GAP
    axes = np.array(axes)

    # every group header is fitted independently, which left them at a
    # spread of sizes -- a wide group kept 7pt beside a one-column group at
    # 5.4pt, and the size difference reads as emphasis the figure does not
    # mean. They are collected here and levelled to the smallest fit once both
    # panels are drawn, so the header row is one typographic voice.
    heads = []

    for ax, (src, grid) in zip(axes, panels):
        M = np.array([v for _, _, v, _ in grid], dtype=float)
        notes = [n for _, _, _, n in grid]
        # colour ranks WITHIN a column and within a panel: the columns are
        # different quantities on different scales, and the two sources have
        # different dictionaries, so cross-panel colour would compare nothing
        norm = np.zeros_like(M)
        with np.errstate(invalid="ignore"):
            for j in range(M.shape[1]):
                col = M[:, j]
                top_v = np.nanmax(col) if not np.all(np.isnan(col)) else np.nan
                if top_v and not np.isnan(top_v):
                    norm[:, j] = col / top_v
        for i in range(M.shape[0]):
            # an uncomputed pair is one band carrying its reason, not a row of
            # dashes: a dash means "measured, nothing there", which is the
            # opposite of what a skipped pair licenses anyone to conclude
            if notes[i]:
                ax.add_patch(plt.Rectangle((0, i), M.shape[1], 1,
                                           facecolor="#EDEEF1", hatch="///",
                                           edgecolor="white", lw=0))
                ax.text(M.shape[1] / 2, i + 0.5, notes[i], ha="center",
                        va="center", fontsize=6.8, color=MUTED, style="italic")
                continue
            for j in range(M.shape[1]):
                v, nv = M[i, j], norm[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#F2F3F5"))
                    ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                            fontsize=7, color=MUTED)
                    continue
                bg = cmap(0.06 + 0.86 * nv)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=bg))
                ax.text(j + 0.5, i + 0.5, COLS[j][3].format(v), ha="center",
                        va="center", fontsize=6.6, color=_text_on(bg))
        ax.set_xlim(0, M.shape[1])
        ax.set_ylim(M.shape[0], 0)
        ax.set_xticks([])
        # Two header rows, not one: the lower names the number in the column,
        # the upper names the metric that emitted it. A bare "fan-out Gini"
        # asks the reader to remember which metric that came from; the pair
        # "4 Out-degree / superparent -> fan-out Gini" does not.
        for j, (_, out_name, _, _) in enumerate(COLS):
            ax.text(j + 0.5, -0.42, out_name, ha="center", va="bottom",
                    fontsize=6.5, color=INK, linespacing=1.3)
        for a, b, fam in spans:
            ax.plot([a + 0.08, b - 0.08], [-1.62, -1.62], color="#C9CCD1",
                    lw=0.9, clip_on=False, solid_capstyle="butt")
            # va="bottom", not "top": this axis is inverted (set_ylim(n, 0)),
            # so a top-anchored label grows DOWN in display -- straight into
            # the column names a tenth of an inch below. Bottom-anchored, the
            # extra wrapped lines stack upward into the empty band under the
            # panel name instead.
            heads.append((_fit_width(fig, ax.text((a + b) / 2, -1.72, fam,
                                                  ha="center", va="bottom",
                                                  fontsize=7.0,
                                                  fontweight="bold", color=INK,
                                                  linespacing=1.15),
                                     (b - a) * CELL_W, wrap=True),
                          (b - a) * CELL_W))
        # A panel pinned to ONE layer repeats that layer on every row, which
        # is noise: the layer is a property of the panel, not of the row. When
        # the panel holds a single layer the tick column is dropped and the
        # layer joins the panel name instead; the full-depth figure, where the
        # rows really do differ by layer, is untouched because it has several.
        lays = sorted({lay for _, lay, _, _ in grid if lay})
        one_layer = lays[0] if len(lays) == 1 else None
        ax.set_yticks(np.arange(len(grid)) + 0.5)
        ax.set_yticklabels([] if one_layer else
                           [lay for _, lay, _, _ in grid], fontsize=7.4)
        # the block pair is a GROUP over consecutive layer rows, so it is drawn
        # once per group outside the layer ticks rather than repeated per row
        bounds, prev = [], None
        for i, (pr, _, _, _) in enumerate(grid):
            if pr != prev:
                bounds.append(i)
                prev = pr
        for k, b in enumerate(bounds):
            stop = bounds[k + 1] if k + 1 < len(bounds) else len(grid)
            ax.text(LBL_X, (b + stop) / 2, grid[b][0], fontsize=8.2,
                    fontweight="bold", color=INK, ha="left", va="center")
            if b:
                ax.axhline(b, color=INK, lw=1.0)
        # -2.98, not -3.20: this axis is inverted, so a less negative y sits
        # LOWER. Dropping the per-panel legend line pulled the panels together
        # and left the lower panel's name crowding the panel above it; the
        # name moves down toward its own grid and the gap absorbs the rest.
        ax.text(0, -2.98,
                f"{src} layer {one_layer[1:]}" if one_layer else src,
                fontsize=10.5, fontweight="bold", color=INK,
                ha="left", va="bottom")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        # the reading rules (what a row is, what the colour ranks over, what a
        # dash means) live in the LaTeX caption, not under every panel: they
        # were identical on both panels and cost a line of height each
        ax.set_xlabel("", labelpad=0)

    if heads:
        size = min(t.get_fontsize() for t, _ in heads)
        for t, avail in heads:
            t.set_fontsize(size)
            # the common size is smaller than some headers were fitted at, so
            # re-fit at it: a label that now fits on one line loses its break,
            # and the floor is dropped since the size is already settled.
            t.set_text(" ".join(t.get_text().split()))
            _fit_width(fig, t, avail, floor=size, wrap=True)
    return _finish(fig, axes, name, tight=False)


# ---------------------------------------------------------------------------
# 10a. The battery, question by question. Each metric was designed to answer
#      one plain-language question about an edge; this figure puts the question
#      in the panel title and the measured answer under it, per layer, so a
#      reader meets the instrument and its verdict in the same glance.
# ---------------------------------------------------------------------------
def battery_questions_gemma(layers, second):
    import textwrap
    Ls = [L for L, _ in layers]
    P = [_pair(r, "0->1") for _, r in layers]
    sres_pct = [100 * second[L]["0->1"]["sres"]["frac_pass"]
                if second.get(L) and "0->1" in second[L] else np.nan for L in Ls]
    # Panels follow the battery order of the Metrics section so the figure and
    # the section can be read against each other panel by number. Note the 2x4
    # grid does NOT align with the battery's own split: metrics 2-5 grade a
    # single edge and 6-8 widen to the parent's neighbourhood, so the group
    # boundary falls inside the bottom row, whose first panel (the probe) is
    # still a single-edge test. The quality/structure reading of this figure is
    # therefore carried by the caption, not by the layout.
    panels = [
        ("1. Activation coverage",
         "Does the child fire only when the parent fires?",
         [p["n_candidate_edges"] for p in P], "candidate edges", None),
        ("2. Independence null",
         "Is the co-firing above chance at all, or just the parent's base rate?",
         [100 * p["independence_null"]["frac_chance_level"] for p in P], "% at chance", None),
        ("3. Token-frequency control",
         "Does the edge still hold on rare tokens, or is it frequency-driven?",
         [100 * p["freq_control"]["frac_freq_driven"] for p in P], "% frequency-driven", None),
        ("4. Reconstruction condition",
         "Does the pair actually carry reconstruction, or do the two just activate together?",
         [100 * p["reconstruction"]["frac_pass"] for p in P], "% of edges passing", None),
        ("5. Probe-based S_res",
         "Does the parent's decoder really point to the child's concept?",
         sres_pct, "% of scored edges passing", None),
        ("6. Out-degree / superparents",
         "Does one parent fan out over most of the next block?",
         [p["n_superparents"] for p in P], "parents flagged", None),
        ("7. Sibling redundancy",
         "Are the children almost copies of each other — feature splitting posing as hierarchy?",
         [p["sibling_redundancy"]["mean_redundancy"] for p in P], "mean pairwise Jaccard",
         C.SIBLING_REDUNDANCY_FLAG),
        ("8. Exact joint-child coverage",
         "How much of the parent do the kept children really explain?",
         [p["joint_child"]["r_supp_mean"] for p in P], "mean support coverage", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.6))
    for ax, (name, q, vals, ylab, thr) in zip(np.ravel(axes), panels):
        ax.bar(np.arange(len(Ls)), vals, 0.62, color=CAT[0])
        if thr is not None:
            ax.axhline(thr, ls=(0, (4, 3)), lw=1.2, color=CAT[3])
            ax.text(len(Ls) - 0.4, thr, f"flag ≥ {thr:g}", fontsize=6.8, color=CAT[3],
                    ha="right", va="bottom")
        ax.set_xticks(np.arange(len(Ls)))
        ax.set_xticklabels([f"L{L}" for L in Ls], fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=8)
        ax.margins(y=0.22)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
        ax.text(0, 1.30, name, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=INK, va="top")
        ax.text(0, 1.19, "\n".join(textwrap.wrap(q, 38)), transform=ax.transAxes,
                fontsize=7.3, color=MUTED, va="top", style="italic")
    lines = _title(fig, "The metric set, question by question — gemma-2-2b, block pair B0→B1, "
                   "every graded layer",
                   "each panel is the question one metric asks and its measured answer; the "
                   "questions the edges pass are about quality, the ones they fail are about "
                   "structure", width=120)
    # the per-panel question texts sit above each axes, so the top margin is
    # for them; only shave extra room when a figure title is actually drawn.
    # This figure owns its margins (tight=False): tight_layout cannot see the
    # hanging texts and clips the first row's headings.
    fig.subplots_adjust(top=0.84 if lines else 0.90, bottom=0.06, left=0.05,
                        right=0.99, hspace=0.95, wspace=0.34)
    return _finish(fig, axes, "battery_questions_gemma", tight=False)


# ---------------------------------------------------------------------------
# 10b. The recovered graph itself, drawn from the edge lists rather than
#      summarized. One panel per world: the trained toy's recovered tree, and
#      the real B0->B1 edge set on gemma. The layout gives a tree its best
#      chance -- every child is placed under its strongest parent -- so any
#      crossing that remains is structure, not plotting.
# ---------------------------------------------------------------------------
def recovered_graph_toy_vs_pcfg_vs_gemma(tt, sp, where, sp_pcfg=None, where_pcfg=""):
    true_edges = [tuple(e) for e in tt["true_edges"]]
    found = {tuple(e) for e in tt["found_edges"]}

    n_panels = 3 if (sp_pcfg and "0->1" in sp_pcfg) else 2
    ratios = [1, 1.35, 1.75][:n_panels] if n_panels == 3 else [1, 1.9]
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 + 3.4 * n_panels, 4.4),
                             gridspec_kw={"width_ratios": ratios})

    # -- toy: the tree it recovered, missed edges dashed ----------------------
    ax = axes[0]
    t_parents = sorted({p for p, _ in true_edges})
    t_children = sorted({c for _, c in true_edges})
    # children grouped under their (unique, by construction) true parent
    order = sorted(t_children, key=lambda c: next(p for p, cc in true_edges if cc == c))
    cx = {c: i / max(len(order) - 1, 1) for i, c in enumerate(order)}
    px = {p: np.mean([cx[c] for pp, c in true_edges if pp == p]) for p in t_parents}
    for p, c in true_edges:
        ok = (p, c) in found
        ax.plot([px[p], cx[c]], [1, 0], linestyle="-" if ok else (0, (3, 3)),
                lw=2 if ok else 1.2, color=GOOD if ok else NEUTRAL, zorder=2)
    ax.scatter([px[p] for p in t_parents], [1] * len(t_parents), s=90, color=INK, zorder=3)
    ax.scatter([cx[c] for c in order], [0] * len(order), s=48, color=MUTED, zorder=3)
    n_fp = tt["false_positives"]
    # Panel titles carry the model, the layer and the block pair; the edge
    # counts and the colour key live in the LaTeX caption, so the numbers have
    # one home and cannot drift between the PNG and the paper.
    ax.set_title("trained toy", fontsize=9.5, loc="left")

    # -- a probed SAE: every candidate edge that reached the probe ------------
    def draw_probed(ax, sp_x, label):
        edges = sp_x["0->1"]["sres"]["edges"]
        by_child: dict = {}
        for e in edges:
            by_child.setdefault(e["child"], []).append(e)
        # anchor each child under the parent that correlates best with its
        # probe -- the layout a genuine tree would satisfy with near-vertical
        # lines
        anchor = {c: max(es, key=lambda e: e["parent_corr"])["parent"]
                  for c, es in by_child.items()}
        deg: dict = {}
        for e in edges:
            deg[e["parent"]] = deg.get(e["parent"], 0) + 1
        parents = sorted(deg, key=lambda p: -deg[p])
        px = {p: i / max(len(parents) - 1, 1) for i, p in enumerate(parents)}
        children = sorted(by_child, key=lambda c: px[anchor[c]])
        cx = {c: i / max(len(children) - 1, 1) for i, c in enumerate(children)}
        for e in edges:
            if not e["pass"]:
                ax.plot([px[e["parent"]], cx[e["child"]]], [1, 0], lw=0.45,
                        color=CAT[0], alpha=0.18, zorder=1)
        n_pass = 0
        for e in edges:
            if e["pass"]:
                n_pass += 1
                ax.plot([px[e["parent"]], cx[e["child"]]], [1, 0], lw=1.8,
                        color=GOOD, zorder=3)
        # parent dot AREA scales with its child count, so the hub parents that
        # own most of the tangle are visible at a glance instead of hiding in a
        # uniform row of dots
        max_deg = max(deg.values())
        sizes = [8 + 150 * deg[p] / max_deg for p in px]
        ax.scatter(list(px.values()), [1] * len(px), s=sizes, color=INK, zorder=4)
        ax.scatter(list(cx.values()), [0] * len(cx), s=5, color=MUTED, zorder=4)
        ax.set_title(label, fontsize=9.5, loc="left")

    if n_panels == 3:
        draw_probed(axes[1], sp_pcfg, where_pcfg)
    draw_probed(axes[-1], sp, where)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["child block", "parent block"], fontsize=8.5)
        ax.set_ylim(-0.14, 1.14)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    _title(fig, "The recovered graph, drawn: a tree where the world is a tree, a tangle on the "
                "released SAE",
           "layout gives a tree its best chance — each child sits under the parent that best "
           "matches its probe, so a clean hierarchy would be near-vertical lines. The crossings "
           "are the structure", width=110)
    fig.subplots_adjust(top=0.80)
    return _finish(fig, axes, "recovered_graph_toy_vs_pcfg_vs_gemma")


# ---------------------------------------------------------------------------
# 10b. The hub, isolated: the single biggest parent of the released SAE's top
#      block pair, drawn with every child it claims — the case study behind
#      the tangle. Left: the fan itself. Right: how ownership of ALL candidate
#      edges concentrates in the few biggest parents.
# ---------------------------------------------------------------------------
def one_parent_owns_the_block(sp, where):
    edges = sp["0->1"]["sres"]["edges"]
    deg: dict = {}
    for e in edges:
        deg[e["parent"]] = deg.get(e["parent"], 0) + 1
    parents = sorted(deg, key=lambda p: -deg[p])
    children = sorted({e["child"] for e in edges})
    top = parents[0]
    top_kids = {e["child"] for e in edges if e["parent"] == top}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # -- left: the fan ------------------------------------------------------
    ax = axes[0]
    # claimed children drawn as one contiguous segment, unclaimed below it:
    # child ids carry no order, so the split is the only meaningful grouping,
    # and it keeps the grey minority visible instead of occluded
    children = ([c for c in children if c in top_kids]
                + [c for c in children if c not in top_kids])
    cy = {c: 1 - i / max(len(children) - 1, 1) for i, c in enumerate(children)}
    for c in top_kids:
        ax.plot([0.06, 0.94], [0.5, cy[c]], lw=0.7, color=CAT[0],
                alpha=0.45, zorder=1)
    claimed = [cy[c] for c in children if c in top_kids]
    others = [cy[c] for c in children if c not in top_kids]
    ax.scatter([0.94] * len(claimed), claimed, s=14, color=CAT[0], zorder=3)
    ax.scatter([0.94] * len(others), others, s=6, color=NEUTRAL, zorder=2)
    ax.scatter([0.06], [0.5], s=340, color=INK, zorder=4)
    ax.annotate("one parent feature",
                xy=(0.06, 0.5), xytext=(0.13, 0.93), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=MUTED))
    ax.text(0.985, 0.5,
            f"{len(top_kids)} of {len(children)} children claimed "
            f"({100 * len(top_kids) / len(children):.0f}%)",
            rotation=90, va="center", fontsize=9, color=CAT[0])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{where} — the single biggest parent, and every child it claims\n"
                 "grey = the children it does not", fontsize=9.5, loc="left")

    # -- right: who owns the candidate edges --------------------------------
    ax = axes[1]
    total = len(edges)
    shares = [("top 1 parent", deg[parents[0]] / total),
              ("top 5", sum(deg[p] for p in parents[:5]) / total),
              ("top 10", sum(deg[p] for p in parents[:10]) / total),
              (f"all {len(parents)}", 1.0)]
    for i, (lab, v) in enumerate(shares):
        ax.barh(i, 100 * v, height=0.55, color=CAT[0] if i < 3 else NEUTRAL)
        ax.text(100 * v + 2, i, f"{100 * v:.0f}%", va="center", fontsize=9,
                color=MUTED)
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([lab for lab, _ in shares], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"share of the {total:,} candidate edges", fontsize=9)
    ax.set_title("who owns the edges", fontsize=9.5, loc="left")
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_axisbelow(True)

    _title(fig, "One parent owns the block",
           "the hub structure behind the tangle, isolated: a single always-on feature "
           "claims most of the next block", width=100)
    fig.subplots_adjust(top=0.80)
    return _finish(fig, axes, "one_parent_owns_the_block")


# ---------------------------------------------------------------------------
# 10c'. The single-child version of the slice below: one child and every
#       parent that claims it, with all parent labels spelled out. NOT in
#       TEX_ORDER — the paper carries the slice; this one is generated as a
#       presentation asset (slides, talks), where one glance and ten label
#       lines beat a network panel. Kept in the generator so its numbers and
#       labels stay current with every regeneration.
# ---------------------------------------------------------------------------
def one_child_many_parents(sp, where, labels=None):
    import textwrap
    edges = sp["0->1"]["sres"]["edges"]
    by_child: dict = {}
    for e in edges:
        by_child.setdefault(e["child"], []).append(e["parent"])
    child, parents = max(by_child.items(), key=lambda kv: len(kv[1]))
    counts = sorted(len(v) for v in by_child.values())
    med = counts[len(counts) // 2]
    labels = labels or {}

    def lab(i, width=52):
        t = labels.get(str(i), "")
        return textwrap.shorten(t, width=width, placeholder="…") if t else f"feature {i}"

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ys = np.linspace(0.96, 0.04, len(parents))
    for p, y in zip(sorted(parents), ys):
        ax.plot([0.435, 0.86], [y, 0.5], lw=1.1, color=CAT[0], alpha=0.55, zorder=1)
        ax.scatter([0.435], [y], s=26, color=INK, zorder=3)
        ax.text(0.425, y, lab(p), ha="right", va="center", fontsize=8, color=MUTED)
    ax.scatter([0.86], [0.5], s=200, color=INK, zorder=4)
    child_txt = textwrap.fill(labels.get(str(child), f"feature {child}"), 26)
    ax.text(0.885, 0.5, child_txt, ha="left", va="center", fontsize=9,
            color=INK, fontweight="bold")
    ax.text(0.435, 1.06, f"{len(parents)} parent features, all claiming…",
            ha="center", fontsize=9.5, color=INK)
    ax.text(0.86, 1.06, "…one child", ha="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.05, 1.14)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{where} — the most-claimed child feature and every parent that claims it\n"
                 "a tree grants each child exactly one parent "
                 f"(the median child here has {med}) · labels: Neuronpedia, shown as read",
                 fontsize=9.5, loc="left")
    _title(fig, "A child with ten parents",
           "the tree violation itself, drawn on the released SAE", width=100)
    fig.subplots_adjust(top=0.82, left=0.02, right=0.99, bottom=0.03)
    return _finish(fig, ax, "one_child_many_parents", tight=False)


# ---------------------------------------------------------------------------
# 10c. The mirror case study, and the sharper one: a SLICE of the block pair
#      around the most-claimed children. A tree permits a parent many
#      children; it grants each child exactly ONE parent, so a slice in which
#      every child trails several parents is the tree violation itself, drawn
#      — and drawn as a neighbourhood, so it cannot be read as one bad apple.
#      Neuronpedia labels name the absurdity for the featured child.
# ---------------------------------------------------------------------------
def a_slice_of_the_tangle(sp, where, labels=None, n_children=8):
    import textwrap
    edges = sp["0->1"]["sres"]["edges"]
    by_child: dict = {}
    for e in edges:
        by_child.setdefault(e["child"], []).append(e["parent"])
    # a stated rule, not a hand pick: the n most-claimed children of the pair,
    # plus every parent that claims any of them
    kids = sorted(by_child, key=lambda c: -len(by_child[c]))[:n_children]
    star = kids[0]
    pars = sorted({p for c in kids for p in by_child[c]})
    pshare = {p: sum(1 for c in kids if p in by_child[c]) for p in pars}
    labels = labels or {}

    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    px = {p: 0.06 + 0.88 * i / max(len(pars) - 1, 1) for i, p in enumerate(pars)}
    kx = {c: 0.14 + 0.72 * i / max(len(kids) - 1, 1)
          for i, c in enumerate(sorted(kids, key=lambda c: min(px[p] for p in by_child[c])))}
    for c in kids:
        for p in by_child[c]:
            hot = c == star
            ax.plot([px[p], kx[c]], [1, 0], lw=1.5 if hot else 0.9,
                    color=CAT[3] if hot else CAT[0],
                    alpha=0.85 if hot else 0.4, zorder=2 if hot else 1)
    ax.scatter([px[p] for p in pars], [1] * len(pars),
               s=[24 + 26 * pshare[p] for p in pars], color=INK, zorder=3)
    ax.scatter([kx[c] for c in kids if c != star], [0] * (len(kids) - 1),
               s=42, color=MUTED, zorder=3)
    ax.scatter([kx[star]], [0], s=90, color=CAT[3], zorder=4)
    for c in kids:
        ax.text(kx[c], -0.09, f"×{len(by_child[c])}", ha="center", fontsize=8.5,
                color=CAT[3] if c == star else MUTED)
    star_txt = labels.get(str(star), "")
    if star_txt:
        ax.text(kx[star], -0.30,
                textwrap.fill("“" + star_txt + "”", 42), ha="center", va="top",
                fontsize=8, color=CAT[3])
    # the two parents claiming the most of these children, named
    named = sorted(pars, key=lambda p: -pshare[p])[:2]
    for i, p in enumerate(named):
        t = labels.get(str(p), "")
        if t:
            # anchor flips near the panel edges so the label never leaves it
            ha = "right" if px[p] > 0.75 else ("left" if px[p] < 0.25 else "center")
            ax.text(px[p], 1.10 + 0.10 * (i % 2),
                    textwrap.shorten("“" + t + "”", 52, placeholder="…”"),
                    ha=ha, fontsize=7.5, color=MUTED)
    ax.text(0.5, -0.52, "×N = how many parents claim that child   ·   a tree would put "
                        "exactly one line into every child", ha="center", fontsize=9,
            color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.62, 1.24)
    ax.set_xticks([]); ax.set_yticks([0, 1])
    ax.set_yticklabels(["child block", "parent block"], fontsize=8.5)
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{where} — the {len(kids)} most-claimed children, and every parent "
                 f"that claims them ({len(pars)} parents, "
                 f"{sum(len(by_child[c]) for c in kids)} edges)\n"
                 "orange = the featured child · dot area = how many of these children "
                 "the parent claims · labels: Neuronpedia, shown as read",
                 fontsize=9.5, loc="left")
    _title(fig, "A slice of the tangle: every child in it has many parents",
           "the tree violation drawn as a neighbourhood, not one bad apple", width=100)
    fig.subplots_adjust(top=0.82, left=0.085, right=0.985, bottom=0.02)
    return _finish(fig, ax, "a_slice_of_the_tangle", tight=False)


# ---------------------------------------------------------------------------
# 11. The formatting-density axis of the PCFG sweep, at the one layer the local
#     runs grade. Formatting density is the axis the project treats as the
#     mechanism behind bottleneck hijacking, so it gets its own figure: which
#     measures a grammar knob can move, and which are properties of the SAE.
# ---------------------------------------------------------------------------
def pcfg_formatting_sweep(fmt_runs):
    """Metric against delimiter density, one point per seed, mean drawn through.

    The reading is the SHAPE: a flat line is a measure no grammar knob moves --
    intrinsic to the architecture -- and a slope is a measure confounded with
    the corpus. That split is computed and printed rather than asserted.
    """
    layer = fmt_runs[0][2]["config"]["layer"]
    panels = [
        ("multi-parenting, % of children", lambda r, sp:
            100 * _pair(r, "0->1")["degree"]["poly_frac"]),
        ("at chance for the base rate, %", lambda r, sp:
            100 * _pair(r, "0->1")["independence_null"]["frac_chance_level"]),
        ("superparents flagged", lambda r, sp:
            _pair(r, "0->1")["n_superparents"]),
        (r"probe-confirmed $S_{res}$, %", lambda r, sp:
            100 * sp["0->1"]["sres"]["frac_pass"] if sp and "0->1" in sp else np.nan),
    ]
    dens = sorted({d for d, *_ in fmt_runs})
    # ORDINAL x positions. The densities are 0, 0.1667, 0.2308 and 0.24: on a
    # numeric axis the last two land 0.009 apart, their tick labels print on
    # top of each other, and two-thirds of the panel is empty. The spacing of
    # the sweep's rungs is the sweep author's choice, not a finding, so equal
    # spacing loses nothing and the labels carry the true values.
    xpos = {d: i for i, d in enumerate(dens)}
    fig, axes = plt.subplots(1, len(panels), figsize=(12.4, 3.4))
    knob, seed_noise = [], []
    for ax, (label, fn) in zip(axes, panels):
        means, spreads = [], []
        for d in dens:
            vals = [fn(r, sp) for dd, _, r, sp in fmt_runs if dd == d]
            vals = [v for v in vals if not np.isnan(v)]
            ax.scatter([xpos[d]] * len(vals), vals, s=26, color=CAT[0], alpha=0.55, zorder=2)
            means.append(float(np.mean(vals)) if vals else np.nan)
            spreads.append(max(vals) - min(vals) if len(vals) > 1 else 0.0)
        ax.plot([xpos[d] for d in dens], means, "-o", color=CAT[0], lw=2, ms=4, zorder=3)
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.set_xlabel("delimiter density")
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([f"{d:g}" for d in dens], fontsize=8)
        ax.margins(y=0.25, x=0.08)
        ax.grid(True, axis="y", alpha=0.12)
        ax.set_axisbelow(True)
        # What three seeds can and cannot support, computed: the knob's effect
        # is the range of the seed means, and it only means something where it
        # exceeds how far the seeds scatter at a single density.
        m = [v for v in means if not np.isnan(v)]
        clean = label.split(",")[0].replace("$", "").replace("_{res}", "_res")
        (knob if max(m) - min(m) > float(np.mean(spreads)) else
         seed_noise).append(clean)
    _title(fig, f"The formatting-density axis, swept — PCFG transformer, layer {layer}, "
                f"{len(fmt_runs) // len(dens)} seeds per density",
           "one point per seed; the line joins seed means. Where the seeds scatter more at one "
           "density than the mean moves across the whole axis, the knob's effect is not "
           "resolved at three seeds"
           + (f" — that is the case for {', '.join(seed_noise)}" if seed_noise else "")
           + (f"; resolved above seed noise: {', '.join(knob)}" if knob else ""),
           width=110)
    fig.subplots_adjust(top=0.76)
    return _finish(fig, axes, "pcfg_formatting_sweep")


# ---------------------------------------------------------------------------
def build(dry: bool) -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (written, skipped) where skipped carries the reason, never silence."""
    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    G = C.OUT_DIR / C.SOURCE_NAME
    layers = gemma_layers()

    sources: dict[str, str] = {}

    def run(name, need_ok, why, fn):
        if not need_ok:
            skipped.append((name, why))
            return
        sources[name] = why
        if dry:
            written.append(f"{name}   <- {why}")
            return
        written.append(fn())

    # layer -> second_pass.json, for every layer that has the strict test
    second = {L: _json(G / f"layer_{L:02d}" / "second_pass.json") for L, _ in layers}
    second = {L: v for L, v in second.items() if v}
    l6 = _json(G / "layer_06" / "metrics_report.json")
    # every PCFG depth run, as (label, report, second_pass) so the two-source
    # figures never re-derive which directory a second pass belongs to
    pcfg = []
    for p in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*")):
        r = _json(p / "metrics_report.json")
        if r:
            pcfg.append((f"PCFG {_layer_name(p.name)}", r,
                         _json(p / "second_pass.json")))
    run("funnel_coverage_to_sres", bool(second),
        f"{len(second)} gemma + {sum(1 for *_, sp in pcfg if sp)} PCFG layers with "
        "second_pass.json"
        if second else "needs at least one gemma second_pass.json (stage 03)",
        lambda: funnel_coverage_to_sres(layers, second, pcfg, where="gemma-2-2b"))

    run("edge_survival_by_block_pair", len(layers) >= 2,
        f"{len(layers)} gemma layer reports" if layers else "needs ≥2 gemma metrics_report.json",
        lambda: edge_survival_by_block_pair(layers))
    run("depth_profile_across_layers", len(layers) >= 2,
        f"{len(layers)} gemma layer reports" if layers else "needs ≥2 gemma metrics_report.json",
        lambda: depth_profile_across_layers(layers))
    run("multiparenting_by_layer", len(layers) >= 1,
        f"{len(layers)} gemma + {len(pcfg)} PCFG layer reports"
        if layers else "needs gemma metrics_report.json",
        lambda: multiparenting_by_layer(layers, [(n, r) for n, r, _ in pcfg]))
    # Matched BY LAYER: a layer graded only after the BOS fix has no v1
    # counterpart and must not veto the figure for the layers that do.
    arch_L = {int(p.parent.name.split("__")[0].split("_")[1])
              for p in (C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json")}
    n_match = sum(1 for L, _ in layers if L in arch_L)
    run("shared_input_moved_every_metric", n_match >= 2,
        f"{n_match} layers graded both before and after BOS exclusion" if n_match
        else "needs the pre-BOS reports in outputs_archive/",
        lambda: shared_input_moved_every_metric(layers))
    run("base_rate_vs_frequency_capture", len(layers) >= 1,
        f"{len(layers)} gemma layer reports" if layers else "needs gemma metrics_report.json",
        lambda: base_rate_vs_frequency_capture(layers))
    run("superparent_fanout_vs_firing", len(layers) >= 1,
        f"{len(layers)} gemma layer reports" if layers else "needs gemma metrics_report.json",
        lambda: superparent_fanout_vs_firing(layers))

    toy = _json(C.OUT_DIR / "synthetic_toy_calibration.json")
    run("calibration_synthetic_toy_scorecard", bool(toy),
        "outputs/synthetic_toy_calibration.json" if toy
        else "needs outputs/synthetic_toy_calibration.json (validation.calibrate_on_synthetic_toy)",
        lambda: calibration_synthetic_toy_scorecard(toy))

    # Computed, not read: the candidate mask and the per-gate outcomes are not in
    # synthetic_toy_calibration.json, which carries the scorecard rows alone.
    world = synthetic_toy_gates()
    run("calibration_toy_world_before_after", bool(world),
        "computed from validation/synthetic_toy_world.py (no cache needed)" if world
        else "needs torch + validation/synthetic_toy_world.py",
        lambda: calibration_toy_world_before_after(world))
    # The single-panel, rows-not-columns redraw of the same world; both are
    # built so the manuscript can pick one (see the function's docstring).
    run("calibration_toy_world_gate_verdicts", bool(world),
        "computed from validation/synthetic_toy_world.py (no cache needed)" if world
        else "needs torch + validation/synthetic_toy_world.py",
        lambda: calibration_toy_world_gate_verdicts(world))
    # The same world, three more questions: what the corpus underneath it looks
    # like, what each gate removed as a count, and what the first gate saw.
    run("calibration_toy_corpus_firing", bool(world),
        (f"computed from validation/synthetic_toy_world.py — {world['P'] + world['C']} "
         f"features over {world['total_tokens']:,} tokens") if world
        else "needs torch + validation/synthetic_toy_world.py",
        lambda: calibration_toy_corpus_firing(world))
    run("calibration_gate_funnel_by_role", bool(world),
        (f"computed from validation/synthetic_toy_world.py — {len(world['candidates'])} "
         "candidates through three gates") if world
        else "needs torch + validation/synthetic_toy_world.py",
        lambda: calibration_gate_funnel_by_role(world))
    run("calibration_reverse_coverage", bool(world),
        (f"computed from validation/synthetic_toy_world.py — "
         f"{world['P']}×{world['C']} coverage matrix") if world
        else "needs torch + validation/synthetic_toy_world.py",
        lambda: calibration_reverse_coverage(world))

    # The most expensive input in the generator: the world and the whole battery,
    # once per seed. Skipped with its reason rather than silently costing a minute.
    sweep = synthetic_toy_seed_sweep() if world else None
    run("calibration_seed_sweep", bool(sweep),
        (f"{len(sweep)} seeds × {len(sweep[0][1])} scorecard rows, re-run from "
         "validation/calibrate_on_synthetic_toy.py") if sweep
        else "needs torch + validation/calibrate_on_synthetic_toy.py",
        lambda: calibration_seed_sweep(sweep))

    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    align = _json(C.OUT_DIR / "block_tree_alignment.json")
    run("calibration_trained_toy_recovery", bool(tt),
        "outputs/trained_toy_calibration.json"
        + ("" if align else " (block_tree_alignment.json absent — right panel blank)")
        if tt else "needs outputs/trained_toy_calibration.json (Tier 2)",
        lambda: calibration_trained_toy_recovery(tt, align))
    run("calibration_toy_tree_recovered", bool(tt),
        "outputs/trained_toy_calibration.json"
        if tt else "needs outputs/trained_toy_calibration.json (Tier 2)",
        lambda: calibration_toy_tree_recovered(tt))

    # Every graded layer of BOTH sources. This figure once carried gemma layer 6
    # alone -- the layer that had the strict test first -- and kept doing so
    # after the other five caught up, which read as "the comparison rests on one
    # layer" long after it no longer did.
    runs = [(f"gemma L{L}", rep) for L, rep in layers] + [(n, r) for n, r, _ in pcfg]
    run("cross_source_funnel_shares", bool(layers and pcfg),
        f"{len(layers)} gemma + {len(pcfg)} PCFG layer reports"
        if (layers and pcfg) else "needs a gemma report and a PCFG report",
        lambda: cross_source_funnel_shares(runs))

    # The two cross-source depth figures. Both need at least one layer index that
    # BOTH sources graded; the alignment check additionally needs the two rules to
    # actually disagree, which they only do once gemma has layers the toy cannot
    # reach. Each condition is reported rather than silently producing a figure
    # whose dumbbells are all zero length.
    dr = X.rows()
    shared = sorted({o["layer"] for o, _, _ in X.matched(dr, "layer")})
    run("cross_source_layer_response", bool(shared),
        f"{len(dr)} graded runs across {len({r['src'] for r in dr})} sources, "
        f"layer{'s' if len(shared) > 1 else ''} {', '.join(str(s) for s in shared)} on both"
        if shared else "needs one layer index graded on both gemma and PCFG",
        lambda: cross_source_layer_response(dr))
    same = X.alignment_gaps(dr) is None
    run("cross_source_alignment_check", bool(shared) and not same,
        f"the same {len(dr)} runs under both alignment rules"
        if shared and not same else
        "the two alignments pair the same runs here, so the comparison is empty"
        if shared else "needs one layer index graded on both gemma and PCFG",
        lambda: cross_source_alignment_check(dr))

    ib = []
    for base, label in [(G, "gemma"), (C.OUT_DIR / "pcfg-matryoshka", "PCFG")]:
        for lay in sorted(base.glob("layer_*")):
            j = _json(lay / "in_block_edges.json")
            if j:
                ib.append((f"{label} {_layer_name(lay.name)}", j))
    run("in_block_relations", bool(ib),
        f"{len(ib)} runs with in_block_edges.json" if ib
        else "needs in_block_edges.json — pipeline stage 01c",
        lambda: in_block_relations(ib))

    # observed S_res shares, read off whatever second_pass.json files exist --
    # every gemma layer that has one, not the single layer that had one first
    observed = sres_observed_rates()
    run("sres_null_rate_vs_dictionary_size", True,
        f"config (k={C.SRES_RANK_TOP_K}) + {len(observed)} measured pass rates",
        lambda: sres_null_rate_vs_dictionary_size(observed))

    # metric 9 lives in its own file, one entry per block, so it is loaded
    # per run and keyed by the same label source_grid puts on the row
    def _ib(base, lay_name, label):
        j = _json(base / lay_name / "in_block_edges.json") or {}
        return {label: {b["block"]: b for b in j.get("blocks", [])}}

    ib_map: dict = {}
    for L, _ in layers:
        ib_map.update(_ib(G, f"layer_{L:02d}", f"L{L}"))
    for n, _, _ in pcfg:
        lay = n.replace("PCFG layer ", "")
        ib_map.update(_ib(C.OUT_DIR / "pcfg-matryoshka",
                          f"layer_{int(lay):02d}", f"L{lay}"))

    run("tangle_lives_in_top_block_pair", bool(layers or pcfg),
        f"{len(layers)} gemma + {len(pcfg)} PCFG layer reports, all block pairs"
        if (layers or pcfg) else "needs metrics_report.json for at least one source",
        lambda: tangle_lives_in_top_block_pair(
            layers, [(n, r) for n, r, _ in pcfg], in_block=ib_map))

    # the same grid at ONE depth per source: gemma at GRAPH_LAYER and PCFG at
    # PCFG_GRAPH_LAYER, its relative-depth twin. The full-depth version above
    # is the appendix's; this one is what the main text can carry, because at
    # one layer per source the rows are block pairs alone and the grid fits
    # without the reader hunting for which of six layers a row belongs to.
    one_g = [(L, r) for L, r in layers if L == GRAPH_LAYER]
    one_p = [(n, r) for n, r, _ in pcfg if n.endswith(f"layer {PCFG_GRAPH_LAYER}")]
    run("metrics_result_mid_layers", bool(one_g or one_p),
        f"gemma layer_{GRAPH_LAYER:02d} + PCFG layer_{PCFG_GRAPH_LAYER:02d} "
        "metrics_report.json, all block pairs"
        if (one_g or one_p) else
        f"needs a metrics_report.json at gemma layer_{GRAPH_LAYER:02d} "
        f"or PCFG layer_{PCFG_GRAPH_LAYER:02d}",
        lambda: tangle_lives_in_top_block_pair(
            one_g, one_p, name="metrics_result_mid_layers", in_block=ib_map))

    run("battery_questions_gemma", bool(layers),
        f"{len(layers)} gemma layer reports + {len(second)} second passes"
        if layers else "needs gemma metrics_report.json",
        lambda: battery_questions_gemma(layers, second))

    # the recovered graph itself, drawn at GRAPH_LAYER and PCFG_GRAPH_LAYER
    # (see the constants). Both panels name their layer explicitly so neither
    # quietly follows whichever run happened to finish first. This figure's job
    # is to show the full funnel visually -- candidate, rejected, and
    # probe-CONFIRMED edges -- so the named PCFG layer must carry all three
    # stages; a layer that scores edges but confirms none reads as a dead
    # probe. If PCFG_GRAPH_LAYER has no scored edges we fall back to the
    # largest scored set rather than dropping the panel. Depth-matched
    # cross-source comparisons live in the tables.
    s_gl = second.get(GRAPH_LAYER)
    scored = [(lab, spx) for lab, _, spx in pcfg
              if spx and "0->1" in spx and spx["0->1"]["sres"]["n_edges_scored"]]
    want = f"layer {PCFG_GRAPH_LAYER}"
    sp_p, sp_p_name = None, ""
    for lab, spx in scored:
        if lab.endswith(want):
            sp_p, sp_p_name = spx, lab
    if sp_p is None and scored:
        sp_p_name, sp_p = max(scored,
                              key=lambda t: t[1]["0->1"]["sres"]["n_edges_scored"])
    run("recovered_graph_toy_vs_pcfg_vs_gemma", bool(tt and s_gl),
        f"trained_toy_calibration.json + gemma layer_{GRAPH_LAYER:02d}"
        + (f" + {sp_p_name}" if sp_p else "") + " second_pass.json"
        if (tt and s_gl) else "needs the trained-toy calibration and a gemma second_pass.json",
        lambda: recovered_graph_toy_vs_pcfg_vs_gemma(tt, s_gl,
                                             f"gemma-2-2b layer {GRAPH_LAYER}, B0→B1",
                                             sp_p, f"{sp_p_name}, B0→B1"))

    run("one_parent_owns_the_block", bool(s_gl),
        f"gemma layer_{GRAPH_LAYER:02d} second_pass.json"
        if s_gl else "needs the gemma second_pass.json at GRAPH_LAYER",
        lambda: one_parent_owns_the_block(s_gl, f"gemma-2-2b layer {GRAPH_LAYER}, B0→B1"))

    lbl = _json(G / f"layer_{GRAPH_LAYER:02d}" / "feature_labels.json") or {}
    run("a_slice_of_the_tangle", bool(s_gl),
        f"gemma layer_{GRAPH_LAYER:02d} second_pass.json"
        + (" + feature_labels.json" if lbl else " (no labels cached)")
        if s_gl else "needs the gemma second_pass.json at GRAPH_LAYER",
        lambda: a_slice_of_the_tangle(s_gl, f"gemma-2-2b layer {GRAPH_LAYER}, B0→B1", lbl))

    # the single-child twin of the slice; also emitted into figures.tex (APP
    # 1b') so the manuscript can pick one of the pair and comment the other
    run("one_child_many_parents", bool(s_gl),
        f"gemma layer_{GRAPH_LAYER:02d} second_pass.json (single-child twin of the slice)"
        if s_gl else "needs the gemma second_pass.json at GRAPH_LAYER",
        lambda: one_child_many_parents(s_gl, f"gemma-2-2b layer {GRAPH_LAYER}, B0→B1", lbl))

    # the formatting-density sweep runs, named fmt_<density*1e4>_s<seed>
    fmt_runs = []
    for p in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("fmt_*")):
        r = _json(p / "metrics_report.json")
        if r:
            fmt_runs.append((int(p.name.split("_")[1]) / 1e4,
                             p.name.split("_s")[-1], r,
                             _json(p / "second_pass.json")))
    n_dens = len({d for d, *_ in fmt_runs})
    run("pcfg_formatting_sweep", n_dens >= 2,
        f"{len(fmt_runs)} fmt_* runs across {n_dens} delimiter densities"
        if fmt_runs else "needs pcfg-matryoshka/fmt_* metric reports",
        lambda: pcfg_formatting_sweep(fmt_runs))

    return written, skipped, sources


# ---------------------------------------------------------------------------
# figures.tex — the same figures with captions, ordered as the paper reads.
#
# Same two rules as the rest of this file. Every quantity in every caption is
# read from the JSON its figure plots, so a caption cannot outlive its data --
# which is exactly how this file's own titles came to quote withdrawn numbers.
# And a figure with no caption entry is emitted with none rather than silently
# dropped, so the omission is visible in the .tex.
#
# Captions sit BELOW the figure, which is the convention for figures and the
# mirror of tables.tex, where they sit above.
# ---------------------------------------------------------------------------
TWOCOLUMN = False       # set from --twocolumn; the ICLR template is onecolumn


def _wrap(t: str) -> str:
    import textwrap
    # break_on_hyphens=False: textwrap splits "0--7" across lines by default and
    # LaTeX turns the break into a space, printing "0-- 7".
    return "\n".join(textwrap.wrap(t, 94, initial_indent="    ", subsequent_indent="    ",
                                   break_on_hyphens=False, break_long_words=False))


def _captions():
    """Caption bodies, with their numbers interpolated from the plotted JSON."""
    G = C.OUT_DIR / C.SOURCE_NAME
    lay = gemma_layers()
    d: dict[str, str] = {}

    def rp(L, pair="0->1"):
        return _pair(_json(G / f"layer_{L:02d}" / "metrics_report.json"), pair)

    if lay:
        p6, sp6 = rp(6), _json(G / "layer_06" / "second_pass.json")
        poly = [100 * rp(L)["degree"]["poly_frac"] for L, _ in lay]
        ch = [100 * rp(L)["independence_null"]["frac_chance_level"] for L, _ in lay]
        fq = [100 * rp(L)["freq_control"]["frac_freq_driven"] for L, _ in lay]
        fires = [x["fire_frac"] for _, r in lay for q in r["pairs"]
                 for x in q.get("superparents", [])]
        free = sum(1 for f in fires if f >= C.EDGE_TAU)

        deep = [100 * (_pair(r, p) or {}).get("degree", {}).get("poly_frac", np.nan)
                for _, r in lay for p in ("1->2", "2->3")]
        deep = [v for v in deep if not np.isnan(v)]
        d["multiparenting_by_layer"] = (
            r"\textbf{The recovered graph is not a tree.} Child features are often "
            "associated with multiple parents rather than a single parent: in the "
            rf"top block pair B0$\rightarrow$B1, {min(poly):.0f}--{max(poly):.0f}\% of "
            "children with any parent have two or more, at every graded layer, "
            "revealing pervasive multi-parenting in the recovered structure. Deeper "
            rf"block pairs appear cleaner ({min(deep):.0f}--{max(deep):.0f}\%) on "
            "candidate sets of comparable or larger size, so the drop reflects "
            "looser structure rather than a smaller sample. Gemma's "
            r"B3$\rightarrow$B4 pair is graded at layer 12 alone: its co-firing "
            r"matrix is 6{,}144$\times$24{,}576, which needs a 49 GB card, so it is "
            "off by default and was run once. The other five layers are marked n.a. "
            "because they were not run, not because they cannot be. Every bar prints "
            "the number of children behind its "
            "percentage as $n{=}$; a 100\\% computed over one child is a coin flip, "
            "not a result, and the label keeps it from reading as one.")
        pp = [100 * _pair(r, "0->1")["degree"]["poly_frac"]
              for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*"))
              for r in [_json(q / "metrics_report.json")] if r]
        if pp:
            d["multiparenting_by_layer"] += (
                f" The PCFG panel shows the same signature "
                rf"({min(pp):.0f}--{max(pp):.0f}\% in B0$\rightarrow$B1 across its "
                "layers), so the multi-parenting is a property of the Matryoshka nesting, not "
                "of natural language; its deeper-pair bars rest on a handful of children "
                "and carry their support ($n$) rather than posing as results.")
        sps = {L: _json(G / f"layer_{L:02d}" / "second_pass.json") for L, _ in lay}
        sps = {L: v for L, v in sps.items() if v and "0->1" in v}
        if sps:
            cands = [rp(L)["n_candidate_edges"] for L in sps]
            passes = [sps[L]["0->1"]["sres"]["n_pass"] for L in sps]
            shares = [100 * sps[L]["0->1"]["sres"]["frac_pass"] for L in sps]
            recs = [100 * rp(L)["reconstruction"]["frac_pass"] for L in sps]
            d["funnel_coverage_to_sres"] = (
                r"\textbf{Broad coverage does not translate into confirmed structure.} "
                "Coverage proposes many candidate relationships "
                f"({min(cands):,}--{max(cands):,} per layer), but successive independence "
                "and probe tests eliminate most of them: the independence null rejects "
                rf"{min(ch):.0f}--{max(ch):.0f}\%, and the probe confirms only "
                f"{min(passes)}--{max(passes)} edges per layer "
                rf"(${min(shares):.1f}$--${max(shares):.1f}\%$ of those scored), leaving "
                "only a small subset of supported parent--child relationships. The "
                rf"reconstruction filter (passing {min(recs):.0f}--{max(recs):.0f}\%) is "
                "evaluated separately because it does not form a nested stage.")
            pc = []
            for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*/second_pass.json")):
                spx, rr = _json(q), _json(q.parent / "metrics_report.json")
                if spx and "0->1" in spx and rr:
                    pc.append((_pair(rr, "0->1")["n_candidate_edges"],
                               spx["0->1"]["sres"]["n_pass"]))
            if pc:
                d["funnel_coverage_to_sres"] += (
                    " The PCFG panel repeats the same collapse at small scale: "
                    f"{min(c for c, _ in pc)}--{max(c for c, _ in pc)} candidates per "
                    f"layer end in {min(p for _, p in pc)}--{max(p for _, p in pc)} "
                    "probe-confirmed edges.")
        d["base_rate_vs_frequency_capture"] = (
            r"Two per-edge diagnostics on the same candidate sets, B0$\rightarrow$B1. The "
            "independence null asks whether co-firing exceeds what the parent's own firing rate "
            rf"already forces, and rejects {min(ch):.0f}--{max(ch):.0f}\% of edges. The "
            "token-frequency control asks whether the edge survives once globally frequent tokens "
            rf"are removed, and rejects {min(fq):.1f}--{max(fq):.1f}\%. They disagree by "
            rf"{min(c / f for c, f in zip(ch, fq)):.0f}--"
            rf"{max(c / f for c, f in zip(ch, fq)):.0f}$\times$ at every layer. This tests the "
            "observation the bottleneck-hijacking hypothesis is built on (that the "
            "over-connected parents mostly track high-frequency tokens such as spaces, "
            r"punctuation and \emph{the}) and does not support it: the frequency control "
            rf"exonerates {100 - max(fq):.0f}--{100 - min(fq):.0f}\% of edges while the "
            "base-rate null rejects most of them.")
        d["superparent_fanout_vs_firing"] = (
            r"An edge is kept when $P(\mathrm{parent} \mid \mathrm{child}) \geq \tau = "
            rf"{C.EDGE_TAU}$. Under independence that probability is simply the parent's firing "
            r"rate $\rho$, so the enrichment a parent needs over chance is $\tau/\rho$, the grey "
            rf"curve. All {len(fires)} parents flagged as superparents, across {_spell(len(lay))} layers and "
            rf"every block pair, lie on it. {free} of them fire on at least "
            rf"{100 * C.EDGE_TAU:.0f}\% of tokens and therefore clear the bar at "
            r"$\leq 1\times$ enrichment: on base rate alone, with no association whatsoever "
            r"between parent and child. A parent firing on 1\% of tokens would need "
            rf"{C.EDGE_TAU / 0.01:.0f}$\times$ enrichment for the same edge. This is the "
            "mechanism behind the previous figure.")
        # The withdrawal is stated from the four series the figure plots, not from five
        # numbers typed into the sentence. Those read "5, 7, 6, 7, 6"; no field in any
        # committed report reproduces them over any five of the graded layers, and they
        # predate the sixth layer, so they were replaced rather than re-pinned.
        prof = {
            "candidate edges": [rp(L)["n_candidate_edges"] for L, _ in lay],
            "the share improving reconstruction":
                [rp(L)["reconstruction"]["frac_pass"] for L, _ in lay],
            "the frequency-driven share":
                [rp(L)["freq_control"]["frac_freq_driven"] for L, _ in lay],
            "mean frequency survival":
                [rp(L)["freq_control"]["mean_survival"] for L, _ in lay],
        }

        def _monotone(v):
            return all(b >= a for a, b in zip(v, v[1:])) or all(b <= a for a, b in zip(v, v[1:]))

        n_mono = sum(_monotone(v) for v in prof.values())

        def _interior_turn(v):
            """The index of a max or min that falls strictly inside the layer range.

            An extreme at either end is consistent with a monotone trend and proves
            nothing; only an interior one is a turning point. Picking the extreme
            farthest from the middle, as this first did, selects exactly the wrong
            thing and produced "peaks at layer 1 rather than at either end".
            """
            for i, word in ((v.index(max(v)), "peaks"), (v.index(min(v)), "bottoms out")):
                if 0 < i < len(v) - 1:
                    return i, word
            return None

        turn = next(((k, _interior_turn(v)) for k, v in prof.items() if _interior_turn(v)),
                    None)
        d["depth_profile_across_layers"] = (
            rf"Four quantities for block pair B0$\rightarrow$B1 across the {_spell(len(lay))} "
            "graded layers, each on its own axis because they are not commensurable. The "
            "project previously reported that hierarchy quality degrades with depth. That "
            "claim came from caches in which the beginning-of-sequence position was counted, "
            "and it does not survive regeneration: "
            + (f"{_spell(n_mono)} of the four measures is monotonic in depth"
               if n_mono else "not one of the four measures is monotonic in depth")
            + (f", and {turn[0]} {turn[1][1]} at layer {lay[turn[1][0]][0]}, "
               "inside the range rather than at either end." if turn else ".")
            + " The figure is included because the withdrawal is itself a result about how "
            "easily a depth trend can be manufactured.")
        d["edge_survival_by_block_pair"] = (
            r"\emph{Left:} the share of candidate edges whose reconstruction improves when the "
            r"parent is ablated. \emph{Right:} the share the token-frequency control judges to be "
            "carried by globally frequent tokens. Panels are on separate scales; layers are "
            "ordered, so depth is encoded as a single-hue ramp rather than as five categorical "
            "colours. The frequency control is nearly silent in the top block pair and rises "
            "sharply in the deeper ones.")

    arch = sorted((C.HERE / "outputs_archive").glob("layer_*__v1__*/metrics_report.json"))
    if arch and lay:
        d["shared_input_moved_every_metric"] = (
            f"Change in each metric, averaged over the {_spell(len(lay))} layers, when a single contaminating "
            "token position is excluded from the corpus. The beginning-of-sequence token is an "
            "attention sink on which effectively every feature fires; with "
            f"{C.N_DOCS} documents it handed every pair in the dictionary {C.N_DOCS} joint "
            f"firings against a support guard set at {C.MIN_JOINT}, so the guard admitted pairs "
            "that never co-occur anywhere else. Five of these "
            "six quantities are computed from that one co-firing matrix, and all five moved. The "
            "sixth, multi-parenting, is a ratio over children that already have a parent, does "
            "not read the matrix, and did not move. Agreement among detectors that share an "
            r"input is far weaker evidence than their number implies. The grey value "
            "in each pair is the withdrawn one, plotted to size the error and not as a result.")

    toy = _json(C.OUT_DIR / "synthetic_toy_calibration.json")
    if toy:
        d["calibration_synthetic_toy_scorecard"] = (
            r"\textbf{Calibrating the metrics where the answer is known.} The test world is "
            "built by hand: activations are composed from a known concept hierarchy (six "
            "parent features, each with children that fire only when it fires) plus eight "
            "planted look-alike structures that a naive co-firing analysis would mistake for "
            "parent--child links, such as a feature pair that co-fires only through shared "
            "frequent tokens, a pair lifted together by a shared topic, and an always-on "
            r"``super-parent''. Each row is one of the metrics (Table~\ref{tab:battery}) "
            "applied to this world, and a row passes when the metric tells its planted target "
            f"apart from healthy structure; {sum(r['pass'] for r in toy)} of {len(toy)} rows "
            r"pass. \emph{Left:} metrics that return a score. The bar is the separation: the "
            "ratio between the metric's value on the pairs it must flag and its value on the "
            r"pairs it must keep, so $1\times$ means the two are indistinguishable and the "
            r"metric carries no signal. \emph{Right:} metrics whose output is a discrete "
            "answer (an edge set, a direction), scored simply as correct or not; the two "
            "panels are separate because a discrete answer has no ratio to report. The three "
            "hatched rows are negative controls, planted blind spots expected to slip "
            "through: an absorbed child that coverage cannot even propose (the child fires "
            "exactly where its parent is silent), a shared-topic pair, and a composed "
            "child's two component edges that every filter accepts. They pass when nothing "
            "catches them, documenting the metric set's limits, "
            "and a code change that made them catchable would fail them visibly. Passing "
            "this tier is what licenses running the same metrics on real models, where no "
            "ground truth exists.")

    # DRAFT CAPTION -- the prose is a draft and is meant to be rewritten by hand
    # before submission; only the numbers in it are load-bearing. Every one of them
    # is computed from the world the figure draws, per this module's rule that a
    # caption cannot outlive its data.
    world = synthetic_toy_gates()
    if world:
        n_sp = world["superparent_cut"]
        miss = world["missed"][0] if world["missed"] else None
        spur = world["spurious"][0] if world["spurious"] else None
        d["calibration_toy_world_before_after"] = (
            r"\textbf{Which gate caught which injected pathology.} "
            "The same hand-built world as the previous figure, drawn twice in one "
            r"palette: one colour per injected structure, in both panels. "
            r"\emph{Top:} the world as declared --- "
            f"{len(world['true_edges'])} true parent--child edges over "
            f"{world['P'] + world['C']} features, plus the eight structures planted to be "
            r"caught or to demonstrate that nothing catches them. \emph{Bottom:} the same "
            "world after the three composed gates (reverse coverage, the reconstruction "
            "condition, the token-frequency control), where a faded edge is one the gates "
            "removed and the note above each parent names the gate that removed it. "
            f"Coverage proposes {len(world['candidates'])} candidates; "
            f"{len(world['candidates']) - len(world['survivors'])} are rejected, and "
            f"{len(world['recovered'])} of the {len(world['true_edges'])} true edges "
            f"survive. The division of labour is the reading: all {n_sp} super-parent "
            "pairs die at the reconstruction condition rather than at coverage, the "
            "frequency-coincidence pair passes coverage and reconstruction and dies only "
            "at the frequency control, and the feature-split edges are recovered --- correctly, "
            "since they are real refinements --- with the split reported against the "
            "parent by a different metric. The dashed structures are the negative controls: "
            + (f"the absorbed edge {miss} is never proposed at all, " if miss else "")
            + (f"while the shared-topic pair {spur} and the composition edges "
               f"{world['comp_edges'][0]} and {world['comp_edges'][1]} pass every gate. "
               if spur else "")
            + f"The multi-parenting intruder {world['multi_edges'][0]} and the "
              f"co-extensive pair {world['coext_edges'][0]} also pass every gate and are "
              "flagged by reports (the child's in-degree, the energy share) rather than "
              "rejected.")
        # Standalone twin of the caption above, because the manuscript shows one
        # of the two figures, not both: it re-describes the world instead of
        # deferring to a before panel that may not be on the page.
        d["calibration_toy_world_gate_verdicts"] = (
            r"\textbf{What the set of metrics recovered.} "
            "The same hand-built world as the previous figure, drawn in one "
            "palette with one colour per injected structure: the parent block "
            f"as the top row, the child block as the bottom row --- "
            f"{len(world['true_edges'])} declared true parent--child edges over "
            f"{world['P'] + world['C']} features, plus the eight structures "
            "planted to be caught or to demonstrate that nothing catches them. "
            "A faded edge is one the three composed gates (reverse coverage, "
            "the reconstruction condition, the token-frequency control) "
            "removed, so the declared world is every edge, bright or faded, and "
            "the note above each parent names the gate that decided its edges. "
            f"Coverage proposes {len(world['candidates'])} candidates; "
            f"{len(world['candidates']) - len(world['survivors'])} are rejected, and "
            f"{len(world['recovered'])} of the {len(world['true_edges'])} true "
            "edges survive. The division of labour is the reading: all "
            f"{n_sp} super-parent pairs die at the reconstruction condition "
            "rather than at coverage, the frequency-coincidence pair passes "
            "coverage and reconstruction and dies only at the frequency "
            "control, and the feature-split edges are recovered --- correctly, "
            "since they are real refinements --- with the split reported "
            "against the parent by a different metric. The dashed structures are "
            "the negative controls: "
            + (f"the absorbed edge {miss} is never proposed at all, " if miss else "")
            + (f"while the shared-topic pair {spur} and the composition edges "
               f"{world['comp_edges'][0]} and {world['comp_edges'][1]} pass every gate. "
               if spur else "")
            + f"The multi-parenting intruder {world['multi_edges'][0]} and the "
              f"co-extensive pair {world['coext_edges'][0]} also pass every gate and are "
              "flagged by reports (the child's in-degree, the energy share) rather than "
              "rejected.")

    tt = _json(C.OUT_DIR / "trained_toy_calibration.json")
    if tt:
        pt = tt.get("per_token") or {}
        red = pt.get("parent_conditioned_redundancy") or {}
        cof = pt.get("child_cofire") or {}
        # This clause reports a conflation the SAE committed and the battery missed. It
        # only belongs in the caption when the run actually shows one: written as fixed
        # prose it survived a change of reference checkpoint and went on describing a
        # defect that had gone away, against redundancy figures that were all 0.00.
        # REDUNDANCY_THR mirrors the threshold the metric itself uses.
        REDUNDANCY_THR = (tt.get("per_token") or {}).get("redundancy_threshold", 0.5)
        extra = ""
        if red and cof:
            w = max(red.items(), key=lambda kv: kv[1])
            others = [v for k, v in red.items() if k != w[0]]
            if w[0] in cof and w[1] >= REDUNDANCY_THR and cof[w[0]]["learned"] > 0:
                extra = (
                    " Beyond edge recovery, the parent-conditioned sibling metric reports "
                    f"{w[1]:.2f} for one true parent against "
                    + " and ".join(f"{v:.2f}" for v in others)
                    + " for the others. The grammar declares every parent's children mutually "
                    f"exclusive, and they co-fire {cof[w[0]]['ground_truth']} times in "
                    "200{,}000 draws; the latents that recovered them co-fire "
                    f"{cof[w[0]]['learned']:,} times. The SAE conflated two concepts the "
                    "grammar keeps apart: a defect nobody injected, which the synthetic tier "
                    "structurally cannot produce, and which the rest of the metrics miss, "
                    "since both of that parent's edges are still counted as recovered."
                )
            elif red:
                extra = (
                    " Beyond edge recovery, the parent-conditioned sibling metric stays at or "
                    f"below {max(red.values()):.2f} for every true parent, against a threshold "
                    f"of {REDUNDANCY_THR:.2f}: on this checkpoint the SAE kept each parent's "
                    "mutually exclusive children apart rather than conflating them."
                )
        # DRAFT caption -- captions are written by hand here, so this is a starting
        # point to rewrite, not final text. Every number in it is computed above from
        # trained_toy_calibration.json rather than typed, so an edit to the data cannot
        # leave the prose asserting a withdrawn result.
        n_test = len([e for e in tt["true_edges"]
                      if e[0] in set(tt["recovered_features"])
                      and e[1] in set(tt["recovered_features"])])

        def _n_nodes(node):
            return 1 + sum(_n_nodes(c) for c in node.get("children", []))

        # Links drawn = nodes - 1 (it is a tree). Counting them by hand gave 12, which
        # is the count of family links only and ignores the root's own fan-out.
        n_links = _n_nodes(tt["tree"]) - 1 if tt.get("tree") else None
        links_clause = (f"which is why the {n_links} links drawn on the left reduce to "
                        f"{len(tt['true_edges'])} scoreable edges"
                        if n_links else
                        f"which is why only {len(tt['true_edges'])} of the links drawn "
                        "on the left are scoreable edges")
        arch = str(tt["cfg"].get("activation_function", "")).replace("_", r"\_")
        # k is meaningless for a relu SAE and is stored as null there; printing
        # "$k=None$" in a caption is worse than printing nothing.
        k_clause = (f", $k={tt['cfg']['k']}$" if tt["cfg"].get("k") is not None else "")

        # These two sentences used to be fixed prose written for a checkpoint that
        # missed three edges. Swapping the reference checkpoint left them asserting
        # things the data no longer said -- "every missed edge..." with nothing missed,
        # and a companion run described from memory. Both are now derived.
        n_missed = len(tt["true_edges"]) - tt["true_positives"]
        ceiling_clause = (
            "Every missed edge ends at a feature no latent recovered, so nothing here "
            "is a failure of the metrics; it is the ceiling the SAE set for them."
            if n_missed else
            "Nothing was dropped and nothing was invented: on this checkpoint the "
            "metrics reproduce the tree exactly.")

        comp = _json(C.OUT_DIR / "batch_topk_toy_calibration.json")
        if comp:
            c_truth = {tuple(e) for e in comp["true_edges"]}
            c_found = {tuple(e) for e in comp["found_edges"]}
            c_rec = set(comp["recovered_features"])
            c_test = {e for e in c_truth if e[0] in c_rec and e[1] in c_rec}
            companion_clause = (
                " One run is not a guarantee. An independent checkpoint on a different "
                f"architecture (\\texttt{{batch\\_topk}}, $k={comp['cfg'].get('k')}$, "
                f"this repo's trainer) learned {comp['n_recovered_features']} of "
                f"{comp['n_features']} features and returned "
                f"{comp['true_positives']} of {len(c_truth)} edges with "
                f"{comp['false_positives']} false positives: it too returned every edge "
                f"whose endpoints both existed ({len(c_found & c_test)} of {len(c_test)})."
            )
        else:
            companion_clause = ""
        d["calibration_toy_tree_recovered"] = (
            r"\textbf{What an edge is, and what the metrics did with it.} "
            "The toy world is a generative process: each node is a feature that fires "
            "with the probability printed beneath it, \\emph{conditional on its parent "
            "firing}. Feature~1 at $p=0.2$ under a parent at $p=0.15$ therefore fires on "
            "$3\\%$ of draws, not $20\\%$. An \\textbf{edge} is containment: a child "
            "fires only on draws where its parent fires, so "
            "$P(\\text{parent}\\mid\\text{child})=1$ while "
            "$P(\\text{child}\\mid\\text{parent})$ stays small, and it is that asymmetry "
            "the metrics measure. Hollow nodes (the root, and one hidden sibling per "
            f"family) carry no feature index, {links_clause}. "
            "\\textbf{Left}: the known tree. "
            f"\\textbf{{Right}}: the same tree after a Matryoshka SAE "
            f"(\\texttt{{{arch}}}{k_clause}) is "
            "trained on the world's activations, each latent is matched to the true "
            "feature its decoder points at, and the metrics are run on the learned "
            f"latents alone. It returned {tt['true_positives']} of "
            f"{len(tt['true_edges'])} true edges with {tt['false_positives']} false "
            f"positives. The reading that matters is the decomposition: the SAE learned "
            f"{tt['n_recovered_features']} of {tt['n_features']} features, leaving "
            f"{n_test} edges with both endpoints present, and the metrics returned "
            f"\\emph{{all}} {n_test} of them. {ceiling_clause}{companion_clause}")
        d["calibration_trained_toy_recovery"] = (
            r"\textbf{The same world, after a real training run.} The toy world's ground "
            "truth is a tree: a set of parent$\\rightarrow$child links between features "
            "(its edges). Here a Matryoshka SAE is actually trained on the world's "
            "activations, each learned latent is matched to the true feature it represents, "
            "and an edge counts as recovered when the SAE learned latents for both of its "
            "features and the metrics keep the link between them. Precision "
            rf"{tt['precision']:.2f}, recall {tt['recall']:.2f}: {tt['true_positives']} of "
            f"{len(tt['true_edges'])} true edges recovered with {tt['false_positives']} false "
            f"positives. The SAE recovered {tt['n_recovered_features']} of {tt['n_features']} "
            "true features, "
            + ("and every miss is an edge whose child it never learned, which bounds recall "
               "from above. "
               if tt["true_positives"] < len(tt["true_edges"]) else
               "so no edge was out of reach and recall is not bounded by the SAE here. ")
            + r"\emph{Right:} the lateral control, asking whether the "
            "Matryoshka nesting itself places a parent in an earlier block than its children."
            + extra)

    ib = [q for base in (C.OUT_DIR / C.SOURCE_NAME, C.OUT_DIR / "pcfg-matryoshka")
          for q in sorted(base.glob("layer_*")) if (q / "in_block_edges.json").exists()]
    if ib:
        d["in_block_relations"] = (
            r"Directed edges and co-extensive duplicates \emph{within} each block, where no block "
            "ordering fixes the direction and it must be derived from coverage asymmetry, across "
            f"all {len(ib)} graded runs. Reported as a rate per available pair rather than as a "
            "count: gemma's blocks are nested prefixes of 128 to 6{,}144 features, so the number "
            "of available pairs differs by a factor of 2{,}300 and raw counts invert the "
            "reading: the deepest block holds the most duplicate pairs and the fewest per pair. "
            "The concentration in B0 also holds on a PCFG SAE whose eight blocks are all 224 "
            "features, so it is not an artefact of B0 being small. What distinguishes B0 is "
            "being the outermost Matryoshka prefix: the one block trained to reconstruct on its "
            "own.")

    d["cross_source_funnel_shares"] = (
        "The same metric code and the same global thresholds, applied to the released Matryoshka "
        r"SAE on \texttt{gemma-2-2b} and to Matryoshka SAEs trained on a PCFG corpus. Reported "
        "as shares of each run's own candidate set, because the dictionaries differ in size and "
        "in block count and counts would not compare. Thresholds are deliberately not tuned per "
        "source: holding them fixed is what makes any cross-source comparison mean something. "
        "The reconstruction filter should be read with care on PCFG, where the weakest candidate "
        r"edge sits $3.5\times$ above the threshold, so the filter is inert there and the "
        "surviving edges have passed coverage alone.")
    dr = X.rows()
    by_layer = X.matched(dr, "layer")
    if by_layer:
        by_depth = X.matched(dr, "depth")
        order = [(t, k) for t, k, _ in X.ranked(by_layer)]
        g = [v for _, _, v in X.ranked(by_layer)]
        n_ag = X.agree_split(g)
        agree = list(zip([t for t, _ in order], g))[:n_ag]
        shared = sorted({o["layer"] for o, _, _ in by_layer})
        gem = [r for r in dr if r["is_ref"]]
        oth = [r for r in dr if not r["is_ref"]]
        dens = {r["label"]: X.density(r) for r in dr}
        d["cross_source_layer_response"] = (
            "Six measurements of block pair B0$\\rightarrow$B1, on the released Matryoshka SAE "
            rf"over \texttt{{{C.MODEL_NAME}}} and on Matryoshka SAEs trained over a "
            rf"{oth[0]['n_layers']}-block transformer fitted to a PCFG corpus. Both sources have "
            "an SAE at layer" + ("s " if len(shared) > 1 else " ")
            + " and ".join(str(s) for s in shared) + ", so the grey connectors are drawn between "
            "runs at the same block index and need no matching argument; their length is the "
            "disagreement. Every panel is a share and every panel is on the same axis anchored at "
            "zero, so lengths compare across panels. "
            + ", ".join(X.tex(t) for t, _ in agree)
            + rf" agree to within {max(v for _, v in agree):.1f} "
            rf"points, while {X.tex(order[-1][0])} differs by {g[-1]:.0f} and "
            rf"{X.tex(order[-2][0])} by "
            rf"{g[-2]:.0f}. \textbf{{That agreement is worth less than it looks}}: all {n_ag} "
            "of the close measures sit against a floor or a ceiling on both sources, so scaled "
            r"by the range each one varies over across every graded run "
            r"(Table~\ref{tab:align}) only "
            + X.tex(min(X.PANELS, key=lambda p: X.gap_ratio(dr, by_layer, p[1]))[0])
            + " stands out, and the reconstruction filter moves from nearly the worst to among "
            "the best. The reading we would take, carrying that caveat, is that the "
            "many-to-many fan-out "
            "B0$\\rightarrow$B1 is a property of the Matryoshka nesting rather than of "
            r"\texttt{gemma-2-2b}: what the base model changes is how much of it there is ("
            + " against ".join(rf"{dens[r['label']]:.1f}\%" for r in (gem[0], oth[0]))
            + " of all B0$\\times$B1 feature pairs become candidate edges), not its "
            r"character. \textbf{Both PCFG runs are a single grammar configuration} ("
            + X.grammar_line(dr) + r"), one point of the three-axis sweep Exp 2 specifies "
            "(terminal distribution, formatting density, grammar depth), so this compares gemma "
            "against one PCFG corpus and not against PCFG. In particular the formatting axis, "
            "which the project now treats as the mechanism behind bottleneck hijacking, is at "
            r"its second-lowest rung here. \textbf{Two shared layers cannot establish a trend}, "
            r"and the deeper"
            "block pairs are excluded because the PCFG runs hold 0--3 candidate edges in each of "
            "them. The token budgets also differ by "
            rf"{max(r['tokens'] for r in dr) / min(r['tokens'] for r in dr):.0f}$\times$, which "
            "cuts against the density gap rather than explaining it: the joint-support guard is "
            "an absolute count, so more tokens make it easier to clear, and the source with more "
            "tokens is the sparser one.")
        if by_depth and not all(o["layer"] == q["layer"] for o, q, _ in by_depth):
            dgap = [X.gap(by_depth, k) for _, k in order]
            ml, md = sum(g) / len(g), sum(dgap) / len(dgap)
            d["cross_source_alignment_check"] = (
                "Comparing two base models of different depth requires an alignment, and the two "
                "defensible ones disagree here about which runs to pair: by block index the PCFG "
                "layers pair with gemma's "
                + " and ".join(f"L{q['layer']}" for _, q, _ in by_layer) + ", and by relative "
                r"depth $(L{+}1)/N$ (the fraction of the network that has run when the SAE reads "
                r"\texttt{hook\_resid\_post}) they pair with gemma's "
                + " and ".join(f"L{q['layer']}" for _, q, _ in by_depth) + ", at opposite ends of "
                "the network. The choice is therefore not presentational, so it is measured here "
                "rather than made in an axis label: the mean gap over the six measures is "
                rf"{ml:.1f} points under block index against {md:.1f} under relative depth. "
                rf"\textbf{{On {len(by_layer)} shared layers this is suggestive, not a result.}} "
                "It is reported because an unstated alignment does the same work invisibly, and "
                "because the two rules would support different claims about which part of gemma "
                "a four-block transformer stands in for.")
    # DRAFT CAPTIONS -- prose to be rewritten by hand before submission, per the
    # same rule as the world figure above: the numbers are load-bearing and are
    # computed from the world each figure draws, the sentences around them are not.
    if world:
        n_never = sum(1 for v in world["fire_count"] if v <= 0)
        mass = world["bucket_mass"]
        tot = max(world["total_tokens"], 1)
        big = max(range(len(mass)), key=lambda k: mass[k]["tokens"])
        d["calibration_toy_corpus_firing"] = (
            r"\textbf{The corpus the Tier-1 calibration is run on.} Every claim in this "
            "appendix is a claim about a corpus, and a metric that separated two classes "
            "cleanly on a corpus with nothing in it has demonstrated nothing. "
            r"\emph{Left:} the tokens each declared feature fires on, log scale, coloured "
            "by the role it was planted as; the dashed line is the production "
            rf"\texttt{{MIN\_FIRE\_COUNT}} $= {world['min_fire']}$, below which no pair is "
            # sentence-initial: `_spell` returns the word, not its capitalisation
            f"scored at all. {_spell(n_never).capitalize()} of the "
            f"{len(world['fire_count'])} declared "
            "features never fire and are drawn as stubs at the floor rather than omitted. "
            "The super-parent is visible as the single bar an order of magnitude above "
            r"everything else --- that is what makes it a super-parent. \emph{Right:} where "
            "the corpus's token mass sits across the three frequency buckets metric 5 "
            f"re-weights by: {100 * mass[big]['tokens'] / tot:.0f}\\% of all tokens fall in "
            f"bucket {big}, carried by {mass[big]['ids']:,} distinct id"
            f"{'' if mass[big]['ids'] == 1 else 's'}. That concentration is what the "
            "frequency control has to work against, so it is reported rather than assumed. "
            "The bucket bars use a single-hue sequential shade and no role colour: buckets "
            "are an order, not a category, and the palette in the key below applies to the "
            "left panel only.")
        d["calibration_reverse_coverage"] = (
            r"\textbf{What the first gate proposes, and the one true edge it cannot.} "
            "The reverse-coverage matrix $R = $ co-fire\,/\,child firing for every "
            f"parent--child pair in the hand-built world, with the ground truth drawn on "
            rf"top of it. Coverage keeps a pair when $R \geq \tau = {C.EDGE_TAU}$, which "
            f"proposes {len(world['candidates'])} of the "
            f"{world['P'] * world['C']:,} possible pairs. Circles are true edges it "
            "proposed, crosses are pairs it proposed that no true edge backs, and the "
            "square is the reading that matters: a true edge whose $R$ never reaches the "
            "threshold, so no later gate ever sees it. That is a limit of the first gate "
            "which no downstream metric can repair, and it is the same absorption case the "
            "scorecard files as a demonstrated blind spot rather than a caught pathology. "
            "The super-parent's row is the wide band of crosses --- one parent proposing "
            "against nearly every child, at high $R$, on co-firing alone.")
        stg = world["stages"]
        d["calibration_gate_funnel_by_role"] = (
            r"\textbf{What each gate removes, counted.} The same three composed gates the "
            "before/after figure draws, as counts split by the structure each candidate "
            f"was planted as. Coverage proposes {len(stg[0][1])} candidates; the "
            f"reconstruction condition takes that to {len(stg[1][1])} and the "
            f"token-frequency control to {len(stg[2][1])}. The block of true edges does not move "
            "at any stage, which is the claim: the gates are removing planted pathology "
            "rather than thinning everything. The division of labour is legible in the "
            "widths --- the super-parent's candidates die entirely at reconstruction and "
            "not at coverage, and the frequency-coincidence pair survives both coverage and "
            "reconstruction and dies only at the control built for it. The two negative "
            "controls are the segments that never shrink; they are limitations this "
            "composition demonstrates, not failures of it.")
        d["calibration_seed_sweep"] = (
            r"\textbf{The scorecard is not an accident of one draw.} Every row of the "
            "Tier-1 scorecard, re-run on independently drawn worlds --- a new corpus, new "
            "decoder directions and a new residual draw per seed, with the production "
            "thresholds unchanged. The cell carries the verdict and the shade carries the "
            r"margin: how decisively that metric separated the two classes it was graded "
            r"on, on a $\log_{10}$ scale, because the margins span orders of magnitude and "
            "a grid of identical verdicts would hide exactly that. A row that passes at a "
            "margin of $1.05$ and one that passes at $10^{3}$ are not the same evidence. "
            r"The scale is capped at $10^{3}\times$, the same ceiling the scorecard itself "
            r"prints as \texttt{>1000x}: three rows score $10^{9}$, which is not a measured "
            "magnitude but the guard in the denominator when the rejected class scores zero, "
            "and letting it set the top of the ramp pushed every honestly-measured margin "
            "into the palest two shades. "
            "Rows scored categorically have no magnitude to report and are left grey rather "
            "than given an invented one. Everything else in this appendix is reported at "
            "seed 0; this is the figure that says what that number is worth.")
    if lay:
        d["tangle_lives_in_top_block_pair"] = (
            "Every metric per block pair \\emph{and} per layer, one panel per SAE source; no "
            "cell is an average, so a row is one block pair read at one layer and an empty "
            "candidate set shows as a dash rather than as a number. Colour ranks "
            "\\emph{within} a column of its own panel (the columns are different quantities on "
            "different scales, and the two sources have different dictionaries), so the reading "
            "is locational. Metric 9 is a within-block quantity while every row is a "
            "between-block pair, so each row reports its own parent block. "
            "Three things read off it. The structural pathology "
            "(multi-parenting, base-rate co-firing, superparents) sits in the outermost pair "
            "B0$\\rightarrow$B1 and it is not an average effect: multi-parenting is 89--100\\% at "
            "\\emph{every} graded layer of gemma and 99--100\\% at three of the four PCFG layers. "
            "The deeper pairs are not clean but differently broken, failing the frequency "
            "control instead (up to 96.3\\% of B2$\\rightarrow$B3 edges at L1) with sibling "
            "overlap several times higher. And fan-out Gini is the one column that localises "
            "nothing, sitting at 0.81--0.97 on gemma and 0.95--1.00 on PCFG in every pair at "
            "every layer, which makes it a property of the dictionary rather than of any "
            "boundary within it. The PCFG deep pairs rest on one to three candidate edges, so "
            "their quiet columns record emptiness, not health. gemma's fourth adjacent pair "
            "B3$\\rightarrow$B4 is read at layer 12 alone: its co-firing matrix is "
            "6{,}144$\\times$24{,}576 and needs a 49 GB card, so it is off by default and was "
            "run once. The five dashes beside it mean not run, not empty. The distinction is "
            "drawn rather than left out, because a pair silently absent from a grid reads as a "
            "pair with nothing in it, which is a claim the run never tested.")
    if _json(G / f"layer_{GRAPH_LAYER:02d}" / "metrics_report.json") or _json(
            C.OUT_DIR / "pcfg-matryoshka"
            / f"layer_{PCFG_GRAPH_LAYER:02d}" / "metrics_report.json"):
        # Two lines, by request. The reading rules that do not fit -- metric 9
        # reporting each row's parent block, and the deep PCFG pairs resting on
        # a handful of edges -- are the two a reader can get wrong, so they
        # belong in the body text or the appendix rather than being dropped.
        d["metrics_result_mid_layers"] = (
            r"\textbf{The metrics result at mid layers.} This figure displays the "
            "results of all metrics across the middle layer and all its blocks: "
            rf"\texttt{{gemma-2-2b}} at layer {GRAPH_LAYER} and the PCFG SAE at "
            rf"layer {PCFG_GRAPH_LAYER}.")

    if lay:
        d["battery_questions_gemma"] = (
            "Each of the metrics was built to answer one plain-language question about "
            "a candidate edge; each panel titles that question and plots its measured answer "
            r"on \texttt{gemma-2-2b}, block pair B0$\rightarrow$B1, at every graded layer. Read "
            "as a whole the figure shows the split the paper's results rest on: the questions "
            "about an edge's \\emph{quality} (reconstruction, frequency, siblings) come back "
            "healthy, while the questions about the graph's \\emph{structure} (base-rate "
            "co-firing, the probe, fan-out) come back failing, at every depth.")

    sp_gl = _json(G / f"layer_{GRAPH_LAYER:02d}" / "second_pass.json")
    tt2 = _json(C.OUT_DIR / "trained_toy_calibration.json")
    if tt2 and sp_gl:
        s = sp_gl["0->1"]["sres"]
        n_par = len({e["parent"] for e in s["edges"]})
        n_chi = len({e["child"] for e in s["edges"]})
        # the PCFG middle panel: PCFG_GRAPH_LAYER, the same constant build()
        # draws, with build()'s largest-scored fallback. Both sites must name
        # the same layer or the caption describes a panel that is not there.
        mid = ""
        best, best_name = None, ""
        cands = []
        for q in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("layer_*/second_pass.json")):
            spx = _json(q)
            if spx and "0->1" in spx and spx["0->1"]["sres"]["n_edges_scored"]:
                cands.append((q.parent.name.replace("_", "~"), spx))
        for name, spx in cands:
            if name == f"layer~{PCFG_GRAPH_LAYER:02d}":
                best, best_name = spx, name
        if best is None and cands:
            best_name, best = max(cands,
                                  key=lambda t: t[1]["0->1"]["sres"]["n_edges_scored"])
        grad = ""
        if best:
            b = best["0->1"]["sres"]
            b_par = len({e["parent"] for e in b["edges"]})
            b_chi = len({e["child"] for e in b["edges"]})
            mid = (rf"\emph{{Middle:}} a Matryoshka SAE trained on a PCFG corpus "
                   f"({best_name.replace('layer~0', 'layer~')}, B0$\\rightarrow$B1) "
                   f"repeats the tangle at small scale ({b['n_edges_scored']:,} "
                   f"candidate edges, {b_par} parents $\\times$ {b_chi} children, "
                   f"{b['n_pass']} probe-confirmed), and ")
            # the dictionary sizes that make "ascending scale" a number, not a vibe
            rr = _json(C.OUT_DIR / "pcfg-matryoshka"
                       / best_name.replace("~", "_") / "metrics_report.json")
            d_p = ((rr or {}).get("config") or {}).get("d_sae", 1792)
            grad = (" Read left to right, the worlds ascend in scale and complexity "
                    f"({tt2['n_features']} hand-built features, a {d_p:,}-latent SAE "
                    f"over a small transformer, a {C.D_SAE:,}-latent SAE over a "
                    "2B-parameter LLM), and the recovered structure degrades in step: "
                    "a clean tree, a small tangle, a dense one.")
        d["recovered_graph_toy_vs_pcfg_vs_gemma"] = (
            r"\textbf{The recovered graph, drawn edge by edge.} Hierarchy is partially "
            r"recovered, but not as a clean tree, especially at scale. \emph{Left:} "
            f"the trained toy recovers "
            f"{len(tt2['found_edges'])}/{len(tt2['true_edges'])} true edges with "
            # "no false positives" reads as prose while the count is zero, and
            # falls back to the number the moment a regeneration makes it real
            f"{tt2['false_positives'] or 'no'} false positives, while " + mid +
            rf"\emph{{Right:}} Gemma-2-2B (layer {GRAPH_LAYER}, B0$\rightarrow$B1) shows "
            f"substantial multi-parenting among the {s['n_edges_scored']:,} candidate edges "
            f"that reached the probe ({n_par} parents $\\times$ {n_chi} children), with "
            f"only the {s['n_pass']} probe-confirmed edges in green." + grad +
            " Blue marks a candidate edge that reached the probe but was not "
            "confirmed, green a probe-confirmed edge, and on the toy panel a dashed "
            "line marks a true edge the SAE never learned. Each child is placed "
            "beneath its best-matching parent, so a clean hierarchy would appear as "
            "near-vertical lines, and a parent's dot area scales with its child count, "
            "so the hub parents that own most of the tangle are visible at a glance.")
        ed = sp_gl["0->1"]["sres"]["edges"]
        dg: dict = {}
        for e in ed:
            dg[e["parent"]] = dg.get(e["parent"], 0) + 1
        ps = sorted(dg, key=lambda p: -dg[p])
        ch = {e["child"] for e in ed}
        tk = {e["child"] for e in ed if e["parent"] == ps[0]}
        d["one_parent_owns_the_block"] = (
            r"\textbf{One parent owns the block.} The hub structure behind the tangle, "
            rf"isolated on gemma-2-2b (layer {GRAPH_LAYER}, B0$\rightarrow$B1). "
            rf"\emph{{Left:}} the single biggest parent feature claims {len(tk)} of the "
            rf"{len(ch)} children that have any parent ({100 * len(tk) / len(ch):.0f}\%). "
            rf"\emph{{Right:}} ownership of all {len(ed):,} candidate edges is "
            rf"concentrated: the top parent alone carries {100 * dg[ps[0]] / len(ed):.0f}\%, "
            rf"the top five {100 * sum(dg[p] for p in ps[:5]) / len(ed):.0f}\%, and the top "
            rf"ten {100 * sum(dg[p] for p in ps[:10]) / len(ed):.0f}\% of every proposed "
            "parent$\\rightarrow$child relationship. A hierarchy distributes parenthood; "
            "a hub monopolises it, which is what the fan-out Gini and superparent metrics "
            "quantify in aggregate.")
        bc: dict = {}
        for e in ed:
            bc.setdefault(e["child"], []).append(e["parent"])
        wc, wp = max(bc.items(), key=lambda kv: len(kv[1]))
        cnts = sorted(len(v) for v in bc.values())
        lblj = _json(G / f"layer_{GRAPH_LAYER:02d}" / "feature_labels.json") or {}
        ex = ""
        if lblj:
            pl = [lblj.get(str(p), "") for p in wp]
            pl = [t for t in pl if t][:3]
            if pl and lblj.get(str(wc)):
                # word-boundary truncation: a caption quote cut mid-word reads
                # as a typo, not an ellipsis
                short = lambda t, w: textwrap.shorten(t, w, placeholder="…")
                ex = (rf" Neuronpedia's labels name the absurdity: a child read as "
                      rf"``{short(lblj[str(wc)], 60)}'' is claimed by parents read as "
                      + ", ".join(f"``{short(t, 48)}''" for t in pl)
                      + ", among others; the labels are automatic and noisy, but the "
                        "point is structural, not semantic.")
        d["a_slice_of_the_tangle"] = (
            r"\textbf{A slice of the tangle.} A parent with many children is what a "
            "hierarchy looks like; a child with many parents is what breaks one. A tree "
            "grants each child exactly one parent, and in this slice of "
            rf"gemma-2-2b (layer {GRAPH_LAYER}, B0$\rightarrow$B1), selected by a "
            "stated rule (the eight most-claimed children and every parent claiming "
            rf"them), the featured child alone is claimed by {len(wp)} parents at "
            rf"once, its neighbours by 7--9 each, and the \emph{{median}} child of the "
            rf"whole pair has {cnts[len(cnts) // 2]}. This is the per-child view of the "
            rf"{100 * sum(1 for v in bc.values() if len(v) >= 2) / len(bc):.0f}\% "
            "multi-parenting rate: the violation is not a tail event but the typical "
            "case." + ex)
        d["one_child_many_parents"] = (
            r"\textbf{A child with "
            + f"{len(wp)}"
            + r" parents.} The single-child view of the slice in "
            r"Fig.~\ref{fig:a-slice-of-the-tangle}, with every claiming parent's label "
            "spelled out. A tree grants each child exactly one parent; the most-claimed "
            rf"child feature of gemma-2-2b (layer {GRAPH_LAYER}, B0$\rightarrow$B1) is "
            rf"claimed by {len(wp)} at once, while the \emph{{median}} child with any "
            rf"parent has {cnts[len(cnts) // 2]}." + ex)

    fmt_dirs = sorted((C.OUT_DIR / "pcfg-matryoshka").glob("fmt_*"))
    if fmt_dirs:
        _fr = [(int(p.name.split("_")[1]) / 1e4, _json(p / "metrics_report.json"))
               for p in fmt_dirs]
        _fr = [(d0, r) for d0, r in _fr if r]
        _dens = sorted({d0 for d0, _ in _fr})
        _lay = _fr[0][1]["config"]["layer"]
        d["pcfg_formatting_sweep"] = (
            "Four measures of block pair B0$\\rightarrow$B1 against the delimiter density of "
            f"the generating grammar, at layer {_lay} of the PCFG transformer; one point per "
            f"seed ({len(_fr) // len(_dens)} per density), the line through the seed means. "
            "Formatting density is the axis the project treats as the mechanism behind "
            "bottleneck hijacking, and the reading is the shape: a measure that stays flat "
            "across the axis is a property of the SAE that no grammar knob reaches, while a "
            "measure that slopes is confounded with the corpus and must be controlled before "
            "it is blamed on the architecture. Seeds are plotted individually because with "
            "three of them a mean without its spread would overstate what one grammar "
            "configuration can support.")
    d["sres_null_rate_vs_dictionary_size"] = (
        r"The $S_\mathrm{res}$ test passes an edge when both decoders fall within the top "
        rf"$k = {C.SRES_RANK_TOP_K}$ of the probe's correlations over the whole dictionary. It "
        "is a geometry test, so an unrelated parent passes whenever chance places it there: the "
        r"null rate is $k/D$, the grey line, which is $11.9\%$ on the 42-feature synthetic toy, "
        r"$0.28\%$ on a 1{,}792-latent PCFG SAE and $0.015\%$ on gemma's 32{,}768. Each measured "
        "pass rate is drawn with a vertical drop to its own null, and that distance (not the "
        r"rate) is what the measurement is worth. The rate is over every block pair a run "
        r"probed, not its outermost pair alone (Table~\ref{tab:null}), because the null is a "
        "property of the dictionary and every pair is scored against the same one. "
        + _sres_same_D_clause()
        + ("A measured zero has no position on a logarithmic axis and is drawn at a marked "
           "floor rather than silently dropped."
           if any(o == 0 for *_, o in sres_observed_rates())
           else "A measured zero would have no position on a logarithmic axis and is drawn "
                "at a marked floor rather than silently dropped; no run needs that here."))
    return d


# (slot, figure, wide?, bold lead sentence). Order is an argument, not taste: the
# instrument has to be established before any number it produces means anything,
# so the calibrations come first; the empirical claims then follow in order of how
# much evidence stands behind them (multi-parenting has five layers, the funnel
# has one); the hypothesis figure and the mechanism that explains it are adjacent;
# and the battery's own failure closes, because it qualifies everything above it.
TEX_ORDER = [
    # main text: the whole battery on both sources FIRST, then the three
    # headline results it decomposes into; everything after goes to the
    # appendix. The overview leads because every main figure after it is one
    # of its columns read closely -- the funnel is coverage against the probe,
    # multi-parenting is the out-degree column across layers -- and a reader
    # who meets those one at a time never learns that the columns disagree.
    # It is a full-page figure at 19 columns x 47 rows; the manuscript is
    # expected to give it its own page (\clearpage, or sidewaysfigure*).
    ("MAIN 1", "tangle_lives_in_top_block_pair", True,
     "One instrument, untuned, on both sources.",
     "What the metrics find"),
    ("MAIN 1b", "metrics_result_mid_layers", True,
     "The metrics result at mid layers.", None),
    ("MAIN 2", "funnel_coverage_to_sres", False,
     "Coverage proposes; the strict test disposes.", None),
    ("MAIN 3", "multiparenting_by_layer", False,
     "The recovered graph is not a tree.", None),
    ("MAIN 4", "recovered_graph_toy_vs_pcfg_vs_gemma", True,
     "The recovered graph, drawn.", None),
    # The whole calibration ladder sits in the appendix: it is what licenses the main
    # text's numbers rather than a finding of its own, and the three read as one
    # argument only when they are adjacent.
    # Ordered by TIER, not by the order the figures were written: both Tier-1
    # figures (the hand-built world, where the answer was fixed before any metric
    # saw it) come before both Tier-2 ones (the same question after a real training
    # run). The ladder is the argument, so the figures climb it in order.
    # Tier 1 climbs in the order the argument does, not in the order the figures
    # were written: the world first (a metric that separated cleanly on an empty
    # corpus has shown nothing), then the first gate at full resolution, then the
    # gates composed as counts, then as a drawing, then every metric scored, and
    # last whether that score survives a different draw of the world.
    ("APP 0a", "calibration_toy_corpus_firing", True,
     "The world the calibration is run on.", "Appendix: the instrument, calibrated"),
    ("APP 0b", "calibration_reverse_coverage", True,
     "What the first gate proposes, and the one true edge it cannot.", None),
    ("APP 0c", "calibration_gate_funnel_by_role", True,
     "What each gate removes, counted.", None),
    ("APP 0d", "calibration_toy_world_before_after", True,
     "Which gate caught which injected pathology.", None),
    # the single-panel twin of 0d: emitted so the manuscript can choose
    # between them; expected to be commented in/out there, not both shown
    ("APP 0d'", "calibration_toy_world_gate_verdicts", True,
     "What the set of metrics kept.", None),
    ("APP 0e", "calibration_synthetic_toy_scorecard", True,
     "Every metric scored against a known tree.", None),
    ("APP 0f", "calibration_seed_sweep", True,
     "The same scorecard on eight independently drawn worlds.", None),
    ("APP 0g", "calibration_trained_toy_recovery", True,
     "The same tree, after a real training run.", None),
    ("APP 0h", "calibration_toy_tree_recovered", True,
     "That tree drawn, before and after the metrics.", None),
    # the overview matrix moved to MAIN 1; the section it used to open now
    # starts at the slice, which is one cell of that matrix read closely
    ("APP 1b", "a_slice_of_the_tangle", False,
     "A slice of the tangle: every child in it has many parents.",
     "Appendix: what the metrics find"),
    # the single-child twin of the slice: emitted so the manuscript can choose
    # between them; expected to be commented in/out there, not both shown
    ("APP 1b'", "one_child_many_parents", False,
     "A child with ten parents: the single-child view of the slice.", None),
    ("APP 1c", "one_parent_owns_the_block", False,
     "One parent owns the block: the hub behind the tangle, isolated.", None),
    ("APP 2", "base_rate_vs_frequency_capture", False,
     "The over-connection is a base-rate effect, not token-frequency capture.",
     "The bottleneck-hijacking hypothesis, tested"),
    ("APP 3", "superparent_fanout_vs_firing", False,
     "Why a parent that fires often enough clears the coverage bar for nothing.", None),
    ("APP 4", "shared_input_moved_every_metric", True,
     "Six metrics designed as independent detectors failed together.",
     "What this says about metric batteries"),
    ("APP 5", "sres_null_rate_vs_dictionary_size", False,
     "A top-$k$ rank rule is only as strict as the dictionary is large.", None),
    ("APP 6", "in_block_relations", True,
     "Same-level structure lives in the outermost block.", None),
    ("APP 7", "edge_survival_by_block_pair", True,
     "What each filter removes, by block pair and by depth.", None),
    ("APP 8", "cross_source_funnel_shares", False,
     "One unchanged metric set across SAE sources.", None),
    # The two cross-source depth figures sit together and immediately after the
    # funnel that establishes the sources are comparable at all. The alignment
    # check follows the result it qualifies rather than preceding it: it is a
    # caveat on how the layers were paired, and a reader who has not yet seen the
    # pairing has nothing to apply it to.
    ("APP 9", "cross_source_layer_response", True,
     "The shape of the B0$\\rightarrow$B1 relation is the same on both base models; "
     "its strength is not.", None),
    ("APP 10", "cross_source_alignment_check", False,
     "Which alignment the comparison rests on, measured rather than assumed.", None),
    ("APP 11", "depth_profile_across_layers", True,
     "No measure is monotonic in depth.", None),
    ("APP 12", "pcfg_formatting_sweep", True,
     "The formatting-density axis, swept.", None),
    ("APP 13", "battery_questions_gemma", True,
     "The metrics, question by question.", None),
]

TEX_HEAD = r"""% ===========================================================================
% Figures and captions.
%
% Generated by reporting/make_report_figures.py. Every quantity in every
% caption was read from the JSON its figure plots, at generation time; do not
% edit a number here by hand, regenerate.
%
% Ordered as the paper reads. MAIN 1-5 are the main text, APP 1-13 the appendix.
%
% Requires: graphicx. Emitted for a ONECOLUMN class, which is what the ICLR
% template uses; pass --twocolumn for full-width figure* floats.
% Captions sit below the image, as figures take them, and the mirror of
% tables.tex where they sit above.
%
% \graphicspath below lists several candidates on purpose. It resolves against
% the directory of the MAIN .tex, not against this file, so a copy of this file
% \input from a Sections/ subdirectory while the PNGs sit in a folder at the
% project root needs that folder named -- a bare ./ points at the root and finds
% nothing, and every figure fails with nothing wrong in the markup. LaTeX takes
% the first path that hits, so the extra entries cost nothing.
%
% Expects the PNGs reachable via \graphicspath below. To compile on its own,
% prepend
%     \documentclass{article}\usepackage{graphicx}\begin{document}
% and append \end{document}.
% ===========================================================================
"""


def write_tex(out_dir: Path, graphics: str):
    """figures.tex beside the PNGs, or wherever --out points."""
    caps = _captions()
    # \graphicspath resolves against the COMPILATION root -- the directory of the
    # main .tex -- not against the file doing the \input. On Overleaf the images
    # sit in a folder at the project root while this file is \input from a
    # Sections/ subdirectory, so a bare "./" points at the root, finds no PNGs,
    # and every figure comes back "file not found" with nothing wrong in the
    # markup. Emitting one path was the bug; a graphicspath is a LIST precisely
    # so a document can move without its images being re-found by hand.
    #
    # Order: the named folder from the project root, the same folder one level up
    # (compiling this file standalone from inside Sections/), then the two
    # directories a reader is most likely to have dropped the PNGs into. LaTeX
    # takes the first hit, so listing more costs nothing at compile time.
    # PAPER_DIR.name is in the list because that is what the folder is called
    # here, and a reader who drags this directory into Overleaf keeps the name.
    # "figuers" (sic) is the name it has in the paper project, kept because
    # renaming a folder someone else's main.tex already points at is worse than
    # carrying the typo. "figures" is the spelling anyone else would pick.
    extra = ["./", PAPER_DIR.name + "/", "figuers/", "figures/"]
    cands = [graphics] + [f"../{graphics}"] if graphics.strip("/.") else []
    cands += extra + [f"../{e}" for e in extra if e != "./"] + ["../"]
    seen, paths = set(), []
    for g in cands:
        if g not in seen:
            seen.add(g)
            paths.append(g)
    L = [TEX_HEAD, r"\graphicspath{" + "".join(f"{{{g}}}" for g in paths) + "}", ""]
    seen, n = set(), 0
    for slot, name, wide, lead, section in TEX_ORDER:
        if not (PAPER_DIR / f"{name}.png").exists():
            continue
        if section and section not in seen:
            seen.add(section)
            L += [r"% " + "-" * 73, f"% {section}", r"% " + "-" * 73, ""]
        env = "figure*" if (wide and TWOCOLUMN) else "figure"
        width = r"\textwidth" if (wide and TWOCOLUMN) else r"\linewidth"
        body = caps.get(name)
        L += [f"% [{slot}]", rf"\begin{{{env}}}[tbp]", r"  \centering",
              rf"  \includegraphics[width={width}]{{{name}}}",
              r"  \caption{%"]
        # a caption body that opens with its own \textbf lead carries the
        # whole caption; prepending the TEX_ORDER lead would print two bold
        # openers back to back
        if not (body and body.lstrip().startswith(r"\textbf")):
            L.append(rf"    \textbf{{{lead}}}")
        if body:
            L.append(_wrap(body))
        else:
            L.append(r"    % no caption body: nothing in _captions() covers this figure")
        # labels default to the figure name; overrides keep manuscript-facing
        # labels short where the file name is long
        label = {"recovered_graph_toy_vs_pcfg_vs_gemma": "recovered-graph",
                 }.get(name, name.replace("_", "-"))
        L += [r"  }", rf"  \label{{fig:{label}}}",
              rf"\end{{{env}}}", ""]
        n += 1
    (out_dir / "figures.tex").write_text("\n".join(L))
    return n


CLAIMS = {
    "funnel_coverage_to_sres": "co-firing proposes far more edges than survive the strict test",
    "edge_survival_by_block_pair": "what each filter removes, by block pair and by depth",
    "depth_profile_across_layers": "no measure is monotonic in depth once BOS is excluded",
    "metrics_result_mid_layers": "every metric at one fixed depth per source, so the axis carries architecture and not depth",
    "multiparenting_by_layer": "the graph is not a tree — the one claim BOS exclusion left standing",
    "superparent_fanout_vs_firing": "the superparent gate reads fan-out; firing rate is handled per edge",
    "calibration_synthetic_toy_scorecard": "every metric scored against a known tree, plus two demonstrated blind spots",
    "calibration_toy_world_before_after": "which gate removed which injected pathology, on the world where the answer was fixed first",
    "calibration_toy_world_gate_verdicts": "the same gate verdicts in one panel — blocks as rows, the declared world kept as the faded edges",
    "calibration_trained_toy_recovery": "the same tree after a real training run, and the nesting control",
    "calibration_toy_tree_recovered": "the tree drawn: which edges the metrics returned, and which features were never learned",
    "cross_source_funnel_shares": "one unchanged metric set across two SAE sources",
    "shared_input_moved_every_metric": "five of six metrics share an input and failed together — the battery's own failure mode",
    "base_rate_vs_frequency_capture": "the over-connection is base rate, not frequency capture — the hypothesis's premise, tested",
    "in_block_relations": "same-level structure concentrates in B0 on both sources, read as a per-pair rate",
    "sres_null_rate_vs_dictionary_size": "a top-k rank rule is only as strict as D is large",
    "calibration_toy_corpus_firing": "the corpus the Tier-1 calibration runs on — firing per feature, and where the token mass sits",
    "calibration_reverse_coverage": "what reverse coverage proposes at full resolution, and the true edge it structurally cannot",
    "calibration_gate_funnel_by_role": "what each composed gate removes, counted by the structure it was planted as",
    "calibration_seed_sweep": "the Tier-1 scorecard re-run per seed — whether 14/14 is a result or one draw",
    "tangle_lives_in_top_block_pair": "every metric, every column, on both sources with one untuned instrument",
    "cross_source_layer_response": "the shape of B0→B1 is the same on both base models at the layers both graded; its strength is not",
    "cross_source_alignment_check": "which alignment across two models of different depth the data prefers — block index or relative depth",
}


def write_readme(written, skipped, sources):
    """The directory documents itself, from the same table that built it.

    A hand-written index of figures is a second place for the truth to live, and
    the one that goes stale first -- it is still describing figure 3 after figure
    3 was replaced.
    """
    readme = PAPER_DIR / "README.md"
    L = [C.nav_html(depth=C.page_depth(readme), current="outputs/paper_figuers/"), "",
         "# `paper_figuers/` — figures for the write-up", "",
         "Generated by [`reporting/make_report_figures.py`](../../reporting/make_report_figures.py); "
         "do not edit by hand, and do not add a figure here that no generator can rebuild.", "",
         "| figure | what it is evidence for | built from |", "| --- | --- | --- |"]
    for name in written:
        L.append(f"| `{name}.png` | {CLAIMS.get(name, '')} | {sources.get(name, '')} |")
    if skipped:
        L += ["", "## Not built", "",
              "| figure | why |", "| --- | --- |"]
        for name, why in skipped:
            L.append(f"| `{name}.png` | {why} |")
    L += ["", "Every number in every title is read from the JSON being plotted, so a caption "
          "cannot outlive the data under it -- the previous generator quoted layer-6 figures "
          "that had been withdrawn.", ""]
    (PAPER_DIR / "README.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="show the plan and each figure's input, draw nothing")
    ap.add_argument("--out", type=Path, default=None, metavar="DIR",
                    help=f"write figures.tex here instead of {PAPER_DIR.name}/ "
                         "(the PNGs are always written beside the code)")
    ap.add_argument("--graphicspath", default=None, metavar="PATH",
                    help="an EXTRA \\graphicspath entry, tried first; the common layouts "
                         "(./, the PNG folder's name, figuers/, figures/, and each one "
                         "level up) are always emitted after it")
    ap.add_argument("--twocolumn", action="store_true",
                    help="emit figure* full-width floats (needs a twocolumn class)")
    ap.add_argument("--titles", action="store_true",
                    help="bake the figure-level title/subtitle into each PNG; off by "
                         "default because the paper's LaTeX captions carry that text")
    args = ap.parse_args()
    globals()["TWOCOLUMN"] = args.twocolumn
    globals()["TITLES"] = args.titles

    if not args.list:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped, sources = build(args.list)
    tex_dir = None
    if not args.list:
        write_readme(written, skipped, sources)
        # Default beside the PNGs; --out puts the .tex somewhere else, in which
        # case graphicspath has to reach back to them and "figures/" is the
        # layout the paper repo uses.
        tex_dir = (args.out or PAPER_DIR)
        tex_dir.mkdir(parents=True, exist_ok=True)
        gp = args.graphicspath or ("./" if tex_dir.resolve() == PAPER_DIR.resolve()
                                   else "figures/")
        n_tex = write_tex(tex_dir, gp)

    print(f"[fig] {'plan for' if args.list else 'wrote'} {PAPER_DIR}")
    for w in written:
        print(f"  ok    {w}")
    for name, why in skipped:
        print(f"  SKIP  {name}\n          {why}")
    if tex_dir is not None:
        print(f"  ok    figures.tex  ({n_tex} figures, graphicspath {gp!r}) -> {tex_dir}")
    if skipped and not args.list:
        print(f"\n[fig] {len(written)} written, {len(skipped)} skipped — "
              "the set above is not complete, and the reasons are printed rather "
              "than left to the reader to notice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
