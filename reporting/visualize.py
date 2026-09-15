"""
Visualise the hierarchy-metric results as self-contained interactive HTML
(plotly.js embedded, works offline - same style as the example sankey script).

Reads the cached statistics in outputs/exp0_stats.pt, recomputes the full
per-block-pair distributions (the JSON report keeps only summaries, not the
distributions), then writes:

    outputs/metrics_dashboard.html    aggregate dashboard: filter funnel + distributions
    outputs/superparent_sankey.html   one superparent's fan-out to its children

    outputs/synthetic_toy_calibration.html      (--calibration) metric scorecard + the
                                      genuine-vs-pathological separation each
                                      metric achieves on the synthetic toy
    outputs/trained_toy_calibration.html (--trained-calibration) Tier-2 scorecard:
                                      edge recovery against the known tree, plus
                                      the Matryoshka nesting check when
                                      outputs/block_tree_alignment.json exists
    outputs/qualitative_dashboard.html (--qualitative) surviving vs rejected real
                                      edges with Neuronpedia labels, colour-coded

Run:
    pip install plotly
    python3 -m reporting.visualize                # real gemma-2-2b dashboards (needs exp0_stats.pt)
    python3 -m reporting.visualize --calibration  # synthetic-toy calibration (needs no cache)
    python3 -m reporting.visualize --qualitative  # real survivor-vs-rejected label table
                                                  # (needs outputs/qualitative_check.json)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

import config as C
# The block structure comes from the file being graded, never from this module:
# config.BLOCK_RANGES is gemma's 32768 latents in 5 blocks, and slicing a PCFG
# dictionary (1792 in 8) at those boundaries returns a full set of plausible
# numbers computed from the wrong features. run_metrics owns the rule; importing
# it keeps one definition rather than a second copy that can drift.
from run_metrics import source_structure
from metrics import (
    coverage_legs,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    keep_edges,
    sibling_redundancy,
)

# House palette (same as the example sankey)
PURPLE, BLUE, TEAL = "#7C22CE", "#2196F3", "#0EA5A4"
AMBER = "#F59E0B"               # 4th pair (B3->B4), enabled with EXP0_B3B4=1
PAIR_COLORS = [PURPLE, BLUE, TEAL, AMBER]
GREEN, GREY, RED = "#22C55E", "#CBD5E1", "#EF4444"
GREEN_DARK = "#15803D"          # readable green for heading text on white


def page_subtitle(stats=None):
    """Context line for any page that describes ONE layer.

    Every per-layer HTML lands in outputs/layer_NN/ with no on-page clue which
    layer it is, so state it up front together with the run knobs a reader
    needs to interpret the numbers.
    """
    tokens = stats.get("total_tokens") if stats is not None else None
    cfg = stats.get("config") if stats is not None else None
    # Plotly does not decode HTML entities, so config.scope_line uses literal glyphs.
    return C.scope_line(tokens, bold=("<b>", "</b>"), config=cfg)


def plotly_asset(page_dir):
    """Ensure OUT_DIR/assets/plotly.min.js exists; return its page-relative src.

    Embedding the 4.6 MB plotly bundle in every page (`include_plotlyjs=True`)
    made each dashboard ~4.8 MB, so every regeneration added ~70 MB of new blobs
    to git. One shared copy keeps the pages ~150 KB and still works offline and
    on GitHub Pages, which cannot serve an LFS pointer (see .gitattributes).
    The src is computed per page rather than from a level count, so it stays
    right under EXP0_OUT, where the run dir is not named `outputs`.
    """
    dest = C.OUT_DIR / "assets" / "plotly.min.js"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(get_plotlyjs(), encoding="utf-8")
    return Path(os.path.relpath(dest, page_dir)).as_posix()


CAPTION_CSS = """
<style>
.x0cap{max-width:1150px;margin:8px auto 42px;padding:0 18px;
font:400 13px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#3F4B57;}
.x0cap h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:#9AA7B3;
font-weight:700;margin:0 0 10px;}
.x0cap dl{margin:0;display:grid;grid-template-columns:minmax(140px,auto) 1fr;
gap:9px 18px;align-items:baseline;}
.x0cap dt{font-weight:700;color:#5A3E8C;}
.x0cap dd{margin:0;}
.x0cap code{font:12px/1.4 "DejaVu Sans Mono",monospace;background:#F6F3FE;
border-radius:4px;padding:1px 4px;}
@media (prefers-color-scheme:dark){
.x0cap{color:#B9C3CE;} .x0cap h2{color:#77828E;} .x0cap dt{color:#C79BF2;}
.x0cap code{background:#1E1830;}}
@media (max-width:720px){.x0cap dl{grid-template-columns:1fr;gap:2px 0;}
.x0cap dd{margin:0 0 12px;}}
</style>
"""


def report_json(run_dir):
    """`metrics_report.json` for a run, or None. The captions' only data source.

    A pair graded in a variant run of the same layer is folded in, so a page says what
    was measured for its own layer. Without this, every layer page read "B3->B4: not
    graded at all" while the paper figures printed layer 12's numbers for it. The merge
    checks that the two runs agree before combining them; see
    `make_report_figures._merge_variant_pairs`.
    """
    p = Path(run_dir) / "metrics_report.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text())
    try:
        from .make_report_figures import _merge_variant_pairs
    except ImportError:
        return rep
    return _merge_variant_pairs(Path(run_dir), rep)


def second_json(run_dir):
    """`second_pass.json` for a run, or None -- stage 03 has not run everywhere."""
    p = Path(run_dir) / "second_pass.json"
    return json.loads(p.read_text()) if p.exists() else None


def caption_block(items) -> str:
    """The descriptive captions that sit under a figure, as real HTML.

    Not plotly annotations. An annotation lives in the figure's coordinate space,
    so it has to be positioned by hand per panel, it does not reflow when the
    window narrows, and its text cannot be selected or found by the browser's
    search. These pages are read, not just looked at, so the captions are a
    normal <dl> under the plot.

    `items` is a sequence of (term, description). A term is the panel as it is
    labelled in the figure, so a reader can match caption to panel by name rather
    than by counting; the description says what the panel *is* -- what one mark
    represents, what the axes carry, what a threshold line is set to. It does not
    say which value is larger than which. That is the reader's job, and a
    generated sentence claiming it would be a finding nobody checked.

    Any number inside a description must be passed in by the caller from the data
    or from `config`, never typed here.

    **The one thing here that can go stale, and what stops it.** Numbers cannot:
    they are read at generation time, so a number that disagrees with the data is
    not reachable. Sentences can. Change what a panel plots -- swap an axis, add
    a sixth filter, reorder the grid -- and the numbers follow the new data while
    the prose goes on describing the panel that used to be there. Nothing about
    deriving a number fixes that, because what went wrong is a claim about
    meaning, and no generator holds one.

    This repo has had exactly that failure twice already in prose no test could
    reach: figure titles quoting withdrawn layer-6 results, and "14/14 across
    seeds 0-7" surviving in seven files after the seed sweep it described had
    stopped existing. Both were found by a person reading, long after.

    So `tests/test_captions_match_panels.py` pins each dashboard's panel
    structure. It does not check that a caption is true -- nothing can -- it
    fails when the panels move, which sends whoever moved them back here. Treat a
    failure as a request to re-read, not as a hash to refresh.
    """
    items = [(t, d) for t, d in items if d]
    if not items:
        return ""
    rows = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items)
    return (CAPTION_CSS + '<section class="x0cap"><h2>What each panel shows</h2>'
            f"<dl>{rows}</dl></section>")


def write_page(fig, path, up=None, captions=None):
    """Write a plotly figure with the site nav bar on top.

    Every page is reachable only by a deep link, so each carries its own nav
    (`config.nav_html`). Which layer, which page and how far the site root is are
    all read off the output path, so callers never have to repeat what the path
    already says — `up` remains only for a caller that must override, and the day
    results were grouped by source every hardcoded `up=2` in here was wrong.

    Injected after write_html because plotly gives no layout slot for it — the
    same slot also carries the <script> tag for the shared plotly bundle that
    `include_plotlyjs=False` leaves out.

    `captions` is the (term, description) sequence documented on `caption_block`,
    rendered under the plot. A page called without them keeps working and simply
    has none, so a new page kind is never blocked on prose being written for it.
    """
    path = Path(path)
    fig.write_html(str(path), include_plotlyjs=False)
    layer, page = _page_identity(path)
    head = (
        f"{C.nav_html(depth=up or C.page_depth(path), layer=layer, page=page, current=_site_path(path))}\n"
        "<script>window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n"
        f'<script charset="utf-8" src="{plotly_asset(path.parent)}"></script>'
    )
    html = path.read_text().replace("<body>", "<body>\n" + head, 1)
    path.write_text(html)
    if captions:
        inject_captions(path, captions)


CAP_RE = re.compile(r"\n?<style>\n\.x0cap.*?</section>", re.S)


def inject_captions(path, items) -> bool:
    """Put the caption block into an already-written page, replacing any previous one.

    Separate from `write_page` because the caption does not depend on the plot.
    The plots on this site are drawn from `exp0_stats.pt`, which is far too large
    for git, so most pages cannot be redrawn from a clone -- but every number in
    their captions is in the committed JSON. Keeping the injection standalone is
    what lets `--captions` caption the whole site from a clone instead of only
    the layers whose cache happens to be present.

    Idempotent: the previous block is matched and replaced, so re-running does
    not stack two copies.
    """
    path = Path(path)
    html = path.read_text()
    block = caption_block(items)
    if not block:
        return False
    html, n = CAP_RE.subn("\n" + block, html, count=1)
    if not n:
        if "</body>" not in html:
            return False
        html = html.replace("</body>", block + "\n</body>", 1)
    path.write_text(html)
    return True


def _page_identity(path):
    """(layer, page-file) for a written page, read off its own path.

    A page under layer_NN/ describes that layer; anything else is site-wide and
    gets no Page row. It can still keep the page KIND, though: from a run
    directory's dashboard the layer pills should land on each gemma layer's
    dashboard, not on its index. `page` only names one of the per-layer
    kinds -- the calibration pages are not among them, and marking one would
    point every pill at a file no layer has.
    """
    name = Path(path).name
    kind = name if name in {f for f, _ in C.NAV_PAGES} else None
    parent = Path(path).parent.name
    if parent.startswith("layer_"):
        try:
            return int(parent.split("_")[1]), kind
        except ValueError:
            pass
    return None, kind


def _site_path(path):
    """This page's path from the site root, for highlighting the global row.

    The page's position in the OUTPUT TREE, not on disk. Every page has one,
    including a layer page: the source entry it sits under lights up from it, so
    the bar answers "which source am I in" as well as "which layer". Deriving it
    from OUT_DIR rather than the repo root keeps a scratch run under EXP0_OUT
    identical to the published page: where the file was written is not what the
    page IS.
    """
    return "outputs/" + Path(os.path.relpath(Path(path).resolve(), C.OUT_DIR)).as_posix()


def scope_subtitle(text):
    """Context line for a page that is NOT tied to one layer.

    Multi-layer and layer-independent pages need the same up-front statement of
    what they cover, so a reader never has to guess the scope from the URL.
    """
    return f"<b>{text}</b>"


def _titled(name, desc, stats=None, subtitle=None):
    """Three-line page header, identical on every generated page.

        1. name      what page this is, so a reader arriving from a link knows
        2. desc      one line on what the figure shows
        3. subtitle  the scope: which layer(s), and the knobs behind the numbers
    """
    sub = subtitle if subtitle is not None else page_subtitle(stats)
    return (f"<span style='font-size:17px;color:{PURPLE}'><b>{name}</b></span>"
            f"<br><span style='font-size:12.5px'>{desc}</span>"
            f"<br><span style='font-size:11.5px;color:{INK}'>{sub}</span>")
INK = "#5A6B7B"
FONT = dict(family="DejaVu Sans Mono, Courier New, monospace", size=11, color=INK)


def binned_bar(values, color, name, rng=None, nbins=40, log_x=False):
    """Pre-bin an array with numpy and return a go.Bar, so millions of raw
    values are never embedded in the HTML (keeps the file small)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    x = np.log10(np.clip(values, 1, None)) if log_x else values
    counts, edges = np.histogram(x, bins=nbins, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if log_x:
        centers = 10 ** centers
    return go.Bar(x=centers, y=counts, marker_color=color, name=name,
                  legendgroup=name, showlegend=False, opacity=0.65)


def compute_pair(stats, p_blk, c_blk):
    """Everything the plots need for one block pair."""
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    ranges, sibling_blocks = source_structure(stats)
    p0, p1 = ranges[p_blk]
    c0, c1 = ranges[c_blk]
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]

    cofire = stats["cofire"][key].double()
    R, F = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                           cofire=cofire, min_joint=C.MIN_JOINT)
    n_edges = int(edge_mask.sum())

    # Filter funnel: how many edges survive each metric.
    recon = edge_reconstruction_condition(
        stats["err_sum_c"][c_blk].double(),
        stats["g_parent_sum"][key].double(),
        stats["g_child_sum"][c_blk].double(),
        C.RECON_REL_GAIN_MIN,
    )
    n_recon = int((recon["passes"] & edge_mask).sum())

    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"][key].double(),
        stats["fire_c_by_bucket"][c_blk].double(),
        edge_mask,
    )
    survival = fcov["survival"]
    surv_vals = survival[~torch.isnan(survival)]
    n_survive = int((survival >= C.FREQ_SURVIVAL_MIN).sum())

    outdeg = edge_mask.sum(dim=1)
    outdeg = outdeg[outdeg > 0].double()

    sib = {}
    if c_blk in sibling_blocks:
        sib = sibling_redundancy(edge_mask, stats["within_cofire"][c_blk].double(), fire_c)
    redundancy = [v["redundancy"] for v in sib.values()]

    superparents = find_superparents(
        edge_mask, fire_p, total, C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC
    )

    return {
        "key": key,
        # First global feature index of each endpoint's block, carried alongside
        # the tensors so every downstream plot labels features from the SAME
        # structure these numbers were computed with.
        "p0": p0,
        "c0": c0,
        "n_edges": n_edges,
        "n_recon": n_recon,
        "n_survive": n_survive,
        "surv_vals": surv_vals.numpy(),
        "outdeg": outdeg.numpy(),
        "redundancy": redundancy,
        "superparents": superparents,
        "R": R,
        "F": F,
        "cofire": cofire,
        "edge_mask": edge_mask,
        "recon_pass": recon["passes"],
        "survival": survival,
    }


def _wrap(text, width=28):
    """Soft-wrap a label onto <br> lines so long descriptions stay readable
    inside plotly hover boxes / sankey nodes."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def build_dashboard(pairs_data, labels=None, stats=None):
    labels = labels or {}
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Edges passing each metric (log scale) — baseline vs strict",
            "Share of candidate edges passing each metric (%)",
            "Out-degree CCDF: P(children per parent ≥ x)",
            "Frequency-survival distribution",
            "Sibling-redundancy distribution",
            "Superparents: child coverage x firing rate",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    stages = ["candidate", "improves recon", "survives freq", "PMI > 0", "pass S_res"]
    for i, pd_ in enumerate(pairs_data):
        # Cycle: the palette holds gemma's four pairs, and a dictionary with more
        # blocks has more pairs (the PCFG SAE's 8 blocks give 7). Repeating a
        # colour is survivable; running off the end of the list is not.
        col = PAIR_COLORS[i % len(PAIR_COLORS)]
        name = pd_["key"]
        n_pmi = pd_.get("n_pmi")
        n_sres = pd_.get("n_sres")
        # (1,1) count of edges passing each metric (independent counts, not strict
        # nesting; recon/freq are the lenient baseline, PMI/S_res the strict tests)
        fig.add_bar(
            x=stages,
            y=[pd_["n_edges"], pd_["n_recon"], pd_["n_survive"], n_pmi, n_sres],
            name=name, marker_color=col, legendgroup=name, row=1, col=1,
        )
        # (1,2) percentages — all metrics, so the strict S_res collapse is visible
        e = max(pd_["n_edges"], 1)
        pmi_pct = 100 * n_pmi / e if n_pmi is not None else None
        sres_pct = 100 * n_sres / e if n_sres is not None else None
        fig.add_bar(
            x=["improves recon %", "survives freq %", "PMI > 0 %", "pass S_res %"],
            y=[100 * pd_["n_recon"] / e, 100 * pd_["n_survive"] / e, pmi_pct, sres_pct],
            name=name, marker_color=col, legendgroup=name, showlegend=False, row=1, col=2,
        )
        # (2,1) out-degree CCDF: P(outdeg >= x) — a heavy right tail = superparents
        od = torch.tensor(pd_["outdeg"]).sort(descending=True).values
        if od.numel():
            n = od.numel()
            ccdf = (torch.arange(1, n + 1).double() / n)
            fig.add_trace(
                go.Scatter(x=od.numpy(), y=ccdf.numpy(), mode="lines",
                           line=dict(color=col, width=2, shape="hv"),
                           name=name, legendgroup=name, showlegend=False),
                row=2, col=1,
            )
            fig.update_xaxes(type="log", row=2, col=1)
            fig.update_yaxes(type="log", row=2, col=1)
        # (2,2) frequency survival
        b = binned_bar(pd_["surv_vals"], col, name, rng=(0.0, 1.5), nbins=40)
        if b:
            fig.add_trace(b, row=2, col=2)
        # (3,1) sibling redundancy
        b = binned_bar(pd_["redundancy"], col, name, rng=(0.0, 1.0), nbins=30)
        if b:
            fig.add_trace(b, row=3, col=1)
        # (3,2) superparents
        sp = pd_["superparents"]
        if sp:
            p_base = pd_["p0"]
            fig.add_trace(
                go.Scatter(
                    x=[s["outdeg_frac"] for s in sp],
                    y=[s["fire_frac"] for s in sp],
                    mode="markers", marker=dict(color=col, size=12, line=dict(width=1, color="white")),
                    text=[f"feature {p_base + s['parent_local']}<br>"
                          f"{_wrap(C.feature_label(p_base + s['parent_local'], labels))}"
                          for s in sp],
                    hovertemplate="%{text}<br>child coverage %{x:.2f}<br>firing rate %{y:.2f}<extra></extra>",
                    name=name, legendgroup=name, showlegend=False,
                ),
                row=3, col=2,
            )

    fig.add_vline(x=C.FREQ_SURVIVAL_MIN, line=dict(color="#EF4444", width=1, dash="dash"), row=2, col=2)
    fig.add_vline(x=C.SIBLING_REDUNDANCY_FLAG, line=dict(color="#EF4444", width=1, dash="dash"), row=3, col=1)

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)  # number of parents
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_xaxes(type="log", title_text="children per parent (log)", row=2, col=1)
    fig.update_xaxes(title_text="survival", row=2, col=2)
    fig.update_xaxes(title_text="redundancy", row=3, col=1)
    fig.update_xaxes(title_text="child coverage (fraction of block)", row=3, col=2)
    fig.update_yaxes(title_text="firing rate", row=3, col=2)

    fig.update_layout(
        # Title and legend both live above the plot area: pin them to separate
        # paper-space rows (title on top, legend under it) or they overlap.
        title=dict(text=_titled(
                       "Metrics dashboard",
                       "Grading coverage edges with the metric battery "
                       f"({pairs_data[0]['n_edges']:,} to {pairs_data[-1]['n_edges']:,} candidate edges "
                       "across the block pairs)",
                       stats),
                   x=0.01, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=14, color=INK)),
        barmode="group", bargap=0.15, font=FONT,
        paper_bgcolor="white", plot_bgcolor="#FbFcFd",
        width=1200, height=1400, margin=dict(l=60, r=40, t=200, b=50),
        legend=dict(orientation="h", y=1.012, yanchor="bottom", x=0.01, xanchor="left"),
    )
    return fig


# ---------------------------------------------------------------------------
# Captions. One builder per page kind, kept together so the prose has one home
# and a reader auditing it does not have to walk six figure builders.
#
# Two rules, both of which the module docstring's "nothing hardcoded" already
# implies and which are easy to break in prose:
#
#   1. Every NUMBER comes from the data or from `config`. A threshold typed into
#      a sentence is a second copy of a constant, and it is the copy that goes
#      stale when the constant moves.
#   2. Captions DESCRIBE, they do not read. "one marker per flagged parent, x is
#      the fraction of the child block it covers" is a description; "most
#      superparents fire on more tokens than genuine parents" is a finding, and
#      a generated finding is one nobody checked.
# ---------------------------------------------------------------------------
def captions_dashboard(report, second=None):
    """Six panels of the per-layer metrics dashboard.

    Reads `metrics_report.json`, never the `.pt` cache -- even though the figure
    itself is drawn from the cache. The caches are not in git (they are hundreds
    of megabytes), so a builder that needed one could only caption the layers
    whose cache happened to be on the machine, and the published site would carry
    captions on some pages and not others with nothing saying why. Every count
    below is already in the committed report, so a bare clone can caption every
    page. `--captions` relies on exactly that.
    """
    pairs = report["pairs"]

    def tot(section, field):
        # A pair with too few scored parents writes `null` for a whole section
        # rather than a zeroed one, so this sums what exists instead of assuming
        # every pair reports every metric.
        return sum((p.get(section) or {}).get(field) or 0 for p in pairs)

    n_par = tot("degree", "n_parents_with_children")
    n_sp = sum(p.get("n_superparents") or 0 for p in pairs)
    n_red = tot("sibling_redundancy", "n_parents_scored")
    n_surv = tot("freq_control", "n_testable")
    keys = ", ".join(p["pair"] for p in pairs)
    scored = [k for k in (p["pair"] for p in pairs) if (second or {}).get(k, {}).get("sres")]
    strict = ("Both strict stages are present for " + ", ".join(scored) + "."
              if scored else
              "The two strict stages are blank here: they come from stage 03 "
              "(<code>second_pass.json</code>), which this run does not have.")
    return [
        ("Edges passing each metric",
         "Number of candidate parent&rarr;child edges left by each test, one bar group per "
         f"block pair ({keys}), on a logarithmic y-axis. The five stages are applied "
         "<b>independently to the same candidate set</b> and do not nest, so the bars are not "
         "a funnel: <code>improves recon</code> and <code>survives freq</code> are the lenient "
         f"filters, <code>PMI &gt; 0</code> and <code>pass S_res</code> the strict ones. {strict}"),
        ("Share passing each metric",
         "The same four tests as a percentage of each pair's own candidate set. Pairs differ in "
         "size by orders of magnitude, so a share is what places them on one axis; a count "
         "would not."),
        ("Out-degree CCDF",
         f"One curve per block pair over {n_par:,} parents in total. A point reads "
         "<i>P(a parent has at least x children)</i>, both axes logarithmic. The unit is a "
         "parent, not an edge."),
        ("Frequency-survival distribution",
         f"One value per testable candidate edge ({n_surv:,} in total): its reverse coverage "
         "computed on low- and mid-frequency tokens only, divided by its reverse coverage on "
         f"all tokens. The dashed line is <code>FREQ_SURVIVAL_MIN = {C.FREQ_SURVIVAL_MIN}</code>, "
         "the cutoff below which an edge is recorded as frequency-driven. Bars are counts of "
         "edges."),
        ("Sibling-redundancy distribution",
         "Mean pairwise Jaccard overlap between the children of one parent, one value per "
         f"scored parent ({n_red:,} in total). The dashed line is "
         f"<code>SIBLING_REDUNDANCY_FLAG = {C.SIBLING_REDUNDANCY_FLAG}</code>. Computed over "
         "whole-corpus co-firing, so it is confounded by how often each child fires at all; "
         "the parent-conditioned version is in <code>second_pass.json</code>."),
        ("Superparents",
         f"One marker per parent flagged as a superparent ({n_sp} across all block pairs). "
         "<i>x</i> is the fraction of the child block that parent has edges to, <i>y</i> is the "
         "fraction of tokens it fires on. The flag is out-degree alone, at "
         f"<code>SUPERPARENT_OUTDEG_FRAC = {100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}%</code> of the "
         "child block; the firing rate is plotted, not gated on. Hover gives the feature index "
         "and its label."),
    ]


def captions_sankey(report, top_n=None):
    """The stacked superparent Sankey page. JSON-only, for the reason above."""
    top_n = SANKEY_TOP_N if top_n is None else top_n
    pairs = report["pairs"]
    graded = [p["pair"] for p in pairs]
    named = [p["pair"] for p in pairs if p.get("n_superparents")]
    empty = [p["pair"] for p in pairs if not p.get("n_superparents")]
    # Adjacent pairs the block structure allows, minus the ones actually graded.
    # A pair can be missing for two different reasons and a reader cannot tell
    # them apart from an absent diagram: graded and found nothing, or never
    # computed at all. B3->B4 is the second kind -- a 6144 x 24576 co-firing
    # matrix, off unless EXP0_B3B4=1 -- and calling both "no diagram" hides that.
    nb = len(report.get("block_ranges") or [])
    absent = [f"{i}->{i + 1}" for i in range(max(nb - 1, 0))
              if f"{i}->{i + 1}" not in graded]

    def blocks(keys):
        return ", ".join("B" + k.replace("->", "&rarr;B") for k in keys)

    why = ""
    if empty:
        why += (f"{blocks(empty)} {'is' if len(empty) == 1 else 'are'} graded but no parent "
                f"there reaches the flag, so {'it gets' if len(empty) == 1 else 'they get'} no "
                "diagram. The gate is fan-out alone, at <code>SUPERPARENT_OUTDEG_FRAC = "
                f"{100 * C.SUPERPARENT_OUTDEG_FRAC:.0f}%</code> of the child block.")
    if absent:
        why += (f" {blocks(absent)} is a different case: not graded at this layer, so absent "
                "rather than empty. It is the largest co-firing matrix in the run and is off "
                "unless <code>EXP0_B3B4=1</code>, which needs a card with room for it and a "
                "stage 01 re-run, not just a redraw. It has been graded once, at layer 12.")
    return [
        ("Each diagram",
         f"One block pair that has a flagged superparent: {blocks(named) or 'none'} &mdash; "
         f"{len(named)} of the {len(pairs)} pairs graded in this run. The left node is the "
         f"parent feature; the right nodes are up to {top_n} of its children, ordered by the "
         "strength of the link."),
        ("Pairs with no diagram", why.strip() or None),
        ("Ribbon width",
         "The reverse coverage of that one parent&rarr;child edge &mdash; "
         "<i>P(parent fires | child fires)</i>, the quantity the edge criterion "
         f"(<code>&ge; {C.EDGE_TAU}</code>) is applied to. Width is per edge; it is not a share "
         "of the parent."),
        ("Ribbon colour",
         "The verdict on that edge. Where stage 03 has run the colour is the probe-based "
         "<code>S_res</code> rank test (both decoders inside the top "
         f"<code>k = {C.SRES_RANK_TOP_K}</code> of the child probe's correlations); where it has "
         "not, it falls back to the reconstruction and frequency filters. The legend above the "
         "first diagram names whichever is in force."),
        ("Node labels",
         "The feature index, plus its Neuronpedia label when "
         "<code>feature_labels.json</code> is present. Bare indices mean the labels were not "
         "fetched for this run, not that the features are unlabelled."),
    ]


def captions_in_block(report):
    """The in-block (same-level) dashboard."""
    blocks = report["blocks"]
    named = ", ".join(f"B{b['block']} ({b['n_features']:,} features)" for b in blocks)
    n_sp = sum(len(b["superparents"]) for b in blocks)
    return [
        ("Scope",
         "Relations <b>inside</b> a single Matryoshka block, where no block ordering fixes the "
         "direction, so it is derived from coverage asymmetry instead. Blocks shown: "
         f"{named}."),
        ("Counts per block",
         "Four independent counts per block on a logarithmic axis. <code>directed edges</code> "
         "is pairs where coverage is asymmetric enough to call one the parent; "
         "<code>survive PMI &gt; 0</code> and <code>pass S_res</code> are the strict tests "
         "applied to those edges; <code>duplicate pairs</code> is pairs whose firing sets are "
         "co-extensive, which are recorded as duplicates and never as edges."),
        ("Comparing blocks",
         "These are raw counts, and the blocks differ in size, so the number of pairs available "
         "to be counted differs with them &mdash; a block of <i>n</i> features offers "
         "<i>n</i>(<i>n</i>&minus;1) ordered pairs. Divide by that before reading one block "
         "against another."),
        ("Superparents",
         f"One bubble per in-block superparent ({n_sp} in total). <i>x</i> is the fraction of "
         "its own block it has edges to, <i>y</i> is the fraction of tokens it fires on, and "
         "the bubble area grows with its out-degree. Hover gives the global feature index and "
         "its child count."),
    ]


def captions_qualitative(report):
    """The per-layer qualitative check: one table per block pair."""
    n = sum(len(v) for v in report.values())
    cats = sorted({r["category"] for v in report.values() for r in v})
    return [
        ("Each table",
         f"One block pair; {n} edges in total across {len(report)} "
         f"{'table' if len(report) == 1 else 'tables'}. Rows are grouped by verdict "
         f"({', '.join(cats)}) and then by descending reverse coverage."),
        ("Columns",
         "<code>R</code> reverse coverage, <i>P(parent | child)</i> &mdash; the quantity the "
         f"edge criterion <code>&ge; {C.EDGE_TAU}</code> tests. <code>F</code> forward coverage, "
         "<i>P(child | parent)</i>, reported only. <code>gain</code> the relative reconstruction "
         "error increase when the parent is ablated on the child's firing tokens, with "
         f"<code>recon</code> its verdict at <code>&ge; {C.RECON_REL_GAIN_MIN}</code>. "
         "<code>surv</code> the frequency-survival ratio, cut at "
         f"<code>{C.FREQ_SURVIVAL_MIN}</code>."),
        ("The two label columns",
         "Neuronpedia's autointerp descriptions of each endpoint. They are model-generated, so "
         "they are what this page is judged <i>against</i>, not a ground truth &mdash; this is "
         "the one tier with no known answer. Labels are truncated for width; the full text and "
         "the clickable links are in <code>qualitative_check.md</code>."),
    ]


def captions_calibration(data):
    """Tier 1, the synthetic-toy scorecard."""
    rows = data["rows"]
    ratio = [r for r in rows if r.get("margin_kind") != "categorical"]
    cat = [r for r in rows if r.get("margin_kind") == "categorical"]
    ctrl = [r for r in rows if r["metric"].lstrip().startswith("—")]
    return [
        ("Scorecard (top)",
         f"All {len(rows)} rows, each a claim one metric makes about a world whose tree is "
         "known by construction. <code>margin</code> is how far apart the metric put the class "
         "it must keep from the class it must reject; <code>detail</code> carries the two "
         "underlying values."),
        ("Separation margin (left)",
         f"The {len(ratio)} rows whose two classes separate by a <i>ratio</i>, on a logarithmic "
         "axis. Bars are clipped for display at 10&#8308;&times;; the text on each bar is the "
         "unclipped value."),
        ("Categorically scored (right)",
         f"The {len(cat)} rows whose answer is not a ratio &mdash; a recovered edge set is right "
         "or it is not. They are kept off the ratio axis because a categorical margin of 1.0 "
         "means <i>correct</i>, which on a log ratio axis would be indistinguishable from "
         "<i>no separation at all</i>."),
        ("Hatched bars",
         f"The {len(ctrl)} negative controls. They pass when the battery does <b>not</b> act, so "
         "they record a demonstrated blind spot rather than a catch."),
        ("Per-metric panels",
         f"One panel for each of the {len(data['panels'])} metrics that has two classes to "
         "separate, showing the individual values behind that row's margin rather than the "
         "summary."),
    ]


def captions_trained_calibration(d, align=None):
    """Tier 2, the trained-toy page."""
    pt = d.get("per_token") or {}
    edges = [e for e in pt.get("edges", []) if e.get("testable")]
    cof = pt.get("child_cofire") or {}
    items = [
        ("Scorecard (top)",
         "One row per check, in the same six columns as Tier 1. The difference from Tier 1 is "
         "the input: every number here is computed from the latents a Matryoshka SAE "
         f"<b>learned</b> on this tree, not from constructed statistics. The SAE recovered "
         f"{d['n_recovered_features']} of {d['n_features']} true features, which is the ceiling "
         "on anything below."),
        ("Probe S_res on learned latents",
         f"One bar per testable true edge ({len(edges)} of "
         f"{len(d['true_edges'])}): where the true parent's decoder ranks among the child "
         "probe's correlations over the whole dictionary. Rank 0 is the closest. The rule "
         f"accepts an edge when the rank is inside the top <code>k = {C.SRES_RANK_TOP_K}</code>; "
         f"an unrelated feature sits at chance in a dictionary of "
         f"{d['cfg']['d_sae']} latents." if edges else None),
        ("Children the tree keeps apart",
         "One bar per parent whose children the grammar declares mutually exclusive: how many "
         "sampled tokens the SAE's recovered latents co-fire on"
         + (f", out of {pt['n_draws']:,} draws" if pt.get("n_draws") else "")
         + ". The ground truth for every one of these is zero, so the bar is the SAE's own "
         "departure from the tree, not a metric's."
         if cof else None),
        ("Edge recovery",
         f"All {len(d['true_edges'])} true edges with the verdict on each. A miss is attributed: "
         "<i>child never learned</i> means the SAE has no latent for that endpoint, so the edge "
         "was unreachable for any metric."),
    ]
    if align:
        items.append((
            "Nesting control",
            f"A separate question from everything above: not whether the metrics work, but "
            "whether the Matryoshka nesting itself places a parent in an earlier block than its "
            f"children. {align['n_testable']} of {len(d['true_edges'])} true edges are testable "
            f"&mdash; the rest have an endpoint the SAE never learned &mdash; over "
            f"{len(align['block_ranges'])} blocks."))
    return items


def _superparent_sankey_trace(stats, pd_, p_blk, c_blk, top_n=25, feat_labels=None):
    """Build the Sankey trace + title for one block pair's top superparent.

    Returns (trace, title) or None when the pair has no superparent.
    Shared by the single-pair figure and the stacked all-pairs figure.
    """
    feat_labels = feat_labels or {}
    sp = pd_["superparents"]
    if not sp:
        return None
    parent_local = sp[0]["parent_local"]
    p0, c0 = pd_["p0"], pd_["c0"]
    gp = p0 + parent_local

    kids = torch.nonzero(pd_["edge_mask"][parent_local]).flatten()
    cof = pd_["cofire"][parent_local, kids]
    order = torch.argsort(cof, descending=True)[:top_n]
    kids = kids[order]

    labels = [f"B{p_blk}:{gp} {C.feature_label(gp, feat_labels)} "
              f"(fires {100 * sp[0]['fire_frac']:.0f}%)"]
    node_colors = [PURPLE]
    source, target, value, link_colors = [], [], [], []
    for i, ci in enumerate(kids.tolist(), start=1):
        gc = c0 + ci
        if "sres_pass" in pd_:                        # second-pass refinement verdict
            real = bool(pd_["sres_pass"][parent_local, ci])
        else:                                         # fallback: weak baseline + freq
            passes = bool(pd_["recon_pass"][parent_local, ci])
            s = pd_["survival"][parent_local, ci]
            real = passes and (not torch.isnan(s)) and float(s) >= C.FREQ_SURVIVAL_MIN
        labels.append(f"B{c_blk}:{gc} {C.feature_label(gc, feat_labels)}")
        node_colors.append(GREEN if real else GREY)
        link_colors.append("rgba(34,197,94,0.55)" if real else "rgba(203,213,225,0.5)")
        source.append(0)
        target.append(i)
        value.append(float(pd_["cofire"][parent_local, ci]))

    trace = go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=node_colors, pad=18, thickness=14, line=dict(width=0)),
        link=dict(source=source, target=target, value=value, color=link_colors,
                  hovertemplate="%{source.label} -> %{target.label}<br>"
                                "co-fire: %{value}<extra></extra>"),
    )
    n_real = sum(1 for c in node_colors[1:] if c == GREEN)
    # Rendered as a panel heading: bold, and the survivor count coloured by
    # whether anything survived at all (green if some, red if none).
    count_color = GREEN_DARK if n_real else RED
    title = (f"<b>B{p_blk} → B{c_blk}</b>"
             f"<span style='color:{INK}'>　 superparent </span>"
             f"<b>B{p_blk}:{gp}</b>"
             f"<span style='color:{INK}'> → top {len(kids)} children 　 pass S_res (genuine refinement): </span>"
             f"<b><span style='color:{count_color}'>{n_real}/{len(kids)}</span></b>")
    return trace, title


def build_superparent_sankey(stats, pd_, p_blk, c_blk, top_n=25, feat_labels=None):
    """One superparent's flow to its top_n children; colour separates edges that
    survive (reconstruction AND frequency) from frequency-captured ones."""
    built = _superparent_sankey_trace(stats, pd_, p_blk, c_blk, top_n, feat_labels)
    if built is None:
        return None
    trace, title = built
    fig = go.Figure(trace)
    _add_sres_legend(fig)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=13, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white", showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.01),
        width=1150, height=680, margin=dict(l=50, r=90, t=90, b=40),
    )
    return fig


def _add_sres_legend(fig):
    """Two off-canvas markers so the green/grey link colouring gets a real legend
    box (Sankey links themselves don't populate the legend)."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=GREEN),
                             name="pass S_res (genuine refinement)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=12, color=GREY),
                             name="fail S_res (co-fires only)"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


SANKEY_TOP_N = 25   # children drawn per superparent; named so the caption cannot drift


def build_all_superparent_sankeys(stats, pairs, pairs_data, top_n=SANKEY_TOP_N,
                                  feat_labels=None):
    """Every block pair's top superparent, stacked in ONE figure.

    Stacking keeps a single embedded plotly.js bundle, so the file stays the
    size of one dashboard instead of one per pair.
    """
    built = []
    for (p, c), pd_ in zip(pairs, pairs_data):
        got = _superparent_sankey_trace(stats, pd_, p, c, top_n, feat_labels)
        if got is not None:
            built.append((p, c, *got))
    if not built:
        return None

    n = len(built)
    panel_h = 660                      # px per panel
    title_gap = 0.055                  # fraction of figure height reserved per heading
    fig = go.Figure()
    annotations = []
    for i, (p, c, trace, title) in enumerate(built):
        top = 1.0 - i / n
        bottom = 1.0 - (i + 1) / n
        trace.domain = dict(x=[0, 1], y=[bottom, top - title_gap])
        fig.add_trace(trace)
        annotations.append(dict(
            text=title, x=0.0, xref="paper", xanchor="left",
            y=top - 0.006, yref="paper", yanchor="top",
            showarrow=False, align="left",
            font=dict(size=14, color=PURPLE),
            bgcolor="#F6F3FE", bordercolor="#E3DAFB", borderwidth=1, borderpad=8,
        ))

    _add_sres_legend(fig)
    fig.update_layout(
        title=dict(text=_titled(
                       "Superparent fan-out",
                       "One high-firing feature adopting most of the next block, "
                       "shown for every block pair (green = the edge passes S_res, "
                       "the genuine-refinement test)",
                       stats),
                   x=0.005, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=14, color=INK)),
        annotations=annotations, showlegend=True,
        legend=dict(orientation="h", y=1.0, x=0.30, yref="container"),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        width=1250, height=160 + panel_h * n, margin=dict(l=50, r=90, t=150, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Calibration dashboard (synthetic ground-truth toy from validation/)
# ---------------------------------------------------------------------------
def _calibration_data():
    """Run the metrics on the toy and split each metric's per-edge score
    into the class it should KEEP (genuine) vs the class it should REJECT."""
    from validation.calibrate_on_synthetic_toy import _render, _run_metrics, _score
    from validation.synthetic_toy_world import build_world

    stats, labels = build_world()
    m = _run_metrics(stats)
    rows = _score(stats, labels, m)

    edge_mask = m["edge_mask"]
    kept = {(int(p), int(c)) for p, c in torch.nonzero(edge_mask).tolist()}
    genuine = labels.genuine & kept
    sp_edges = [(p, c) for (p, c) in kept if p in labels.superparent_parents]
    freq_edges = [(p, c) for (p, c) in labels.freq_edges if (p, c) in kept]

    pgain = m["recon"]["parent_gain"]
    survival = m["fcov"]["survival"]
    outdeg = edge_mask.sum(dim=1)
    sib = m["sib"]
    genuine_parents = sorted({p for p, _ in genuine} - labels.superparent_parents)

    return {
        "rows": rows,
        "md": _render(rows),
        # (metric 2) reconstruction parent-gain, log scale
        "recon_keep": [float(pgain[p, c]) for p, c in genuine],
        "recon_reject": [float(pgain[p, c]) for p, c in sp_edges],
        "recon_thr": C.RECON_REL_GAIN_MIN,
        # (metric 5) frequency survival
        "freq_keep": [float(survival[p, c]) for p, c in genuine
                      if not torch.isnan(survival[p, c])],
        "freq_reject": [float(survival[p, c]) for p, c in freq_edges],
        "freq_thr": C.FREQ_SURVIVAL_MIN,
        # (metric 3) sibling redundancy
        "sib_keep": [sib[p]["redundancy"] for p in genuine_parents if p in sib],
        "sib_reject": [v["redundancy"] for p, v in sib.items() if p in labels.split_parents],
        "sib_thr": C.SIBLING_REDUNDANCY_FLAG,
        # (metric 4) out-degree
        "deg_keep": [int(outdeg[p]) for p in genuine_parents],
        "deg_reject": [int(outdeg[p]) for p in labels.superparent_parents],
        "deg_thr": C.SUPERPARENT_OUTDEG_FRAC * stats["C"],
        # Panels are built from whatever rows carry a `series`, plus the four
        # above that are computed here because their two classes are edge sets
        # rather than a per-row list. The page used to hardcode exactly those
        # four, so it showed 4 of 14 graded metrics and looked like the battery
        # was five metrics wide.
        "panels": [
            ("2a. reconstruction — parent gain (log)", data_recon := {
                "keep": [float(pgain[p, c]) for p, c in genuine],
                "reject": [float(pgain[p, c]) for p, c in sp_edges],
                "threshold": C.RECON_REL_GAIN_MIN, "log": True}),
            ("5. frequency control — survival", {
                "keep": [float(survival[p, c]) for p, c in genuine
                         if not torch.isnan(survival[p, c])],
                "reject": [float(survival[p, c]) for p, c in freq_edges],
                "threshold": C.FREQ_SURVIVAL_MIN}),
            ("3. sibling redundancy — global Jaccard", {
                "keep": [sib[p]["redundancy"] for p in genuine_parents if p in sib],
                "reject": [v["redundancy"] for p, v in sib.items()
                           if p in labels.split_parents],
                "threshold": C.SIBLING_REDUNDANCY_FLAG, "reject_above": True}),
            ("4. out-degree — children per parent", {
                "keep": [int(outdeg[p]) for p in genuine_parents],
                "reject": [int(outdeg[p]) for p in labels.superparent_parents],
                "threshold": C.SUPERPARENT_OUTDEG_FRAC * stats["C"],
                "reject_above": True}),
        ] + [(r["metric"] + " — " + r["series"]["name"], r["series"])
             for r in rows if "series" in r],
    }


def _strip(fig, row, col, keep, reject, log_y=False, rng=None):
    """Two jittered point clouds: green = should-keep, red = should-reject."""
    jit = np.random.default_rng(0)
    for vals, color, name, xc in ((keep, GREEN, "genuine (keep)", 0),
                                  (reject, RED, "pathology (reject)", 1)):
        if not vals:
            continue
        x = xc + jit.uniform(-0.12, 0.12, size=len(vals))
        fig.add_trace(
            go.Scatter(
                x=x, y=vals, mode="markers",
                marker=dict(color=color, size=11, line=dict(width=1, color="white"), opacity=0.85),
                name=name, legendgroup=name, showlegend=(row == 2 and col == 1),
                hovertemplate=f"{name}<br>%{{y:.4g}}<extra></extra>",
            ),
            row=row, col=col,
        )
    fig.update_xaxes(tickvals=[0, 1], ticktext=["genuine", "pathology"],
                     range=[-0.5, 1.5], row=row, col=col)
    if log_y:
        fig.update_yaxes(type="log", row=row, col=col)
    if rng:
        fig.update_yaxes(range=rng, row=row, col=col)


def build_calibration_dashboard(data):
    """Scorecard for every row, a margin overview, then one panel per metric that
    has two classes to separate.

    The margin overview is split in two, because two kinds of row live in this
    scorecard and only one belongs on a ratio axis. A row scored categorically
    (the recovered edge set is right or it is not) carries margin 1.0 meaning
    "correct", which on a log ratio axis is indistinguishable from "separated by
    a factor of one" -- that is, from no separation at all.
    """
    rows = data["rows"]
    ranked = sorted(rows, key=lambda r: (-int(r["pass"]), -r["margin"]))
    panels = data["panels"]
    ncol = 2
    prow = (len(panels) + ncol - 1) // ncol

    specs = ([[{"type": "table", "colspan": 2}, None],
              [{}, {}]] + [[{}, {}] for _ in range(prow)])
    titles = ["", "separation margin (log) — ratio-scored rows",
              "categorically scored — right or not"]
    titles += [t for t, _ in panels] + [""] * (prow * ncol - len(panels))
    fig = make_subplots(
        rows=2 + prow, cols=2, specs=specs, subplot_titles=titles,
        vertical_spacing=0.055, horizontal_spacing=0.12,
        row_heights=[0.30, 0.16] + [0.54 / prow] * prow,
    )

    fig.add_trace(
        go.Table(
            columnwidth=[36, 150, 200, 50, 55, 380],
            header=dict(
                values=["#", "metric", "job", "verdict", "margin", "detail"],
                fill_color="#EEF2F6", align="left",
                font=dict(color=INK, size=11), height=26,
            ),
            cells=dict(
                values=[
                    [i for i in range(1, len(ranked) + 1)],
                    [r["metric"] for r in ranked],
                    [r["job"] for r in ranked],
                    ["PASS" if r["pass"] else "FAIL" for r in ranked],
                    [">1000x" if r["margin"] >= 1000 else f"{r['margin']:.1f}x" for r in ranked],
                    [r["detail"] for r in ranked],
                ],
                align="left", height=42,
                font=dict(color=[["#166534" if r["pass"] else "#991B1B" for r in ranked]
                                 if j == 3 else INK for j in range(6)], size=10),
                fill_color=[["white"] * len(ranked) if j != 3 else
                            ["#DCFCE7" if r["pass"] else "#FEE2E2" for r in ranked]
                            for j in range(6)],
            ),
        ),
        row=1, col=1,
    )

    ratio = sorted((r for r in rows if r.get("margin_kind") != "categorical"),
                   key=lambda r: r["margin"])
    cat = [r for r in rows if r.get("margin_kind") == "categorical"]
    for col, group, logx in ((1, ratio, True), (2, cat, False)):
        if not group:
            continue
        fig.add_trace(
            go.Bar(
                x=[min(r["margin"], 1e4) if logx else 1 for r in group],
                y=[r["metric"] for r in group], orientation="h",
                marker=dict(color=[GREEN if r["pass"] else RED for r in group],
                            pattern=dict(shape=["/" if r["metric"].lstrip().startswith("—")
                                                else "" for r in group])),
                text=[(">1000x" if r["margin"] >= 1000 else f"{r['margin']:.1f}x") if logx
                      else ("correct" if r["pass"] else "wrong") for r in group],
                textposition="auto", hoverinfo="y+text", showlegend=False,
            ),
            row=2, col=col,
        )
        if logx:
            fig.update_xaxes(type="log", row=2, col=col, title_text="× separation")
        else:
            fig.update_xaxes(visible=False, row=2, col=col)
        fig.update_yaxes(tickfont=dict(size=9), row=2, col=col)

    for i, (_, ser) in enumerate(panels):
        r, c = 3 + i // ncol, 1 + i % ncol
        _strip(fig, r, c, ser.get("keep", []), ser.get("reject", []),
               log_y=bool(ser.get("log")))
        thr = ser.get("threshold")
        if thr is not None:
            fig.add_trace(
                go.Scatter(x=[-0.5, 1.5], y=[thr, thr], mode="lines", hoverinfo="skip",
                           line=dict(color=RED, width=1, dash="dash"), showlegend=False),
                row=r, col=c,
            )

    n_pass = sum(r["pass"] for r in rows)
    fig.update_layout(
        height=430 + 300 * prow, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=60, r=30, t=190, b=40), plot_bgcolor="white",
        title=dict(text=_titled(
            "Synthetic toy calibration",
            f"every metric scored against a known tree — <b>{n_pass}/{len(rows)}</b> rows, "
            f"covering all 21 metric functions; the two hatched rows are negative controls "
            f"that pass when the battery does <b>not</b> act",
            subtitle=scope_subtitle(
                "No layer: a hand-built world with a known 5-parent tree and six injected "
                "structures"),
        ), x=0.01, xanchor="left", font=dict(size=15)),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor="#EEF1F5", zeroline=False)
    return fig


def run_calibration():
    data = _calibration_data()
    fig = build_calibration_dashboard(data)
    out = C.OUT_DIR / "synthetic_toy_calibration.html"
    write_page(fig, out, captions=captions_calibration(data))   # lives in outputs/
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Tier 2 dashboard: metrics on a Matryoshka SAE actually trained on the toy.
# Mirrors the Tier-1 scorecard, but the question is edge recovery vs a known
# tree, not pathology separation.
# ---------------------------------------------------------------------------
CAL2_STYLE = {                       # (label, fill, ink)
    "recovered":                  ("recovered ✓",           "#DCFCE7", "#166534"),
    "missed: child not learned":  ("missed (SAE gap)",       "#FEF3C7", "#92400E"),
    "missed":                     ("missed",                 "#FEE2E2", "#991B1B"),
    "spurious":                   ("spurious (false pos)",   "#FEE2E2", "#991B1B"),
}
CAL2_ORDER = ["recovered", "missed: child not learned", "missed", "spurious"]


# A scorecard where every row says PASS is the same survivorship problem the error
# log has: it reads as reassurance. Some rows here are not tests at all -- feature
# recovery is a fact about the SAE, and feature splitting is a caveat on how
# generously the nesting check was read. Grading them would claim a verdict nobody
# measured, so they get a third state and are visibly not scored.
def _verdict_text(r):
    return "noted" if not r.get("graded", True) else ("PASS" if r["pass"] else "FAIL")


def _verdict_ink(r):
    return "#9AA7B3" if not r.get("graded", True) else ("#166534" if r["pass"] else "#991B1B")


def _verdict_fill(r):
    return "#F1F5F9" if not r.get("graded", True) else ("#DCFCE7" if r["pass"] else "#FEE2E2")


def trained_scorecard(d, align):
    """Tier 2's checks as scored rows, in the shape Tier 1 uses.

    A verdict alone is not calibration: a check that passed by a hair is not
    evidence, it is luck that has not run out. So every row carries the same
    `margin` idea as the Tier-1 scorecard — how far this check was from failing —
    and the margins here are small integers because the toy is small. That is the
    honest reading, not a defect of the display.

    The nesting rows come from block_tree_alignment.json when it exists.
    `calibrate_on_trained_toy` indexes by ground truth and never by block, on
    purpose: mixing them would confound "is the metric right?" with "did Matryoshka
    order the features right?". Those are two questions and they get two rows.
    """
    tp, fp2, fn = d["true_positives"], d["false_positives"], d["false_negatives"]
    F = d["n_features"]
    n_gap = sum(1 for r in d["edge_rows"] if "not learned" in r["category"])
    testable = (tp + fn) - n_gap

    rows = [
        {"check": "1. precision — no invented edges",
         "job": "keep nothing the tree does not contain",
         "pass": fp2 == 0,
         "value": f"{d['precision']:.2f}",
         "margin": f"{fp2} false positives",
         "detail": f"{tp} edges kept, {fp2} of them absent from the true tree"},
        {"check": "2. recall against the SAE's ceiling",
         "job": "recover every edge whose endpoints the SAE actually learned",
         "pass": tp == testable,
         "value": f"{tp}/{testable}",
         "margin": f"{testable - tp} testable misses",
         "detail": f"raw recall {d['recall']:.2f} ({tp}/{tp + fn}); {n_gap} of the "
                   f"{fn} misses are edges whose child the SAE never learned, so no "
                   f"metric could have found them"},
        {"check": "3. feature recovery (the ceiling itself)",
         "job": "state what the SAE gave the metrics to work with",
         "graded": False,
         "value": f"{d['n_recovered_features']}/{F}",
         "margin": "—",
         "detail": f"{F - d['n_recovered_features']} true features were never learned. "
                   f"This bounds recall from above: it is a fact about the SAE, so it is "
                   f"reported and not graded"},
    ]

    pt = d.get("per_token")
    if pt:
        red = pt.get("parent_conditioned_redundancy") or {}
        worst = max(red.items(), key=lambda kv: kv[1], default=(None, 0.0))
        clean = [k for k, v in red.items() if v < C.SIBLING_REDUNDANCY_FLAG]
        rows += [
            {"check": "3b. probe S_res on LEARNED latents",
             "job": "accept a parent the SAE had to find, not one we constructed",
             "pass": pt["n_pass"] == pt["n_testable"] and pt["n_testable"] > 0,
             "value": f"{pt['n_pass']}/{pt['n_testable']}",
             "margin": f"chance {pt['chance_pass_rate']:.0%}",
             "detail": "the true parent lands at rank 0 or 1 of the whole dictionary. "
                       f"With {int(5 / max(pt['chance_pass_rate'], 1e-9))} latents a top-5 "
                       f"rank rule passes an unrelated parent at k/D = "
                       f"{pt['chance_pass_rate']:.0%} by chance, so this tier shows a learned "
                       f"true parent is ACCEPTED and cannot show an unrelated one is rejected; "
                       "gemma's 32768 puts the same null at 0.015%. Untestable edges: "
                       + "; ".join(f"{r['edge']} ({r['why']})"
                                   for r in pt["edges"] if not r["testable"])},
            {"check": "3c. sibling redundancy inside each true parent",
             "job": "the tree says every parent's children are mutually exclusive",
             "pass": bool(worst[0]) and worst[1] >= C.SIBLING_REDUNDANCY_FLAG,
             "value": f"{worst[1]:.2f} vs {', '.join(f'{v:.2f}' for k, v in red.items() if k in clean)}",
             "margin": "found, not injected",
             "detail": f"parent {worst[0]} scores {worst[1]:.3f} where the other parents score "
                       f"{', '.join(f'{v:.3f}' for k, v in red.items() if k in clean)}. The tree "
                       "declares all of them `mutually_exclusive_children`, so this is a real "
                       "conflation the SAE introduced — a defect nobody injected, which the "
                       "synthetic tier structurally cannot produce. Edge recovery calls both of "
                       "that parent's edges recovered and precision stays 1.00, so this metric "
                       "is adding a column coverage and reconstruction do not have"},
        ]

    if align:
        gaps = [r["child_block"] - r["parent_block"]
                for r in align["edge_rows"] if r["child_block"] is not None]
        n_t, n_ok = align["n_testable"], align["n_respected"]
        splits = align.get("split_features") or {}
        rows += [
            {"check": "4. Matryoshka nesting respects the tree",
             "job": "parent lands in an earlier block than its children",
             "pass": n_ok == n_t and n_t > 0,
             "value": f"{n_ok}/{n_t}",
             "margin": f"min gap {min(gaps)} block" + ("" if min(gaps) == 1 else "s"),
             "detail": f"block gaps {sorted(gaps)}; mean parent block "
                       f"{align['mean_parent_block']:.1f} vs child "
                       f"{align['mean_child_block']:.1f}. The narrowest edge clears by "
                       f"{min(gaps)} — the claim holds, but not by much at its tightest"},
            {"check": "5. feature splitting (caveat on 4)",
             "job": "say how generously check 4 was read",
             "graded": False,
             "value": f"{len(splits)} split",
             "margin": "—",
             "detail": ("no true feature was recovered by more than one latent, so check 4 "
                        "had no choice to make" if not splits
                        else "recovered by >1 latent: "
                             + ", ".join(f"feature {k} in blocks {v}" for k, v in splits.items())
                             + ". Check 4 takes the EARLIEST block for each feature, which is "
                               "the reading most favourable to the architecture — a later "
                               "choice could turn a respected edge into a violation. Not a "
                               "metric failure, but it is why 6/6 is weaker than it looks")},
        ]
    return rows


def build_trained_calibration_dashboard(d, align=None):
    """Tier 2 in Tier 1's shape: a scored card, then the evidence behind it.

    Not the layer dashboard's shape. That one plots distributions over millions of
    candidate edges, where a CCDF means something. Here there are nine true edges and
    twenty features — a distribution over nine points is decoration. The question is
    also different in kind: the layer pages ask what happened to a population, both
    calibration tiers ask whether each check did its job where the answer is known.
    """
    sc = trained_scorecard(d, align)
    rows = sorted(d["edge_rows"], key=lambda r: (CAL2_ORDER.index(r["category"]),
                                                 r["parent"], r["child"]))
    have_align = bool(align)

    # Row 2 is this tier's own question -- do the metrics work on LEARNED
    # features -- and rows 3-4 are the nesting control, which asks about the
    # architecture instead. The page used to give the control three panels and
    # the calibration one, so the two results added on 7 August (S_res on learned
    # latents, and the conflation the sibling metric found) lived only in table
    # cells while three charts described something else.
    pt = d.get("per_token") or {}
    # The nesting control owns row 3's right half and all of row 4. When its file is
    # absent or stale those panels are not drawn, and a fixed four-row grid then
    # printed three titles over empty space and left a blank band at the bottom --
    # the layout claiming results the page does not have. So the grid is built to
    # fit what will actually be plotted: the edge-recovery table widens to the full
    # row and the figure ends there.
    if have_align:
        specs = [[{"type": "table", "colspan": 2}, None],
                 [{}, {}],
                 [{"type": "table"}, {}],
                 [{"type": "table"}, {}]]
        titles = ("", "Probe S_res on LEARNED latents: where the true parent ranks",
                  "Children the tree keeps apart, and what the SAE did with them",
                  "Edge recovery vs the known tree",
                  "Matryoshka nesting: parent block vs child block",
                  "Nesting, edge by edge", "Where the tree landed in the blocks")
        row_heights = [0.28, 0.24, 0.24, 0.24]
    else:
        specs = [[{"type": "table", "colspan": 2}, None],
                 [{}, {}],
                 [{"type": "table", "colspan": 2}, None]]
        titles = ("", "Probe S_res on LEARNED latents: where the true parent ranks",
                  "Children the tree keeps apart, and what the SAE did with them",
                  "Edge recovery vs the known tree", "")
        row_heights = [0.34, 0.30, 0.36]

    fig = make_subplots(
        rows=len(row_heights), cols=2, specs=specs, subplot_titles=titles,
        vertical_spacing=0.06, horizontal_spacing=0.08, row_heights=row_heights,
    )

    # (row 2, left) rank of the true parent per edge, against the two lines that
    # decide it: the top-k cutoff, and D -- the floor an unrelated feature sits at.
    edges = [e for e in pt.get("edges", []) if e.get("testable")]
    if edges:
        names = [e["edge"] for e in edges]
        fig.add_trace(go.Bar(x=names, y=[e["parent_rank"] for e in edges],
                             marker_color=GREEN, name="true parent's rank",
                             text=[e["parent_rank"] for e in edges], textposition="outside"),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=names, y=[e["child_rank"] for e in edges],
                             marker_color=TEAL, name="child's rank",
                             text=[e["child_rank"] for e in edges], textposition="outside"),
                      row=2, col=1)
        D = int(round(5 / max(pt.get("chance_pass_rate", 0.25), 1e-9)))
        for y, col, lab in ((C.SRES_RANK_TOP_K - 0.5, RED, f"top-{C.SRES_RANK_TOP_K} cutoff"),
                            (D - 1, GREY, f"bottom of the dictionary (D = {D})")):
            fig.add_trace(go.Scatter(x=names, y=[y] * len(names), mode="lines",
                                     line=dict(color=col, width=1, dash="dash"),
                                     name=lab, hoverinfo="skip"), row=2, col=1)
        fig.update_yaxes(title_text="rank among all latents (0 = top)",
                         range=[-1.5, D], row=2, col=1)

    # (row 2, right) the fact under the redundancy number. The tree makes every
    # parent's children mutually exclusive, so the ground-truth column is zero by
    # construction and any bar on the right is the SAE's own conflation.
    cof = pt.get("child_cofire") or {}
    if cof:
        ps = sorted(cof, key=lambda k: -cof[k]["learned"])
        red = pt.get("parent_conditioned_redundancy") or {}
        fig.add_trace(go.Bar(x=[f"parent {k}" for k in ps],
                             y=[cof[k]["ground_truth"] for k in ps],
                             marker_color=GREY, name="co-firings the grammar allows"),
                      row=2, col=2)
        fig.add_trace(go.Bar(x=[f"parent {k}" for k in ps],
                             y=[cof[k]["learned"] for k in ps],
                             marker_color=[RED if cof[k]["learned"] else GREEN for k in ps],
                             name="co-firings the SAE produced",
                             text=[f"{cof[k]['learned']:,}<br>redundancy "
                                   f"{red.get(k, 0):.2f}" for k in ps],
                             textposition="outside"), row=2, col=2)
        fig.update_yaxes(title_text="co-firing tokens (200,000 draws)", row=2, col=2)

    # (1) the scorecard — same six columns as Tier 1, same verdict colouring
    fig.add_trace(
        go.Table(
            columnwidth=[36, 190, 210, 50, 92, 420],
            header=dict(values=["#", "check", "job", "verdict", "margin", "detail"],
                        fill_color="#EEF2F6", align="left",
                        font=dict(color=INK, size=11), height=26),
            cells=dict(
                values=[
                    list(range(1, len(sc) + 1)),
                    [r["check"] for r in sc],
                    [r["job"] for r in sc],
                    [_verdict_text(r) for r in sc],
                    [r["margin"] for r in sc],
                    [r["detail"] for r in sc],
                ],
                align="left", height=54,
                font=dict(size=10,
                          color=[[_verdict_ink(r) for r in sc] if j == 3 else INK
                                 for j in range(6)]),
                fill_color=[["white"] * len(sc) if j != 3 else
                            [_verdict_fill(r) for r in sc] for j in range(6)],
            ),
        ), row=1, col=1,
    )

    # (2,1) edge recovery, unchanged in content — the evidence for rows 1-3
    edge, verdict, note, vcol, fill = [], [], [], [], []
    for r in rows:
        label, tint, ink = CAL2_STYLE[r["category"]]
        edge.append(r["edge"])
        verdict.append(label)
        note.append("kept by all metrics, matches the tree" if r["category"] == "recovered"
                    else "child feature never learned by the SAE" if "not learned" in r["category"]
                    else "metric kept an edge not in the tree" if r["category"] == "spurious"
                    else "in the tree but not kept")
        vcol.append(ink)
        fill.append(tint)
    fig.add_trace(
        go.Table(
            columnwidth=[70, 150, 300],
            header=dict(values=["edge (parent → child)", "verdict", "why"],
                        fill_color="#EEF2F6", align="left",
                        font=dict(color=INK, size=11), height=26),
            cells=dict(values=[edge, verdict, note], align="left", height=26,
                       font=dict(size=10, color=[[INK] * len(edge), vcol, [INK] * len(edge)]),
                       fill_color=[fill, fill, fill], line=dict(color="white", width=1)),
        ), row=3, col=1,
    )

    if have_align:
        nb = len(align["block_ranges"])
        tested = [r for r in align["edge_rows"] if r["child_block"] is not None]

        # (2,2) the nesting claim as a picture. Every point must sit ABOVE the
        # dotted diagonal; a violation would be a point below it, which is easier
        # to disbelieve than the word "respected" in a cell.
        fig.add_trace(go.Scatter(x=[0, nb - 1], y=[0, nb - 1], mode="lines",
                                 line=dict(color="#CBD5E1", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=3, col=2)
        fig.add_trace(
            go.Scatter(
                x=[r["parent_block"] for r in tested],
                y=[r["child_block"] for r in tested],
                mode="markers+text",
                text=[r["edge"].replace(" -> ", "→") for r in tested],
                textposition="top center", textfont=dict(size=9, color="#9AA7B3"),
                marker=dict(size=13, line=dict(width=1, color="white"),
                            color=["#166534" if r["verdict"] == "respected" else "#991B1B"
                                   for r in tested]),
                hovertemplate="%{text}<br>parent B%{x} → child B%{y}<extra></extra>",
                showlegend=False), row=3, col=2)
        fig.update_xaxes(title_text="parent block", range=[-0.7, nb - 0.3], dtick=1,
                         row=3, col=2)
        fig.update_yaxes(title_text="child block", range=[-0.7, nb + 0.3], dtick=1,
                         row=3, col=2)

        # (3,1) the same six edges as a table, plus the three that cannot be tested
        av, ap, ac, aw, afill, aink = [], [], [], [], [], []
        for r in align["edge_rows"]:
            ok = r["verdict"] == "respected"
            untest = r["child_block"] is None
            av.append(r["edge"].replace(" -> ", " → "))
            ap.append("-" if r["parent_block"] is None else f"B{r['parent_block']}")
            ac.append("-" if untest else f"B{r['child_block']}")
            aw.append("untestable — child never learned" if untest
                      else f"respected (gap {r['child_block'] - r['parent_block']})" if ok
                      else "VIOLATED")
            afill.append("#F1F5F9" if untest else "#DCFCE7" if ok else "#FEE2E2")
            aink.append("#9AA7B3" if untest else "#166534" if ok else "#991B1B")
        fig.add_trace(
            go.Table(
                columnwidth=[70, 50, 50, 220],
                header=dict(values=["edge", "parent", "child", "verdict"],
                            fill_color="#EEF2F6", align="left",
                            font=dict(color=INK, size=11), height=26),
                cells=dict(values=[av, ap, ac, aw], align="left", height=26,
                           font=dict(size=10, color=[[INK] * len(av)] * 3 + [aink]),
                           fill_color=[afill] * 4, line=dict(color="white", width=1)),
            ), row=4, col=1,
        )

        # (3,2) parents early, children late — the nesting claim in aggregate
        fb = {int(k): v for k, v in align["first_block_of_feature"].items()}
        parents = sorted({p for p, _ in d["true_edges"]})
        children = sorted({c for _, c in d["true_edges"]})
        for name, feats, col in (("parents", parents, "#7C22CE"),
                                 ("children", children, "#F59E0B")):
            fig.add_bar(x=list(range(nb)),
                        y=[sum(1 for f in feats if fb.get(f) == b) for b in range(nb)],
                        name=name, marker_color=col, row=4, col=2)
        fig.update_xaxes(title_text="earliest Matryoshka block holding the feature",
                         dtick=1, row=4, col=2)
        fig.update_yaxes(title_text="features", row=4, col=2)
        fig.update_layout(barmode="group")

    tp, fp2, fn = d["true_positives"], d["false_positives"], d["false_negatives"]
    F = d["n_features"]
    nest = ""
    if have_align:
        nest = (f"　·　<b><span style='color:#166534'>nesting {align['n_respected']}"
                f"/{align['n_testable']}</span></b>")
    fig.update_layout(
        title=dict(text=_titled(
            "Trained toy calibration",
            f"metrics run on a Matryoshka SAE trained on Bussmann's tree　·　"
            f"<b><span style='color:#166534'>precision {d['precision']:.2f}</span></b>, "
            f"<b>recall {d['recall']:.2f}</b> "
            f"({tp}/{tp + fn} edges, {fp2} false positives){nest}",
            subtitle=scope_subtitle("No layer: Bussmann toy, a Matryoshka SAE trained on it")
            + f"　·　SAE recovered {d['n_recovered_features']}/{F} true features　·　"
              "the misses are edges whose child the SAE never learned, not metric failures"),
            x=0.01, xanchor="left", yref="container", y=0.99, yanchor="top",
            font=dict(size=14, color=INK)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        # a fourth row was added for this tier's own two results; without the
        # extra height every panel is squeezed and the rank bars lose their labels
        width=1180, height=(1560 if have_align else 1160) if pt else (1180 if have_align else 780),
        # The key sits BELOW the figure, not inside it. Placed in-figure it was
        # pinned at (0.62, 0.055) -- coordinates that land on the bottom-left panel,
        # which is a table. Eight entries wrapped leftward across its rows and hid
        # the edge verdicts underneath a translucent box. Nothing in the layout
        # reserved that space, so the collision was a matter of how many entries the
        # run happened to produce.
        legend=dict(orientation="h", x=0.01, xanchor="left", y=-0.045, yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=40, t=150, b=90),
    )
    return fig


def run_trained_calibration():
    path = C.OUT_DIR / "trained_toy_calibration.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run validation/calibrate_on_trained_toy.py first")
    # Optional: the lateral control's output. Absent on a checkout that has not run
    # validation.block_tree_alignment, and the page degrades to the recovery half
    # rather than failing -- the two are separate questions and separate scripts.
    ap = C.OUT_DIR / "block_tree_alignment.json"
    align = json.loads(ap.read_text()) if ap.exists() else None
    if align is None:
        print("note: outputs/block_tree_alignment.json not found - "
              "run `python3 -m validation.block_tree_alignment` for the nesting panels")
    d = json.loads(path.read_text())

    # Absent is handled above; STALE was not, and stale is worse -- it renders. The
    # control ran on 9 August against a checkpoint that learned 17 of 20 features, so
    # its panel said "6 of 9 edges testable, the rest have an endpoint the SAE never
    # learned" on a page whose other half says all 20 were learned. One page, two
    # checkpoints, contradicting each other. The two files record which features were
    # recovered, so they can be asked whether they describe the same run.
    if align is not None:
        aligned_feats = {int(k) for k in align.get("first_block_of_feature", {})}
        graded_feats = set(d.get("recovered_features", []))
        if aligned_feats != graded_feats:
            print(f"note: block_tree_alignment.json describes a different checkpoint "
                  f"({len(aligned_feats)} features recovered, this run has "
                  f"{len(graded_feats)}) - the nesting panels are dropped rather than "
                  f"shown against numbers they do not belong to. Re-run "
                  f"`python3 -m validation.block_tree_alignment` to restore them.")
            align = None
    fig = build_trained_calibration_dashboard(d, align)
    out = C.OUT_DIR / "trained_toy_calibration.html"
    write_page(fig, out, captions=captions_trained_calibration(d, align))
    print(f"saved: {out}")


# ---------------------------------------------------------------------------
# Qualitative dashboard (real gemma edges + Neuronpedia labels, colour-coded)
# ---------------------------------------------------------------------------
# One tint per verdict: survivors green, the three rejected categories in
# distinct warm/purple tints so a glance separates "kept" from "why-killed".
CAT_STYLE = {
    "survivor":          ("survivor",       "#DCFCE7", "#166534"),
    "reject:superparent": ("superparent",   "#EDE9FE", "#5B21B6"),
    "reject:freq-driven": ("freq-driven",   "#FEE2E2", "#991B1B"),
    "reject:no-recon":    ("no-recon",      "#FEF3C7", "#92400E"),
}
CAT_ORDER = ["survivor", "reject:superparent", "reject:freq-driven", "reject:no-recon"]


def _clip(s, n=64):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def build_qualitative_dashboard(report):
    pairs = list(report.keys())
    fig = make_subplots(
        rows=len(pairs), cols=1,
        specs=[[{"type": "table"}] for _ in pairs],
        subplot_titles=[f"Block pair {k}" for k in pairs],
        vertical_spacing=0.04,
    )

    for ri, key in enumerate(pairs, start=1):
        rows = sorted(report[key], key=lambda r: (CAT_ORDER.index(r["category"]),
                                                   -r["reverse_cov"]))
        edge, verdict, Rv, Fv, gain, recon, surv, plab, clab = ([] for _ in range(9))
        vcolor, rowfill = [], []
        for r in rows:
            short, tint, tcol = CAT_STYLE[r["category"]]
            # Plain text: plotly table cells render HTML as literal text, so a
            # link here would print its own markup. The clickable Neuronpedia
            # links live in the markdown report instead.
            edge.append(f"{r['parent']} → {r['child']}")
            verdict.append(short)
            Rv.append(f"{r['reverse_cov']:.2f}")
            Fv.append(f"{r['forward_cov']:.2f}")
            gain.append(f"{r['recon_parent_gain']:.2f}")
            recon.append("Y" if r["recon_pass"] else "n")
            surv.append("-" if r["freq_survival"] is None else f"{r['freq_survival']:.2f}")
            plab.append(_clip(r.get("parent_label")))
            clab.append(_clip(r.get("child_label")))
            vcolor.append(tcol)
            rowfill.append(tint)

        cols = [edge, verdict, Rv, Fv, gain, recon, surv, plab, clab]
        ncol = len(cols)
        fig.add_trace(
            go.Table(
                columnwidth=[80, 88, 32, 32, 44, 40, 40, 226, 226],
                header=dict(
                    values=["edge", "verdict", "R", "F", "gain", "recon", "surv",
                            "parent label (Neuronpedia)", "child label (Neuronpedia)"],
                    fill_color="#EEF2F6", align="left",
                    font=dict(color=INK, size=11), height=26,
                ),
                cells=dict(
                    values=cols,
                    align="left", height=24,
                    font=dict(size=10,
                              color=[[INK] * len(edge) if j != 1 else vcolor for j in range(ncol)]),
                    fill_color=[rowfill for _ in range(ncol)],
                    line=dict(color="white", width=1),
                ),
            ),
            row=ri, col=1,
        )

    n_surv = sum(1 for k in pairs for r in report[k] if r["category"] == "survivor")
    n_rej = sum(1 for k in pairs for r in report[k] if r["category"] != "survivor")
    fig.update_layout(
        title=dict(text=_titled(
                       "Qualitative agreement",
                       f"<span style='color:#166534'>{n_surv} survivors</span> vs "
                       f"<span style='color:#991B1B'>{n_rej} rejected</span>, read against Neuronpedia "
                       "labels: survivors should be semantically related, rejected should look like artifacts"
                       "<br><span style='font-size:11px'>R = reverse coverage　·　F = forward coverage　·　"
                       "gain = parent's reconstruction gain　·　surv = frequency survival　·　"
                       "the markdown report has the same rows with clickable Neuronpedia links</span>"),
                   x=0.01, xanchor="left", yref="container", y=0.985, yanchor="top",
                   font=dict(size=13, color=INK)),
        font=FONT, paper_bgcolor="white",
        width=1200, height=270 + 640 * len(pairs), margin=dict(l=30, r=30, t=140, b=30),
    )
    return fig


def run_qualitative():
    path = C.RUN_DIR / "qualitative_check.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - run python3 -m validation.qualitative_check first")
    report = json.loads(path.read_text())
    fig = build_qualitative_dashboard(report)
    # Named to mirror metrics_dashboard/metrics_report: writing this as
    # qualitative_check.html would shadow Jekyll's render of qualitative_check.md,
    # leaving the text report unreachable on the published site.
    out = C.RUN_DIR / "qualitative_dashboard.html"
    write_page(fig, out, captions=captions_qualitative(report))
    print(f"saved: {out}")


ORANGE = "#D98A3D"


def build_in_block_dashboard(report):
    """In-block (same-level) edges dashboard from in_block_edges.json. Two panels,
    both with an explicit legend: (1) per-block edge/duplicate/gate counts,
    (2) the in-block superparents' out-degree x firing rate."""
    blocks = report["blocks"]
    names = [f"B{b['block']} ({b['n_features']})" for b in blocks]
    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.16,
        subplot_titles=(
            "Same-level relations per block: directed edges, duplicates, and how "
            "many survive PMI then S_res (log)",
            "In-block superparents: out-degree x firing rate (bubble = children)",
        ),
    )
    series = [
        ("directed edges", GREY, [b["n_edges"] for b in blocks]),
        ("survive PMI > 0", PURPLE, [b["n_after_pmi"] for b in blocks]),
        ("pass S_res (genuine)", GREEN, [(b["sres"]["n_pass"] if b.get("sres") else 0) for b in blocks]),
        ("duplicate pairs (rename/split)", ORANGE, [b["n_duplicates"] for b in blocks]),
    ]
    for nm, col, y in series:
        fig.add_bar(x=names, y=y, name=nm, marker_color=col, legendgroup=nm, row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    # (2) superparents across all blocks, sized by out-degree
    any_sp = False
    for bi, b in enumerate(blocks):
        sps = b["superparents"]
        if not sps:
            continue
        any_sp = True
        fig.add_trace(
            go.Scatter(
                x=[s["outdeg_frac"] for s in sps], y=[s["fire_frac"] for s in sps],
                mode="markers",
                marker=dict(color=PAIR_COLORS[bi % len(PAIR_COLORS)], size=[12 + s["outdeg"] / 12 for s in sps],
                            line=dict(width=1, color="white")),
                text=[f"B{b['block']}:{s['global']} {_wrap(s.get('label', ''))}<br>"
                      f"{s['outdeg']} children, fires {100*s['fire_frac']:.0f}%" for s in sps],
                hovertemplate="%{text}<extra></extra>",
                name=f"B{b['block']} superparents", legendgroup=f"sp{b['block']}",
            ), row=3, col=1)
    if not any_sp:
        fig.add_annotation(text="no in-block superparents", row=3, col=1,
                           showarrow=False, font=dict(color=INK))
    fig.update_xaxes(title_text="out-degree fraction of block", row=3, col=1)
    fig.update_yaxes(title_text="firing rate", row=3, col=1)

    fig.update_layout(
        title=dict(text=_titled(
            "In-block (same-level) edges",
            "Directed parent→child WITHIN a block (asymmetric containment); "
            "co-extensive pairs are duplicates, never edges.",
            stats={"total_tokens": report["total_tokens"]},
            subtitle=page_subtitle({"total_tokens": report["total_tokens"]})),
            x=0.01, xanchor="left", font=dict(size=15, color=PURPLE)),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        barmode="group", showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.01),
        width=1150, height=820, margin=dict(l=60, r=60, t=140, b=80),
    )
    return fig


def recaption_site():
    """Refresh the caption block on every page under OUT_DIR, from committed JSON.

    The plots cannot be redrawn from a clone -- `exp0_stats.pt` is not in git --
    but the captions can, so this walks the published tree and rewrites just the
    caption section in place. That is what keeps the site from carrying captions
    on the pages whose cache happened to be present and none on the rest.

    Reports what it skipped and why, rather than passing over a page in silence:
    a page missing its caption because its JSON is absent is a fact worth seeing.
    """
    done, skipped = [], []

    def do(path, items, why_missing):
        if not path.exists():
            return
        if items is None:
            skipped.append((path, why_missing))
        elif inject_captions(path, items):
            done.append(path)
        else:
            skipped.append((path, "no </body> to inject into"))

    for src in sorted(C.SOURCES):
        for run in sorted((C.OUT_DIR / src).glob("layer_*")):
            rep, sec = report_json(run), second_json(run)
            do(run / "metrics_dashboard.html",
               captions_dashboard(rep, sec) if rep else None, "no metrics_report.json")
            do(run / "superparent_sankey.html",
               captions_sankey(rep) if rep else None, "no metrics_report.json")
            q = run / "qualitative_check.json"
            do(run / "qualitative_dashboard.html",
               captions_qualitative(json.loads(q.read_text())) if q.exists() else None,
               "no qualitative_check.json")
            ib = run / "in_block_edges.json"
            do(run / "in_block_dashboard.html",
               captions_in_block(json.loads(ib.read_text())) if ib.exists() else None,
               "no in_block_edges.json")

    cal = C.OUT_DIR / "synthetic_toy_calibration.json"
    do(C.OUT_DIR / "synthetic_toy_calibration.html",
       captions_calibration(_calibration_data()) if cal.exists() else None,
       "no synthetic_toy_calibration.json")
    tt = C.OUT_DIR / "trained_toy_calibration.json"
    al = C.OUT_DIR / "block_tree_alignment.json"
    do(C.OUT_DIR / "trained_toy_calibration.html",
       captions_trained_calibration(json.loads(tt.read_text()),
                                    json.loads(al.read_text()) if al.exists() else None)
       if tt.exists() else None, "no trained_toy_calibration.json")

    for pth in done:
        print(f"  captioned  {pth.relative_to(C.OUT_DIR)}")
    for pth, why in skipped:
        print(f"  SKIP       {pth.relative_to(C.OUT_DIR)}\n               {why}")
    print(f"[cap] {len(done)} captioned, {len(skipped)} skipped")
    return len(skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, help="which layer to render (overrides EXP0_LAYER)")
    ap.add_argument("--in-block", action="store_true",
                    help="visualise in-block same-level edges (needs in_block_edges.json)")
    ap.add_argument("--calibration", action="store_true",
                    help="visualise the synthetic-toy metric calibration (no cache needed)")
    ap.add_argument("--qualitative", action="store_true",
                    help="visualise real survivor-vs-rejected edges (needs qualitative_check.json)")
    ap.add_argument("--trained-calibration", action="store_true",
                    help="Tier-2: edge recovery on a trained toy SAE (needs trained_toy_calibration.json)")
    ap.add_argument("--captions", action="store_true",
                    help="refresh the caption block on every published page, from the "
                         "committed JSON; does not redraw any plot, so it needs no cache")
    args = ap.parse_args()
    C.use_layer(args.layer)          # re-execs when it changes anything; see config.use_layer
    if args.captions:
        recaption_site()
        return
    if args.calibration:
        run_calibration()
        return
    if args.qualitative:
        run_qualitative()
        return
    if args.trained_calibration:
        run_trained_calibration()
        return
    if args.in_block:
        if not C.IN_BLOCK_PATH.exists():
            raise SystemExit(f"missing {C.IN_BLOCK_PATH} - run in_block_edges.py first")
        ib = json.loads(C.IN_BLOCK_PATH.read_text())
        fig = build_in_block_dashboard(ib)
        path = C.RUN_DIR / "in_block_dashboard.html"
        write_page(fig, path, captions=captions_in_block(ib))
        print(f"saved: {path}")
        return

    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(C.missing_stats_msg())
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    pairs = stats["pairs"]
    pairs_data = [compute_pair(stats, p, c) for (p, c) in pairs]

    # Attach per-edge S_res verdicts (second pass) so the Sankey colours by genuine
    # refinement, not the weak run_metrics contribution filter. Absent -> Sankey
    # falls back to the recon+frequency colouring (older caches / no second pass).
    if C.SECOND_PASS_PATH.exists():
        second = json.loads(C.SECOND_PASS_PATH.read_text())
        for pd_ in pairs_data:
            sp_entry = second.get(pd_["key"], {}).get("sres", {})
            p0, c0 = pd_["p0"], pd_["c0"]
            mat = torch.zeros_like(pd_["edge_mask"], dtype=torch.bool)
            for e in sp_entry.get("edges", []):
                if e["pass"]:
                    mat[e["parent"] - p0, e["child"] - c0] = True
            pd_["sres_pass"] = mat
            pd_["n_pmi"] = sp_entry.get("n_edges_scored")   # PMI>0 shortlist size
            pd_["n_sres"] = sp_entry.get("n_pass")          # pass probe-S_res

    feat_labels = C.load_feature_labels()
    if not feat_labels:
        print("note: outputs/feature_labels.json not found - run fetch_labels.py "
              "to show descriptions instead of bare indices")

    dash = build_dashboard(pairs_data, feat_labels, stats=stats)
    dash_path = C.RUN_DIR / "metrics_dashboard.html"
    write_page(dash, dash_path,
               captions=captions_dashboard(report_json(C.RUN_DIR), second_json(C.RUN_DIR)))
    print(f"saved: {dash_path}")

    # One Sankey per block pair that has a superparent, stacked in a single file.
    sk = build_all_superparent_sankeys(stats, pairs, pairs_data, feat_labels=feat_labels)
    if sk is None:
        print("note: no superparent found in any block pair - skipping sankey")
    else:
        # Sankeys only: _add_sres_legend puts two off-canvas Scatter markers in
        # the same figure, so len(sk.data) reported two panels more than there are.
        n_panels = sum(1 for t in sk.data if t.type == "sankey")
        sk_path = C.RUN_DIR / "superparent_sankey.html"
        write_page(sk, sk_path, captions=captions_sankey(report_json(C.RUN_DIR)))
        print(f"saved: {sk_path}  ({n_panels} block pair{'s' if n_panels != 1 else ''})")


if __name__ == "__main__":
    main()
