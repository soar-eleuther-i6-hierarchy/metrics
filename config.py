"""
Central configuration for the hierarchy-metric suite.

Everything tunable lives here: the model/SAE pair, the corpus slice, the block
structure, and every metric threshold. The thresholds are global and identical
for every SAE source and layer — holding them fixed is what makes cross-source
comparisons meaningful — and reporting/make_report_tables.py generates the
paper's threshold macros from this file, so a value changed here re-numbers
the paper on the next build.

The metric implementations are pure functions in metrics/; the pipeline stages
sit at the repo root; per-run outputs land under outputs/<source>/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model + SAE
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-2-2b-res-matryoshka-dc"

# Depth of the base transformer, read off `google/gemma-2-2b`'s own
# `config.json` (`num_hidden_layers: 26`). A literal because nothing this repo
# caches carries it: `metrics_report.json`'s config block records which layer was
# graded, never how many the model has. The PCFG runs DO carry theirs, under
# `config.base_model.n_layers`, so that side is read from the report and only
# gemma needs a constant here.
#
# Needed by exactly one thing: putting two models of very different depth on one
# axis. A graded layer's position in its own network is (LAYER + 1) / N -- the
# SAE reads `hook_resid_post`, the output of block LAYER, so LAYER + 1 of the N
# blocks have run. That definition, not LAYER / N, is what makes PCFG's layer 1
# of 4 land on gemma's layer 12 of 26: both sit at exactly half depth.
BASE_N_LAYERS = 26

# Which transformer layer's residual-stream Matryoshka SAE to analyse.
# Override per run with the EXP0_LAYER env var, e.g. `EXP0_LAYER=12 python3 ...`.
# The matryoshka SAE is released for layers 0-24; the SAE id, hook name, the
# Neuronpedia / dataset "source" name, and the per-layer output dir are all
# derived from this single number so nothing has to be edited by hand per layer.
LAYER = int(os.environ.get("EXP0_LAYER", "12"))
SAE_ID = f"blocks.{LAYER}.hook_resid_post"
HOOK_NAME = SAE_ID
SAE_SOURCE = f"{LAYER}-res-matryoshka-dc"  # e.g. "12-res-matryoshka-dc"
MODEL_KWARGS = {"center_writing_weights": False}
PREPEND_BOS = True


def use_layer(layer: int | None) -> None:
    """Make a script's `--layer N` authoritative, by re-execing with EXP0_LAYER set.

    LAYER is read from the environment above, and everything downstream — SAE_ID,
    HOOK_NAME, SAE_SOURCE, RUN_DIR and every path under it — is derived from it at
    import time. A script's argparse runs long after that, so a `--layer` flag
    cannot simply assign: the constants are already fixed, and any module that did
    `from config import RUN_DIR` would keep the old value while its neighbour saw
    the new one. That is a wrong path with no error, which is the failure this repo
    keeps logging.

    Re-exec instead. The replacement process derives every constant from the flag
    in the normal way, and child stages inherit it through the environment, so the
    flag and the env var can never disagree — passing both is not an error, the
    flag simply wins.

    Called after parse_args() and before the layer is used. A no-op when the flag
    is absent or already agrees, which is what stops it re-execing forever.
    """
    if layer is None or os.environ.get("EXP0_LAYER") == str(layer):
        return
    os.environ["EXP0_LAYER"] = str(layer)
    # `python3 -m pkg.mod` must be rebuilt as `-m`, not as a path: running the file
    # directly puts its own directory on sys.path instead of the repo root, and
    # `import config` then fails.
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    head = [sys.executable, "-m", spec.name] if spec else [sys.executable, sys.argv[0]]
    os.execve(sys.executable, head + sys.argv[1:], os.environ)


MATRYOSHKA_STEPS = [128, 512, 2048, 8192, 32768]
D_SAE = 32768


def _block_ranges(steps):
    ranges, prev = [], 0
    for s in steps:
        ranges.append((prev, s))
        prev = s
    return ranges


BLOCK_RANGES = _block_ranges(MATRYOSHKA_STEPS)
N_BLOCKS = len(BLOCK_RANGES)


def block_of(feature_idx: int) -> int:
    for b, (start, end) in enumerate(BLOCK_RANGES):
        if start <= feature_idx < end:
            return b
    raise ValueError(f"feature {feature_idx} out of range [0,{D_SAE})")


# ---------------------------------------------------------------------------
# Dataset (one fixed slice, so counts are comparable across layers and runs)
# ---------------------------------------------------------------------------
DATASET = "NeelNanda/pile-10k"
N_DOCS = 400
CONTEXT_SIZE = 128
BATCH_DOCS = 8

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# The gate thresholds come from the named set `metrics.rules.GEMMA_MATRYOSHKA`, shared with
# the scoring pipeline. The names below are unchanged. Any changes should be made there, not here, and bump `metrics.rules.RULESET_VERSION` with it.
from metrics.rules.constants import GEMMA_MATRYOSHKA as _GATES  # noqa: E402
from metrics.rules.constants import METRIC_SETTINGS as _METRIC  # noqa: E402

FIRE_THRESHOLD = _GATES.fire_threshold   # feature "fires" above this (post-JumpReLU)
EDGE_TAU = _GATES.edge_tau               # reverse-coverage edge criterion
MIN_FIRE_COUNT = _GATES.min_fire_count   # rare-feature guard
# Joint-support guard: a child firing MIN_FIRE_COUNT times
# inside a near-always-on parent hits R = 1.0 by chance; requiring a minimum
# co-fire count kills those. Excluded edges are REPORTED, not silently dropped.
# NOTE: any value <= EDGE_TAU * MIN_FIRE_COUNT (= 10) is vacuous — every kept
# edge already satisfies it.
# The converse also holds: with MIN_JOINT = 30 > MIN_FIRE_COUNT, the per-endpoint
# guard is subsumed in the joint gate (cofire <= min(fire_p, fire_c), so
# co-fire >= 30 forces both endpoints >= 30). MIN_FIRE_COUNT still binds on its
# own wherever keep_edges runs WITHOUT the joint guard (the calibration stages
# and the legacy_guards path), and as the per-feature testability filter — it is
# a feature-level guard, MIN_JOINT an edge-level one. State the pair gate as two
# effective conditions: R >= EDGE_TAU and co-fire >= MIN_JOINT.
MIN_JOINT = _GATES.min_joint

# Which adjacent block pairs to compute. B3->B4 is the 6144 x 24576 monster;
# it does not fit on a 4 GB GPU. On the A40 it fits: enable with EXP0_B3B4=1.
# Measured 2026-09-11 on one A40, peak about 26 GB of 49, collection 210 s.
#
# The caveat here used to predict "a thin, noisy pair", on the reasoning that B4
# is 24576 mostly-rare features so most of its edges would be unsupported. Half
# of that held. The pair is thin: 3,277 edges out of 151 million possible, which
# is 0.002 percent. It is not noisy. Those 3,277 are the largest candidate set of
# the four pairs, and B3->B4 tracks B2->B3 closely on every metric. Skipping it
# costs the fourth point that separates a plateau from a slope.
# See research-log/EXPERIMENT_LOG.md, 2026-09-11, "The fourth block pair".
INCLUDE_B3_B4 = os.environ.get("EXP0_B3B4", "0") == "1"

# --- Metric 2a: reconstruction-ablation contribution filter -----------------
# (Tree-SAE-INSPIRED baseline, not the paper's S_res — see metrics/reconstruction.py.)
# An edge passes when ablating the parent hurts reconstruction on the child's
# firing tokens by at least this relative amount (and same for the child).
RECON_REL_GAIN_MIN = _GATES.recon_rel_gain_min   # >=1% relative error increase = "contributes"

# --- Metric 2b: probe-based S_res (Tree SAE Eq. 5, rank-scored) --------------
# S_res(p,c) = min((d_c*)ᵀ d_c, (d_c*)ᵀ d_p) with d_c* a linear-probe direction
# for the child concept. Scored by the RANK rule (both decoders in the top-k
# probe correlations over all features), not a threshold: healthy pairs have
# d_p ⟂ d_c which caps the min at 1/√2, so thresholds above that reject
# everything. Probes need enough positives to train on.
SRES_RANK_TOP_K = _GATES.sres_rank_top_k        # Tree SAE's operational rule: both in top-5
# The probe settings come from `metrics.rules.METRIC_SETTINGS`, shared with scoring/.
MIN_PROBE_POS = _METRIC.probe_min_pos          # min child-firing tokens to train a probe
SRES_MAX_CHILDREN_PER_PAIR = 4000   # cost guard; log when hit
SRES_NEG_RATIO = _METRIC.probe_neg_ratio       # negatives sampled per positive
SRES_MAX_PROBE_TOKENS = _METRIC.probe_max_tokens   # cap on (pos + neg) tokens per probe
SRES_MIN_NEG = _METRIC.probe_min_neg           # fewer negatives than this -> child untestable
SRES_STEPS = _METRIC.probe_steps               # probe Adam steps
SRES_LR = _METRIC.probe_lr                     # probe Adam lr

# --- Joint-child: share-energy split flag -----------------------------------
# One child holding at least this share of its parent's activation energy is a
# copy of the parent, not a refinement (feature splitting). Was hard-coded as
# 0.9 in run_metrics ("n_share_energy_ge_09") and as SPLIT_SHARE_MIN in the
# Tier-1 calibration; lives here so the paper's η, the pipeline and the
# calibration can never disagree.
SHARE_ENERGY_SPLIT = 0.9

# --- Metric 3: sibling redundancy ------------------------------------------
# Mean pairwise child-child co-activation (Jaccard) above this = feature
# splitting in disguise rather than real refinement.
SIBLING_REDUNDANCY_FLAG = 0.5
# Within-block co-firing matrices are needed for sibling stats. B4's 24576^2
# does not fit in RAM comfortably; we compute B1, B2, B3 only.
SIBLING_BLOCKS = [1, 2, 3]

# --- In-block (same-level) edges (in_block_edges.py) ------------------------
# Hierarchy need not respect block boundaries: two features in the SAME block
# can stand in a parent/child (refinement) or duplicate relation. These blocks
# get a within-block directed-edge analysis. collect_statistics caches within-cofire for
# SIBLING_BLOCKS ∪ IN_BLOCK_BLOCKS, so B0 is cached after a rerun; on older caches
# in_block_edges falls back to rebuilding B0 from the token cache. B3/B4 skipped:
# B4's 24576^2 matrix is ~4.8 GB, not worth it.
IN_BLOCK_BLOCKS = [0, 1, 2]

# --- Metric 4: out-degree / superparents ------------------------------------
SUPERPARENT_OUTDEG_FRAC = _GATES.superparent_outdeg_frac
SUPERPARENT_FIRE_FRAC = 0.10

# --- Metric 5: token-frequency control --------------------------------------
# Token ids ranked by corpus frequency; buckets split by cumulative token MASS:
#   bucket 0 (high) = most frequent ids covering the top HIGH_MASS of tokens
#   bucket 1 (mid)  = next ids up to HIGH_MASS + MID_MASS
#   bucket 2 (low)  = the rest
FREQ_HIGH_MASS = _METRIC.freq_high_mass
FREQ_MID_MASS = _METRIC.freq_mid_mass
N_FREQ_BUCKETS = _METRIC.n_freq_buckets
# An edge is "frequency-driven" when its reverse coverage on low+mid tokens
# drops below this fraction of its all-token reverse coverage.
FREQ_SURVIVAL_MIN = _GATES.freq_survival_min

# ---------------------------------------------------------------------------
# Paths + device
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# EXP0_OUT redirects ALL outputs to a directory of your choice, so a scratch run
# never touches the git-tracked outputs/ published on GitHub Pages. Default: unchanged.
OUT_DIR = Path(os.environ.get("EXP0_OUT", HERE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Results are grouped by SOURCE first, layer second: outputs/gemma-2-2b/layer_NN/.
# They used to sit at outputs/layer_NN/, from when gemma was the only source there
# could be -- which stopped being true the moment a PCFG run was published beside
# them, and left the site claiming that "layer 6" is a global fact rather than a
# fact about one model.
SOURCE_NAME = "gemma-2-2b"
LAYER_RUN = f"{SOURCE_NAME}/layer_{LAYER:02d}"

# EXP0_RUN names the directory instead, for a run that is not a gemma layer: the
# PCFG SAE is 1792 latents in 8 blocks over its own base model, and `layer_01`
# would both collide with a future gemma layer and misdescribe it. Any depth under
# OUT_DIR works -- every page derives its own distance to the site root from its
# path -- but the LAST component decides how it is read: a directory named
# `layer_NN` gets the layer nav, anything else gets the site-wide one.
RUN_NAME = os.environ.get("EXP0_RUN", LAYER_RUN)
RUN_DIR = OUT_DIR / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
# Layer-independent artifacts (the toy calibrations) stay directly in OUT_DIR.
IS_LAYER_RUN = Path(RUN_NAME).name.startswith("layer_")


def page_depth(path) -> int:
    """Levels from a generated page's directory up to the site root.

    The site root is the repo root, so the answer is simply how deep the file sits
    inside it -- which also covers the pages that are NOT under outputs/ at all:
    the package READMEs, and outputs_archive/. Derived rather than passed, because
    the depth changed for 25 published pages the day results were grouped by
    source, and every caller that had hardcoded 2 was then wrong.

    A page written under EXP0_OUT is outside the repo; there OUT_DIR stands in for
    `outputs/`, so a scratch run produces the same bar as the published page.
    """
    p = Path(path).resolve()
    try:
        return len(p.relative_to(HERE.resolve()).parts) - 1
    except ValueError:
        return len(p.relative_to(OUT_DIR.resolve()).parts)

def scope_line(total_tokens=None, bold=("**", "**"), sep="　·　", n_docs=None, config=None):
    """Which layer, and the knobs a reader needs to interpret the numbers.

    Single source for the context line shown on every page and report, so the
    dashboards and the markdown digests can never drift apart. `bold` selects the
    emphasis syntax: ("**", "**") for markdown, ("<b>", "</b>") for HTML.

    `config` is the stats file's own config block. Pass it and the line describes
    the run that produced the numbers rather than this module's gemma defaults --
    a PCFG page reading "gemma-2-2b" states the wrong model, and a reader has no
    way to tell from the page that it is wrong. Absent, nothing changes: every
    caller that omits it is grading gemma.
    """
    cfg = config or {}
    b0, b1 = bold
    base = cfg.get("base_model") or {}
    model = MODEL_NAME.split("/")[-1]
    if cfg.get("source") == "pcfg":
        model = (f"PCFG toy {base.get('n_layers', '?')}L "
                 f"d_model={base.get('d_model', '?')}")
    bits = [f"{b0}Layer {cfg.get('layer', LAYER)}{b1}",
            f"{model} / {cfg.get('sae_source', SAE_SOURCE)}",
            cfg.get("sae_id", SAE_ID)]
    steps = cfg.get("matryoshka_steps")
    if steps and list(steps) != MATRYOSHKA_STEPS:
        bits.append(f"{steps[-1]:,} latents in {len(steps)} blocks")
    if total_tokens:
        bits.append(f"{int(total_tokens):,} tokens over "
                    f"{n_docs or cfg.get('n_docs') or N_DOCS} docs")
    # Every caller of this line gates edges with the FULL keep_edges gate
    # (including MIN_JOINT), so the line must state all of it: it used to omit
    # the co-fire guard and thereby described a looser gate than the one that
    # produced the numbers on the page.
    bits.append(f"edge: reverse coverage ≥ {EDGE_TAU}, co-fire ≥ {MIN_JOINT}, "
                f"both endpoints fire ≥ {MIN_FIRE_COUNT}")
    return sep.join(bits)


# --- site navigation --------------------------------------------------------
# Every page is reachable only by a deep link, so each carries its own nav bar.
# Emitted into the markdown reports too: Jekyll passes raw HTML through when it
# renders them (and serves them as .html, which is why the report links below
# end in .html, not .md). All hrefs are relative to the site root, so the bar
# works on GitHub Pages and when a file is opened straight off disk.
#
# It is `sticky`, not `fixed`: sticky occupies layout space, so the bar can
# never cover the top of a plot the way the old floating back-button did.
NAV_LAYERS = [1, 3, 6, 12, 18, 24]

# (file, label) for the five pages every layer_NN/ has. `file` doubles as the
# identity of "which page am I on", so switching layer lands on the same kind.
NAV_PAGES = [
    ("metrics_dashboard.html", "Dashboard"),
    ("superparent_sankey.html", "Superparents"),
    ("in_block_dashboard.html", "In-block"),
    ("qualitative_dashboard.html", "Qualitative"),
    ("metrics_report.html", "Metrics report"),
    ("in_block_edges.html", "In-block report"),
    ("qualitative_check.html", "Qualitative report"),
]

# Global links: (path from site root, label). "" is the repo README = the index.
# No "Overview" entry: the brand on the left already links to the site root, and
# two adjacent links to the same place is noise. No "Per layer" either: it went
# to the layer *table*, while the pills in the second row open each layer
# directly, which is strictly better.
# kill_rates.html and cross_depth_comparison.html are gone from this list. Both
# were hand-built with no generator, so neither could be rebuilt by rerunning a
# stage, and both were written against caches that counted BOS. BOS is an
# attention sink -- every feature fires there -- so it manufactured 400 joint
# firings for every pair in the dictionary and defeated the joint-support guard.
# Excluding it inverted the numbers those two pages were built to display: the
# deep-pair reconstruction share, the frequency-driven share, the death rate.
# They now sit in outputs_archive/ carrying a banner. A page that states a
# withdrawn result next to regenerated ones is worse than no page; put them back
# only behind a generator, so a rerun can never leave them stale again.
# Each SOURCE is a directory of layers under outputs/, and one entry in the nav's
# global row. Clicking it opens that source; its own layer pills then appear.
# Nothing here is gemma-specific any more: the PCFG SAE has one layer trained so
# far and gets a one-pill row, which is the honest picture of that sweep.
#
# `pages` differs per source because Neuronpedia labels exist for gemma's
# dictionary and no other, so the two qualitative pages exist only there. Listing
# a page a source does not have would put a 404 in its own nav bar.
SOURCES = {
    "pcfg-matryoshka": {
        "label": "pcfg-matryoshka",
        # All four residual streams of the base transformer (it has four layers,
        # 0-3, so gemma's 6/12/18/24 do not exist here and never will) -- the
        # zipf run's SAEs on every one of them.
        "layers": [0, 1, 2, 3],
        # The formatting sweep (Exp 2, axis b): four densities x three seeds,
        # every one trained and graded at layer 2 of its own base model. Same
        # density + different seed is a different run, not a different layer,
        # so they get their own pill group rather than rows in "layers".
        # (dirname under outputs/pcfg-matryoshka/, pill label)
        "runs": [(f"fmt_{code}_s{seed}", f"{label}-s{seed}")
                 for code, label in [("0000", "0.00"), ("1667", "0.17"),
                                     ("2308", "0.23"), ("2400", "0.24")]
                 for seed in (0, 1, 2)],
        "pages": [p for p in NAV_PAGES if not p[0].startswith("qualitative")],
    },
    SOURCE_NAME: {
        "label": "gemma-2-2b",
        # NAV_LAYERS, not a second literal: the reporting scripts iterate
        # NAV_LAYERS and the nav bar reads this entry — one list, one truth.
        "layers": NAV_LAYERS,
        "pages": NAV_PAGES,
    },
}

# Global links: (path from site root, label).
NAV_GLOBAL = [
    ("outputs/", "Results"),
    # Both tiers are named "<which> Toy Calibration" so the bar reads as two
    # rungs of one ladder. "Toy calibration" beside "Trained toy" implied the
    # first was THE toy calibration and the second was a different kind of
    # thing, when they are the same calibration on hand-built and on learned
    # statistics. The file names are unchanged: they are published URLs.
    ("outputs/synthetic_toy_calibration.html", "Synthetic Toy Calibration"),
    ("outputs/trained_toy_calibration.html", "Trained Toy Calibration"),
] + [(f"outputs/{name}/", cfg["label"]) for name, cfg in SOURCES.items()]


def source_of(current: str | None) -> str | None:
    """Which source a page belongs to, from its path under the site root.

    Derived rather than passed: the answer is already in the path, and a second
    place to state it is a second place for it to be wrong.
    """
    for name in SOURCES:
        if (current or "").startswith(f"outputs/{name}/"):
            return name
    return None

# Inlined rather than linked: every page must render identically offline and on
# Pages, and one <img> to an external host would be the only request any of
# these pages makes.
GITHUB_MARK = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>'

NAV_CSS = """<style>
.x0nav{position:sticky;top:0;z-index:999;background:#fff;border-bottom:1px solid #E3DAFB;
font:500 13px/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 14px;}
.x0nav .row{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding:9px 18px;}
.x0nav .row+.row{border-top:1px solid #F1ECFD;}
.x0nav a{text-decoration:none;color:#5A6B7B;}
.x0nav a:hover{color:#7C22CE;}
.x0nav .brand{font-weight:700;color:#7C22CE;letter-spacing:.2px;}
.x0nav .on{color:#7C22CE;font-weight:700;}
.x0nav .lbl{color:#9AA7B3;font-size:11px;text-transform:uppercase;letter-spacing:.7px;}
.x0nav .pill{border:1px solid #E3DAFB;border-radius:7px;padding:5px 10px;background:#F6F3FE;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{width:1px;height:17px;background:#E3DAFB;}
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;margin-left:auto;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
.x0nav details.dens{flex:1 1 100%;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;display:inline-block;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
.x0nav .drow{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding-top:9px;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style>"""


def nav_html(depth: int = 2, layer: int | None = None, page: str | None = None,
             current: str | None = None) -> str:
    """The site nav bar for one page.

    depth    levels from this page's directory up to the site root
             (1 for outputs/x.html, 3 for outputs/gemma-2-2b/layer_NN/x.html);
             `page_depth(path)` computes it, and callers should use that rather
             than count by hand
    layer    the layer this page describes, or None for a site-wide page
    page     which of NAV_PAGES this is, so the row can mark it current
    current  this page's own path from the site root, used to highlight its
             entry in the global row (site-wide pages only)

    The second row belongs to a SOURCE and appears only inside one, carrying that
    source's layers. Within a layer page it also carries that source's pages, and
    those two groups are what make navigation two-dimensional — change the layer
    and stay on the same page, or change the page and stay on the same layer.

    Which source is read off `current`, so a page that passes no `current` gets
    the single row. That is why every generator now passes one.
    """
    # depth 0 is the site root itself (the repo README): "" is not a usable href.
    root = "../" * depth or "./"

    top = [f'<a class="brand" href="{root}">SOAR I-6 · metrics</a>']
    for href, label in NAV_GLOBAL:
        # A directory entry owns every page under it, so a run that publishes a
        # whole directory (outputs/pcfg-matryoshka/) lights up its own nav entry from any of
        # its pages, not only from the index. `outputs/` itself is excluded --
        # it is the site index and prefixes every page there is, so it would be
        # permanently lit next to the entry that is actually current.
        owns = href.endswith("/") and href.count("/") > 1
        here = current is not None and (href == current or
                                        (owns and current.startswith(href)))
        on = "on" if here else ""
        top.append(f'<a class="{on}" href="{root}{href}">{label}</a>')
    # Right end, past everything else: it is the one link that leaves the site,
    # so it does not belong in the sequence of places within it. `margin-left:auto`
    # does the pushing, which keeps it last in the DOM too -- it should also be
    # last for a keyboard or a screen reader, not first.
    top.append(f'<a class="gh" href="{REPO_URL}" title="Browse the code on GitHub">'
               f'{GITHUB_MARK}Code</a>')
    rows = ['<div class="row">' + "".join(top) + "</div>"]

    # The layer row belongs to ONE source, so it appears only inside that source:
    # on a gemma layer page, and on the source's own index. It used to be on every
    # page, from when gemma was the only thing here -- which put five gemma pills
    # on the PCFG dashboard, offering to "switch layer" to a different model.
    # Reaching a layer from elsewhere is now one hop through the source entry.
    #
    # The "Page" group needs a layer to stay within, so it is the narrower case:
    # a layer page only. A pill keeps the current page kind when there is one;
    # from the source index it opens that layer's index, which is why every
    # layer_NN/ has a README.
    source = source_of(current)
    if source is None:
        return NAV_CSS + '<nav class="x0nav">' + "".join(rows) + "</nav>"
    cfg = SOURCES[source]

    def has_page(dirname: str, f: str) -> bool:
        d = OUT_DIR / source / dirname
        return (d / f).exists() or (d / f.replace(".html", ".md")).exists()

    def pill_href(dirname: str) -> str:
        """Keep the page kind when the target directory has that page; land on
        its index when it does not. Not every run produced every page (a Sankey
        needs a superparent to exist), and a 404 in the site's own chrome is
        worse than arriving at the directory."""
        if page and has_page(dirname, page):
            return f"{root}outputs/{source}/{dirname}/{page}"
        return f"{root}outputs/{source}/{dirname}/"

    second = ['<span class="lbl">Layer</span>']
    for L in cfg["layers"]:
        on = " on" if L == layer else ""
        second.append(
            f'<a class="pill{on}" href="{pill_href(f"layer_{L:02d}")}">{L}</a>'
        )

    # Which directory under the source this page is inside, read off `current`
    # exactly as source_of reads the source: for layer pages it repeats `layer`,
    # for a sweep run it is the run's dirname, and on the source index it is "".
    runs = cfg.get("runs", ())
    here = (current or "")[len(f"outputs/{source}/"):].split("/")[0] \
        if (current or "").startswith(f"outputs/{source}/") else ""
    in_run = any(d == here for d, _ in runs)

    # The Page group needs a directory to stay within: a layer page's layer_NN,
    # or a sweep run's own dirname. On a run page it lives in the sweep row
    # below, next to the pills that place the run.
    in_dir = f"layer_{layer:02d}" if layer is not None else (here if in_run else None)
    if in_dir and not in_run:
        second.append('<span class="sep"></span><span class="lbl">Page</span>')
        for f, label in cfg["pages"]:
            if not has_page(in_dir, f):     # pages-on-disk only, like the indexes
                continue
            on = " on" if f == page else ""     # page=None -> the layer index, nothing marked
            second.append(
                f'<a class="{on.strip()}" href="{root}outputs/{source}/{in_dir}/{f}">{label}</a>'
            )
    rows.append('<div class="row">' + "".join(second) + "</div>")

    # The formatting sweep gets a row of its own, folded behind the word
    # "Density": twelve pills on every page of the source would drown the four
    # layer pills that place the zipf run. <details> keeps it dependency-free --
    # it renders identically on Pages, on github.com and straight off disk --
    # and it starts open exactly when the reader is inside a sweep run, so the
    # lit pill that answers "where am I" is never hidden.
    if runs:
        # The flex row is an inner <div>, never <details> itself: Safari does
        # not lay out details' slotted content as flex children, and the pills
        # end up overlapping the Page group. A plain block child avoids the
        # slot entirely and renders identically everywhere.
        third = [f'<details class="dens"{" open" if in_run else ""}>',
                 '<summary class="lbl">Density</summary>', '<div class="drow">']
        for dirname, label in runs:
            on = " on" if dirname == here else ""
            third.append(
                f'<a class="pill{on}" href="{pill_href(dirname)}">{label}</a>'
            )
        if in_run:
            third.append('<span class="sep"></span><span class="lbl">Page</span>')
            for f, label in cfg["pages"]:
                if not has_page(in_dir, f):
                    continue
                on = " on" if f == page else ""
                third.append(
                    f'<a class="{on.strip()}" href="{root}outputs/{source}/{in_dir}/{f}">{label}</a>'
                )
        third.append('</div></details>')
        rows.append('<div class="row">' + "".join(third) + "</div>")

    return NAV_CSS + '<nav class="x0nav">' + "".join(rows) + "</nav>"

# --- keeping previous runs -------------------------------------------------
# A run writes into a FIXED path (outputs/<source>/layer_NN/) because the published site
# links to it by name -- timestamping that directory would 404 every page. So a
# rerun would otherwise replace the previous run's numbers with no trace. This
# takes a dated copy first. Copy, not move, so the site stays whole even if the
# run dies.
#
# Tracked, not gitignored, on purpose. A superseded run is evidence: when a number
# changes, the version it replaced has to stay citable rather than sit in one
# person's working copy. outputs_local/ was the old home and is ignored, so an
# archive there vanished on a fresh clone -- the one case you need it.
ARCHIVE_DIR = HERE / "outputs_archive"

# Never archived. Two different reasons, both worth stating.
#
# exp0_stats.pt / token_cache / figures: too big to keep per run. The ~700 MB
# cache is on the Hub and the token cache is rebuildable from it.
#
# feature_labels.json / npedia_labels_cache.json: not run output at all. They are
# Neuronpedia's labels for this layer's SAE, a property of the dictionary, and
# nothing we compute touches them -- so an archived copy is byte-identical to the
# live one and archives nothing. Left in, they were 9.8 MB of the archive's 11 MB.
# Safe to drop because the dashboards inline their labels at generation time; no
# archived page reads these files when it is opened.
ARCHIVE_SKIP = ("*.pt", "token_cache", "figures",
                "feature_labels.json", "npedia_labels_cache.json")


def archive_run_dir(stamp: str) -> "Path | None":
    """Copy this layer's current artifacts to ARCHIVE_DIR/layer_NN__<stamp>/.

    `stamp` is passed in rather than read from the clock here, so the caller
    decides the run's identity and every stage of one run can share it.
    Returns the archive path, or None when there is nothing to keep yet.
    """
    import shutil

    if not RUN_DIR.exists() or not any(RUN_DIR.iterdir()):
        return None
    dest = ARCHIVE_DIR / f"{RUN_DIR.name}__{stamp}"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, dest, ignore=shutil.ignore_patterns(*ARCHIVE_SKIP))
    return dest


EXP0_STATS_PATH = RUN_DIR / "exp0_stats.pt"          # written by collect_statistics.py
METRICS_JSON_PATH = RUN_DIR / "metrics_report.json"  # written by run_metrics.py
METRICS_MD_PATH = RUN_DIR / "metrics_report.md"      # written by run_metrics.py

# The ~700 MB/layer stats caches are too big for git, so they are gitignored and
# hosted on the Hub instead. A fresh clone has no exp0_stats.pt, so every consumer
# points at both ways to get one: download it, or recompute it.
HF_STATS_DATASET = "soar-eleuther-i6-hierarchy/experiment_0-stats"

# The source this site is generated from. Absolute, so it is the one nav link
# whose depth does not matter, and the only one that leaves the site.
REPO_URL = "https://github.com/soar-eleuther-i6-hierarchy/metrics"


def missing_stats_msg() -> str:
    """Error text for the metric scripts when exp0_stats.pt isn't there yet."""
    return (
        f"missing {EXP0_STATS_PATH}\n"
        f"  download it:  hf download {HF_STATS_DATASET} --repo-type dataset "
        f'--include "{RUN_DIR.name}/*" --local-dir {RUN_DIR.parent}/\n'
        f"  or rebuild it: EXP0_LAYER={LAYER} python3 collect_statistics.py"
    )

# Token-level caches (written by collect_statistics.py when CACHE_RESIDUALS):
# fp16 residuals + sparse latents let run_token_metrics.py train S_res probes and
# compute parent-conditioned sibling stats WITHOUT re-running the model.
CACHE_RESIDUALS = os.environ.get("EXP0_CACHE_RESIDUALS", "1") != "0"
TOKEN_CACHE_DIR = RUN_DIR / "token_cache"
SECOND_PASS_PATH = RUN_DIR / "second_pass.json"      # model-free token-cache pass (S_res + parent-conditioned siblings); written by run_token_metrics.py
IN_BLOCK_PATH = RUN_DIR / "in_block_edges.json"      # written by in_block_edges.py

# Force a device with the EXP0_DEVICE env var or a script's --device flag:
#   local Mac      -> auto-picks "mps"
#   server (A40)   -> auto-picks "cuda"; pin your assigned GPU with either
#                     CUDA_VISIBLE_DEVICES=<n> (recommended) or EXP0_DEVICE=cuda:<n>
DEVICE_OVERRIDE: str | None = os.environ.get("EXP0_DEVICE")


def pick_device() -> str:
    import torch

    if DEVICE_OVERRIDE:
        if DEVICE_OVERRIDE.startswith("mps"):
            os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return DEVICE_OVERRIDE
    if torch.backends.mps.is_available():
        os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_mps(device: str) -> bool:
    """True for 'mps' (and any 'mps:0' form). MPS has no float64, so accumulators
    fall back to float32 there; CUDA/CPU keep float64."""
    return str(device).startswith("mps")


NEURONPEDIA_BASE = f"https://www.neuronpedia.org/gemma-2-2b/{SAE_SOURCE}"
NEURONPEDIA_API = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{SAE_SOURCE}/{{}}"

# Bulk autointerp explanation export on S3 (one gzipped JSONL batch / 128 feats).
S3_EXPLANATIONS = (
    "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
    f"v1/gemma-2-2b/{SAE_SOURCE}/explanations/batch-{{}}.jsonl.gz"
)


def npedia_url(feature_idx: int) -> str:
    return f"{NEURONPEDIA_BASE}/{int(feature_idx)}"


# ---------------------------------------------------------------------------
# Feature labels (autointerp descriptions), one per feature.
# Written by fetch_labels.py from the Neuronpedia dataset export; a dict
# {index_str: description}. A handful of features have no export description
# and simply fall back to "feature <idx>".
# ---------------------------------------------------------------------------
FEATURE_LABELS_PATH = RUN_DIR / "feature_labels.json"


def load_feature_labels() -> dict[str, str]:
    """Return {index_str: description}, or {} if fetch_labels.py hasn't run."""
    import json

    if FEATURE_LABELS_PATH.exists():
        return json.loads(FEATURE_LABELS_PATH.read_text())
    return {}


def feature_label(feature_idx: int, labels: dict[str, str] | None = None) -> str:
    """Human-readable label for a global feature index, with a graceful
    fallback for the ~26 features that have no export description."""
    if labels:
        text = labels.get(str(int(feature_idx)))
        if text:
            return text
    return f"feature {int(feature_idx)}"
