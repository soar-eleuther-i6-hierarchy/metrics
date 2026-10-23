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
FREQ_MIN_FIRE_LOW = _METRIC.freq_min_fire_low   # survival floor on the child's firing
# An edge is "frequency-driven" when its reverse coverage on low+mid tokens
# drops below this fraction of its all-token reverse coverage.
FREQ_SURVIVAL_MIN = _GATES.freq_survival_min

# ---------------------------------------------------------------------------
# Paths + device
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# EXP0_OUT redirects ALL outputs to a directory of your choice, so a scratch run
# never touches the git-tracked outputs/. Default: unchanged.
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


def scope_line(total_tokens=None, bold=("**", "**"), sep="　·　", n_docs=None, config=None):
    """Which layer, and the knobs a reader needs to interpret the numbers.

    Single source for the context line shown on every report, so the reports
    can never drift apart. `bold` selects the
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


# --- gemma layers -----------------------------------------------------------
# The gemma-2-2b layers graded with the Matryoshka SAE. The reporting scripts
# iterate over this list.
NAV_LAYERS = [1, 3, 6, 12, 18, 24]

# --- keeping previous runs -------------------------------------------------
# A run writes into a FIXED path (outputs/<source>/layer_NN/) because the reports and
# figures find it by name. So a
# rerun would otherwise replace the previous run's numbers with no trace. This
# takes a dated copy first. Copy, not move, so the outputs stay whole even if the
# run dies.
#
# Gitignored since 23 Oct 2026: it is a local safety copy, not a result. Old
# versions of tracked outputs are in git history (and on the `backup` branch).
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
