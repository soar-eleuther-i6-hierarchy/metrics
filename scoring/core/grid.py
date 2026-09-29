"""The labeled pair frame over recovered latents, and detector-by-class AUROC grids.

AUROCs keep the frozen orientation from `scoring.core.registry` (never flipped), with logit-scale
CIs, a by-feature cluster bootstrap, two controls that should sit at ~0.5, and across-seed pooling.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

from scoring.core.detectors import DetectorInputs
from scoring.core.registry import (
    CONSTANTS,
    POSITIVE_LABEL,
    SCORED_COLUMNS,
    SYMMETRIC_DETECTORS,
)
from toygen import labels

DT = torch.float64

# Offset from the training seed, so the held-out draw is not the one the SAE trained on.
HELD_OUT_SEED_OFFSET = 10_000


# --- recovered latents and the pair frame ---
def reduce_to_recovered(held_acts: torch.Tensor, oriented: torch.Tensor, raw: torch.Tensor,
                        match: torch.Tensor, recovered: torch.Tensor,
                        h: torch.Tensor | None = None, b_dec: torch.Tensor | None = None,
                        tokens: torch.Tensor | None = None, vocab: int = 0
                        ) -> tuple[list[int], DetectorInputs, dict[int, int]]:
    """`(feats, inputs, index_map)` for the recovered features that have a match (>= 0).

    `inputs` is indexed by position k in `feats`; `index_map` maps a feature id to k.
    """
    feats = [int(f) for f in range(int(match.shape[0]))
             if bool(recovered[f]) and int(match[f]) >= 0]
    idx = torch.tensor([int(match[f]) for f in feats], dtype=torch.long)
    di = DetectorInputs(
        acts_rec=held_acts[:, idx], W_unit=oriented[idx], W_raw=raw[idx],
        h=h, b_dec=b_dec, tokens=tokens, vocab=vocab,
    )
    return feats, di, {f: k for k, f in enumerate(feats)}


def pair_frame(recovered_feats: list[int], pair_labels: torch.Tensor
               ) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """All ordered off-diagonal pairs, as positions into `recovered_feats`, with their class
    index from `pair_labels`."""
    R = len(recovered_feats)
    pairs, ys = [], []
    for a in range(R):
        for b in range(R):
            if a == b:
                continue
            pairs.append((a, b))
            ys.append(int(pair_labels[recovered_feats[a], recovered_feats[b]]))
    return pairs, torch.tensor(ys, dtype=torch.long)


def class_members(y_label: torch.Tensor, name: str) -> torch.Tensor:
    """Boolean mask over pairs: which ones carry the `name` class."""
    return y_label == labels._index(name)


def split_scores(vals: torch.Tensor, y_label: torch.Tensor,
                 pos_name: str, col_name: str
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair scores split into the `pos_name` and `col_name` populations of one cell."""
    return vals[class_members(y_label, pos_name)], vals[class_members(y_label, col_name)]


# --- AUROC and logit CI ---
def _tie_averaged_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks with tied values sharing their mean rank (scipy `rankdata` semantics)."""
    order = x.argsort()
    xs = x[order]
    _, inv, counts = torch.unique(xs, return_inverse=True, return_counts=True)
    csum = torch.cumsum(counts, 0).double()
    starts = csum - counts.double()
    group_mean = (starts + 1.0 + csum) / 2.0            # mean of (start+1 .. csum)
    ranks = torch.empty(x.numel(), dtype=DT)
    ranks[order] = group_mean[inv]
    return ranks


def auroc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    """AUROC of positives vs negatives via Mann-Whitney U with tie-averaged ranks (matches sklearn).

    Never flipped: below 0.5 is reported as-is. NaN scores are dropped; NaN if a side is then empty.
    """
    pos = scores_pos[~torch.isnan(scores_pos)].double()
    neg = scores_neg[~torch.isnan(scores_neg)].double()
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _tie_averaged_ranks(torch.cat([pos, neg]))
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def logit_ci(auroc_value: float, n_pos: int, n_neg: int, clamp: float,
             z: float = 1.96) -> tuple[float, float]:
    """95% AUROC CI on the logit scale (Hanley-McNeil SE), mapped back to (0, 1).

    The AUROC is clamped to [clamp, 1 - clamp] so the logit stays finite at perfect separation.
    """
    if not math.isfinite(auroc_value) or n_pos <= 0 or n_neg <= 0:
        return (float("nan"), float("nan"))
    a = min(max(auroc_value, clamp), 1.0 - clamp)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    logit = math.log(a / (1.0 - a))
    se_logit = se / (a * (1.0 - a))
    lo = 1.0 / (1.0 + math.exp(-(logit - z * se_logit)))
    hi = 1.0 / (1.0 + math.exp(-(logit + z * se_logit)))
    return (lo, hi)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _split_scores(mat: torch.Tensor, pairs: list[tuple[int, int]], y_label: torch.Tensor,
                  pos_name: str, col_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    vals = mat[pa, pb].double()
    return split_scores(vals, y_label, pos_name, col_name)


def auroc_matrix(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                 y_label: torch.Tensor, columns: tuple[str, ...]) -> dict[str, dict[str, dict]]:
    """AUROC(is_a vs column) for every detector-by-column cell, with counts and a logit CI.

    The per-cell CI assumes independent pairs, so it is too narrow for pairs that share
    features; use `aggregate_seeds` or `cluster_bootstrap_auroc` for an interval to report.
    """
    out: dict[str, dict[str, dict]] = {}
    for det, mat in detectors.items():
        symmetric = det in SYMMETRIC_DETECTORS
        n_pos_ci_hoist = _effective_n(mat, pairs, y_label, POSITIVE_LABEL) if symmetric else None
        out[det] = {}
        for col in columns:
            pos, neg = _split_scores(mat, pairs, y_label, POSITIVE_LABEL, col)
            n_dropped = int(torch.isnan(pos).sum() + torch.isnan(neg).sum())
            a = auroc(pos, neg)
            n_pos = int((~torch.isnan(pos)).sum())
            n_neg = int((~torch.isnan(neg)).sum())
            # a symmetric detector scores (a,b) and (b,a) alike, so its CI uses the unordered count
            if symmetric:
                n_pos_ci = n_pos_ci_hoist
                n_neg_ci = _effective_n(mat, pairs, y_label, col)
            else:
                n_pos_ci, n_neg_ci = n_pos, n_neg
            # clamp to [1/(2n), 1-1/(2n)], n the smaller class, so a saturated cell's CI tracks n
            n_ci = (max(1, min(int(n_pos_ci), int(n_neg_ci)))
                    if (n_pos_ci is not None and n_neg_ci is not None) else 1)
            cell_clamp = max(CONSTANTS["auroc_clamp"], 1.0 / (2.0 * n_ci))
            lo, hi = logit_ci(a, n_pos_ci, n_neg_ci, cell_clamp)
            out[det][col] = {"auroc": a, "n_pos": n_pos, "n_neg": n_neg,
                             "n_pos_ci": n_pos_ci, "n_neg_ci": n_neg_ci,
                             "ci_lo": lo, "ci_hi": hi, "n_dropped": n_dropped}
    return out


def _effective_n(mat: torch.Tensor, pairs: list[tuple[int, int]], y_label: torch.Tensor,
                 name: str) -> int:
    """Unordered count of finite-scored pairs in class `name`: the CI n for a symmetric detector."""
    member = class_members(y_label, name).tolist()
    seen = set()
    for (a, b), inside in zip(pairs, member):
        if inside and not math.isnan(float(mat[a, b])):
            seen.add(frozenset((a, b)))
    return len(seen)


def _effective_n_mask(mat: torch.Tensor, pairs: list[tuple[int, int]],
                      mask: torch.Tensor) -> int:
    """`_effective_n` over a boolean pair mask, for the rest-of-pairs negative class."""
    seen = set()
    for (a, b), inside in zip(pairs, mask.tolist()):
        if inside and not math.isnan(float(mat[a, b])):
            seen.add(frozenset((a, b)))
    return len(seen)


def property_vs_rest_grid(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                          y_label: torch.Tensor, columns: tuple[str, ...],
                          label_masks: dict[str, torch.Tensor] | None = None
                          ) -> dict[str, dict[str, dict]]:
    """AUROC(column vs every other pair) for every detector-by-column cell, never flipped.

    Positives come from `y_label`, or from `label_masks[col]` for latent-side columns
    (`absorbed`, `merged`). An empty side gives a NaN cell and one warning, which is expected.
    """
    label_masks = label_masks or {}
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    col_masks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for col in columns:
        if col in label_masks:
            pos_mask = label_masks[col].to(torch.bool)
        else:
            pos_mask = class_members(y_label, col)
        neg_mask = ~pos_mask                         # everything else (pairs are all off-diagonal)
        if int(pos_mask.sum()) == 0:
            logger.warning("property_vs_rest_grid: column %r has 0 positive pairs — NaN cell", col)
        elif int(neg_mask.sum()) == 0:
            logger.warning("property_vs_rest_grid: column %r has 0 negative pairs — NaN cell", col)
        col_masks[col] = (pos_mask, neg_mask)

    out: dict[str, dict[str, dict]] = {}
    for det, mat in detectors.items():
        symmetric = det in SYMMETRIC_DETECTORS
        vals_all = mat[pa, pb].double()
        out[det] = {}
        for col in columns:
            pos_mask, neg_mask = col_masks[col]
            pos, neg = vals_all[pos_mask], vals_all[neg_mask]
            n_dropped = int(torch.isnan(pos).sum() + torch.isnan(neg).sum())
            a = auroc(pos, neg)
            n_pos = int((~torch.isnan(pos)).sum())
            n_neg = int((~torch.isnan(neg)).sum())
            # unordered counts for a symmetric detector, as in auroc_matrix
            if symmetric:
                n_pos_ci = _effective_n_mask(mat, pairs, pos_mask)
                n_neg_ci = _effective_n_mask(mat, pairs, neg_mask)
            else:
                n_pos_ci, n_neg_ci = n_pos, n_neg
            # same n-aware clamp as auroc_matrix
            n_ci = (max(1, min(int(n_pos_ci), int(n_neg_ci)))
                    if (n_pos_ci is not None and n_neg_ci is not None) else 1)
            cell_clamp = max(CONSTANTS["auroc_clamp"], 1.0 / (2.0 * n_ci))
            lo, hi = logit_ci(a, n_pos_ci, n_neg_ci, cell_clamp)
            out[det][col] = {"auroc": a, "n_pos": n_pos, "n_neg": n_neg,
                             "n_pos_ci": n_pos_ci, "n_neg_ci": n_neg_ci,
                             "ci_lo": lo, "ci_hi": hi, "n_dropped": n_dropped}
    return out


# --- within-seed cluster bootstrap by feature ---
def cluster_bootstrap_auroc(detector_scalar: torch.Tensor, recovered_feats: list[int],
                            y_label_fn, column: str, n_boot: int = 200,
                            rng_seed: int = 0) -> tuple[float, float]:
    """Percentile CI for one cell by resampling features with replacement, never pairs.

    Pairs are formed within the sampled multiset, so a feature drawn twice counts twice.
    `y_label_fn(a, b)` gives the class name of a position pair. (nan, nan) for R < 2.
    """
    R = len(recovered_feats)
    if R < 2:
        return (float("nan"), float("nan"))
    # label code, filled lazily: -2 unfilled, 1 is_a, 0 the column, -1 other (ignored)
    lab = torch.full((R, R), -2, dtype=torch.long)

    def _fill(uniq: list[int]) -> None:
        for a in uniq:
            for b in uniq:
                if a != b and int(lab[a, b]) == -2:
                    nm = y_label_fn(a, b)
                    lab[a, b] = 1 if nm == POSITIVE_LABEL else (0 if nm == column else -1)

    g = torch.Generator().manual_seed(int(rng_seed))
    aurocs: list[float] = []
    for _ in range(n_boot):
        sample = torch.randint(0, R, (R,), generator=g)          # with replacement
        _fill(torch.unique(sample).tolist())
        sub = detector_scalar[sample][:, sample]                 # [R,R] multiset values
        sub_lab = lab[sample][:, sample]
        finite = ~torch.isnan(sub)                               # drops same-feature cells
        pos = sub[(sub_lab == 1) & finite]
        neg = sub[(sub_lab == 0) & finite]
        if pos.numel() and neg.numel():
            a = auroc(pos, neg)
            if math.isfinite(a):
                aurocs.append(a)
    if not aurocs:
        return (float("nan"), float("nan"))
    t = torch.tensor(aurocs, dtype=DT)
    return (float(torch.quantile(t, 0.025)), float(torch.quantile(t, 0.975)))


# --- label-free percentiles ---
def component_percentiles(detectors: dict[str, torch.Tensor],
                          pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Per-detector percentile of each pair among all pairs: the fraction of finite scores <=
    its own. NaN where its own score is NaN."""
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    out: dict[str, torch.Tensor] = {}
    for name, mat in detectors.items():
        vec = mat[pa, pb].double()                       # [n_pairs]
        finite = ~torch.isnan(vec)
        pct = torch.full_like(vec, float("nan"))
        fv = vec[finite]
        if fv.numel():
            sorted_fv, _ = torch.sort(fv)
            pct[finite] = torch.searchsorted(sorted_fv, fv, right=True).double() / fv.numel()
        out[name] = pct
    return out


# --- controls (should sit at AUROC ~0.5) ---
def random_scalar_control(pairs: list[tuple[int, int]], y_label: torch.Tensor,
                          column: str, rng: torch.Generator) -> float:
    """AUROC of a random per-pair scalar; should be ~0.5."""
    scores = torch.rand(len(pairs), generator=rng, dtype=DT)
    pos, neg = split_scores(scores, y_label, POSITIVE_LABEL, column)
    return auroc(pos, neg)


def shuffled_label_control(detector_scalar: torch.Tensor, pairs: list[tuple[int, int]],
                           y_label: torch.Tensor, column: str, rng: torch.Generator) -> float:
    """Detector AUROC after permuting the labels over the same pairs; should be ~0.5, or the
    pipeline is drawing signal from the label map."""
    perm = torch.randperm(y_label.numel(), generator=rng)
    shuffled = y_label[perm]
    vals = torch.tensor([float(detector_scalar[a, b]) for (a, b) in pairs], dtype=DT)
    pos, neg = split_scores(vals, shuffled, POSITIVE_LABEL, column)
    return auroc(pos, neg)


# --- held-out seed, dispersion readout, redundancy ---
def held_out_sample_seed(train_seed: int, offset: int = HELD_OUT_SEED_OFFSET) -> int:
    """The sampling seed of the held-out draw; a zero offset is refused, since it reuses the
    training draw."""
    if offset == 0:
        raise ValueError("held_out offset must be non-zero: a 0 offset reuses the training draw")
    return int(train_seed) + int(offset)


def dispersion_split(isa_pairs: list[tuple[int, int]], r_disp: dict[int, float],
                     detector_scalar: torch.Tensor,
                     neg_pairs: list[tuple[int, int]] | None = None) -> dict:
    """Split the is-a pairs at the median child `r_disp` and summarize each half.

    `r_disp` reads the true directions, so this is a readout diagnostic, not a detector. Each
    half is its AUROC against `neg_pairs` if given, else its mean detector score.
    """
    disps = torch.tensor([r_disp[c] for _, c in isa_pairs], dtype=DT)
    median = float(disps.median())

    def _summary(half: list[tuple[int, int]]) -> float:
        if not half:
            return float("nan")
        if neg_pairs is not None:
            pos = torch.tensor([float(detector_scalar[a, b]) for a, b in half], dtype=DT)
            neg = torch.tensor([float(detector_scalar[a, b]) for a, b in neg_pairs], dtype=DT)
            return auroc(pos, neg)
        vals = torch.tensor([float(detector_scalar[a, b]) for a, b in half], dtype=DT)
        vals = vals[~torch.isnan(vals)]
        return float(vals.mean()) if vals.numel() else float("nan")

    high = [(p, c) for (p, c) in isa_pairs if r_disp[c] >= median]
    low = [(p, c) for (p, c) in isa_pairs if r_disp[c] < median]
    return {"high": _summary(high), "low": _summary(low), "median": median,
            "n_high": len(high), "n_low": len(low)}


def redundancy_map(detectors: dict[str, torch.Tensor], pairs: list[tuple[int, int]],
                   y_label: torch.Tensor, columns: tuple[str, ...],
                   grid: dict | None = None) -> dict:
    """Pairwise detector rank correlation, and each detector's AUROC per column minus the best
    other detector's (NaN when there is nothing to compare). `grid` reuses an AUROC matrix."""
    names = list(detectors)
    comps = component_percentiles(detectors, pairs)
    rank_corr: dict[str, dict[str, float]] = {}
    for a in names:
        rank_corr[a] = {}
        for b in names:
            va, vb = comps[a], comps[b]
            m = (~torch.isnan(va)) & (~torch.isnan(vb))
            rank_corr[a][b] = float(_spearman(va[m], vb[m])) if int(m.sum()) > 2 else float("nan")

    grid = auroc_matrix(detectors, pairs, y_label, columns) if grid is None else grid
    marginal: dict[str, dict[str, float]] = {}
    for det in names:
        marginal[det] = {}
        for col in columns:
            others = [grid[o][col]["auroc"] for o in names if o != det
                      and math.isfinite(grid[o][col]["auroc"])]
            a = grid[det][col]["auroc"]
            if not others or not math.isfinite(a):
                marginal[det][col] = float("nan")
            else:
                marginal[det][col] = a - max(others)
    return {"rank_corr": rank_corr, "marginal_auroc": marginal}


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman correlation with tie-averaged ranks (matches scipy); NaN on zero variance or NaN.

    Position-broken ties (`argsort().argsort()`) would fabricate correlation between the
    tie-heavy per-parent broadcasts.
    """
    if a.numel() < 2 or torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan")
    ra = _tie_averaged_ranks(a)
    rb = _tie_averaged_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    da, db = ra.norm(), rb.norm()
    if float(da) == 0.0 or float(db) == 0.0:
        return float("nan")
    return float((ra @ rb) / (da * db))


# --- across-seed aggregation ---
# Student-t 0.975 quantiles by df, hardcoded to avoid scipy.
_T_CRIT_975: dict[int, float] = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199,
    17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
    25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
}


def _t_crit_975(n_seeds: int) -> float:
    """Student-t 0.975 quantile for df = n_seeds - 1; 1.96 when df < 1 or df > 30."""
    return _T_CRIT_975.get(int(n_seeds) - 1, 1.96)


def aggregate_seeds(reports: list[dict]) -> dict:
    """Across-seed AUROC per cell: mean on the logit scale with a Student-t CI (df = n_seeds - 1).

    Refuses duplicate train seeds and reports that differ in (config, variant, k).
    """
    if not reports:
        return {}
    # a duplicate seed would count as independent and tighten the CI; seedless reports pass
    seeds = [r["meta"]["train_seed"] for r in reports
             if isinstance(r.get("meta"), dict) and r["meta"].get("train_seed") is not None]
    if len(set(seeds)) < len(seeds):
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        raise ValueError(f"aggregate_seeds: non-distinct train_seed(s) {dupes} in the report set — "
                         "seeds must be distinct (a checkpoint_dirname collision or a duplicate report)")
    # reports without meta pass this check
    sigs = {(m.get("config"), m.get("variant"), m.get("k"))
            for r in reports if isinstance((m := r.get("meta")), dict)}
    if len(sigs) > 1:
        raise ValueError(f"aggregate_seeds: reports disagree on (config, variant, k) {sorted(sigs)} — "
                         "a --out directory mixing two experiments; aggregate each run separately")
    dets = list(reports[0]["grid"].keys())
    cols = list(reports[0]["grid"][dets[0]].keys())
    agg: dict[str, dict[str, dict]] = {}
    clamp = CONSTANTS["auroc_clamp"]
    for det in dets:
        agg[det] = {}
        for col in cols:
            cells = [r["grid"][det][col] for r in reports
                     if math.isfinite(r["grid"][det][col]["auroc"])]
            if not cells:
                agg[det][col] = {"mean": float("nan"), "ci_lo": float("nan"),
                                 "ci_hi": float("nan"), "median": float("nan"), "n_seeds": 0}
                continue
            vals = [c["auroc"] for c in cells]
            # n-aware clamp per seed, so one saturated seed cannot swamp the mean
            logits = []
            for c in cells:
                # prefer the CI n; never default n to 1, which would clamp the AUROC to 0.5
                np_ = c.get("n_pos_ci", c.get("n_pos"))
                nn_ = c.get("n_neg_ci", c.get("n_neg"))
                if np_ is None or nn_ is None:
                    cl = clamp
                else:
                    n = max(1, min(int(np_), int(nn_)))
                    cl = max(clamp, 1.0 / (2.0 * n))
                a = min(max(c["auroc"], cl), 1.0 - cl)
                logits.append(math.log(a / (1.0 - a)))
            mean_l = sum(logits) / len(logits)
            sd = (sum((x - mean_l) ** 2 for x in logits) / max(len(logits) - 1, 1)) ** 0.5
            se = sd / math.sqrt(len(logits))
            t_crit = _t_crit_975(len(logits))
            svals = sorted(vals)
            median = svals[len(svals) // 2] if len(svals) % 2 else \
                0.5 * (svals[len(svals) // 2 - 1] + svals[len(svals) // 2])
            if sd == 0.0 and len(logits) > 1:
                # zero spread (e.g. every seed saturated): per-seed CI envelope, not a zero-width CI
                lo_env = min((c["ci_lo"] for c in cells if c.get("ci_lo") is not None),
                             default=_sigmoid(mean_l))
                hi_env = max((c["ci_hi"] for c in cells if c.get("ci_hi") is not None),
                             default=_sigmoid(mean_l))
                ci_lo, ci_hi = lo_env, hi_env
            else:
                ci_lo, ci_hi = _sigmoid(mean_l - t_crit * se), _sigmoid(mean_l + t_crit * se)
            agg[det][col] = {"mean": _sigmoid(mean_l), "ci_lo": ci_lo, "ci_hi": ci_hi,
                             "median": median, "n_seeds": len(vals)}
    return agg
