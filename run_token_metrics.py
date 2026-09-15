"""
The second pass: a model-free sweep over the token cache for the metrics that
need per-token detail the count matrices cannot provide. Nothing here touches
the model again — it all runs off the token cache (fp16 residuals + sparse
latents) written by collect_statistics.py:

    1. S_res, probe-based and RANK-scored (metrics/sres.py) on the shortlist
       of edges that survive coverage + the independence null. Self-labeled
       probes — see the circularity caveat in metrics/sres.py.
    2. Parent-conditioned sibling redundancy for flagged superparents
       (Jaccard restricted to the parent's firing set).
    3. Exact joint-child union over the KEPT children only (collect_statistics
       streams the all-children union; the kept-children one depends on the edge
       set, which exists only after run_metrics).

Needs:  outputs/layer_NN/token_cache/ (collect_statistics) + run_metrics's edge set
Writes: outputs/layer_NN/second_pass.json  (+ "second_pass" key merged into
        metrics_report.json when present)
Run:    python3 run_token_metrics.py                # all pairs
        python3 run_token_metrics.py --pairs 0->1   # subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import config as C
from run_metrics import source_structure
from utils import sae_utils as U
from utils.io import TokenCache
from metrics import (
    coverage_legs,
    find_superparents,
    independence_scores,
    keep_edges,
    parent_conditioned_redundancy,
    sres_rank_check,
    train_probe,
)


# ---------------------------------------------------------------------------
# Per-pair work
# ---------------------------------------------------------------------------
def blocks(stats, *blks):
    """(start, end) per requested block, read off the file being graded.

    Same rule as run_metrics and the dashboards: config.BLOCK_RANGES is gemma's
    32768 in 5 blocks, and a PCFG cache (1792 in 8) sliced there yields probes
    trained on the wrong features -- with no error, because every index is in
    range for the first four blocks and the numbers stay plausible.
    """
    # This path slices by (start, end) RANGES. A non-contiguous source declares
    # config.block_indices (run_metrics honors it via block_selector); here the range slices +
    # `p0 + local` global ids would silently address the WRONG features (every index is in range,
    # numbers stay plausible, no error). Refuse rather than emit silently-wrong second-pass metrics.
    if ((stats.get("config") or {}).get("block_indices")) is not None:
        raise ValueError(
            "run_token_metrics does not support non-contiguous config.block_indices: its range-slice "
            "path (rebuild_edges / sres_for_pair / conditioned redundancy / kept-union) would compute "
            "on the WRONG feature ids with no error. Use run_metrics (first pass honors block_indices "
            "via block_selector), or index this path by block_indices before trusting it.")
    ranges, _ = source_structure(stats)
    return [ranges[b] for b in blks]


def rebuild_edges(stats, p_blk, c_blk):
    """Same edge set + shortlist as run_metrics (single source: metrics/)."""
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    (p0, p1), (c0, c1) = blocks(stats, p_blk, c_blk)
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]
    cofire = stats["cofire"][key].double()
    R, _ = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                           cofire=cofire, min_joint=C.MIN_JOINT)
    null = independence_scores(cofire, fire_p, fire_c, total, C.MIN_JOINT)
    shortlist = edge_mask & (null["pmi"] > 0.0)
    return edge_mask, shortlist, fire_p, fire_c, R


def sres_for_pair(cache, W_dec, stats, p_blk, c_blk, device):
    """Probe-based S_res over the shortlist. One probe per unique child."""
    _, shortlist, _, fire_c, _ = rebuild_edges(stats, p_blk, c_blk)
    (p0, _), (c0, _) = blocks(stats, p_blk, c_blk)

    child_locals = torch.nonzero(shortlist.any(dim=0)).flatten()
    child_locals = child_locals[fire_c[child_locals] >= C.MIN_PROBE_POS]
    if child_locals.numel() > C.SRES_MAX_CHILDREN_PER_PAIR:
        print(f"[03] {p_blk}->{c_blk}: capping probes to "
              f"{C.SRES_MAX_CHILDREN_PER_PAIR}/{child_locals.numel()} children (by fire count)")
        top = torch.argsort(fire_c[child_locals], descending=True)
        child_locals = child_locals[top[: C.SRES_MAX_CHILDREN_PER_PAIR]]

    resid = cache.resid.to(device)
    Wd = W_dec.to(device)
    results, n_pass, n_untestable_children = [], 0, 0
    for i, cl in enumerate(child_locals.tolist()):
        gc = c0 + cl
        pos = cache.feature_mask(gc).to(device)
        probe = train_probe(resid, pos, seed=gc,
                            neg_ratio=C.SRES_NEG_RATIO,
                            max_tokens=C.SRES_MAX_PROBE_TOKENS,
                            min_neg=C.SRES_MIN_NEG)
        if probe is None:                                 # too few negatives -> untestable
            n_untestable_children += 1
            continue
        corr = (probe @ Wd.T).cpu()                       # [D_SAE]
        for pl in torch.nonzero(shortlist[:, cl]).flatten().tolist():
            gp = p0 + pl
            ok, detail = sres_rank_check(corr, gp, gc, C.SRES_RANK_TOP_K)
            n_pass += int(ok)
            results.append({"parent": gp, "child": gc, "pass": ok, **detail})
        if (i + 1) % 50 == 0:
            print(f"[03]   {p_blk}->{c_blk}: {i + 1}/{child_locals.numel()} probes")
    return {
        "n_shortlist_edges": int(shortlist.sum()),
        "n_probed_children": int(child_locals.numel()),
        "n_untestable_children": n_untestable_children,   # skipped: < SRES_MIN_NEG negatives
        "n_edges_scored": len(results),
        "n_pass": n_pass,
        "frac_pass": n_pass / len(results) if results else 0.0,
        "edges": results,
    }


def conditioned_redundancy_for_pair(cache, stats, p_blk, c_blk):
    """Parent-conditioned sibling Jaccard for this pair's flagged superparents."""
    edge_mask, _, fire_p, fire_c, _ = rebuild_edges(stats, p_blk, c_blk)
    (p0, _), (c0, _) = blocks(stats, p_blk, c_blk)
    total = int(stats["total_tokens"])
    sps = find_superparents(edge_mask, fire_p, total,
                            C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC)
    out = []
    for sp in sps:
        pl = sp["parent_local"]
        kids = torch.nonzero(edge_mask[pl]).flatten()
        if kids.numel() > 512:                            # cap children per parent so the pairwise Jaccard stays cheap
            top = torch.argsort(fire_c[kids], descending=True)[:512]
            kids = kids[top]
        fires_p = cache.feature_mask(p0 + pl)
        kid_masks = torch.stack([cache.feature_mask(c0 + int(k)) for k in kids], dim=1)
        red = parent_conditioned_redundancy(fires_p, kid_masks)
        out.append({**sp, "parent_global": p0 + pl,
                    "n_kids_scored": int(kids.numel()),
                    "conditioned_redundancy": red})
    return out


def kept_union_for_pair(cache, stats, p_blk, c_blk):
    """Exact R_supp / R_mass over the KEPT children only (chunked scan)."""
    edge_mask, _, fire_p, _, _ = rebuild_edges(stats, p_blk, c_blk)
    key = f"{p_blk}->{c_blk}"
    (p0, p1), (c0, c1) = blocks(stats, p_blk, c_blk)
    K = edge_mask.double()                                # [P, C]
    union_count = torch.zeros(p1 - p0, dtype=torch.float64)
    union_energy = torch.zeros(p1 - p0, dtype=torch.float64)
    child_chunks = cache.chunks_dense(c0, c1)
    parent_chunks = cache.chunks_dense(p0, p1, values=True)
    for (_, fc), (_, ep) in zip(child_chunks, parent_chunks):
        any_kept = (fc @ K.T.float() > 0).double()        # [n, P]
        e = (ep.double() ** 2)                            # [n, P]
        union_count += ((e > 0).double() * any_kept).sum(dim=0)
        union_energy += (e * any_kept).sum(dim=0)
    # denominator from collect_statistics (over ALL tokens) so r_mass_kept is
    # directly comparable with the all-children r_mass in the run_metrics report
    energy_total = stats["energy_total"][key].double()
    has = edge_mask.any(dim=1)
    r_supp_kept = union_count / fire_p.clamp(min=1.0)
    r_mass_kept = union_energy / energy_total.clamp(min=1e-12)
    return {
        "r_supp_kept_mean": float(r_supp_kept[has].mean()) if has.any() else float("nan"),
        "r_mass_kept_mean": float(r_mass_kept[has].mean()) if has.any() else float("nan"),
        "per_parent": [
            {"parent_global": p0 + int(p), "r_supp_kept": float(r_supp_kept[p]),
             "r_mass_kept": float(r_mass_kept[p])}
            for p in torch.nonzero(has).flatten().tolist()
        ],
    }


def load_w_dec(path=None):
    """The decoder [D_SAE, d_model], which turns a probe direction into per-feature
    correlations.

    Defaults to the released gemma SAE, as before. A source with its own
    dictionary leaves a `w_dec.pt` in the run dir (adapters/from_pcfg.py writes
    one) and it is used automatically -- otherwise a PCFG run silently pulls
    gemma's 32768x2304 decoder from the Hub and dies on a shape mismatch inside
    sres_for_pair, several frames from the decision that caused it.
    """
    if path is None:
        default = C.RUN_DIR / "w_dec.pt"
        path = default if default.exists() else None
    if path is None:
        return U.load_sae("cpu").W_dec.detach().float()
    W = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(W, dict):                               # a state dict, or the SAE's own save format
        W = W["W_dec"]
    print(f"[03] decoder: {path}")
    return W.detach().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None, help='e.g. 0->1 1->2')
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-sres", action="store_true")
    ap.add_argument("--w-dec", type=Path, default=None,
                    help="decoder [D_SAE, d_model] for a non-gemma dictionary "
                         "(default: RUN_DIR/w_dec.pt if present, else the released gemma SAE)")
    args = ap.parse_args()

    device = args.device or C.pick_device()
    # The run name, not C.LAYER: under EXP0_RUN the layer constant describes
    # nothing this pass is reading.
    print(f"[03] run = {C.RUN_NAME}  device = {device}")
    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(f"[03] {C.missing_stats_msg()}")
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    if not (C.TOKEN_CACHE_DIR / "meta.json").exists():
        raise SystemExit(f"[03] no token cache at {C.TOKEN_CACHE_DIR} - rerun collect_statistics.py")
    cache = TokenCache(C.TOKEN_CACHE_DIR)
    print(f"[03] token cache: {cache.n_tokens} tokens")

    W_dec = load_w_dec(args.w_dec)                        # [D_SAE, d]
    d_sae = int(stats["fire_count"].numel())
    if W_dec.shape[0] != d_sae:
        raise SystemExit(
            f"[03] decoder has {W_dec.shape[0]} features but the statistics have {d_sae}. "
            "Pass --w-dec pointing at this SAE's decoder; the default is gemma's."
        )

    pairs = stats["pairs"]
    if args.pairs:
        want = set(args.pairs)
        pairs = [pr for pr in pairs if f"{pr[0]}->{pr[1]}" in want]

    from run_metrics import json_safe

    # UPDATE any existing second_pass.json rather than replace it, so a
    # partial --pairs rerun never wipes other pairs' results.
    report = {}
    if C.SECOND_PASS_PATH.exists():
        report = json.loads(C.SECOND_PASS_PATH.read_text())

    for (p, c) in pairs:
        key = f"{p}->{c}"
        print(f"[03] pair {key}")
        entry = {}
        entry["kept_union"] = kept_union_for_pair(cache, stats, p, c)
        entry["superparent_conditioned_redundancy"] = conditioned_redundancy_for_pair(cache, stats, p, c)
        if not args.skip_sres:
            entry["sres"] = sres_for_pair(cache, W_dec, stats, p, c, device)
        report[key] = entry
        # flush after EVERY pair — a crash later never loses finished pairs
        C.SECOND_PASS_PATH.write_text(json.dumps(json_safe(report), indent=2))
        ku = entry["kept_union"]
        print(f"[03]   R_supp_kept mean {ku['r_supp_kept_mean']:.3f} | "
              f"R_mass_kept mean {ku['r_mass_kept_mean']:.3f} | "
              + (f"S_res pass {entry['sres']['n_pass']}/{entry['sres']['n_edges_scored']}"
                 if not args.skip_sres else "S_res skipped"))

    print(f"[03] wrote {C.SECOND_PASS_PATH}")
    if C.METRICS_JSON_PATH.exists():                      # merge for one-stop reading
        full = json.loads(C.METRICS_JSON_PATH.read_text())
        merged = dict(full.get("second_pass") or {})
        merged.update({
            k: {kk: (vv if kk != "sres" else {x: y for x, y in vv.items() if x != "edges"})
                for kk, vv in v.items()}
            for k, v in report.items()
        })
        full["second_pass"] = merged
        C.METRICS_JSON_PATH.write_text(json.dumps(json_safe(full), indent=2))
        print(f"[03] merged summary into {C.METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
