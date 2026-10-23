"""
Run the hierarchy metrics on the cached statistics.

Loads outputs/layer_NN/exp0_stats.pt (from collect_statistics.py) and, for every
adjacent block pair, treats the metrics as competing measurements of the same
candidate edge set:

    1. Activation coverage (reverse R / forward F / joint-child J)  -> edge set
    2. Reconstruction condition (Tree-SAE)  -> does the edge improve recon?
    3. Sibling redundancy                   -> are the children real refinements?
    4. Out-degree distribution              -> superparents / poly-parenting
    5. Token-frequency-controlled coverage  -> does the edge survive on rare tokens?

The edge SET comes from metric 1 (reverse coverage >= EDGE_TAU, both endpoints
firing >= MIN_FIRE_COUNT); metrics 2-5 then grade those edges.

Needs:  outputs/layer_NN/exp0_stats.pt
Writes: outputs/layer_NN/metrics_report.{json,md}
Run:    python3 run_metrics.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import config as C
from utils import sae_utils as U
from metrics import (
    coverage_legs,
    degree_stats,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    independence_scores,
    joint_child_coverage_upper,
    keep_edges,
    r_mass,
    r_supp,
    share_energy,
    sibling_redundancy,
)


def _f(x) -> float:
    return float(x)


def json_safe(obj):
    """Recursively replace NaN/inf floats with None so json.dumps emits valid
    JSON (bare NaN tokens break strict parsers, e.g. JS on the Pages site).
    Applied at serialization time only — markdown rendering happens before."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def _nanmean(t: torch.Tensor) -> float:
    v = t[~torch.isnan(t)]
    return float(v.mean()) if v.numel() else float("nan")


def _clip(s: str, n: int = 40) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def source_structure(stats):
    """Block structure from the stats file itself, falling back to this config.

    The metric *thresholds* (EDGE_TAU, MIN_JOINT, ...) stay global on purpose:
    holding them fixed across sources is the claim the paper makes. But the block
    boundaries belong to whichever SAE produced the file. Reading them from
    config.py meant a PCFG dictionary (1792 latents in 8 blocks) got sliced with
    gemma's ranges (32768 in 5) and still returned numbers -- wrong ones, with
    nothing to signal it.

    Files written before `config` carried these keys fall back to the module,
    which is correct for them: they are all gemma.
    """
    cfg = stats.get("config") or {}
    ranges = cfg.get("block_ranges") or C.BLOCK_RANGES
    ranges = [tuple(r) for r in ranges]
    siblings = cfg.get("sibling_blocks")
    return ranges, (C.SIBLING_BLOCKS if siblings is None else list(siblings))


def block_selector(stats):
    """How to cut block b out of a [D_SAE] vector, for this file.

    Blocks are sets of feature indices; Matryoshka's are contiguous prefixes and
    every other reader can stay a slice. A source whose groups are not contiguous
    declares `config.block_indices`, and it wins over `block_ranges` when present.
    """
    cfg = stats.get("config") or {}
    idx = cfg.get("block_indices")
    if idx is not None:
        sel = [torch.as_tensor(ix, dtype=torch.long) for ix in idx]
        return lambda b: sel[b]
    ranges, _ = source_structure(stats)
    return lambda b: slice(*ranges[b])


def analyse_pair(stats, p_blk, c_blk, labels=None, legacy_guards=False):
    """One block pair through every metric.

    legacy_guards=True reproduces the pre-tranche edge criterion (no
    min_joint) — used only for regression checks against old runs.
    """
    labels = labels or {}
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    block_ranges, sibling_blocks = source_structure(stats)
    sel = block_selector(stats)
    fire_p = fire[sel(p_blk)]
    fire_c = fire[sel(c_blk)]
    # Global feature ids for the report. With explicit indices the block's own list
    # gives them; with ranges it is the offset. `p0` is used for both the superparent
    # rows and the top-edge ids below.
    cfg_idx = (stats.get("config") or {}).get("block_indices")
    if cfg_idx is not None:
        p_ids, c_ids = list(cfg_idx[p_blk]), list(cfg_idx[c_blk])
    else:
        p_lo, c_lo = block_ranges[p_blk][0], block_ranges[c_blk][0]
        p_ids, c_ids = None, None

    def gid_p(i):
        """Local column -> the feature's id in the full dictionary."""
        return p_ids[i] if p_ids is not None else p_lo + i

    def gid_c(i):
        return c_ids[i] if c_ids is not None else c_lo + i

    cofire = stats["cofire"][key].double()

    # --- Metric 1: coverage + edge set --------------------------------------
    # R = P(parent | child) ("reverse"), F = P(child | parent) ("forward");
    # direction-neutral names: Cont(c,p) and Share_freq(c,p).
    R, F = coverage_legs(cofire, fire_p, fire_c)
    min_joint = 0 if legacy_guards else C.MIN_JOINT
    edge_mask = keep_edges(
        R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
        cofire=cofire, min_joint=min_joint,
    )
    # edges the old criterion kept but the joint-support guard kills (reported)
    edge_mask_nojoint = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT)
    n_dropped_min_joint = int((edge_mask_nojoint & ~edge_mask).sum())
    joint_upper = joint_child_coverage_upper(F, edge_mask)  # [P]
    n_edges = int(edge_mask.sum())

    # --- Independence null (PMI / Dev) --------------------------------------
    null = independence_scores(cofire, fire_p, fire_c, total, C.MIN_JOINT)
    pmi = null["pmi"]
    edge_pmi = pmi[edge_mask]
    edge_pmi_ok = edge_pmi[~torch.isnan(edge_pmi)]
    # PMI ~ 0 = co-firing at chance level given base rates (C-freq artifact)
    n_chance_level = int((edge_pmi_ok < 0.5).sum())

    # --- Energy share + exact joint-child (schema v2 caches) ----------------
    joint_child = None
    if "union_count" in stats:
        e_cof = stats["energy_cofire"][key].double()
        e_tot = stats["energy_total"][key].double()
        rs = r_supp(stats["union_count"][key].double(), fire_p)
        rm = r_mass(stats["union_energy"][key].double(), e_tot)
        S = share_energy(e_cof, e_tot)
        has_edges = edge_mask.any(dim=1)
        max_share = S.max(dim=1).values
        joint_child = {                      # JSON-safe scalar summaries only
            "r_supp_mean": _nanmean(rs[has_edges]) if n_edges else float("nan"),
            "r_mass_mean": _nanmean(rm[has_edges]) if n_edges else float("nan"),
            # rename/duplicate candidates: one child holds ~all parent energy
            "n_share_energy_ge_09": int((max_share[has_edges] >= C.SHARE_ENERGY_SPLIT).sum()) if n_edges else 0,
        }

    # --- Metric 4: out-degree / superparents --------------------------------
    deg = degree_stats(edge_mask)
    superparents = find_superparents(
        edge_mask, fire_p, total, C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC
    )

    # --- Metric 2: reconstruction condition ---------------------------------
    recon = edge_reconstruction_condition(
        stats["err_sum_c"][c_blk].double(),
        stats["g_parent_sum"][key].double(),
        stats["g_child_sum"][c_blk].double(),
        C.RECON_REL_GAIN_MIN,
    )
    recon_pass = recon["passes"] & edge_mask
    n_recon_pass = int(recon_pass.sum())

    # --- Metric 5: token-frequency-controlled coverage ----------------------
    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"][key].double(),
        stats["fire_c_by_bucket"][c_blk].double(),
        edge_mask,
        min_fire_low=C.FREQ_MIN_FIRE_LOW,
    )
    survival = fcov["survival"]                               # [P, C] nan off-edge/untestable
    surv_vals = survival[~torch.isnan(survival)]
    n_freq_driven = int((survival < C.FREQ_SURVIVAL_MIN).sum())
    n_testable = int(surv_vals.numel())

    # --- Metric 5b: the same control, bucketed within each window --------------
    # Only present when stage 01 accumulated it. The two differ in nothing but what
    # counts as "frequent": a token id everywhere, versus a position in its own
    # context. On a corpus whose global marginal is flat but whose within-document
    # distribution is not, they can disagree completely — which is the point.
    freq_local = None
    if "cofire_by_local_bucket" in stats:
        lcov = frequency_controlled_coverage(
            stats["cofire_by_local_bucket"][key].double(),
            stats["fire_c_by_local_bucket"][c_blk].double(),
            edge_mask,
            min_fire_low=C.FREQ_MIN_FIRE_LOW,
        )
        lsurv = lcov["survival"]
        lvals = lsurv[~torch.isnan(lsurv)]
        n_local_testable = int(lvals.numel())
        freq_local = {
            "n_testable": n_local_testable,
            "mean_survival": _nanmean(lvals) if n_local_testable else float("nan"),
            "n_freq_driven": int((lsurv < C.FREQ_SURVIVAL_MIN).sum()),
            "frac_freq_driven": _f(int((lsurv < C.FREQ_SURVIVAL_MIN).sum()) / n_local_testable)
            if n_local_testable else 0.0,
        }

    # --- Metric 3: sibling redundancy ---------------------------------------
    # GLOBAL Jaccard only — confounded for Matryoshka: it
    # scores co-firing anywhere, not disjointness within the parent's support.
    # This is a cheap diagnostic, NOT the splitting verdict; the corrected
    # parent-conditioned Jaccard is in the second pass. Reported so
    # both numbers are auditable, but must not be read as "flagged splitting".
    sib = {}
    sib_summary = None
    if c_blk in sibling_blocks:
        sib = sibling_redundancy(edge_mask, stats["within_cofire"][c_blk].double(), fire_c)
        if sib:
            reds = [v["redundancy"] for v in sib.values()]
            sib_summary = {
                "method": "global_jaccard_confounded",   # NOT the verdict — see second_pass
                "n_parents_scored": len(reds),
                "mean_redundancy": _f(sum(reds) / len(reds)),
                "n_over_global_threshold": int(sum(r >= C.SIBLING_REDUNDANCY_FLAG for r in reds)),
            }

    # --- top edges (by reverse coverage) with every metric attached ---------
    top_edges = []
    if n_edges:
        pidx, cidx = torch.nonzero(edge_mask, as_tuple=True)
        order = torch.argsort(R[pidx, cidx], descending=True)[:15]
        for j in order.tolist():
            pi, ci = int(pidx[j]), int(cidx[j])
            gp, gc = gid_p(pi), gid_c(ci)
            s = survival[pi, ci]
            pm = pmi[pi, ci]
            top_edges.append(
                {
                    "parent": gp,
                    "child": gc,
                    "reverse_cov": _f(R[pi, ci]),
                    "forward_cov": _f(F[pi, ci]),
                    "pmi": None if torch.isnan(pm) else _f(pm),
                    "recon_parent_gain": _f(recon["parent_gain"][pi, ci]),
                    "recon_child_gain": _f(recon["child_gain"][ci]),
                    "recon_pass": bool(recon["passes"][pi, ci]),
                    "freq_survival": None if torch.isnan(s) else _f(s),
                    "parent_redundancy": (sib.get(pi, {}) or {}).get("redundancy"),
                    "parent_label": C.feature_label(gp, labels),
                    "child_label": C.feature_label(gc, labels),
                    "parent_url": C.npedia_url(gp),
                    "child_url": C.npedia_url(gc),
                }
            )

    return {
        "pair": key,
        "n_candidate_edges": n_edges,
        # support-guard accounting (never silently drop)
        "n_dropped_min_joint": n_dropped_min_joint,
        "n_pairs_below_min_joint": null["n_excluded"],
        "independence_null": {
            "mean_edge_pmi": _nanmean(edge_pmi_ok) if n_edges else float("nan"),
            "n_chance_level": n_chance_level,           # edges with PMI < 0.5
            "frac_chance_level": _f(n_chance_level / n_edges) if n_edges else 0.0,
        },
        "joint_child": joint_child,
        "degree": {
            "n_parents_with_children": deg["n_parents_with_children"],
            "n_children_with_parent": deg["n_children_with_parent"],
            "n_multi_parented": deg["n_multi_parented"],
            "poly_frac": _f(deg["poly_frac"]),
            "top1_edge_share": _f(deg["top1_edge_share"]),
            "outdeg_gini": _f(deg["outdeg_gini"]),
            "max_outdeg": int(deg["outdeg"].max()) if n_edges else 0,
        },
        "joint_child_cov_mean": _nanmean(joint_upper[edge_mask.any(dim=1)]) if n_edges else float("nan"),
        "reconstruction": {
            "n_pass": n_recon_pass,
            "frac_pass": _f(n_recon_pass / n_edges) if n_edges else 0.0,
        },
        "freq_control": {
            "n_testable": n_testable,
            "mean_survival": _nanmean(surv_vals) if n_testable else float("nan"),
            "n_freq_driven": n_freq_driven,
            "frac_freq_driven": _f(n_freq_driven / n_testable) if n_testable else 0.0,
        },
        "freq_control_local": freq_local,
        "sibling_redundancy": sib_summary,
        "n_superparents": len(superparents),                      # outdeg-only flag
        "n_superparents_strict": sum(sp["strict"] for sp in superparents),  # old AND gate
        # A Neuronpedia link only means anything for the released gemma SAE; labels
        # come from that same export, so their absence is the honest signal that
        # this dictionary is not on Neuronpedia.
        "superparents": [
            {
                **sp,
                "parent_global": gid_p(sp["parent_local"]),
                "label": C.feature_label(gid_p(sp["parent_local"]), labels),
                **({"url": C.npedia_url(gid_p(sp["parent_local"]))} if labels else {}),
            }
            for sp in superparents[:10]
        ],
        "top_edges": top_edges,
    }


def to_markdown(report) -> str:
    # Jekyll renders this file to .html on GitHub Pages; the raw HTML passes
    # through and gives the report the same nav bar as the generated dashboards,
    # so navigation is consistent across the site.
    # A run that is not a gemma layer takes the site-wide nav, same rule the
    # dashboards use: marking a layer here would point the "Page" row at
    # outputs/layer_NN/ files that describe a different SAE entirely.
    is_layer = C.IS_LAYER_RUN
    nav = C.nav_html(depth=C.page_depth(C.METRICS_MD_PATH),
                     layer=report["config"].get("layer", C.LAYER) if is_layer else None,
                     page="metrics_report.html",
                     current=f"outputs/{C.RUN_NAME}/metrics_report.html")
    L = [nav, "", "# Exp 0 - metrics report", ""]
    L.append(C.scope_line(report["total_tokens"], n_docs=report["config"].get("n_docs"),
                          config=report["config"]))
    L.append("")
    for pr in report["pairs"]:
        d = pr["degree"]
        L.append(f"## Block pair {pr['pair']}  -  {pr['n_candidate_edges']} candidate edges")
        L.append("")
        L.append(f"- **Out-degree**: {d['n_parents_with_children']} parents, "
                 f"{d['n_children_with_parent']} children, {d['n_multi_parented']} multi-parented "
                 f"(PolyFrac {100 * d['poly_frac']:.1f}%); "
                 f"top-1 parent holds {100 * d['top1_edge_share']:.1f}% of edges, "
                 f"Gini {d['outdeg_gini']:.3f}, max out-degree {d['max_outdeg']}.")
        L.append(f"- **Superparents** (out-degree flag): {pr['n_superparents']} "
                 f"({pr['n_superparents_strict']} also pass the old fire-rate AND-gate)"
                 + (f" — e.g. feature {pr['superparents'][0]['parent_global']} "
                    f"_{_clip(pr['superparents'][0].get('label'), 60)}_: "
                    f"{pr['superparents'][0]['outdeg']} children, fires on "
                    f"{100 * pr['superparents'][0]['fire_frac']:.1f}% of tokens" if pr["superparents"] else ""))
        nl = pr["independence_null"]
        L.append(f"- **Independence null**: mean edge PMI {nl['mean_edge_pmi']:.2f}; "
                 f"{nl['n_chance_level']} edges ({100 * nl['frac_chance_level']:.1f}%) at chance "
                 f"level (PMI < 0.5). {pr['n_dropped_min_joint']} edges dropped by the "
                 f"joint-support guard (n_joint < {C.MIN_JOINT}).")
        rec = pr["reconstruction"]
        L.append(f"- **Recon-ablation contribution filter** (Tree-SAE-inspired baseline): "
                 f"{rec['n_pass']}/{pr['n_candidate_edges']} edges pass "
                 f"({100 * rec['frac_pass']:.1f}%).")
        fq = pr["freq_control"]
        ms = fq["mean_survival"]
        L.append(f"- **Frequency control**: mean survival {ms:.3f} over {fq['n_testable']} testable edges; "
                 f"{fq['n_freq_driven']} ({100 * fq['frac_freq_driven']:.1f}%) are frequency-driven "
                 f"(survival < {C.FREQ_SURVIVAL_MIN}).")
        sb = pr["sibling_redundancy"]
        if sb:
            L.append(f"- **Sibling redundancy** (global Jaccard — confounded proxy, not the "
                     f"splitting verdict; the parent-conditioned version is in the "
                     f"stage-03 second pass): mean {sb['mean_redundancy']:.3f} over "
                     f"{sb['n_parents_scored']} parents; {sb['n_over_global_threshold']} over the "
                     f"{C.SIBLING_REDUNDANCY_FLAG} global threshold.")
        else:
            L.append("- **Sibling redundancy**: n/a (child block not in SIBLING_BLOCKS).")
        jc2 = pr.get("joint_child")
        if jc2:
            L.append(f"- **Joint-child (exact union, parents with edges)**: "
                     f"R_supp mean {jc2['r_supp_mean']:.3f}, R_mass mean {jc2['r_mass_mean']:.3f}; "
                     f"{jc2['n_share_energy_ge_09']} parents with one child holding >=90% of their "
                     f"energy (rename candidates).")
        jc = pr["joint_child_cov_mean"]
        L.append(f"- **Joint-child coverage** (min(1, ΣF) upper bound — saturates when children "
                 f"co-fire, kept only for contrast with the exact union): "
                 + (f"{jc:.3f}" if not math.isnan(jc) else "n/a") + ".")
        if pr["top_edges"]:
            L.append("")
            L.append("| parent -> child | R | F | PMI | recon P/C gain | recon? | surv | sib | parent label | child label |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for e in pr["top_edges"][:8]:
                surv = "-" if e["freq_survival"] is None else f"{e['freq_survival']:.2f}"
                red = "-" if e["parent_redundancy"] is None else f"{e['parent_redundancy']:.2f}"
                pm = "-" if e.get("pmi") is None else f"{e['pmi']:.2f}"
                L.append(f"| {e['parent']} -> {e['child']} | {e['reverse_cov']:.2f} | "
                         f"{e['forward_cov']:.2f} | {pm} | {e['recon_parent_gain']:.2f}/{e['recon_child_gain']:.2f} | "
                         f"{'Y' if e['recon_pass'] else 'n'} | {surv} | {red} | "
                         f"{_clip(e.get('parent_label'))} | {_clip(e.get('child_label'))} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Grade one cached-statistics file.")
    ap.add_argument("--stats", type=Path, default=None,
                    help="stats file to grade (default: this layer's outputs/exp0_stats.pt)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the report (default: beside the stats file's run dir)")
    args = ap.parse_args()

    stats_path = args.stats or C.EXP0_STATS_PATH
    if not stats_path.exists():
        raise SystemExit(f"[02] {C.missing_stats_msg()}" if args.stats is None
                         else f"[02] no stats file at {stats_path}")
    print(f"[02] loading {stats_path}")
    stats = torch.load(stats_path, weights_only=False)
    pairs = stats["pairs"]

    labels = C.load_feature_labels()
    if not labels:
        print("[02] note: outputs/feature_labels.json not found - run fetch_labels.py "
              "to include feature descriptions")
    per_pair = [analyse_pair(stats, p, c, labels) for (p, c) in pairs]
    block_ranges, _ = source_structure(stats)
    report = {
        "config": stats["config"],
        "total_tokens": int(stats["total_tokens"]),
        "pairs": per_pair,
        "block_ranges": block_ranges,
    }

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        md_path, json_path = args.out_dir / "metrics_report.md", args.out_dir / "metrics_report.json"
    else:
        md_path, json_path = C.METRICS_MD_PATH, C.METRICS_JSON_PATH

    md_path.write_text(to_markdown(report))                    # md handles NaN itself
    json_path.write_text(json.dumps(json_safe(report), indent=2))
    print(f"[02] wrote {json_path}")
    print(f"[02] wrote {md_path}")
    for pr in per_pair:
        print(f"[02]   {pr['pair']}: {pr['n_candidate_edges']} edges, "
              f"{pr['reconstruction']['n_pass']} recon-pass, "
              f"{pr['n_superparents']} superparents")


if __name__ == "__main__":
    main()
