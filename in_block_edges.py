"""
In-block (same-level) directed-edge analysis for config.IN_BLOCK_BLOCKS.

Complements the cross-block graph: within each block it finds directed
parent→child edges (asymmetric containment) and co-extensive duplicates
(renames/splits), then grades the edges with the same gates as the cross-block
pipeline — PMI (chance-level) and probe-S_res (genuine refinement). The
direction/duplicate rule itself is metrics/in_block.py; this module is the
stage runner around it.

Within-block co-firing comes from collect_statistics's `within_cofire` when
available (B1, B2, B3); for B0 (not cached by default) it is rebuilt from the
token cache.

Needs:  outputs/layer_NN/exp0_stats.pt (+ token_cache/ for B0 and S_res)
Writes: outputs/layer_NN/in_block_edges.{json,md}
Run:    python3 in_block_edges.py              # all IN_BLOCK_BLOCKS, with S_res
        python3 in_block_edges.py --skip-sres  # coverage + PMI only (no probes)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import config as C
from run_metrics import source_structure
from run_token_metrics import load_w_dec
from metrics import (
    degree_stats,
    directed_coverage,
    duplicate_pairs,
    find_superparents,
    independence_scores,
    sres_rank_check,
    train_probe,
)
from utils.io import TokenCache


def within_cofire_from_cache(cache: TokenCache, ranges, b: int) -> torch.Tensor:
    """[Cb, Cb] within-block co-firing rebuilt from the sparse token cache."""
    s, e = ranges[b]
    Cb = e - s
    acc = torch.zeros(Cb, Cb, dtype=torch.float64)
    for _, fb in cache.chunks_dense(s, e):            # [n, Cb] fired indicators
        acc += fb.double().T @ fb.double()
    return acc


def _clip(s, n=46):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def analyse_block(b, stats, ranges, cache, labels, W_dec, device, do_sres):
    s0, s1 = ranges[b]
    fire = stats["fire_count"][s0:s1].double()
    total = int(stats["total_tokens"])

    if "within_cofire" in stats and b in stats["within_cofire"]:
        wc = stats["within_cofire"][b].double()
    else:                                             # B0 isn't cached by collect_statistics -> rebuild from token cache
        wc = within_cofire_from_cache(cache, ranges, b)

    d = directed_coverage(wc, fire, C.EDGE_TAU, C.MIN_FIRE_COUNT, C.MIN_JOINT)
    parent_of, dup = d["parent_of"], d["duplicate"]

    null = independence_scores(wc, fire, fire, total, C.MIN_JOINT)
    shortlist = parent_of & (null["pmi"] > 0.0)       # above chance

    deg = degree_stats(parent_of)
    sps = find_superparents(parent_of, fire, total,
                            C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC)
    dups = duplicate_pairs(dup)

    sres = None
    if do_sres and int(shortlist.sum()):
        resid = cache.resid.to(device)
        Wd = W_dec.to(device)
        n_scored = n_pass = n_untestable = 0
        edges_out = []
        child_locals = torch.nonzero(shortlist.any(dim=0)).flatten()
        child_locals = child_locals[fire[child_locals] >= C.MIN_PROBE_POS]
        for cl in child_locals.tolist():
            gc = s0 + cl
            probe = train_probe(resid, cache.feature_mask(gc).to(device), seed=gc,
                                neg_ratio=C.SRES_NEG_RATIO,
                                max_tokens=C.SRES_MAX_PROBE_TOKENS, min_neg=C.SRES_MIN_NEG,
                                steps=C.SRES_STEPS, lr=C.SRES_LR)
            if probe is None:
                n_untestable += 1
                continue
            corr = (probe @ Wd.T).cpu()
            for pl in torch.nonzero(shortlist[:, cl]).flatten().tolist():
                ok, det = sres_rank_check(corr, s0 + pl, gc, C.SRES_RANK_TOP_K)
                n_scored += 1
                n_pass += int(ok)
                if ok:
                    edges_out.append({"parent": s0 + pl, "child": gc, **det})
        sres = {"n_scored": n_scored, "n_pass": n_pass,
                "n_untestable_children": n_untestable, "edges": edges_out}

    # top directed edges (by reverse coverage) with labels
    R = d["R"]
    pi, ci = torch.nonzero(parent_of, as_tuple=True)
    order = torch.argsort(R[pi, ci], descending=True)[:15]
    top = [{"parent": s0 + int(pi[k]), "child": s0 + int(ci[k]),
            "R": float(R[pi[k], ci[k]]),
            "pmi": (None if torch.isnan(null["pmi"][pi[k], ci[k]])
                    else float(null["pmi"][pi[k], ci[k]]))}
           for k in order.tolist()]

    return {
        "block": b,
        "n_features": s1 - s0,
        "n_edges": int(parent_of.sum()),
        "n_duplicates": len(dups),
        "n_after_pmi": int(shortlist.sum()),
        "poly_frac": float(deg["poly_frac"]),
        "outdeg_gini": float(deg["outdeg_gini"]),
        "n_superparents": len(sps),
        "superparents": [{**sp, "global": s0 + sp["parent_local"],
                          "label": C.feature_label(s0 + sp["parent_local"], labels)}
                         for sp in sps[:10]],
        "top_edges": [{**t, "parent_label": C.feature_label(t["parent"], labels),
                       "child_label": C.feature_label(t["child"], labels)} for t in top],
        "duplicate_examples": [
            {"a": s0 + i, "b": s0 + j,
             "a_label": C.feature_label(s0 + i, labels),
             "b_label": C.feature_label(s0 + j, labels)}
            for i, j in dups[:15]],
        "sres": sres,
    }


def to_md(report):
    # The nav bar, same as every other generated page. This file went without one
    # because it was written while the script sat outside the pipeline, and a
    # markdown page with no bar is a dead end: `refresh_nav` REPLACES a nav block
    # and cannot add a missing one, so nothing downstream would ever notice.
    # It IS one of NAV_PAGES now, so the pill is marked current and switching
    # layer from here lands on the same report rather than on that layer's index.
    md_path = C.RUN_DIR / "in_block_edges.md"
    nav = C.nav_html(depth=C.page_depth(md_path),
                     layer=(report["config"] or {}).get("layer", C.LAYER)
                     if C.IS_LAYER_RUN else None,
                     page="in_block_edges.html",
                     current=f"outputs/{C.RUN_NAME}/in_block_edges.html")
    L = [nav, "", "# In-block (same-level) directed edges", "",
         C.scope_line(report["total_tokens"], n_docs=report.get("n_docs"),
                      config=report.get("config")), "",
         "Parent→child *within* a block (asymmetric containment); co-extensive "
         "pairs are reported as duplicates (renames/splits), never edges.", ""]
    for r in report["blocks"]:
        L.append(f"## Block B{r['block']}  ({r['n_features']} features)")
        L.append(f"- **{r['n_edges']}** directed edges, **{r['n_duplicates']}** duplicate pairs, "
                 f"{r['n_after_pmi']} survive PMI>0; PolyFrac {100*r['poly_frac']:.0f}%, "
                 f"Gini {r['outdeg_gini']:.3f}.")
        if r["sres"]:
            sr = r["sres"]
            L.append(f"- **S_res: {sr['n_pass']}/{sr['n_scored']}** edges are genuine refinements "
                     f"({sr['n_untestable_children']} children untestable).")
        if r["superparents"]:
            sp = r["superparents"][0]
            L.append(f"- In-block superparents: {r['n_superparents']} "
                     f"(e.g. F{sp['global']} _{_clip(sp.get('label'))}_: "
                     f"{sp['outdeg']} children, fires {100*sp['fire_frac']:.0f}%).")
        if r["top_edges"]:
            L.append("")
            L.append("| parent → child | R | PMI | parent label | child label |")
            L.append("|---|---|---|---|---|")
            for e in r["top_edges"][:8]:
                pm = "-" if e["pmi"] is None else f"{e['pmi']:.2f}"
                L.append(f"| {e['parent']} → {e['child']} | {e['R']:.2f} | {pm} | "
                         f"{_clip(e['parent_label'])} | {_clip(e['child_label'])} |")
        if r["duplicate_examples"]:
            L.append("")
            L.append("_Duplicate pairs (rename/split candidates):_ "
                     + "; ".join(f"{d['a']}≈{d['b']} ({_clip(d['a_label'],24)})"
                                 for d in r["duplicate_examples"][:6]))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, help="which layer to grade (overrides EXP0_LAYER)")
    ap.add_argument("--skip-sres", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--w-dec", type=Path, default=None,
                    help="decoder [D_SAE, d_model] for a non-gemma dictionary "
                         "(default: RUN_DIR/w_dec.pt, else gemma's)")
    args = ap.parse_args()
    C.use_layer(args.layer)          # re-execs when it changes anything; see config.use_layer
    device = args.device or C.pick_device()

    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    labels = C.load_feature_labels()
    cache = TokenCache(C.TOKEN_CACHE_DIR) if (C.TOKEN_CACHE_DIR / "meta.json").exists() else None

    # Structure from the file being graded, never from this module: gemma is 32768
    # latents in 5 blocks, a PCFG SAE 1792 in 8, and slicing one with the other's
    # ranges returns a full report computed from the wrong feature columns. Same
    # reason stages 02, 03 and 04 stopped holding it as a constant.
    ranges, _ = source_structure(stats)

    # Which blocks: every one whose within-block matrix the file carries, plus any
    # the token cache can rebuild. `config.IN_BLOCK_BLOCKS` is not consulted -- it
    # is a *collection* directive telling stage 01 which matrices to accumulate,
    # and reusing it here would reimpose gemma's [0, 1, 2] on a source with eight
    # blocks. Analysis reads what was collected. B4 is absent on gemma by
    # construction: 24576^2 was never cached.
    cached = sorted(stats.get("within_cofire", {}))
    have = cached if cache is None else sorted(set(cached) | set(range(len(ranges))))
    have = [b for b in have if b < len(ranges)]
    if not have:
        raise SystemExit("[ib] no within-block matrices and no token cache - rerun collect_statistics.py")
    if cache is None and not args.skip_sres:
        raise SystemExit("[ib] token cache missing - rerun collect_statistics.py or pass --skip-sres")

    # The decoder feeds the S_res probes and nothing else, so --skip-sres must
    # not require it: the default load pulls the released SAE through sae_lens,
    # which a coverage-only machine need not have installed.
    W_dec = None
    if not args.skip_sres:
        W_dec = load_w_dec(args.w_dec)
        d_sae = int(stats["fire_count"].numel())
        if W_dec.shape[0] != d_sae:
            raise SystemExit(
                f"[ib] decoder has {W_dec.shape[0]} features but the statistics have {d_sae}. "
                "Pass --w-dec pointing at this SAE's decoder; the default is gemma's."
            )

    blocks = [analyse_block(b, stats, ranges, cache, labels, W_dec, device, not args.skip_sres)
              for b in have]
    report = {"total_tokens": int(stats["total_tokens"]),
              "n_docs": stats["config"].get("n_docs"),
              # Carried so the digest can name the source it graded; without it
              # to_md() falls back to this module's gemma defaults.
              "config": stats["config"], "blocks": blocks}

    from run_metrics import json_safe
    C.IN_BLOCK_PATH.write_text(json.dumps(json_safe(report), indent=2))
    (C.RUN_DIR / "in_block_edges.md").write_text(to_md(report))
    print(f"[ib] wrote {C.IN_BLOCK_PATH}")
    for r in blocks:
        print(f"[ib]   B{r['block']}: {r['n_edges']} edges, {r['n_duplicates']} dups, "
              f"{r['n_superparents']} superparents"
              + (f", S_res {r['sres']['n_pass']}/{r['sres']['n_scored']}" if r["sres"] else ""))


if __name__ == "__main__":
    main()
