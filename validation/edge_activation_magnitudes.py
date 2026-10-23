"""
Edge activation magnitudes: how strongly do a kept edge's two endpoints
activate on the tokens they share?

Two questions this answers with numbers instead of intuition:

    1. Do children activate more strongly than their parents? Per kept
       cross-block edge, the mean activation of parent and child over their
       co-firing tokens, and the child/parent ratio of those means.
    2. Does the parent fire only weakly exactly where the child fires? Per
       edge, the parent's mean on shared tokens over its mean on ALL its
       firing tokens. Values well below 1 are the regime where the
       reconstruction contribution filter can fail an ACTIVE feature: its
       ablation gain scales with the activation value, so a parent at the
       edge of being on contributes almost nothing on the child's tokens.

Per block, the quantiles of all firing values place "edge of being on"
against the JumpReLU threshold (firing values start just above it).

Needs:  outputs/<run>/exp0_stats.pt + token_cache/ (collect_statistics.py)
Writes: outputs/<run>/edge_activation_magnitudes.{json,md}
Run:    python3 -m validation.edge_activation_magnitudes
        python3 -m validation.edge_activation_magnitudes --pairs 0->1 1->2
"""

from __future__ import annotations

import argparse
import json

import torch

import config as C
from run_metrics import source_structure, json_safe
from run_token_metrics import rebuild_edges
from utils.io import TokenCache


def quantiles(t: torch.Tensor, qs=(0.10, 0.50, 0.90)) -> dict[str, float]:
    """{p10, p50, p90} of a 1-D tensor. torch.quantile refuses very large
    inputs, so anything past 10M values is subsampled — quantiles of a uniform
    subsample that size are exact to well past the digits reported."""
    t = t.float()
    if t.numel() == 0:
        return {}
    if t.numel() > 10_000_000:
        t = t[torch.randperm(t.numel())[:10_000_000]]
    return {f"p{int(q * 100)}": float(torch.quantile(t, q)) for q in qs}


def block_value_quantiles(cache: TokenCache, ranges) -> list[dict]:
    """Per block: quantiles of ALL firing values. f_vals is feature-sorted and
    blocks are contiguous feature ranges, so each block is one slice."""
    out = []
    for b, (s, e) in enumerate(ranges):
        lo, hi = int(cache.f_bounds[s]), int(cache.f_bounds[e])
        vals = cache.f_vals[lo:hi]
        out.append({"block": b, "n_firings": int(vals.numel()), **quantiles(vals)})
    return out


def pair_magnitudes(stats, cache, p_blk, c_blk, memo):
    """Per kept edge of one block pair: endpoint means on shared tokens.

    memo caches (rows, vals, mean-over-all-firings) per global feature —
    a parent with many children is looked up once, not once per child.
    """

    def feat(g):
        if g not in memo:
            rows, vals = cache.feature_vals(g)
            memo[g] = (rows, vals.float(), float(vals.float().mean()) if vals.numel() else 0.0)
        return memo[g]

    edge_mask, _, _, _, _ = rebuild_edges(stats, p_blk, c_blk)
    ranges, _ = source_structure(stats)
    (p0, _), (c0, _) = ranges[p_blk], ranges[c_blk]

    edges = []
    for pl, cl in zip(*[t.tolist() for t in torch.nonzero(edge_mask, as_tuple=True)]):
        gp, gc = p0 + pl, c0 + cl
        rows_p, vals_p, mean_p_all = feat(gp)
        rows_c, vals_c, _ = feat(gc)
        on_c = torch.isin(rows_p, rows_c)
        vp = vals_p[on_c]
        vc = vals_c[torch.isin(rows_c, rows_p)]
        if vp.numel() == 0:
            continue                                       # kept edges have co-fire >= MIN_JOINT; only a stats/cache mismatch lands here
        mp, mc = float(vp.mean()), float(vc.mean())
        edges.append({
            "parent": gp, "child": gc, "n_shared": int(vp.numel()),
            "mean_parent_shared": mp,
            "mean_child_shared": mc,
            "child_over_parent": mc / mp if mp > 0 else float("inf"),
            "parent_shared_over_all": mp / mean_p_all if mean_p_all > 0 else float("nan"),
        })
    return edges


def summarize(edges: list[dict]) -> dict:
    if not edges:
        return {"n_edges": 0}
    ratio = torch.tensor([e["child_over_parent"] for e in edges])
    psa = torch.tensor([e["parent_shared_over_all"] for e in edges])
    return {
        "n_edges": len(edges),
        "child_over_parent": quantiles(ratio),
        "frac_child_stronger": float((ratio > 1.0).float().mean()),
        "parent_shared_over_all": quantiles(psa),
        # the parent is markedly weaker on the child's tokens than on its own
        # support — the absorption-flavoured regime the contribution filter
        # is blind to when the residual activation sits near the threshold
        "frac_parent_halved_on_child": float((psa < 0.5).float().mean()),
        "mean_parent_shared": quantiles(torch.tensor([e["mean_parent_shared"] for e in edges])),
        "mean_child_shared": quantiles(torch.tensor([e["mean_child_shared"] for e in edges])),
    }


def extremes(edges: list[dict], labels, n=8) -> dict:
    """The edges worth reading by hand: the parents most weakened on their
    child's tokens, and the children most above their parents."""

    def named(e):
        return {**e, "parent_label": C.feature_label(e["parent"], labels),
                "child_label": C.feature_label(e["child"], labels)}

    by_psa = sorted(edges, key=lambda e: e["parent_shared_over_all"])
    by_ratio = sorted(edges, key=lambda e: e["child_over_parent"], reverse=True)
    return {"weakest_parent_on_child": [named(e) for e in by_psa[:n]],
            "strongest_child_over_parent": [named(e) for e in by_ratio[:n]]}


def to_md(report) -> str:
    L = ["# Edge activation magnitudes", "",
         C.scope_line(report["total_tokens"], n_docs=report.get("n_docs"),
                      config=report.get("config")), "",
         "Per kept edge: endpoint mean activations on the tokens both fire on. "
         "`child/parent` compares the two means; `parent shared/all` is the "
         "parent's mean there against its mean over all its firings — well "
         "below 1 means the parent goes quiet exactly where the child fires, "
         "the regime where the reconstruction contribution filter can fail an "
         "active feature.", ""]

    L.append("## Firing values per block")
    L.append("")
    L.append("| block | firings | p10 | p50 | p90 |")
    L.append("|---|---|---|---|---|")
    for r in report["block_values"]:
        if not r["n_firings"]:
            continue
        L.append(f"| B{r['block']} | {r['n_firings']:,} | "
                 f"{r['p10']:.2f} | {r['p50']:.2f} | {r['p90']:.2f} |")
    L.append("")

    L.append("## Kept edges per pair")
    L.append("")
    L.append("| pair | edges | child/parent p50 | % child stronger | "
             "parent shared/all p50 | % parent halved |")
    L.append("|---|---|---|---|---|---|")
    for key, s in report["pairs"].items():
        if not s["summary"]["n_edges"]:
            L.append(f"| {key} | 0 | – | – | – | – |")
            continue
        m = s["summary"]
        L.append(f"| {key} | {m['n_edges']} | {m['child_over_parent']['p50']:.2f} | "
                 f"{100 * m['frac_child_stronger']:.0f}% | "
                 f"{m['parent_shared_over_all']['p50']:.2f} | "
                 f"{100 * m['frac_parent_halved_on_child']:.0f}% |")
    L.append("")

    def clip(s, n=40):
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    for key, s in report["pairs"].items():
        ex = s.get("extremes")
        if not ex or not ex["weakest_parent_on_child"]:
            continue
        L.append(f"### {key}: parents most weakened on their child's tokens")
        L.append("")
        L.append("| parent → child | shared/all | child/parent | parent label | child label |")
        L.append("|---|---|---|---|---|")
        for e in ex["weakest_parent_on_child"]:
            L.append(f"| {e['parent']} → {e['child']} | {e['parent_shared_over_all']:.2f} | "
                     f"{e['child_over_parent']:.2f} | {clip(e['parent_label'])} | "
                     f"{clip(e['child_label'])} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, help="which layer to grade (overrides EXP0_LAYER)")
    ap.add_argument("--pairs", nargs="*", default=None, help="e.g. 0->1 1->2")
    args = ap.parse_args()
    C.use_layer(args.layer)          # re-execs when it changes anything; see config.use_layer

    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(f"[am] {C.missing_stats_msg()}")
    if not (C.TOKEN_CACHE_DIR / "meta.json").exists():
        raise SystemExit(f"[am] no token cache at {C.TOKEN_CACHE_DIR} - rerun collect_statistics.py")
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)
    cache = TokenCache(C.TOKEN_CACHE_DIR)
    labels = C.load_feature_labels()
    ranges, _ = source_structure(stats)

    pairs = stats["pairs"]
    if args.pairs:
        want = set(args.pairs)
        pairs = [pr for pr in pairs if f"{pr[0]}->{pr[1]}" in want]

    memo: dict[int, tuple] = {}
    report = {
        "total_tokens": int(stats["total_tokens"]),
        "n_docs": stats["config"].get("n_docs"),
        "config": stats["config"],
        "block_values": block_value_quantiles(cache, ranges),
        "pairs": {},
    }
    for p, c in pairs:
        key = f"{p}->{c}"
        edges = pair_magnitudes(stats, cache, p, c, memo)
        report["pairs"][key] = {"summary": summarize(edges),
                                "extremes": extremes(edges, labels) if edges else None}
        s = report["pairs"][key]["summary"]
        print(f"[am] {key}: {s['n_edges']} edges"
              + (f", child/parent p50 {s['child_over_parent']['p50']:.2f}, "
                 f"parent shared/all p50 {s['parent_shared_over_all']['p50']:.2f}"
                 if s["n_edges"] else ""))

    out = C.RUN_DIR / "edge_activation_magnitudes.json"
    # UPDATE any existing report rather than replace it, so a partial --pairs
    # rerun never wipes other pairs' results (same rule as run_token_metrics).
    if out.exists():
        prev = json.loads(out.read_text())
        report["pairs"] = {**prev.get("pairs", {}), **report["pairs"]}
    out.write_text(json.dumps(json_safe(report), indent=2))
    (C.RUN_DIR / "edge_activation_magnitudes.md").write_text(to_md(report))
    print(f"[am] wrote {out}")


if __name__ == "__main__":
    main()
