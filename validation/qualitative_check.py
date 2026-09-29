"""
Qualitative agreement check on the real gemma-2-2b Matryoshka SAE.

The synthetic calibration (validation/) proved the five metrics recover a known tree
and reject injected pathologies. This is the OTHER half of "how we decide
which metric works": for edges the metrics flag good vs bad on the *real* SAE,
check the feature labels against human / Neuronpedia intuition. A clean metric
number on a semantically unrelated pair means the metric is failing.

We contrast, per block pair:

    SURVIVOR   edges that pass coverage AND reconstruction AND frequency control,
               whose parent is NOT a superparent  -> should read as real refinement
               (parent label and child label semantically related).

    REJECTED   edges that pass the crude coverage criterion (so they LOOK strong)
               but are killed by a richer metric:
                 - superparent : parent fans out over most of the block
                 - freq-driven : coverage collapses on rare tokens (survival < min)
                 - no-recon    : ablating the parent doesn't hurt the child's recon
               -> should read as frequency/co-occurrence artifacts (unrelated labels).

Feature labels come from the bulk export in outputs/feature_labels.json (built
by fetch_labels.py - all 32768 descriptions in one sweep, no per-feature HTTP).
For the ~26 features missing from that export we fall back to Neuronpedia's
public API (cached to outputs/npedia_labels_cache.json). Pass --no-fetch to skip
the API fallback and emit URLs for anything the bulk file doesn't cover.

This is Tier 4 of the validation ladder (Tiers 1-2 are the toys next door; Tier 3
is the PCFG SAE, which has no script here). It is
also pipeline stage 02b: unlike the toys it needs the real cache and writes a
published artifact into RUN_DIR, so it is the one tier that is not self-contained.

Needs:  outputs/layer_NN/exp0_stats.pt + feature_labels.json (fetch_labels.py)
Writes: outputs/layer_NN/qualitative_check.{json,md}
Run:    python3 -m validation.qualitative_check                 # pair 0->1, fetch labels
        python3 -m validation.qualitative_check --pairs 0->1 1->2
        python3 -m validation.qualitative_check --no-fetch      # offline: URLs only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a file, too

import config as C  # noqa: E402
from metrics import (  # noqa: E402
    coverage_legs,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    keep_edges,
)

NPEDIA_API = C.NEURONPEDIA_API
CACHE_PATH = C.RUN_DIR / "npedia_labels_cache.json"


# ---------------------------------------------------------------------------
# Neuronpedia label fetch (cached, polite, degrades gracefully offline)
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def fetch_label(idx: int, cache: dict, enabled: bool, bulk: dict | None = None) -> str:
    key = str(idx)
    # Prefer the bulk export (complete, offline); it covers all but ~26 features.
    if bulk:
        text = bulk.get(key)
        if text:
            return text
    if key in cache:
        return cache[key]
    if not enabled:
        return "(not fetched)"
    try:
        req = urllib.request.Request(NPEDIA_API.format(idx), headers={"User-Agent": "refusal-lens-exp0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        exps = d.get("explanations") or []
        label = exps[0]["description"].strip() if exps else "(no label on Neuronpedia)"
    except Exception as e:  # network / rate limit / missing feature
        label = f"(fetch failed: {type(e).__name__})"
        return label  # don't cache failures
    cache[key] = label
    time.sleep(0.2)
    return label


# ---------------------------------------------------------------------------
# Per-pair metric computation + edge selection
# ---------------------------------------------------------------------------
def compute(stats, p_blk, c_blk):
    key = f"{p_blk}->{c_blk}"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    p0, p1 = C.BLOCK_RANGES[p_blk]
    c0, c1 = C.BLOCK_RANGES[c_blk]
    fire_p, fire_c = fire[p0:p1], fire[c0:c1]

    cofire = stats["cofire"][key].double()
    R, F = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT,
                           cofire=cofire, min_joint=C.MIN_JOINT)

    recon = edge_reconstruction_condition(
        stats["err_sum_c"][c_blk].double(),
        stats["g_parent_sum"][key].double(),
        stats["g_child_sum"][c_blk].double(),
        C.RECON_REL_GAIN_MIN,
    )
    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"][key].double(),
        stats["fire_c_by_bucket"][c_blk].double(),
        edge_mask,
        min_fire_low=C.FREQ_MIN_FIRE_LOW,
    )
    superparents = find_superparents(
        edge_mask, fire_p, total, C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC
    )
    sp_locals = torch.zeros(edge_mask.shape[0], dtype=torch.bool)
    for sp in superparents:
        sp_locals[sp["parent_local"]] = True

    return {
        "key": key, "p0": p0, "c0": c0, "total": total,
        "R": R, "F": F, "edge_mask": edge_mask,
        "passes": recon["passes"], "parent_gain": recon["parent_gain"],
        "survival": fcov["survival"], "fire_p": fire_p,
        "sp_locals": sp_locals, "superparents": superparents,
    }


def _rows(pi, ci, d, category):
    p0, c0 = d["p0"], d["c0"]
    out = []
    for p, c in zip(pi.tolist(), ci.tolist()):
        s = d["survival"][p, c]
        out.append({
            "category": category,
            "parent": p0 + p, "child": c0 + c,
            "reverse_cov": float(d["R"][p, c]),
            "forward_cov": float(d["F"][p, c]),
            "recon_parent_gain": float(d["parent_gain"][p, c]),
            "recon_pass": bool(d["passes"][p, c]),
            "freq_survival": None if torch.isnan(s) else float(s),
            "parent_fire_rate": float(d["fire_p"][p] / max(d["total"], 1)),
        })
    return out


def select(d, n_survivor=8, n_each_reject=4):
    """Pick survivor edges and the three rejected categories, each ranked by
    reverse coverage so the rejected ones look strongest under the crude metric."""
    em = d["edge_mask"]
    testable = ~torch.isnan(d["survival"])
    survive = testable & (d["survival"] >= C.FREQ_SURVIVAL_MIN)
    sp_edge = d["sp_locals"].unsqueeze(1) & em

    survivor = em & d["passes"] & survive & ~d["sp_locals"].unsqueeze(1)
    freq_driven = em & testable & (d["survival"] < C.FREQ_SURVIVAL_MIN) & ~d["sp_locals"].unsqueeze(1)
    no_recon = em & ~d["passes"] & survive & ~d["sp_locals"].unsqueeze(1)

    def top(mask, n, score):
        pi, ci = torch.nonzero(mask, as_tuple=True)
        if pi.numel() == 0:
            return pi, ci
        order = torch.argsort(score[pi, ci], descending=True)[:n]
        return pi[order], ci[order]

    rows = []
    rows += _rows(*top(survivor, n_survivor, d["parent_gain"]), d, "survivor")
    rows += _rows(*top(sp_edge, n_each_reject, d["R"]), d, "reject:superparent")
    rows += _rows(*top(freq_driven, n_each_reject, d["R"]), d, "reject:freq-driven")
    rows += _rows(*top(no_recon, n_each_reject, d["R"]), d, "reject:no-recon")
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def to_markdown(all_rows, fetched) -> str:
    # Raw HTML passes through Jekyll's markdown rendering, giving this report the
    # same nav bar as the generated dashboards.
    nav = C.nav_html(depth=C.page_depth(C.RUN_DIR / "qualitative_check.md"),
                     layer=C.LAYER, page="qualitative_check.html",
                     current=f"outputs/{C.RUN_NAME}/qualitative_check.html")
    L = [nav, "", "# Exp 0 - qualitative agreement check (real gemma-2-2b SAE)", "",
         C.scope_line(), ""]
    L.append("For each block pair we compare edges the metrics KEEP (survivors) "
             "against edges they REJECT despite passing the crude coverage test. "
             "Read the parent/child labels: survivors should be semantically "
             "related; rejected edges should look like frequency / co-occurrence "
             "artifacts. Labels from Neuronpedia"
             + ("" if fetched else " (not fetched - URLs only)") + ".")
    L.append("")
    for key, rows in all_rows.items():
        L.append(f"## Block pair {key}")
        L.append("")
        for cat in ["survivor", "reject:superparent", "reject:freq-driven", "reject:no-recon"]:
            crows = [r for r in rows if r["category"] == cat]
            if not crows:
                continue
            L.append(f"### {cat}  ({len(crows)})")
            L.append("")
            for r in crows:
                surv = "-" if r["freq_survival"] is None else f"{r['freq_survival']:.2f}"
                L.append(f"- **{r['parent']} -> {r['child']}**  "
                         f"`R={r['reverse_cov']:.2f} F={r['forward_cov']:.2f} "
                         f"recon_gain={r['recon_parent_gain']:.3f} recon={'Y' if r['recon_pass'] else 'n'} "
                         f"surv={surv} p_fires={100 * r['parent_fire_rate']:.0f}%`")
                L.append(f"    - parent [{r['parent']}]({C.npedia_url(r['parent'])}): "
                         f"_{r.get('parent_label', '')}_")
                L.append(f"    - child  [{r['child']}]({C.npedia_url(r['child'])}): "
                         f"_{r.get('child_label', '')}_")
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["0->1"], help="e.g. 0->1 1->2")
    ap.add_argument("--no-fetch", action="store_true", help="skip Neuronpedia, emit URLs only")
    ap.add_argument("--n-survivor", type=int, default=8)
    ap.add_argument("--n-reject", type=int, default=4)
    args = ap.parse_args()

    if not C.EXP0_STATS_PATH.exists():
        raise SystemExit(C.missing_stats_msg())
    print(f"[02b] loading {C.EXP0_STATS_PATH}")
    stats = torch.load(C.EXP0_STATS_PATH, weights_only=False)

    cache = load_cache()
    bulk = C.load_feature_labels()
    if not bulk:
        print("[02b] note: outputs/feature_labels.json not found - run fetch_labels.py "
              "for offline bulk labels (falling back to Neuronpedia API)")
    fetch_enabled = not args.no_fetch
    all_rows: dict[str, list] = {}
    for key in args.pairs:
        p_blk, c_blk = (int(x) for x in key.split("->"))
        d = compute(stats, p_blk, c_blk)
        rows = select(d, args.n_survivor, args.n_reject)
        # attach labels (bulk export first, Neuronpedia API only for gaps)
        for r in rows:
            r["parent_label"] = fetch_label(r["parent"], cache, fetch_enabled, bulk)
            r["child_label"] = fetch_label(r["child"], cache, fetch_enabled, bulk)
        all_rows[key] = rows
        print(f"[02b]   {key}: {len(rows)} edges selected")

    if fetch_enabled:
        CACHE_PATH.write_text(json.dumps(cache, indent=2))

    md = to_markdown(all_rows, fetch_enabled)
    (C.RUN_DIR / "qualitative_check.md").write_text(md)
    (C.RUN_DIR / "qualitative_check.json").write_text(json.dumps(all_rows, indent=2))
    print(f"[02b] wrote {C.RUN_DIR / 'qualitative_check.md'}")
    print(f"[02b] wrote {C.RUN_DIR / 'qualitative_check.json'}")


if __name__ == "__main__":
    main()
