#!/usr/bin/env python3
"""Sensitivity of the edge-acceptance thresholds, measured on the trained toy.

Why this exists. The methodology states tau = 0.5 and a minimum co-fire count of 30 without
saying why those values and not others. A reviewer asked what would happen at tau = 0.9. The
trained toy is the only setting where that question has a checkable answer, because the true
parent-to-child edges are known by construction, so every threshold can be scored by the
precision and recall it produces.

An edge is accepted when three gates pass together:

    keep_edges(R, fire_p, fire_c, tau, min_fire, cofire, min_joint)
    edge_reconstruction_condition(..., recon_gain)["passes"]
    frequency_controlled_coverage(...)["survival"] >= freq_survival

This sweeps each gate on its own, holding the others at their defaults, and then sweeps the two
the reviewer named as a grid. The expensive part, sampling and encoding, runs once.

The script does not import the gating code by copy. It calls the same functions
`calibrate_on_trained_toy` calls, and asserts that the default point reproduces that script's
published precision and recall. If the two ever diverge, this run fails rather than quietly
reporting a second set of numbers.

    python3 validation/threshold_sweep.py [--out outputs/threshold_sweep.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics import (                                # noqa: E402
    coverage_legs, keep_edges, edge_reconstruction_condition,
    frequency_controlled_coverage, frequency_buckets,
)
from metrics.reconstruction import per_token_ablation_gain   # noqa: E402

# The published operating point. Defined here so every sweep row is read against one baseline.
DEFAULTS = dict(tau=0.5, min_fire=20, min_joint=0, recon_gain=0.01, freq_survival=0.5)

GRIDS = {
    "tau": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "min_fire": [5, 10, 20, 50, 100, 200],
    "min_joint": [0, 10, 30, 50, 100],
    "recon_gain": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
    "freq_survival": [0.0, 0.25, 0.4, 0.5, 0.6, 0.75],
}


def build_tensors(n: int = 200_000):
    """Everything the gates consume. Mirrors calibrate_on_trained_toy.main up to the gates.

    Imported here rather than at module load: only the toy mode needs these helpers, and an
    older checkout that lacks one of them would otherwise break the cached-stats mode too.
    """
    from validation.calibrate_on_trained_toy import (   # noqa: E402
        build_tree, true_edges, n_features, sample, load_sae, encode_decode, match_latents,
    )
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    tree = build_tree()
    truth = true_edges(tree)
    F = n_features(tree)

    w, cfg = load_sae()
    true_dirs = w["true_feats"] if "true_feats" in w else torch.eye(F)
    match = match_latents(w, true_dirs)

    gt = sample(tree, n, F, gen)
    x = gt @ true_dirs
    acts, resid = encode_decode(w, x, cfg)

    fired = (acts > 1e-3).double()
    g = per_token_ablation_gain(acts.double(), resid.double(), w["W_dec"].double())
    err = (resid.double() ** 2).sum(dim=1)

    m = match.tolist()
    parent_lat = [i for i, t in enumerate(m) if t in {p for p, _ in truth}]
    child_lat = [i for i, t in enumerate(m) if t in {c for _, c in truth}]

    fp, fc = fired[:, parent_lat], fired[:, child_lat]
    cofire = fp.T @ fc
    fire_p, fire_c = fp.sum(0), fc.sum(0)
    R, _ = coverage_legs(cofire, fire_p, fire_c)

    recon_inputs = (fc.T @ err, g[:, parent_lat].T @ fc, (fc * g[:, child_lat]).sum(0))

    token_ids = fired.argmax(dim=1).long()
    counts = torch.zeros(int(token_ids.max()) + 1, dtype=torch.float64)
    counts.scatter_add_(0, token_ids, torch.ones(n, dtype=torch.float64))
    buckets = frequency_buckets(counts, 0.5, 0.4)[token_ids]
    cbb = torch.zeros(3, fp.shape[1], fc.shape[1], dtype=torch.float64)
    fcb = torch.zeros(3, fc.shape[1], dtype=torch.float64)
    for k in range(3):
        sel = (buckets == k).double().unsqueeze(1)
        cbb[k] = fp.T @ (fc * sel)
        fcb[k] = (fc * sel).sum(0)

    return dict(truth=truth, m=m, parent_lat=parent_lat, child_lat=child_lat,
                R=R, fire_p=fire_p, fire_c=fire_c, cofire=cofire,
                recon_inputs=recon_inputs, cbb=cbb, fcb=fcb)


def score(T, tau, min_fire, min_joint, recon_gain, freq_survival) -> dict:
    """Precision and recall of the accepted edge set at one threshold setting."""
    edge_mask = keep_edges(T["R"], T["fire_p"], T["fire_c"], tau, min_fire,
                           cofire=T["cofire"], min_joint=min_joint)
    recon = edge_reconstruction_condition(*T["recon_inputs"], recon_gain)
    fcov = frequency_controlled_coverage(T["cbb"], T["fcb"], edge_mask)
    survivors = edge_mask & recon["passes"] & (fcov["survival"] >= freq_survival)

    found = {(T["m"][T["parent_lat"][pi]], T["m"][T["child_lat"][ci]])
             for pi in range(survivors.shape[0]) for ci in range(survivors.shape[1])
             if survivors[pi, ci]}
    tp = found & T["truth"]
    return dict(accepted=len(found), true_positives=len(tp),
                false_positives=len(found - T["truth"]),
                false_negatives=len(T["truth"] - found),
                precision=len(tp) / max(len(found), 1),
                recall=len(tp) / max(len(T["truth"]), 1))


def _json_safe(obj):
    """NaN is not valid JSON, whatever `json.dump` will happily write.

    A chance share is undefined when no edges survived, and there is nothing to take a share of.
    Python writes that as the bare token `NaN`, which every strict reader rejects: JavaScript's
    `JSON.parse`, jq in strict mode, and most typed loaders. `null` is the encoding JSON has for
    "no value", so we use it and keep the file readable by anything.
    """
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def sweep_cached_stats(stats_path: Path) -> dict:
    """Sensitivity on a run with no ground truth, where precision and recall do not exist.

    All three acceptance gates are applied, matching `run_metrics.analyse_pair`. An earlier
    version of this function applied only the coverage gate, which made the accepted set look
    larger and dirtier than it is. The frequency-survival gate is the one that removes
    frequency-driven edges, so leaving it out measured the wrong set.

    What replaces precision and recall: the share of accepted edges whose co-firing is close to
    what their base rates alone predict, reported BEFORE and AFTER the frequency-survival gate.

    One caution on that measure, which the paper states about itself. The frequency-survival
    gate and the independence null both shift weight away from frequent tokens. They are not
    independent, so the drop from `before` to `after` overstates how much of the frequency
    problem the gate removes. The two numbers bound it rather than measure it.
    """
    import torch as _t
    from metrics import independence_scores                            # noqa: E402

    stats = _t.load(stats_path, map_location="cpu", weights_only=False)
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])

    # Block boundaries are read from the stats file itself rather than imported from
    # run_metrics. The helper there is newer than some checkouts, and a sweep that cannot run
    # on an older copy of the repository is a sweep nobody can reproduce on the node.
    cfg = stats.get("config") or {}
    ranges = cfg.get("block_ranges")
    if ranges is None:
        steps = cfg.get("matryoshka_steps")
        if steps is None:
            raise SystemExit(f"{stats_path}: no block_ranges or matryoshka_steps in config")
        ranges, prev = [], 0
        for st_ in steps:
            ranges.append((prev, st_)); prev = st_
    ranges = [(int(a), int(b)) for a, b in ranges]

    def sel(b):
        return slice(*ranges[b])
    out = []
    for (p_blk, c_blk) in stats["pairs"]:
        key = f"{p_blk}->{c_blk}"
        if key not in stats.get("cofire", {}):
            continue
        cofire = stats["cofire"][key].double()
        fire_p, fire_c = fire[sel(p_blk)], fire[sel(c_blk)]
        R, _F = coverage_legs(cofire, fire_p, fire_c)
        pmi = independence_scores(cofire, fire_p, fire_c, total, DEFAULTS["min_joint"])["pmi"]
        recon = edge_reconstruction_condition(
            stats["err_sum_c"][c_blk].double(), stats["g_parent_sum"][key].double(),
            stats["g_child_sum"][c_blk].double(), DEFAULTS["recon_gain"])

        def chance(mask):
            v = pmi[mask]
            v = v[~_t.isnan(v)]
            return (float((v < 0.5).sum()) / len(v)) if len(v) else float("nan")

        # One knob at a time, the others held at their defaults. This is the sensitivity
        # ablation proper: it attributes a change to a single threshold. The recon gate is
        # recomputed only when its own threshold moves, since it does not depend on the others.
        for knob, values in GRIDS.items():
            for v in values:
                cfg = {**DEFAULTS, knob: v}
                rec = (recon if knob != "recon_gain" else edge_reconstruction_condition(
                    stats["err_sum_c"][c_blk].double(), stats["g_parent_sum"][key].double(),
                    stats["g_child_sum"][c_blk].double(), v))
                cov = keep_edges(R, fire_p, fire_c, cfg["tau"], cfg["min_fire"],
                                 cofire=cofire, min_joint=cfg["min_joint"])
                before = cov & rec["passes"]
                fcov = frequency_controlled_coverage(
                    stats["cofire_by_bucket"][key].double(),
                    stats["fire_c_by_bucket"][c_blk].double(), cov)
                after = before & (fcov["survival"] >= cfg["freq_survival"])
                out.append(dict(pair=key, knob=knob, value=v, **cfg,
                                n_coverage=int(cov.sum()), n_before_sigma=int(before.sum()),
                                n_accepted=int(after.sum()),
                                chance_before=chance(before), chance_after=chance(after)))
    return {"stats_path": str(stats_path), "rows": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/threshold_sweep.json")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--stats", type=Path, default=None,
                    help="Cached exp0_stats.pt. Runs the no-ground-truth sweep instead of the toy.")
    args = ap.parse_args()

    if args.stats:
        paths = sorted(args.stats.glob("**/exp0_stats.pt")) if args.stats.is_dir() else [args.stats]
        print(f"{len(paths)} run(s)\n")
        all_rows = []
        for sp in paths:
            r = sweep_cached_stats(sp)
            for row in r["rows"]:
                row["run"] = sp.parent.name
            all_rows += r["rows"]

        # Pooled across runs, at the default min_joint. Per-run spread is what says whether the
        # shape below is a property of the method or of one grammar configuration.
        import statistics as st
        pairs = sorted({r["pair"] for r in all_rows}, key=lambda x: int(x.split("->")[0]))
        for pair in pairs:
            base = [r for r in all_rows if r["pair"] == pair]
            if not base or all(r["n_accepted"] == 0 for r in base):
                continue
            print(f"=== {pair} === ({len({r['run'] for r in base})} runs, mean +/- sd)\n")
            for knob in GRIDS:
                sub = [r for r in base if r["knob"] == knob]
                print(f"  --- {knob} (others at default) ---")
                print(f"  {'value':>8} {'accepted':>16} {'chance before':>18} {'chance after':>18}")
                for v in GRIDS[knob]:
                    rs = [r for r in sub if r["value"] == v]
                    if not rs:
                        continue
                    n = [r["n_accepted"] for r in rs]
                    cb = [r["chance_before"] for r in rs if r["chance_before"] == r["chance_before"]]
                    ca = [r["chance_after"] for r in rs if r["chance_after"] == r["chance_after"]]
                    f = lambda z, q=2: (f"{st.mean(z):.{q}f} +/- {st.pstdev(z):.{q}f}" if z else "      --")
                    mark = "  <- default" if v == DEFAULTS[knob] else ""
                    print(f"  {v:>8} {f(n,1):>16} {f(cb):>18} {f(ca):>18}{mark}")
                print()
        print("'chance' = share of accepted edges with PMI < 0.5, i.e. co-firing at about the")
        print("rate their base rates alone predict. Before and after the sigma gate.")
        # Provenance travels with the numbers. A sweep file that says only "rows" cannot be
        # told apart from one run on a different cache, and one was: the gemma layer-12 sweep
        # was first run on a pre-BOS cache and nobody could see that from the file. Recording
        # the token count and the guards lets the comparability audit check it.
        prov = []
        for sp in paths:
            st = torch.load(sp, map_location="cpu", weights_only=False)
            cfg = st.get("config", {}) or {}
            prov.append({"stats_path": str(sp), "total_tokens": int(st.get("total_tokens", 0)),
                         "bos_excluded": cfg.get("bos_excluded"), "min_joint": cfg.get("min_joint"),
                         "sweep_defaults": DEFAULTS})
        Path(args.out).write_text(json.dumps(_json_safe({"provenance": prov, "rows": all_rows}),
                                             indent=2, allow_nan=False))
        print(f"\nwrote {args.out}")
        return

    print("building tensors (sampling and encoding once) ...")
    T = build_tensors(args.n)
    print(f"toy: {len(T['truth'])} true edges, "
          f"{len(T['parent_lat'])} parent latents, {len(T['child_lat'])} child latents\n")

    base = score(T, **DEFAULTS)
    print(f"default point {DEFAULTS}")
    print(f"  precision {base['precision']:.2f}  recall {base['recall']:.2f}  "
          f"accepted {base['accepted']}")

    # The guard against this script becoming a second source of truth. If the gating here ever
    # drifts from calibrate_on_trained_toy, the run stops instead of publishing a second set of
    # numbers for the same checkpoint.
    published = Path("outputs/trained_toy_calibration.json")
    if published.is_file():
        ref = json.loads(published.read_text())
        for k in ("precision", "recall", "true_positives", "false_positives"):
            if abs(base[k] - ref[k]) > 1e-9:
                sys.exit(f"[sweep] default point disagrees with {published} on {k}: "
                         f"{base[k]} vs {ref[k]}. One of the two is wrong; fix before reporting.")
        print(f"  reproduces {published.name} at the default point\n")
    else:
        print(f"  WARNING: {published} absent, default point unverified\n")

    results = {"n_samples": args.n, "defaults": DEFAULTS, "default_point": base,
               "true_edges": len(T["truth"]), "sweeps": {}, "grid_tau_min_joint": []}

    for knob, values in GRIDS.items():
        rows = []
        print(f"--- {knob} (others at default) ---")
        print(f"{'value':>8} {'accepted':>9} {'TP':>4} {'FP':>4} {'FN':>4} "
              f"{'precision':>10} {'recall':>8}")
        for v in values:
            r = score(T, **{**DEFAULTS, knob: v})
            rows.append({"value": v, **r})
            mark = "  <- default" if v == DEFAULTS[knob] else ""
            print(f"{v:>8} {r['accepted']:>9} {r['true_positives']:>4} "
                  f"{r['false_positives']:>4} {r['false_negatives']:>4} "
                  f"{r['precision']:>10.2f} {r['recall']:>8.2f}{mark}")
        results["sweeps"][knob] = rows
        print()

    print("--- tau x min_joint grid (the two raised in review) ---")
    print(f"{'tau':>6} " + "".join(f"{f'mj={j}':>12}" for j in GRIDS["min_joint"]))
    for tau in GRIDS["tau"]:
        cells = []
        line = f"{tau:>6} "
        for mj in GRIDS["min_joint"]:
            r = score(T, **{**DEFAULTS, "tau": tau, "min_joint": mj})
            cells.append({"tau": tau, "min_joint": mj, **r})
            line += f"{r['precision']:>5.2f}/{r['recall']:<6.2f}"
        results["grid_tau_min_joint"].extend(cells)
        print(line)
    print("\ncells are precision/recall")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
