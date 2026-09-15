"""Score trained toy-SAE checkpoints (one per seed) and aggregate the results.

Runs three scorers per seed: ``run_recovery`` (feature matching + realized L0/arch),
``run_retrieval`` (per-class detector AUROC grid), ``run_absorption`` (absorption/decoder
multiplicity decomposition). Each report is saved as JSON; ``aggregate_seeds`` then combines
retrieval into across-seed AUROC with Student-t CIs, and a compact summary is printed.

Usage (from the experiment_0 directory)::

    python -m scoring.run_scoring \
        --ckpt-glob "/path/checkpoints/seed{seed}/full-matryoshka-k11-x4-pow-sp3" \
        --seeds 0 1 2 --out outputs_local/final
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from scoring.trained.absorption import run_absorption
from scoring.trained.recovery import run_recovery
from scoring.core.registry import DETECTORS, LATENT_COLUMNS
from scoring.core.grid import aggregate_seeds
from scoring.trained.retrieval import run_retrieval
from toygen import labels

# Retrieval grid columns: generative classes plus latent-side columns.
_GRID_COLS = tuple(labels.LABELS) + LATENT_COLUMNS


def score_seeds(ckpt_glob: str, seeds: list[int], out: Path, n_tokens: int) -> dict:
    """Run all three scorers per seed, save each report as JSON, and aggregate across seeds.

    ``ckpt_glob`` must contain a ``{seed}`` placeholder.
    """
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recovery, retrieval, absorption = {}, {}, {}
    for s in seeds:
        ckpt = ckpt_glob.format(seed=s)
        print(f"[seed {s}] scoring {ckpt} ...", flush=True)
        recovery[s] = run_recovery(ckpt, n_tokens=n_tokens)
        (out / f"recovery_seed{s}.json").write_text(json.dumps(recovery[s], indent=2))
        retrieval[s] = run_retrieval(ckpt, n_tokens=n_tokens)
        (out / f"retrieval_seed{s}.json").write_text(json.dumps(retrieval[s], indent=2))
        absorption[s] = run_absorption(ckpt, n_tokens=n_tokens)
        (out / f"absorption_seed{s}.json").write_text(json.dumps(absorption[s], indent=2))
        print(f"[seed {s}] done ({time.time() - t0:.0f}s elapsed)", flush=True)

    aggregate = aggregate_seeds([retrieval[s] for s in seeds])
    (out / "aggregate_seeds.json").write_text(json.dumps(aggregate, indent=2))
    return {"recovery": recovery, "retrieval": retrieval, "absorption": absorption,
            "aggregate": aggregate, "seeds": seeds}


def print_summary(res: dict) -> None:
    """Print a compact human-readable summary of the scored seeds."""
    seeds, agg = res["seeds"], res["aggregate"]
    print("\n" + "=" * 88)
    print(f"SCORING SUMMARY — {len(seeds)} seeds")
    print("=" * 88)

    # [1] Per seed: realized L0, source architecture, and how many features were recovered.
    print("\n[1] per-seed provenance")
    for s in seeds:
        rec, ret = res["recovery"][s], res["retrieval"][s]
        print(f"  seed{s}: L0={rec.get('realized_l0'):.2f} arch={rec.get('architecture')} "
              f"n_recovered={ret.get('n_recovered')}")

    # [2] The retrieval grid: mean AUROC across seeds for each detector x column (column vs. the rest).
    print("\n[2] across-seed property-vs-rest AUROC (mean; each column vs the rest)")
    print("  " + " " * 20 + "".join(f"{c[:7]:>8}" for c in _GRID_COLS))
    for det in DETECTORS:
        row = agg.get(det, {})
        cells = "".join(
            (f"{m:>8.2f}" if isinstance((m := row.get(c, {}).get("mean")), float) and m == m
             else f"{'--':>8}")
            for c in _GRID_COLS)
        print(f"  {det:20s}{cells}")

    # [2b] Reportable interval = across-seed Student-t CI; per-cell logit CIs assume independence and read too tight, so we skip them.
    print("\n[2b] is_a column — across-seed AUROC [Student-t 95% CI] (the reportable interval)")
    for det in DETECTORS:
        cell = agg.get(det, {}).get("is_a", {})
        m, lo, hi = cell.get("mean"), cell.get("ci_lo"), cell.get("ci_hi")
        ns = cell.get("n_seeds", 0)
        if isinstance(m, float) and m == m:
            ci = f"[{lo:.2f}, {hi:.2f}]" if isinstance(lo, float) and lo == lo else "[--, --]"
            print(f"  {det:20s} {m:.2f}  {ci}  (n_seeds={ns})")

    # [2c] Firing-count floor: trivial baseline from firing rate alone (max over child/parent variants, averaged over seeds) — beat this, not 0.5.
    def _col_floor(c: str) -> float:
        vals = []
        for s in seeds:
            nb = res["retrieval"][s].get("nuisance_baselines", {}).get(c, {})
            cand = [v for v in nb.values() if isinstance(v, float) and v == v]
            if cand:
                vals.append(max(cand))
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n[2c] baselines per column (mean across seeds) — beat best_firing_baseline, not 0.50")
    print("  " + " " * 20 + "".join(f"{c[:7]:>8}" for c in _GRID_COLS))
    print(f"  {'random_baseline':22s}" + "".join(f"{0.50:>8.2f}" for _ in _GRID_COLS))
    floors = "".join((f"{f:>8.2f}" if (f := _col_floor(c)) == f else f"{'--':>8}") for c in _GRID_COLS)
    print(f"  {'best_firing_baseline':22s}{floors}")

    # [3] Per seed: absorption counts (overall and by relation) and how many features split.
    print("\n[3] absorption decomposition + split readout per seed")
    for s in seeds:
        ab = res["absorption"][s]
        sp = res["retrieval"][s].get("split_readout", {})
        print(f"  seed{s}: counts={ab['counts']} by_relation={ab['absorbed_by_relation']} "
              f"n_split={sp.get('n_split', '--')}")

    # [4] Per seed: the is_a deployment cascade — a greedy Boolean rule built from both tails.
    print("\n[4] is_a deployment cascade per seed (greedy both-tails Boolean rule)")
    for s in seeds:
        isa = res["retrieval"][s].get("cascade", {}).get("is_a", {})
        if "skipped" in isa or not isa:
            print(f"  seed{s}: {isa.get('skipped', 'no cascade')}")
            continue
        hn = isa.get("hard_negative", {})
        trivial = "  (no filter found — base-rate only)" if isa.get("trivial") else ""
        print(f"  seed{s}: prec={isa['final_precision']:.2f} survival={isa['final_survival']:.2f} "
              f"enrich={isa['enrichment']:.1f}x hard_neg_prec={hn.get('precision', float('nan')):.2f} "
              f"n_metrics={isa['n_metrics']}{trivial}")
        print(f"          rule: {isa['final_rule']}")


def _is_num(x) -> bool:
    """True for a real (non-NaN) float."""
    return isinstance(x, float) and x == x


def _col_floor(res: dict, seeds: list[int], c: str) -> float:
    """Firing-count nuisance floor for one column: max child/parent baseline, averaged over seeds."""
    vals = []
    for s in seeds:
        nb = res["retrieval"][s].get("nuisance_baselines", {}).get(c, {})
        cand = [v for v in nb.values() if _is_num(v)]
        if cand:
            vals.append(max(cand))
    return sum(vals) / len(vals) if vals else float("nan")


def format_summary_md(res: dict) -> str:
    """Render print_summary's tables as Markdown (reads the SAME report keys; keep the two in sync)."""
    seeds, agg = res["seeds"], res["aggregate"]
    cols = _GRID_COLS
    sep = "|" + "---|" * (len(cols) + 1)
    L = [f"# Stage 2 — scoring summary ({len(seeds)} seeds)", ""]

    L += ["## [1] per-seed provenance", "",
          "| seed | realized L0 | architecture | n_recovered |", "|---|---|---|---|"]
    for s in seeds:
        rec, ret = res["recovery"][s], res["retrieval"][s]
        L.append(f"| {s} | {rec.get('realized_l0'):.2f} | {rec.get('architecture')} | {ret.get('n_recovered')} |")

    L += ["", "## [2] across-seed property-vs-rest AUROC (mean; each column vs the rest)", "",
          "| detector | " + " | ".join(cols) + " |", sep]
    for det in DETECTORS:
        row = agg.get(det, {})
        cells = " | ".join(f"{m:.2f}" if _is_num(m := row.get(c, {}).get("mean")) else "--" for c in cols)
        L.append(f"| {det} | {cells} |")

    L += ["", "## [2b] is_a column — across-seed AUROC [Student-t 95% CI]", "",
          "| detector | AUROC | 95% CI | n_seeds |", "|---|---|---|---|"]
    for det in DETECTORS:
        cell = agg.get(det, {}).get("is_a", {})
        m, lo, hi, ns = cell.get("mean"), cell.get("ci_lo"), cell.get("ci_hi"), cell.get("n_seeds", 0)
        if _is_num(m):
            ci = f"[{lo:.2f}, {hi:.2f}]" if _is_num(lo) else "[--, --]"
            L.append(f"| {det} | {m:.2f} | {ci} | {ns} |")

    L += ["", "## [2c] baselines per column — beat best_firing_baseline, not 0.50", "",
          "_random_baseline = chance (0.50); best_firing_baseline = the strongest single-feature "
          "firing-count AUROC (max over child-only / parent-only, both tails). A detector is "
          "informative only ABOVE best_firing_baseline._",
          "", "| | " + " | ".join(cols) + " |", sep,
          "| random_baseline | " + " | ".join("0.50" for _ in cols) + " |"]
    floors = " | ".join(f"{f:.2f}" if _is_num(f := _col_floor(res, seeds, c)) else "--" for c in cols)
    L.append(f"| best_firing_baseline | {floors} |")

    L += ["", "## [3] absorption decomposition + split readout per seed", "",
          "| seed | counts | by_relation | n_split |", "|---|---|---|---|"]
    for s in seeds:
        ab, sp = res["absorption"][s], res["retrieval"][s].get("split_readout", {})
        L.append(f"| {s} | {ab['counts']} | {ab['absorbed_by_relation']} | {sp.get('n_split', '--')} |")

    L += ["", "## [4] is_a deployment cascade per seed (greedy both-tails Boolean rule)", ""]
    for s in seeds:
        isa = res["retrieval"][s].get("cascade", {}).get("is_a", {})
        if "skipped" in isa or not isa:
            L.append(f"- **seed{s}**: {isa.get('skipped', 'no cascade')}")
            continue
        hn = isa.get("hard_negative", {})
        trivial = "  (no filter found — base-rate only)" if isa.get("trivial") else ""
        L.append(f"- **seed{s}**: prec={isa['final_precision']:.2f} survival={isa['final_survival']:.2f} "
                 f"enrich={isa['enrichment']:.1f}x hard_neg_prec={hn.get('precision', float('nan')):.2f} "
                 f"n_metrics={isa['n_metrics']}{trivial}")
        L.append(f"  - rule: `{isa['final_rule']}`")
    return "\n".join(L) + "\n"


def write_grid_csv(res: dict, path: Path) -> None:
    """Across-seed mean AUROC grid (detector x column) + the nuisance-floor row, for spreadsheets."""
    seeds, agg = res["seeds"], res["aggregate"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", *_GRID_COLS])
        for det in DETECTORS:
            row = agg.get(det, {})
            w.writerow([det, *(f"{m:.4f}" if _is_num(m := row.get(c, {}).get("mean")) else ""
                               for c in _GRID_COLS)])
        w.writerow(["random_baseline", *(f"{0.5:.4f}" for _ in _GRID_COLS)])
        w.writerow(["best_firing_baseline", *(f"{f:.4f}" if _is_num(f := _col_floor(res, seeds, c)) else ""
                                              for c in _GRID_COLS)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Score trained toy-SAE checkpoints across seeds.")
    ap.add_argument("--ckpt-glob", required=True,
                    help="checkpoint path containing a literal {seed} placeholder")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("outputs_local/final"))
    ap.add_argument("--n-tokens", type=int, default=200_000)
    args = ap.parse_args()

    res = score_seeds(args.ckpt_glob, args.seeds, args.out, args.n_tokens)
    print_summary(res)
    (args.out / "stage2_summary.md").write_text(format_summary_md(res), encoding="utf-8")
    write_grid_csv(res, args.out / "stage2_grid.csv")
    print(f"\nreports written to {args.out}")


if __name__ == "__main__":
    main()
