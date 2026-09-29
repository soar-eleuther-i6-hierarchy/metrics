"""The cross-world verdict: one rule graded over all five worlds of one (seed, read).

A rule's target lives in one world and its leakage in the others, so only the rollup sees both.
The null FPR bar takes the worst world, leakage pools counts across worlds (PRECOMMIT.md s8).
"""

from __future__ import annotations

import csv
from pathlib import Path

from scoring.benchmark.evaluate import (BAR_CONFOUND_LEAK, VERDICTS, leak_exceedances, verdict)
from scoring.benchmark.registry import MIN_SCORABLE_SUPPORT, NULL_CLASS, REPORT_SCHEMA, TOYS

# This pools worlds of one seed, so `toy` must differ: the opposite of `grid.aggregate_seeds`,
# which pools seeds of one world. Do not move `toy` into MUST_MATCH.
MUST_DIFFER: tuple[str, ...] = ("toy",)

# `report_schema` is checked separately and more strictly: every world on the current schema.
MUST_MATCH: tuple[str, ...] = ("seed", "read", "freeze_tag", "settings_sha256")

VERDICT_SCOPE = "benchmark-wide"


def _blocks(worlds: list[dict], name: str) -> list[tuple[str, dict]]:
    out = []
    for w in worlds:
        blk = (w.get("expressions") or {}).get(name)
        if blk is not None:
            out.append((w["toy"], blk))
    return out


def _check_guards(worlds: list[dict]) -> None:
    """Raise on a pool of incomparable blocks: a wrong pooled number still renders."""
    if not worlds:
        raise ValueError("nothing to combine")
    # MUST_MATCH first, so mixing oracle and trained reads names the read, not a repeated toy.
    for key in MUST_MATCH:
        seen = {str(w.get(key)) for w in worlds}
        if len(seen) > 1:
            raise ValueError(
                f"{key} must be IDENTICAL across the blocks being pooled, got {sorted(seen)}")
    # The current schema, not merely a shared one: an older tree agrees with itself but keys
    # rules by old names, so every lookup would miss and read UNTESTABLE instead of failing.
    stale = sorted({str(w.get("report_schema")) for w in worlds} - {str(REPORT_SCHEMA)})
    if stale:
        raise ValueError(f"report_schema {stale} is not the current {REPORT_SCHEMA}; see "
                         f"scoring/benchmark/registry.py REPORT_SCHEMA for what changed")
    for key in MUST_DIFFER:
        seen = [w.get(key) for w in worlds]
        dupes = {v for v in seen if seen.count(v) > 1}
        if dupes:
            raise ValueError(
                f"{key} must DIFFER across the blocks being pooled, but {sorted(dupes)} appears "
                f"more than once. This pools WORLDS of one seed; grid.aggregate_seeds pools "
                f"SEEDS of one world and has the opposite requirement.")


def _pool_target(blocks: list[tuple[str, dict]]) -> dict:
    """Sum target counts across worlds before taking rates, so each world weighs by its size."""
    tot = rec = sc = ps = 0
    for _toy, blk in blocks:
        r = blk["target_rollup"]
        tot += int(r["N_total"]); rec += int(r["N_recovered"])
        sc += int(r["N_scorable"]); ps += int(r["N_pass"])
    return {
        "N_total": tot, "N_recovered": rec, "N_scorable": sc, "N_pass": ps,
        # as within a world: recall rides on N_recovered, and is None when nothing was scorable
        "recall_given_recovery": (None if sc == 0 else (ps / rec if rec else None)),
        "pass_rate_given_scorable": (ps / sc) if sc else None,
        "recall_end_to_end": (ps / tot) if tot else None,
        "scorable_fraction": (sc / rec) if rec else None,
        "status": ("UNTESTABLE" if sc == 0 else
                   "UNDER_SUPPORTED" if sc < MIN_SCORABLE_SUPPORT else "MEASURED"),
        "min_scorable_support": MIN_SCORABLE_SUPPORT,
    }


def combine(worlds: list[dict], name: str, expected_toys: tuple[str, ...] = TOYS) -> dict:
    """One expression's benchmark-wide row, over the five worlds of one (seed, read)."""
    _check_guards(worlds)
    blocks = _blocks(worlds, name)
    present = [t for t, _ in blocks]
    missing = [t for t in expected_toys if t not in present]

    roll = _pool_target(blocks)

    # ---- eval-null FPR: worst world faces the bar ----
    per_world_fpr: dict[str, float] = {}
    no_null: list[str] = []
    ev_pass = ev_sc = 0
    for toy, blk in blocks:
        # `unrelated_eval` is the whole null; the key name is kept for the artifacts on disk
        ev = blk["counts"]["unrelated_eval"]
        f = ev.get("fpr_given_scorable")
        if f is None or int(ev.get("N_scorable", 0)) == 0:
            no_null.append(toy)
            continue
        per_world_fpr[toy] = float(f)
        ev_pass += int(ev["N_pass"]); ev_sc += int(ev["N_scorable"])
    worst_world = max(per_world_fpr, key=per_world_fpr.get) if per_world_fpr else None
    fpr_worst = per_world_fpr[worst_world] if worst_world else None
    fpr_pooled = (ev_pass / ev_sc) if ev_sc else None

    # ---- leakage: pooled by counts faces the bar, worst world reported beside it ----
    agg: dict[str, dict] = {}
    for toy, blk in blocks:
        for cls, row in blk["counts"].items():
            if cls in ("unrelated_eval", NULL_CLASS):   # null rows, not confounds
                continue
            if cls in blk["target"]:
                continue
            n_sc = int(row.get("N_scorable", 0))
            if n_sc == 0:
                continue
            a = agg.setdefault(cls, {"pass": 0, "sc": 0, "per_world": {}})
            a["pass"] += int(row["N_pass"]); a["sc"] += n_sc
            a["per_world"][toy] = int(row["N_pass"]) / n_sc
    leak_pooled = {c: a["pass"] / a["sc"] for c, a in agg.items() if a["sc"]}
    leak_worst = {c: max(a["per_world"].values()) for c, a in agg.items() if a["per_world"]}
    leak_worst_world = {c: max(a["per_world"], key=a["per_world"].get)
                        for c, a in agg.items() if a["per_world"]}
    leak_worlds = {c: sorted(a["per_world"]) for c, a in agg.items()}

    # ---- the verdict ----
    if missing:
        # No verdict from a partial rollup, but the row keeps what was measured.
        v = "UNTESTABLE"
        reason = (f"the rollup is missing {', '.join(missing)}; a benchmark-wide verdict needs "
                  f"all {len(expected_toys)} worlds, because a rule's confounds live in the worlds "
                  f"its target does not (PRECOMMIT s3)")
    else:
        reason = ""
        under = tuple(c for c, a in agg.items() if 0 < a["sc"] < MIN_SCORABLE_SUPPORT)
        v = verdict(
            roll["recall_given_recovery"], fpr_worst, leak_pooled,
            testable=roll["status"] in ("MEASURED", "UNDER_SUPPORTED"),
            supported=(roll["status"] == "MEASURED"),
            unmeasurable_confounds=(),
            null_supported=bool(per_world_fpr) and not no_null,
            under_supported_confounds=under,
        )

    within = [blk.get("verdict") for _t, blk in blocks]
    return {
        "expression": name, "verdict": v, "verdict_scope": VERDICT_SCOPE,
        "report_schema": REPORT_SCHEMA,
        "worlds": present, "missing_worlds": missing, "untestable_reason": reason,
        "target_rollup": roll,
        "recall_given_recovery": roll["recall_given_recovery"],
        # the worst world faces the bar; the pooled rate is a diagnostic
        "eval_null_fpr_worst": fpr_worst,
        "eval_null_fpr_worst_world": worst_world,
        "eval_null_fpr_pooled": fpr_pooled,
        "eval_null_fpr_for_bar": fpr_worst,
        "eval_null_fpr_per_world": per_world_fpr,
        "worlds_without_a_measurable_null": no_null,
        # the pooled rate faces the bar; `reversed` is the only class where the two can differ
        "leakage_pooled": leak_pooled,
        "leakage_worst": leak_worst,
        "leakage_worst_world": leak_worst_world,
        "leakage_worlds": leak_worlds,
        "leakage_for_bar": leak_pooled,
        "leak_exceedances": leak_exceedances(leak_pooled),
        "multi_world_confounds": [c for c in leak_worlds if len(leak_worlds[c]) > 1],
        "within_world_verdicts": within,
        "disagrees_with_within_world": ("MET CRITERIA" in within and v != "MET CRITERIA"),
    }


def combine_all(worlds: list[dict], names, expected_toys: tuple[str, ...] = TOYS) -> dict:
    return {n: combine(worlds, n, expected_toys=expected_toys) for n in names}


# --- renderers ---
def _f(x, nd=4):
    return "--" if x is None else f"{x:.{nd}f}"


def format_cross_world_md(rows: dict, seed: int, read: str) -> str:
    L = [f"# Benchmark-wide verdict -- seed {seed}, {read} read", "",
         f"Scope: **{VERDICT_SCOPE}**. A within-world verdict cannot see a rule's leakage, "
         f"because a rule's target lives in one world and its confounds live in the others.", "",
         "The two combination rules are deliberately different:", "",
         "- **eval-null FPR: the WORST world**, because PRECOMMIT s8 words the bar as holding "
         "*in each tested world*. The **pooled** rate is a labelled diagnostic, never the bar.",
         "- **leakage: POOLED by counts**, because the confound bar has no per-world wording. "
         "The worst world is reported beside it.", ""]

    seen_multi = sorted({c for r in rows.values() for c in r.get("multi_world_confounds", [])})
    L += ["Pooled and worst-world leakage can differ only for a confound class present in more "
          "than one world.",
          "",
          (f"In this rollup the classes present in more than one world: "
           f"`{'`, `'.join(seen_multi)}`." if seen_multi else
           "In this rollup no confound class appears in more than one world, so the two leakage "
           "columns coincide everywhere below."),
          ""]

    L += ["| expression | verdict | recall | eval-null FPR (worst) | worst world | "
          "eval-null FPR (pooled) | leaks over bar |",
          "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        over = r.get("leak_exceedances") or {}
        leaks = ", ".join(f"{k} {v:.3f}" for k, v in sorted(over.items())) or "--"
        L.append(f"| `{name}` | {r['verdict']} | {_f(r['recall_given_recovery'], 3)} | "
                 f"{_f(r['eval_null_fpr_worst'])} | {r['eval_null_fpr_worst_world'] or '--'} | "
                 f"{_f(r['eval_null_fpr_pooled'])} | {leaks} |")

    dis = [n for n, r in rows.items() if r.get("disagrees_with_within_world")]
    if dis:
        L += ["", "## Disagrees with the within-world verdict", "",
              "These read MET CRITERIA in at least one world and do not meet criteria "
              "benchmark-wide. That disagreement is the reason this table exists.", ""]
        L += [f"- `{n}`: " + ", ".join(f"{k} {v:.3f}" for k, v in
                                       sorted((rows[n].get('leak_exceedances') or {}).items()))
              for n in dis]

    miss = {n: r["missing_worlds"] for n, r in rows.items() if r["missing_worlds"]}
    if miss:
        L += ["", "## Incomplete rollups", ""]
        for n, m in miss.items():
            L.append(f"- `{n}`: UNTESTABLE -- missing {', '.join(m)}. {rows[n]['untestable_reason']}")
    return "\n".join(L) + "\n"


_CSV_COLS = ("seed", "read", "expression", "verdict", "verdict_scope", "report_schema",
             "N_total", "N_recovered", "N_scorable", "N_pass",
             "recall_given_recovery", "pass_rate_given_scorable", "recall_end_to_end",
             "eval_null_fpr_worst", "eval_null_fpr_worst_world", "eval_null_fpr_pooled",
             "leaks_over_bar", "leakage_pooled", "leakage_worst",
             "missing_worlds", "worlds")


def write_cross_world_csv(rows: dict, path: Path, seed: int, read: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_CSV_COLS))
        w.writeheader()
        for name, r in rows.items():
            roll = r["target_rollup"]
            w.writerow({
                "seed": seed, "read": read, "expression": name, "verdict": r["verdict"],
                "verdict_scope": r["verdict_scope"], "report_schema": r["report_schema"],
                "N_total": roll["N_total"], "N_recovered": roll["N_recovered"],
                "N_scorable": roll["N_scorable"], "N_pass": roll["N_pass"],
                "recall_given_recovery": _f(roll["recall_given_recovery"], 6),
                "pass_rate_given_scorable": _f(roll["pass_rate_given_scorable"], 6),
                "recall_end_to_end": _f(roll["recall_end_to_end"], 6),
                "eval_null_fpr_worst": _f(r["eval_null_fpr_worst"], 6),
                "eval_null_fpr_worst_world": r["eval_null_fpr_worst_world"] or "",
                "eval_null_fpr_pooled": _f(r["eval_null_fpr_pooled"], 6),
                "leaks_over_bar": ";".join(f"{k}={v:.6f}" for k, v in
                                           sorted((r.get("leak_exceedances") or {}).items())),
                "leakage_pooled": ";".join(f"{k}={v:.6f}" for k, v in
                                           sorted(r["leakage_pooled"].items())),
                "leakage_worst": ";".join(f"{k}={v:.6f}" for k, v in
                                          sorted(r["leakage_worst"].items())),
                "missing_worlds": ";".join(r["missing_worlds"]),
                "worlds": ";".join(r["worlds"]),
            })
    return path
