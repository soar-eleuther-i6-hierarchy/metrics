"""Dose-response collector: walk a synthdict tag tree, emit dose_response.csv + REPORT.md.

One row per (artifact, target class). Each row splits the target class into three arms:
`corrupted` (the damage's own pair rule), `touched` (a damaged feature on either end, but not
corrupted) and `intact` (no damaged feature at all), read against the evaluation null, so a
detector's response is read within one world against a clean control. Expression rows report
recall-given-recovery restricted to each arm, plus the evaluator's own eval-null FPR (never
recomputed here). Configuration columns are copied from the point's `run_config`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scoring.benchmark.registry import EXPRESSIONS
from toygen import labels

# The pair classes each toy's damages are read against. only_superparent has two: its dense
# features carry the `superparent` confound, and its dense edges are `firing_only`.
TARGET_CLASSES_BY_TOY = {"only_isa": ("is_a",), "only_firing": ("firing_only",),
                         "only_superparent": ("superparent", "firing_only"),
                         "only_frequency": ("frequency",), "only_topical": ("topical",)}
KEY_DETECTORS = ("G", "S_res", "coverage_R", "asymmetry_R", "pmi", "recon_2a",
                 "token_freq_survival", "wide")
DIAL_COLUMNS = ("beta", "eta", "edge_fraction", "gamma_rel", "k", "roles", "skew",
                "fraction", "pi")
DIALS_BY_KIND = {"absorption": ["beta", "eta", "edge_fraction"],
                 "hedging": ["gamma_rel", "edge_fraction"],
                 "split": ["k", "roles", "skew", "fraction"],
                 "composition": ["pi", "fraction"]}


def _median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _median_or_none(values) -> float | None:
    vals = [float(v) for v in values]
    return float(np.median(vals)) if vals else None


def collect_point(d: Path) -> list[dict]:
    """One artifact dir -> one flat row per target class."""
    npz = np.load(d / "scores.npz", allow_pickle=False)
    meta = json.loads(str(npz["__meta__"]))
    report = json.loads((d / "expressions.json").read_text())
    census = (json.loads((d / "census.json").read_text())
              if (d / "census.json").exists() else None)
    rc = meta.get("run_config") or {}
    derived = rc.get("derived") or {}

    y = npz["y"]
    corr_mask = npz["corrupted_pair"].astype(bool)
    # Older artifacts carry no touched mask; for them the two arms coincide.
    touched_mask = (npz["touched_pair"].astype(bool) if "touched_pair" in npz.files
                    else corr_mask) | corr_mask
    null_eval = (y == labels._index("unrelated")) & (npz["split"] == 2)

    dials = meta.get("dials") or {}
    base = {
        "toy": meta["toy"], "seed": meta["seed"],
        "kind": meta.get("corruption", "absorption"),
        # The world shape is NOT in the artifact path: without these a shrunken run is
        # indistinguishable from a full one in the CSV.
        "n_tokens": meta.get("n_tokens"),
        "n_roots": (meta.get("cfg_overrides") or {}).get("n_roots"),
        "readout": meta.get("readout", meta.get("match_mode")),
        "acts_model": meta.get("acts_model", meta.get("acts_mode", "ridge")),
        "nnls_lambda": (rc.get("acts_model") or {}).get("lambda"),
    }
    # Each damage carries only its own dials; a missing knob is absent, not zero.
    for name in DIAL_COLUMNS:
        v = dials.get(name)
        base[name] = "+".join(v) if isinstance(v, list) else v
    pairs = derived.get("partner_density") or []
    base |= {
        "severity": meta["realized_severity_median"],
        "severity_kind": meta.get("severity_kind", "edge_cos_parent"),
        "pair_rule": meta.get("corrupted_pair_rule", "ordered_edge"),
        "fvu": meta["fvu"]["scoring"],
        "zeroed_rate": (meta.get("zeroed_rate") or meta.get("support_flip_rate"))["scoring"],
        "zeroed_rate_damaged": (meta.get("zeroed_rate_damaged") or {}).get("scoring"),
        "latent_l0": meta.get("latent_l0", meta.get("realized_l0")),
        "feature_l0": meta.get("feature_l0"),
        "n_latents": meta.get("n_latents"), "F": report["F"],
        "L_over_F": rc.get("L_over_F"),
        "n_lost_features": rc.get("n_lost_features"),
        "n_corrupted_edges": meta.get("n_corrupted_edges"),
        "gamma_star_median": _median_or_none(derived.get("gamma_star", [])),
        "gamma_median": _median_or_none(derived.get("gamma", [])),
        "n_composition_pairs": len((rc.get("selection") or {}).get("composition_pairs", [])),
        "partner_rule": derived.get("partner_rule"),
        "partner_density_median": _median_or_none(v for pr in pairs for v in pr),
        "n_recovered_features": report["n_recovered_features"],
    }

    rows = []
    for target in TARGET_CLASSES_BY_TOY.get(meta["toy"], (None,)):
        is_target = (y == labels._index(target)) if target else np.zeros_like(y, dtype=bool)
        corrupted = is_target & corr_mask
        touched = is_target & touched_mask & ~corr_mask
        intact = is_target & ~touched_mask
        arms = (("corrupted", corrupted), ("touched", touched), ("intact", intact))
        row = dict(base) | {
            "target_class": target,
            "n_corrupted_pairs": int(corrupted.sum()), "n_touched_pairs": int(touched.sum()),
            "n_intact_pairs": int(intact.sum()),
            "target_recovered": report["class_recovered"].get(target) if target else None,
            "target_total": report["class_totals"].get(target) if target else None,
        }
        for det in KEY_DETECTORS:
            v = npz[det]
            for side, m in arms:
                row[f"{det}__{side}"] = _median(v[m])
            row[f"{det}__null"] = _median(v[null_eval])
        # The fitted null thresholds: the evaluator refits them from THIS read's own null, so a
        # damaged dictionary moves the bar as well as the target.
        for m in ("G", "coverage_R", "pmi", "abs_asymmetry_R", "S_res"):
            th = report["thresholds"].get(m, {})
            row[f"{m}__q99"] = th.get("q99")
            if m == "G":
                row["G__q01"] = th.get("q01")
        for name in EXPRESSIONS:
            e = report["expressions"][name]
            row[f"{name}__recall"] = e["target_rollup"].get("recall_given_recovery")
            row[f"{name}__fpr"] = e["counts"]["unrelated_eval"].get("fpr_given_scorable")
            p, s = npz[f"pass__{name}"].astype(bool), npz[f"scorable__{name}"].astype(bool)
            for side, m in arms:
                sc = int((s & m).sum())
                row[f"{name}__pass_{side}"] = (float((p & m).sum()) / sc if sc else float("nan"))
        if census is not None:
            row["census_absorbed"] = census["counts"]["absorbed"]
            row["census_absorbed_among_planted"] = census["n_census_absorbed_among_planted"]
            row["census_clean"] = census["counts"]["clean"]
        rows.append(row)
    return rows


def _sort_key(r: dict) -> tuple:
    """Stable ordering across damages; absent numeric dials sort together as -inf."""
    def num(v):
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    return (r["toy"], r["kind"], r["readout"] or "", r["target_class"] or "", r["roles"] or "",
            num(r.get("edge_fraction")), num(r.get("fraction")), num(r.get("k")),
            num(r.get("gamma_rel")), num(r.get("pi")), num(r.get("eta")), num(r.get("beta")))


def collect(tag_dir: Path) -> list[dict]:
    rows = [row for p in sorted(Path(tag_dir).rglob("scores.npz"))
            for row in collect_point(p.parent)]
    rows.sort(key=_sort_key)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(rows[0].keys()) + sorted({k for r in rows for k in r} - set(rows[0]))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


READING_RULES = [
    "- `edge_cos_parent` (absorption): median over absorbed EDGES of cos(child row, g_p).",
    "- `child_own_component` (hedging): median over hedged EDGES of the parent row's component "
    "along the child's own direction (g_c minus its g_p part).",
    "- `none` (splitting, composition): no severity axis; the dose is the dial (k, pi).",
    "- Arms: `corrupted` follows the damage's pair rule; `touched` holds target pairs with a "
    "damaged feature on either end that the rule does not mark (classes labelled in both "
    "orderings: superparent, frequency, topical); `intact` has no damaged feature. Read the "
    "dose-response against `intact`, never against `intact` + `touched`.",
    "- `zeroed_rate` is the share of planted firing the NNLS fit set to exactly 0, pooled over "
    "the whole support; `zeroed_rate_damaged` is the same share on the damaged features' "
    "latents, where the zeroing concentrates.",
    "- Absorption: `G__corrupted` vs severity is DEFINITIONALLY y=x on an orthogonal edge (the "
    "generator sets that cosine), a manipulation check, not evidence a metric responds. The "
    "hole removes the parent on eta of the tokens where it co-fires with any absorbed child, so "
    "`coverage_R__corrupted` sits near (1 - eta) x undamaged coverage only at beta = 0: a child "
    "row carrying g_p also drives the parent's NNLS strength to 0 on co-firing tokens, which "
    "lowers coverage further as beta grows (see `zeroed_rate_damaged`). Read firing-channel "
    "curves against the realized coverage, not against eta.",
    "- Absorption on only_superparent: the `superparent` row has no corrupted pairs (absorbed "
    "edges are firing_only); its pairs touching a holed dense parent are in `touched`.",
    "- Hedging: a hedged child has no latent and leaves the scored frame. On one-child trees "
    "no target pair touches a hedged parent afterwards, so the corrupted arm is empty by "
    "construction and the intact arm holds the unhedged edges; read the damage on "
    "`n_lost_features` and `target_recovered`, and on `dose_response.csv` rows only together "
    "with the pair-level `corrupted_pair` mask in `scores.npz`.",
    "- Splitting: the corrupted arm is the pairs whose CANDIDATE PARENT was split. `union` "
    "reproduces the unsplit firing channel exactly, so a union row moves only through "
    "geometry; `strongest_shard` is what a one-to-one pipeline sees.",
    "- Composition: the combination latent is not a feature and its own pairs are unscored. "
    "Under `own` a composed feature's own coverage is 1 - pi x partner density. Under `union` "
    "rows are averaged by activation mass, so the combination direction enters each feature's "
    "row in proportion to how much it fires (not at all at pi = 0).",
    "- edge_fraction = 1.0 or fraction = 1.0: the per-read calibration null is itself damaged "
    "(see the `__q99` columns) and the intact arm may be empty.",
    "- `token_freq_survival` cannot inform a claim about random holes: they are "
    "frequency-uniform by construction, so its flatness licenses nothing about systematic holes.",
    "- Units: census `theta_hat` is RADIANS, `severity` is a COSINE; severity = sin(theta_hat) "
    "on an orthogonal edge.",
    "- Severity on only_isa includes the DESIGNED alpha = 0.48 overlap at beta = 0; cross-toy "
    "curves are aligned by the dials, not by raw severity.",
]


def write_md(rows: list[dict], path: Path, tag: str) -> None:
    lines = [f"# Dose-response — synthetic dictionary damage ({tag})", ""]
    lines += ["Every detector/expression cell is median-over-pairs or a rate, split corrupted "
              "vs intact within the target class. A flat curve is a result, not a bug.", ""]
    lines += ["Reading rules:", ""] + READING_RULES + [""]
    groups = sorted({(r["toy"], r["kind"], r["readout"] or "", r["target_class"] or "")
                     for r in rows})
    for toy, kind, readout, target in groups:
        sub = [r for r in rows if (r["toy"], r["kind"], r["readout"] or "",
                                   r["target_class"] or "") == (toy, kind, readout, target)]
        dial_cols = DIALS_BY_KIND.get(kind, list(DIAL_COLUMNS))
        lines += [f"## {toy} — {kind} — {readout} readout — {target}", ""]
        det_cols = dial_cols + ["severity"] + \
            [f"{d}__{s}" for d in ("G", "S_res", "coverage_R", "pmi")
             for s in ("corrupted", "touched", "intact", "null")]
        ex_cols = dial_cols + \
            [f"{n}__{s}" for n in ("overlap_v6", "orthogonal_v6", "containment_baseline",
                                   "probe_overlap_v3", "probe_orthogonal_v3")
             for s in ("recall", "pass_corrupted", "pass_touched", "pass_intact", "fpr")]
        # Arm sizes belong in the rendered table: an empty arm must read as "nothing to say",
        # not as "the metric said nothing".
        rec_cols = dial_cols + ["n_corrupted_pairs", "n_touched_pairs", "n_intact_pairs",
                                "n_recovered_features", "n_lost_features", "target_recovered",
                                "target_total", "n_latents", "L_over_F", "latent_l0",
                                "feature_l0", "fvu", "zeroed_rate", "zeroed_rate_damaged",
                                "census_absorbed", "census_absorbed_among_planted"]
        for cols in (det_cols, ex_cols, rec_cols):
            lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in sub]
            lines += [""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-dir", required=True, help="e.g. outputs_local/synthdict/SYNTH-R1")
    args = ap.parse_args()
    tag_dir = Path(args.tag_dir)
    rows = collect(tag_dir)
    if not rows:
        raise SystemExit(f"no scores.npz under {tag_dir}")
    write_csv(rows, tag_dir / "dose_response.csv")
    write_md(rows, tag_dir / "REPORT.md", tag_dir.name)
    print(f"{len(rows)} rows -> {tag_dir}/dose_response.csv + REPORT.md")


if __name__ == "__main__":
    main()
