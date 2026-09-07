"""Dose-response collector: walk a synthdict tag tree, emit dose_response.csv + REPORT.md.

Each artifact row splits the TARGET class by the planted mask: `corrupted` (the edges the
dictionary damaged) vs `intact` (same class, untouched) vs the evaluation null — so a
detector's response to the pathology is read within one world, against its own controls.
Expression rows report recall-given-recovery restricted to each side, plus the evaluator's
own eval-null FPR (never recomputed here).

House pattern (`scoring/run_scoring.py`): collector -> csv + md, both written into the tag
directory beside the artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scoring.benchmark.registry import EXPRESSIONS, METRICS
from toygen import labels

TARGET_BY_TOY = {"only_isa": "is_a", "only_firing": "firing_only"}
KEY_DETECTORS = ("G", "S_res", "coverage_R", "asymmetry_R", "pmi", "recon_2a",
                 "token_freq_survival", "wide")


def _median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def collect_point(d: Path) -> dict:
    """One artifact dir -> one flat row."""
    npz = np.load(d / "scores.npz", allow_pickle=True)
    meta = json.loads(str(npz["__meta__"]))
    report = json.loads((d / "expressions.json").read_text())
    census = (json.loads((d / "census.json").read_text())
              if (d / "census.json").exists() else None)

    y = npz["y"]
    corr_mask = npz["corrupted_pair"].astype(bool)
    target = TARGET_BY_TOY[meta["toy"]]
    t_idx = labels._index(target)
    is_target = y == t_idx
    null_eval = (y == labels._index("unrelated")) & (npz["split"] == 2)
    corrupted = is_target & corr_mask
    intact = is_target & ~corr_mask

    row = {
        "toy": meta["toy"], "seed": meta["seed"], "mode": meta["match_mode"],
        "acts": meta.get("acts_mode", "ridge"),
        "beta": meta["dials"]["beta"], "eta": meta["dials"]["eta"],
        "f": meta["dials"]["edge_fraction"],
        "severity": meta["realized_severity_median"],
        "n_corrupted_pairs": int(corrupted.sum()), "n_intact_pairs": int(intact.sum()),
        "fvu": meta["fvu"]["scoring"], "flip_rate": meta["support_flip_rate"]["scoring"],
        "realized_l0": meta["realized_l0"],
        "n_recovered_features": report["n_recovered_features"], "F": report["F"],
        "target_recovered": report["class_recovered"].get(target),
        "target_total": report["class_totals"].get(target),
    }
    for det in KEY_DETECTORS:
        v = npz[det]
        row[f"{det}__corrupted"] = _median(v[corrupted])
        row[f"{det}__intact"] = _median(v[intact])
        row[f"{det}__null"] = _median(v[null_eval])
    # The fitted null thresholds, surfaced per dial point: the evaluator refits them from THIS
    # read's own calibration null, so a corrupted dictionary moves the bar as well as the
    # target (measured: at f=1.0 the G null q99 moves +28% and coverage_R q99 -30%). A
    # threshold-crossing read without its threshold column conflates the two.
    for m in ("G", "coverage_R", "pmi", "abs_asymmetry_R", "S_res"):
        th = report["thresholds"].get(m, {})
        row[f"{m}__q99"] = th.get("q99")
        if m == "G":
            row["G__q01"] = th.get("q01")            # IN-BAND(G) reads both ends
    for name in EXPRESSIONS:
        e = report["expressions"][name]
        roll = e["target_rollup"]
        row[f"{name}__recall"] = roll.get("recall_given_recovery")
        row[f"{name}__fpr"] = e["counts"]["unrelated_eval"].get("fpr_given_scorable")
        p, s = npz[f"pass__{name}"].astype(bool), npz[f"scorable__{name}"].astype(bool)
        for side, m in (("corrupted", corrupted), ("intact", intact)):
            sc = int((s & m).sum())
            row[f"{name}__pass_{side}"] = (float((p & m).sum()) / sc if sc else float("nan"))
    if census is not None:
        row["census_absorbed"] = census["counts"]["absorbed"]
        row["census_absorbed_among_planted"] = census["n_census_absorbed_among_planted"]
        row["census_clean"] = census["counts"]["clean"]
    return row


def collect(tag_dir: Path) -> list[dict]:
    rows = [collect_point(p.parent) for p in sorted(Path(tag_dir).rglob("scores.npz"))]
    rows.sort(key=lambda r: (r["toy"], r["mode"], r["f"], r["eta"], r["beta"]))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    keys = list(rows[0].keys()) + [k for k in keys if k not in rows[0]]
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


def write_md(rows: list[dict], path: Path, tag: str) -> None:
    lines = [f"# Dose-response — synthetic absorption ({tag})", ""]
    lines += ["Severity = realized median cos(d_c', g_p); every detector/expression cell is "
              "median-over-pairs or a rate, split corrupted vs intact within the target class. "
              "Predictions P1-P5 and the claim wording are frozen in synthdict/SYNTH_PRECOMMIT.md; "
              "a flat curve is a result, not a bug.", ""]
    lines += ["Reading rules (instrument-audit findings, recorded before the grid was read):", ""]
    lines += [
        "- `G__corrupted` vs severity is DEFINITIONALLY y=x in identity mode (the generator sets "
        "that cosine); it is a manipulation check anchoring the dose axis, not evidence a metric "
        "'responds'. The informative G content is the calibrated threshold crossings "
        "(HIGH/IN-BAND against `G__q99`/`G__q01`), the intact and null columns, and hungarian mode. "
        "`coverage_R__corrupted ~ (1-eta)` is likewise the manipulation check for the hole.",
        "- f=1.0 rows: the per-read calibration null is itself corrupted (thresholds move "
        "materially; see the `__q99` columns) and the intact control is empty. f=0.1 rows are "
        "the primary dose-response; f=1.0 rows characterize the instrument-as-deployed on a "
        "fully corrupted dictionary.",
        "- `token_freq_survival` cannot inform an absorption claim here: random holes are "
        "frequency-uniform by construction and these worlds have no token-frequency structure. "
        "Its flatness licenses nothing about real (systematic) holes.",
        "- Units: census `theta_hat` is RADIANS (atan2), `severity` is a COSINE; "
        "severity = sin(theta_hat) on an orthogonal edge. Do not read the two as one number.",
        "- Severity on only_isa includes the DESIGNED alpha=0.48 overlap at beta=0; cross-toy "
        "curves are aligned by (beta, eta), not by raw severity.",
        "",
    ]
    for toy in sorted({r["toy"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows if r["toy"] == toy}):
            sub = [r for r in rows if r["toy"] == toy and r["mode"] == mode]
            lines += [f"## {toy} — {mode} match", ""]
            det_cols = ["beta", "eta", "f", "severity"] + \
                [f"{d}__{s}" for d in ("G", "S_res", "coverage_R", "pmi")
                 for s in ("corrupted", "intact", "null")]
            lines += ["| " + " | ".join(det_cols) + " |",
                      "|" + "---|" * len(det_cols)]
            lines += ["| " + " | ".join(_fmt(r.get(c)) for c in det_cols) + " |" for r in sub]
            lines += [""]
            ex_cols = ["beta", "eta", "f"] + \
                [f"{n}__{s}" for n in ("overlap_v6", "orthogonal_v6", "containment_baseline",
                                       "probe_overlap_v3", "probe_orthogonal_v3")
                 for s in ("recall", "pass_corrupted", "pass_intact", "fpr")]
            lines += ["| " + " | ".join(ex_cols) + " |", "|" + "---|" * len(ex_cols)]
            lines += ["| " + " | ".join(_fmt(r.get(c)) for c in ex_cols) + " |" for r in sub]
            lines += [""]
            rec_cols = ["beta", "eta", "f", "n_recovered_features", "target_recovered",
                        "target_total", "fvu", "flip_rate",
                        "census_absorbed", "census_absorbed_among_planted"]
            lines += ["| " + " | ".join(rec_cols) + " |", "|" + "---|" * len(rec_cols)]
            lines += ["| " + " | ".join(_fmt(r.get(c)) for c in rec_cols) + " |" for r in sub]
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
    print(f"{len(rows)} dial-point artifacts -> {tag_dir}/dose_response.csv + REPORT.md")


if __name__ == "__main__":
    main()
