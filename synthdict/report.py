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

# The pair class each toy's expressions are aimed at. The three added for round 2 are the
# toy's own declared property (`toygen.labels.LABELS`); a toy absent from this map still
# collects, but without the corrupted/intact split — better an incomplete row than a KeyError
# that loses the whole artifact, and better than inventing a target class silently.
TARGET_BY_TOY = {"only_isa": "is_a", "only_firing": "firing_only",
                 "only_superparent": "superparent", "only_frequency": "frequency",
                 "only_topical": "topical"}
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
    target = TARGET_BY_TOY.get(meta["toy"])
    is_target = (y == labels._index(target)) if target else np.zeros_like(y, dtype=bool)
    null_eval = (y == labels._index("unrelated")) & (npz["split"] == 2)
    corrupted = is_target & corr_mask
    intact = is_target & ~corr_mask

    dials = meta.get("dials") or {}
    row = {
        "toy": meta["toy"], "seed": meta["seed"],
        "kind": meta.get("corruption", "absorption"),
        # The world shape, because it is NOT in the artifact path: without these columns a
        # shrunken `--n-roots` dry run collects into the CSV indistinguishable from a full run.
        "n_tokens": meta.get("n_tokens"),
        "n_roots": (meta.get("cfg_overrides") or {}).get("n_roots"),
        # Round-1 artifacts stamped `match_mode`; the matcher-free pipeline stamps
        # `readout`. Both are read so the existing tags still collect.
        "mode": meta.get("readout", meta.get("match_mode")),
        "acts": meta.get("acts_mode", "ridge"),
        # Each damage carries only its own dials; a missing knob is absent, not zero.
        "beta": dials.get("beta"), "eta": dials.get("eta"),
        "f": dials.get("edge_fraction"),
        "k": dials.get("k"), "skew": dials.get("skew"), "sigma": dials.get("sigma"),
        "fraction": dials.get("fraction"),
        "severity": meta["realized_severity_median"],
        # WHAT the severity column measures and WHICH pairs the mask marks — per damage, so
        # two rows of this CSV can hold different populations under one column name.
        "severity_kind": meta.get("severity_kind", "edge_cos_parent"),
        "pair_rule": meta.get("corrupted_pair_rule", "ordered_edge"),
        "target_class": target,
        "n_corrupted_pairs": int(corrupted.sum()), "n_intact_pairs": int(intact.sum()),
        "fvu": meta["fvu"]["scoring"], "flip_rate": meta["support_flip_rate"]["scoring"],
        # `realized_l0` is round 1's name for the latent-frame number; new artifacts stamp
        # both, and they diverge exactly when a feature is split across shards.
        "latent_l0": meta.get("latent_l0", meta.get("realized_l0")),
        "feature_l0": meta.get("feature_l0"),
        "n_latents": meta.get("n_latents"),
        "n_recovered_features": report["n_recovered_features"], "F": report["F"],
        "target_recovered": report["class_recovered"].get(target) if target else None,
        "target_total": report["class_totals"].get(target) if target else None,
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


def _sort_key(r: dict) -> tuple:
    """Stable ordering across damages. Each damage carries different dials, so the numeric
    keys are None on most rows; `-inf` sorts those together instead of raising on None < float.
    """
    def num(v):
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    return (r["toy"], r["kind"], r["acts"], r["mode"] or "",
            num(r.get("f")), num(r.get("fraction")), num(r.get("sigma")),
            num(r.get("k")), num(r.get("eta")), num(r.get("beta")))


def collect(tag_dir: Path) -> list[dict]:
    rows = [collect_point(p.parent) for p in sorted(Path(tag_dir).rglob("scores.npz"))]
    rows.sort(key=_sort_key)
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
    lines = [f"# Dose-response — synthetic dictionary damage ({tag})", ""]
    lines += ["Every detector/expression cell is median-over-pairs or a rate, split corrupted "
              "vs intact within the target class. Predictions P1-P5 and the claim wording are "
              "frozen in synthdict/SYNTH_PRECOMMIT.md; a flat curve is a result, not a bug.", ""]
    lines += ["What `severity` means depends on the damage — read the `severity_kind` column "
              "of `dose_response.csv`, never the number alone:", ""]
    lines += [
        "- `edge_cos_parent` (absorption): median over corrupted EDGES of cos(d_c', g_p).",
        "- `row_cos_true` (noise): median over decoder ROWS of cos(W'_j, g_j). A different "
        "population in the same column.",
        "- `none` (splitting, missing latent): the damage has no severity axis; the dose is "
        "the dial itself (k, fraction).",
        "",
    ]
    lines += ["Reading rules (instrument-audit findings, recorded before the grid was read):", ""]
    lines += [
        "- A damage that REMOVES features from the scored frame has an empty corrupted arm by "
        "construction: a deleted feature is not in `feats()`, so no scored pair touches one and "
        "every `__corrupted` cell is nan with `n_corrupted_pairs = 0`. That is not a silent "
        "metric failure — the damage lands on RECOVERY, so read `target_recovered` against "
        "`target_total`, and expect the expression recalls to be censored (`-`) once the "
        "recovered target falls below the evaluator's scorable-support floor.",
        "- A damage with `pair_rule = all` (noise) has an empty INTACT arm for the mirror "
        "reason: every feature is damaged, so there is no same-world control and the "
        "comparison is against the evaluation null only.",
        "- `G__corrupted` vs severity is DEFINITIONALLY y=x in identity mode (the generator sets "
        "that cosine); it is a manipulation check anchoring the dose axis, not evidence a metric "
        "'responds'. The informative G content is the calibrated threshold crossings "
        "(HIGH/IN-BAND against `G__q99`/`G__q01`) and the intact and null columns. "
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
        "- In CLEAN acts mode, `flip_rate` and `coverage_R__corrupted` = (1-eta) are "
        "near-definitional (the parent channel is set by construction); comparing them "
        "against ridge rows as evidence of 'cleanliness' is circular. The ridge-vs-clean "
        "delta lands on coverage_R/pmi/asymmetry_R/recon_2a/fvu only; G and S_res are "
        "byte-identical across the two modes.",
        "",
    ]
    # Dial columns per damage: printing every knob on every table would fill the absorption
    # rows with "-" and hide which knobs the point actually moved.
    dials_by_kind = {"absorption": ["beta", "eta", "f"], "split": ["k", "skew", "fraction"],
                     "missing": ["fraction"], "noise": ["sigma"]}
    groups = sorted({(r["toy"], r["kind"], r["acts"], r["mode"] or "") for r in rows})
    for toy, kind, acts, mode in groups:
            sub = [r for r in rows if r["toy"] == toy and r["kind"] == kind
                   and r["mode"] == mode and r["acts"] == acts]
            dial_cols = dials_by_kind.get(kind, ["beta", "eta", "f"])
            lines += [f"## {toy} — {kind} — {acts} acts — {mode} readout", ""]
            det_cols = dial_cols + ["severity"] + \
                [f"{d}__{s}" for d in ("G", "S_res", "coverage_R", "pmi")
                 for s in ("corrupted", "intact", "null")]
            lines += ["| " + " | ".join(det_cols) + " |",
                      "|" + "---|" * len(det_cols)]
            lines += ["| " + " | ".join(_fmt(r.get(c)) for c in det_cols) + " |" for r in sub]
            lines += [""]
            ex_cols = dial_cols + \
                [f"{n}__{s}" for n in ("overlap_v6", "orthogonal_v6", "containment_baseline",
                                       "probe_overlap_v3", "probe_orthogonal_v3")
                 for s in ("recall", "pass_corrupted", "pass_intact", "fpr")]
            lines += ["| " + " | ".join(ex_cols) + " |", "|" + "---|" * len(ex_cols)]
            lines += ["| " + " | ".join(_fmt(r.get(c)) for c in ex_cols) + " |" for r in sub]
            lines += [""]
            # n_corrupted_pairs / n_intact_pairs belong in the RENDERED table, not only the
            # CSV: without them an empty arm reads as "the metric said nothing" rather than
            # "there was nothing in this arm to say it about".
            rec_cols = dial_cols + ["n_corrupted_pairs", "n_intact_pairs",
                        "n_recovered_features", "target_recovered",
                        "target_total", "n_latents", "latent_l0", "feature_l0",
                        "fvu", "flip_rate",
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
