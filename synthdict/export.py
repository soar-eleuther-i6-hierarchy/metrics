"""Flatten a tag's synthdict artifacts into `dose_response.csv`, one row per point and target class.

Each target class is split into three arms: `corrupted` (the damage's own pair rule), `touched`
(a damaged feature on either end, not corrupted) and `intact` (no damaged feature).

    python -m synthdict.export --tag-dir outputs_local/synthdict/<TAG>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics.rules import GATE_NAMES, LABELS, RULES

# only_superparent plants two classes: its dense features are superparent, its dense edges firing_only
TARGET_CLASSES_BY_TOY = {"only_isa": ("is_a",), "only_firing": ("firing_only",),
                         "only_superparent": ("superparent", "firing_only"),
                         "only_frequency": ("frequency",), "only_topical": ("topical",)}
DIALS = ("beta", "eta", "edge_fraction", "gamma_rel", "k", "roles", "skew", "fraction", "pi")
METRICS = ("coverage_R", "G", "S_res")        # the firing, decoder and probe channels


def _median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _pass_rate(x: np.ndarray) -> float:
    """Share of the measurable cells of a gate that passed."""
    x = x[np.isfinite(x)]
    return float((x > 0.5).mean()) if x.size else float("nan")


def point_rows(d: Path) -> list[dict]:
    """The rows for one artifact directory."""
    npz = np.load(d / "scores.npz", allow_pickle=False)
    meta = json.loads(str(npz["__meta__"]))
    report = json.loads((d / "expressions.json").read_text())
    y = npz["y"]
    corr = npz["corrupted_pair"].astype(bool)
    damaged = npz["touched_pair"].astype(bool) | corr
    null = (y == LABELS.index("unrelated")) & (npz["split"] > 0)

    dials = meta.get("dials") or {}
    base = {"toy": meta["toy"], "seed": meta["seed"], "kind": meta["corruption"],
            "readout": meta["readout"], "acts_model": meta["acts_model"]}
    for name in DIALS:
        v = dials.get(name)
        base[name] = "+".join(v) if isinstance(v, list) else v

    rows = []
    for target in TARGET_CLASSES_BY_TOY.get(meta["toy"], ()):
        is_t = y == LABELS.index(target)
        arms = {"corrupted": is_t & corr, "touched": is_t & damaged & ~corr,
                "intact": is_t & ~damaged}
        row = dict(base, target_class=target)
        row |= {f"n_{a}_pairs": int(m.sum()) for a, m in arms.items()}
        cells = arms | {"null": null}
        for name in METRICS:
            if name in npz.files:
                row |= {f"{name}__{a}": _median(npz[name][m]) for a, m in cells.items()}
        for name in GATE_NAMES:
            if name in npz.files:
                row |= {f"{name}__{a}": _pass_rate(npz[name][m]) for a, m in cells.items()}
        for name in RULES:
            e = report["expressions"][name]
            row[f"{name}__recall"] = e["target_rollup"].get("recall_given_recovery")
            row[f"{name}__fpr"] = e["counts"]["unrelated_eval"].get("fpr_given_scorable")
            p, s = npz[f"pass__{name}"].astype(bool), npz[f"scorable__{name}"].astype(bool)
            for a, m in arms.items():
                n = int((s & m).sum())
                row[f"{name}__pass_{a}"] = float((p & m).sum()) / n if n else float("nan")
        rows.append(row)
    return rows


def export(tag_dir: Path) -> Path:
    """Write `<tag_dir>/dose_response.csv` from every artifact under the tag."""
    tag_dir = Path(tag_dir)
    rows = [r for p in sorted(tag_dir.rglob("scores.npz")) for r in point_rows(p.parent)]
    if not rows:
        raise SystemExit(f"no scores.npz under {tag_dir}")
    out = tag_dir / "dose_response.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-dir", required=True)
    print(export(Path(ap.parse_args().tag_dir)))
