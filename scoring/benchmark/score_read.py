"""Grade one scored `Read` and write its artifacts, `scores.npz` and `expressions.json`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from metrics.rules import GATE_NAMES, RULES
from metrics.rules.grading import (DESIGNATED, NULL_CLASS, REPORT_SCHEMA, grade_rules,
                                   rule_overlap)
from scoring.benchmark import evaluate as E
from scoring.benchmark import reads as RD
from scoring.benchmark.null import null_mask
from scoring.benchmark.registry import METRICS
from scoring.core.gates import GATE_SOURCES

REQUIRED_META = ("toy", "seed", "read", "n_tokens", "s_res_mode", "freeze_tag",
                 "git_sha", "git_dirty", "checkpoint", "checkpoint_weights_sha256")


def write_artifacts(d: Path, arrays: dict, report: dict, meta: dict,
                    force: bool = False) -> Path:
    """Write `scores.npz` and `expressions.json`, refusing to overwrite either without `force`.

    A forced overwrite is stamped `forced_overwrite: true`.
    """
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"incomplete provenance: missing {missing}. Every artifact must be "
                         f"self-describing; the pilot's provenance-free files are why.")
    # Enforced at the writer: without `report_schema` an artifact cannot be told from an older
    # contract whose rate keys hold different quotients.
    if meta.get("report_schema") != REPORT_SCHEMA:
        raise ValueError(
            f"report_schema is {meta.get('report_schema')!r}, expected {REPORT_SCHEMA}. Every "
            f"artifact must record the reporting contract it was written under; see "
            f"REPORT_SCHEMA in metrics/rules/grading.py and its history in PRECOMMIT.md s8.")
    d = Path(d)
    npz, js = d / "scores.npz", d / "expressions.json"
    if (npz.exists() or js.exists()) and not force:
        raise FileExistsError(
            f"refusing to overwrite the existing benchmark artifact at {d}. A rerun that "
            f"replaces a result in place is indistinguishable from a fresh one. Pass --force "
            f"to overwrite.")
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(meta) | {"forced_overwrite": bool(force and (npz.exists() or js.exists())),
                         "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(npz, **payload)
    js.write_text(json.dumps(report | {"__meta__": meta}, indent=2), encoding="utf-8")
    return d


def run_read(read: RD.Read) -> tuple[dict, dict]:
    """Grade one read and assemble the report and the arrays to persist."""
    vals = {m: read.vals[m] for m in METRICS}
    gate_vals = {g: read.gate_vals[g] for g in GATE_NAMES}
    null = null_mask(read.pairs, read.feats, read.pair_labels, NULL_CLASS)
    eval_null_idx = [i for i in range(len(read.pairs)) if bool(null[i])]

    in_universe = read.recovered
    n_total = RD.class_totals(read.pair_labels)
    n_recovered = RD.class_recovered(read.pair_labels, in_universe)

    # clauses read the gates; the metrics ride along for the diagnostics
    exprs = grade_rules(vals | gate_vals, read.y, n_total, eval_null_idx,
                        RULES, probe_available=(read.s_res_mode == "probe"))
    # `class_counts` counts recovered pairs on the scored frame; cross-check the answer key's count
    for name, c in next(iter(exprs.values()))["counts"].items():
        if name in n_recovered and c["N_recovered"] != n_recovered[name]:
            raise RuntimeError(f"recovered-pair count disagrees for {name}: scored frame says "
                               f"{c['N_recovered']}, answer key says {n_recovered[name]}")

    # scorability splits "no rule fired" into rejected-by-all and unscorable-for-all
    overlap = rule_overlap({k: v["_mask"] for k, v in exprs.items()}, read.y, eval_null_idx,
                           scorables={k: v["_scorable"] for k, v in exprs.items()})
    expr_masks = {k: v.pop("_mask") for k, v in exprs.items()}      # not JSON-serialisable
    expr_scorable = {k: v.pop("_scorable") for k, v in exprs.items()}
    # Each metric's `constant_target` uses the first designated rule reaching it. Clauses name
    # gates, so metrics inherit through `GATE_SOURCES`; without that every `constant_target`
    # would be None and the degenerate-separation flag would never fire.
    metric_target: dict[str, tuple[str, ...]] = {}
    for ename in DESIGNATED:
        target = RULES[ename]["target"]
        for _pred, g in RULES[ename]["clauses"]:
            metric_target.setdefault(g, target)
            for m in GATE_SOURCES.get(g, ()):
                metric_target.setdefault(m, target)

    report = {
        "toy": read.toy, "seed": read.seed, "read": read.read, "n_tokens": read.n_tokens,
        "s_res_mode": read.s_res_mode,
        "F": read.F, "n_recovered_features": read.n_recovered,
        "n_pairs": len(read.pairs),
        "n_null": int(null.sum()), "n_not_null": int((~null).sum()),
        "class_totals": n_total, "class_recovered": n_recovered,
        # gates too: a rule firing on nothing and a gate NaN everywhere look alike in a verdict
        "metric_diagnostics": E.metric_diagnostics(vals | gate_vals, read.y, eval_null_idx,
                                                   targets=metric_target),
        # how much of the frame the support guard removed, by cause
        "support": read.extra.get("support"),
        "expressions": exprs,
        "rule_overlap": overlap,
        "extra": {k: v for k, v in read.extra.items()
                  if not isinstance(v, torch.Tensor)},
    }
    arrays = {
        "pairs": np.array(read.pairs, dtype=np.int32),
        "feats": np.array(read.feats, dtype=np.int32),
        "y": read.y.numpy().astype(np.int8),
        # 0 = not null, 1 = null; schema-2 artifacts used 1 and 2 for the two null halves
        "split": null.numpy().astype(np.int8),
        "recovered": in_universe.numpy(),
    }
    # full precision, so a borderline decision is reproducible (PRECOMMIT s7)
    for m in METRICS:
        arrays[m] = vals[m].numpy()
    # float64 tristates, not bools: a bool cannot hold the NaN for "never measurable", which
    # would read back as a rejection
    for g in GATE_NAMES:
        arrays[g] = gate_vals[g].numpy().astype(np.float64)
    # per-rule pass and scorable masks, so a decision is reproducible without the scores
    for name, m in expr_masks.items():
        arrays[f"pass__{name}"] = m.numpy()
        arrays[f"scorable__{name}"] = expr_scorable[name].numpy()
    if isinstance(read.extra.get("G_g_matched"), torch.Tensor):
        arrays["G_g_matched"] = read.extra["G_g_matched"].numpy()
    if isinstance(read.extra.get("match"), torch.Tensor):
        arrays["match"] = read.extra["match"].numpy().astype(np.int32)
    return report, arrays
