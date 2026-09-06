"""CLI driver: one (toy, seed, read) -> one seed-namespaced, provenance-stamped artifact dir.

Layout::

    <out>/<TAG>/seed<N>/<toy>/<read>/scores.npz
    <out>/<TAG>/seed<N>/<toy>/<read>/expressions.json

`<TAG>` is the freeze tag: the git SHA the registry and evaluator were committed at. The seed
is in the PATH, not only inside `__meta__`.

That is the structural fix for a near-miss rather than a nicety. No pilot scoring artifact
carried a seed in its filename, and four artifact families carried no provenance at all, so a
second-seed run would have overwritten the arrays in place and left the provenance-free files
beside them still describing seed 0, with nothing in the tree to reveal the mismatch. Naming
discipline had already failed once, so this refuses to overwrite an existing artifact without
`--force`, mirroring `training/train_toy.py`'s checkpoint guard.

The trained read additionally passes a HARNESS GATE before anything is written: the nine
non-`s_res` detectors must reproduce the checkpoint's own saved scoring arrays element-wise,
including length and finite-pattern equality. Agreement pins the matcher, the recovered
universe, the held-out draw and the pair ordering to the saved run. A failure aborts; it is not
a warning, because every downstream number would be computed on a different universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scoring.benchmark import calibrate as C
from scoring.benchmark import evaluate as E
from scoring.benchmark import manifest as M
from scoring.benchmark import reads as RD
from scoring.benchmark.registry import (CAL_SPLIT_SEED, DESIGNATED, EXPRESSIONS, METRICS,
                                        MIN_CAL_SUPPORT, MIN_SCORABLE_SUPPORT, NULL_CLASS,
                                        Q_HI, Q_LO, REPORT_SCHEMA, TAU_SURV, TOYS)
from scoring.core.registry import DETECTORS
from toygen import labels

GATE_TOL = 1e-5          # the saved trained arrays are stored float32
REQUIRED_META = ("toy", "seed", "read", "n_tokens", "s_res_mode", "freeze_tag",
                 "git_sha", "git_dirty", "checkpoint", "checkpoint_weights_sha256")


# --------------------------------------------------------------------------
# layout + write guard
# --------------------------------------------------------------------------
def artifact_dir(out: Path, tag: str, seed: int, toy: str, read: str) -> Path:
    """`<out>/<tag>/seed<N>/<toy>/<read>`. The seed and the read are both in the path, so two
    seeds cannot share a filename and the two reads cannot overwrite each other's arrays."""
    return Path(out) / str(tag) / f"seed{int(seed)}" / str(toy) / str(read)


def write_artifacts(d: Path, arrays: dict, report: dict, meta: dict,
                    force: bool = False) -> Path:
    """Write `scores.npz` + `expressions.json`, refusing to overwrite without `force`.

    The guard trips on EITHER file existing: a run killed between the two leaves a half-written
    directory, and that is exactly when a silent overwrite would be most confusing. A forced
    overwrite is stamped `forced_overwrite: true` so a rewritten directory is distinguishable
    from one written once.
    """
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"incomplete provenance: missing {missing}. Every artifact must be "
                         f"self-describing; the pilot's provenance-free files are why.")
    # Enforced at the WRITER so no producer can skip it. An artifact with no `report_schema` is
    # indistinguishable on disk from a schema-1 one, and `recall_given_recovery` means a
    # different quotient under each -- so pooling the two is a silent denominator mix.
    if meta.get("report_schema") != REPORT_SCHEMA:
        raise ValueError(
            f"report_schema is {meta.get('report_schema')!r}, expected {REPORT_SCHEMA}. Every "
            f"artifact must record the reporting contract it was written under; see "
            f"scoring/benchmark/registry.py REPORT_SCHEMA for what changed between versions.")
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


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def git_provenance(cwd: Path) -> dict:
    """`{git_sha, git_dirty}`, or `"unavailable"` / `None` off a git checkout.

    The server tree is an rsync copy with no `.git`, so this must say so rather than report an
    empty string that reads like a real SHA. The freeze tag is passed in explicitly instead.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return {"git_sha": "unavailable", "git_dirty": None}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_weights_sha256(ckpt: str | None) -> str:
    """Hash the checkpoint's weight file. The directory name carries config/variant/k/seed but
    a RETRAINED checkpoint at the same name would be indistinguishable without this."""
    if not ckpt:
        return "n/a"
    d = Path(ckpt)
    for name in ("sae_weights.safetensors", "sae.safetensors", "sae_weights.pt"):
        if (d / name).exists():
            return file_sha256(d / name)
    cands = sorted(list(d.glob("*.safetensors")) + list(d.glob("*.pt")))
    return file_sha256(cands[0]) if cands else "unavailable"


# --------------------------------------------------------------------------
# the harness gate (trained read only)
# --------------------------------------------------------------------------
def gate_max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a-b|, but INF if the two disagree about length or about WHICH entries are finite.

    Comparing only where both happen to be finite would let a different recovered universe or a
    changed support gate pass with a reported difference of 0.0.
    """
    if a.numel() != b.numel():
        return float("inf")
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    if not bool((fa == fb).all()):
        return float("inf")
    return float((a[fa] - b[fb]).abs().max()) if int(fa.sum()) else 0.0


def harness_gate(read: RD.Read, saved_path: Path, tol: float = GATE_TOL) -> dict:
    """Hold the nine non-`s_res` detectors to the checkpoint's own saved arrays, element-wise.

    `s_res` is excluded: the saved run used PROBE mode and this run computes cosine and probe
    separately under different names. The other nine share identical inputs, so agreement pins
    the matcher, the recovered universe, the held-out draw and the pair ordering. Hard abort.
    """
    saved = np.load(saved_path, allow_pickle=True)
    worst, where = 0.0, ""
    checked = 0
    for det in DETECTORS:
        if det == "s_res":
            continue
        vals = read.vals[det]
        for name in labels.LABELS:
            key = f"{det}__{name}"
            if key not in saved.files:
                continue
            mine = vals[read.y == labels._index(name)]
            theirs = torch.from_numpy(saved[key].astype(np.float64))
            d = gate_max_diff(mine, theirs)
            checked += 1
            if d > worst:
                worst, where = d, key
    if checked == 0:
        raise AssertionError(f"harness gate found no comparable keys in {saved_path}; "
                             f"a silently skipped gate is not a passed gate")
    if not worst <= tol:
        raise AssertionError(
            f"HARNESS GATE FAILED at {where}: max diff {worst:.3e} > {tol:.0e}. The universe, "
            f"matcher or held-out draw does not reproduce the saved run, so every number "
            f"downstream would be computed on a different population.")
    return {"gate_max_diff": worst, "gate_worst_key": where, "gate_keys_checked": checked,
            "gate_reference": str(saved_path)}


# --------------------------------------------------------------------------
# one read, end to end
# --------------------------------------------------------------------------
def run_read(read: RD.Read, cal_split_seed: int = CAL_SPLIT_SEED) -> tuple[dict, dict]:
    """Calibrate, evaluate, and assemble the report + the arrays to persist."""
    assignment = C.null_split_assignment(read.pair_labels, seed=cal_split_seed,
                                         null_class=NULL_CLASS)
    cal_mask, ev_mask, unassigned = C.split_masks(read.pairs, read.feats, assignment)
    vals = {m: read.vals[m] for m in METRICS}
    thresholds = C.fit_thresholds(vals, cal_mask, MIN_CAL_SUPPORT, Q_LO, Q_HI)
    cal_support = C.calibration_support(vals, cal_mask)
    eval_null_idx = [i for i in range(len(read.pairs)) if bool(ev_mask[i])]
    cal_null_idx = [i for i in range(len(read.pairs)) if bool(cal_mask[i])]
    # Fingerprint of the assignment itself. `cal_split_seed` alone does not prove two reads used
    # the same split: `torch.randperm(n)` is not prefix-stable, so any drift in the world's
    # feature count reshuffles everything (measured: F 240 -> 239 moves 50% of shared keys).
    # With this in each artifact, "the two reads shared a split" is checkable, not asserted.
    split_sha = hashlib.sha256(
        ",".join(f"{a}:{b}:{int(v)}" for (a, b), v in sorted(assignment.items()))
        .encode()).hexdigest()

    in_universe = read.recovered
    n_total = RD.class_totals(read.pair_labels)
    n_recovered = RD.class_recovered(read.pair_labels, in_universe)

    exprs = E.evaluate_read(vals, read.y, thresholds, n_total, eval_null_idx,
                            EXPRESSIONS, TAU_SURV,
                            probe_available=(read.s_res_mode == "probe"),
                            cal_null_idx=cal_null_idx)
    # `class_counts` counts recovered pairs from the SCORED frame; cross-check it against the
    # answer key so a pair-frame bug cannot agree with itself.
    for name, c in next(iter(exprs.values()))["counts"].items():
        if name in n_recovered and c["N_recovered"] != n_recovered[name]:
            raise RuntimeError(f"recovered-pair count disagrees for {name}: scored frame says "
                               f"{c['N_recovered']}, answer key says {n_recovered[name]}")

    # Scorability is supplied so "no rule fired" can be split into "every rule rejected it" and
    # "no rule could measure it" -- a pass mask alone cannot tell those apart.
    overlap = E.rule_overlap({k: v["_mask"] for k, v in exprs.items()}, read.y, eval_null_idx,
                             scorables={k: v["_scorable"] for k, v in exprs.items()})
    expr_masks = {k: v.pop("_mask") for k, v in exprs.items()}      # not JSON-serialisable
    expr_scorable = {k: v.pop("_scorable") for k, v in exprs.items()}
    # Each metric's `constant_target` needs a target, and a metric can appear in rules with
    # different targets; use the first designated rule that reads it, which is the one whose
    # verdict the flag will sit beside.
    metric_target: dict[str, tuple[str, ...]] = {}
    for ename in DESIGNATED:
        for _pred, m in EXPRESSIONS[ename]["clauses"]:
            metric_target.setdefault(m, EXPRESSIONS[ename]["target"])

    report = {
        "toy": read.toy, "seed": read.seed, "read": read.read, "n_tokens": read.n_tokens,
        "s_res_mode": read.s_res_mode,
        "F": read.F, "n_recovered_features": read.n_recovered,
        "n_pairs": len(read.pairs),
        "n_cal_null": int(cal_mask.sum()), "n_eval_null": int(ev_mask.sum()),
        "n_not_null": int(unassigned.sum()),
        "cal_split_seed": cal_split_seed,
        "class_totals": n_total, "class_recovered": n_recovered,
        "thresholds": {k: {"q01": v[0], "q99": v[1], "n_cal_finite": cal_support[k]}
                       for k, v in thresholds.items()},
        "metric_diagnostics": E.metric_diagnostics(vals, read.y, eval_null_idx,
                                                   thresholds, cal_support,
                                                   targets=metric_target),
        "null_split_sha256": split_sha,
        "expressions": exprs,
        "rule_overlap": overlap,
        "extra": {k: v for k, v in read.extra.items()
                  if not isinstance(v, torch.Tensor)},
    }
    arrays = {
        "pairs": np.array(read.pairs, dtype=np.int32),
        "feats": np.array(read.feats, dtype=np.int32),
        "y": read.y.numpy().astype(np.int8),
        # 0 = not null, 1 = calibration, 2 = evaluation
        "split": (cal_mask.numpy().astype(np.int8) + 2 * ev_mask.numpy().astype(np.int8)),
        "recovered": in_universe.numpy(),
    }
    # Original precision, every metric an expression can read, so a borderline decision is
    # reproducible without re-deriving it from a rounded cache (PRECOMMIT s7).
    for m in METRICS:
        arrays[m] = vals[m].numpy()
    # Per-expression pass masks, so a decision can be reproduced without re-deriving it from
    # the scores (PRECOMMIT s7 box 3: "sufficient scorable/pass masks").
    for name, m in expr_masks.items():
        arrays[f"pass__{name}"] = m.numpy()
        arrays[f"scorable__{name}"] = expr_scorable[name].numpy()
    if isinstance(read.extra.get("G_g_matched"), torch.Tensor):
        arrays["G_g_matched"] = read.extra["G_g_matched"].numpy()
    if isinstance(read.extra.get("match"), torch.Tensor):
        arrays["match"] = read.extra["match"].numpy().astype(np.int32)
    return report, arrays


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _manifest_main(args) -> None:
    """`--write-manifest` / `--verify-manifest`: the freeze record and its check."""
    from scoring.benchmark import manifest as M

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [0]
    toys = args.toys.split(",") if args.toys else list(TOYS)
    if args.write_manifest:
        m = M.build_manifest(tag=args.tag, seeds=seeds, toys=toys, notes=args.notes)
        path = M.write_manifest(args.out, m, force=args.force)
        print(f"freeze record -> {path}")

    # LOAD, never rebuild. Rebuilding compares the evaluator with itself, which passed on a tree
    # that had no MANIFEST.json at all -- see scoring/benchmark/manifest.py's docstring.
    try:
        saved = M.load_manifest(args.out, args.tag)
    except FileNotFoundError as exc:
        print(f"manifest check: FAILED\n    {exc}")
        raise SystemExit(1)
    res = M.verify_manifest(args.out, saved)
    print(f"manifest check: {res['n_artifacts']} artifacts, "
          f"{'OK' if res['ok'] else str(len(res['problems'])) + ' PROBLEMS'}")
    for prob in res["problems"]:
        print(f"    {prob}")
    if not res["ok"]:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--toy", choices=list(TOYS))
    ap.add_argument("--read", choices=("oracle", "trained"))
    ap.add_argument("--seed", type=int, default=None,
                    help="world/train seed; required for the oracle read, read from the "
                         "checkpoint for the trained read")
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (trained read)")
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--tag", required=True,
                    help="freeze tag: the git SHA the registry and evaluator were committed at")
    ap.add_argument("--out", type=Path, default=Path("outputs_local/benchmark"))
    ap.add_argument("--gate-against", type=Path, default=None,
                    help="saved trained scores npz the nine non-s_res detectors must reproduce")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip probe training. Records s_res_mode=absent and marks the two "
                         "probe comparators INVALID MEASUREMENT; not a full benchmark run")
    ap.add_argument("--probe-fit-seed", type=int, default=None,
                    help=("override the probe FITTING draw (default: seed + 20000). Set it "
                          "equal to the scoring draw (seed + 10000) for the BRIDGE run that "
                          "reproduces the unseparated pilot s_res arrays."))
    ap.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    ap.add_argument("--write-manifest", action="store_true",
                    help="write the freeze record for --tag, then verify the artifact tree")
    ap.add_argument("--verify-manifest", action="store_true",
                    help="verify the artifact tree against the freeze record without writing")
    ap.add_argument("--seeds", default=None, help="comma-separated seed plan, e.g. 1,2,3")
    ap.add_argument("--toys", default=None, help="comma-separated toy plan")
    ap.add_argument("--notes", default="", help="free text recorded in the manifest")
    args = ap.parse_args()

    if args.write_manifest or args.verify_manifest:
        _manifest_main(args)
        return
    if not args.toy or not args.read:
        raise SystemExit("--toy and --read are required unless writing/verifying a manifest")

    t0 = time.time()
    with_probe = not args.no_probe
    if args.read == "oracle":
        if args.seed is None:
            raise SystemExit("--seed is required for the oracle read")
        read = RD.oracle_read(args.toy, args.seed, args.n_tokens, with_probe=with_probe,
                              probe_fit_seed=args.probe_fit_seed)
        ckpt = None
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required for the trained read")
        read = RD.trained_read(args.ckpt, args.n_tokens, with_probe=with_probe,
                               probe_fit_seed=args.probe_fit_seed)
        ckpt = args.ckpt
        if read.toy != args.toy:
            raise SystemExit(f"checkpoint is for {read.toy!r}, not {args.toy!r}")
        if args.seed is not None and read.seed != args.seed:
            raise SystemExit(f"checkpoint train_seed is {read.seed}, not --seed {args.seed}")

    report, arrays = run_read(read)
    if args.gate_against:
        report["harness_gate"] = harness_gate(read, args.gate_against)
        print(f"[{args.toy}/{args.read}] harness gate OK "
              f"(max diff {report['harness_gate']['gate_max_diff']:.2e}, "
              f"{report['harness_gate']['gate_keys_checked']} keys)")
    elif args.read == "trained":
        report["harness_gate"] = {"status": "NOT RUN -- no --gate-against reference supplied"}

    meta = {"toy": read.toy, "seed": read.seed, "read": read.read,
            "n_tokens": read.n_tokens, "s_res_mode": read.s_res_mode,
            # The reporting contract these numbers were written under. `write_artifacts`
            # refuses any other value; see registry.REPORT_SCHEMA for what changed.
            "report_schema": REPORT_SCHEMA,
            # Content hash of the evaluator source. This, not `git_sha`, is what ties an artifact
            # to a revision: the server tree is an rsync copy with no `.git`, so `git_sha` reads
            # "unavailable" on every real run.
            "evaluator_sha256": M.evaluator_sha256(Path(__file__).resolve().parents[2]),
            # Lets `verify_manifest` check that the two reads of one (seed, toy) built the SAME
            # world, which is the property the whole multi-seed comparison rests on and which no
            # single-read path can see.
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(read.extra.get("resolved_config", {}), sort_keys=True,
                           default=str).encode()).hexdigest(),
            "freeze_tag": args.tag, "checkpoint": ckpt,
            "checkpoint_weights_sha256": checkpoint_weights_sha256(ckpt),
            "cal_split_seed": CAL_SPLIT_SEED, "tau_surv": TAU_SURV,
            "q": [Q_LO, Q_HI], "min_cal_support": MIN_CAL_SUPPORT,
            "min_scorable_support": MIN_SCORABLE_SUPPORT,
            "scoring_precision": "float64",
            "null_split_sha256": report.get("null_split_sha256"),
            # BOTH reads score the held-out draw as of B2.1. Read off the Read rather than
            # branched on `read.read`, so the recorded number is the one that was actually used.
            "scoring_sample_seed": read.extra.get("scoring_sample_seed"),
            "matching_sample_seed": read.extra.get("matching_sample_seed"),
            "held_out_draw_rule": "seed + 10000 (scoring.core.grid.held_out_sample_seed)",
            "probe_seed": ("the recovered position of the child "
                           "(scoring/core/detectors.py:fit_probe_directions)"),
            "probe_fit_sample_seed": read.extra.get("probe_fit_sample_seed"),
            "probe_fit_labels": read.extra.get("probe_fit_labels"),
            "probe_fit_rule": "seed + 20000 (scoring.benchmark.registry.probe_fit_sample_seed)",
            # Recorded rather than fixed: `signed_normalized_decoder` orients decoder ROWS using
            # the SCORING draw, so S_res keeps a scoring-draw dependence through the decoder
            # SIGN even once the fitted direction is frozen. The fitting separation does not
            # remove that, and does not touch the self-LABEL circularity either.
            "probe_caveat": ("decoder row signs come from the scoring draw, so S_res retains a "
                             "scoring-draw dependence through the sign; the self-label "
                             "circularity of probe_self_W is untouched"),
            } | git_provenance(Path(__file__).resolve().parents[2])
    report["secs"] = round(time.time() - t0, 1)
    d = write_artifacts(artifact_dir(args.out, args.tag, read.seed, read.toy, read.read),
                        arrays, report, meta, force=args.force)

    print(f"[{read.toy}/{read.read}] seed {read.seed}  "
          f"{read.n_recovered}/{read.F} features  {len(read.pairs)} pairs  "
          f"{report['secs']}s  -> {d}")
    for name, blk in report["expressions"].items():
        r = blk["target_rollup"]
        rec = "--" if r["recall_given_recovery"] is None else f"{r['recall_given_recovery']:.3f}"
        fpr = blk["counts"]["unrelated_eval"]["fpr_given_scorable"]
        fpr_s = "--" if fpr is None else f"{fpr:.4f}"
        # The verdict is within-world. An expression whose target is absent here reads
        # UNTESTABLE while still accepting confound pairs, so the exceedances are printed
        # beside it rather than left in the json (PRECOMMIT s3).
        over = blk.get("leak_exceedances") or {}
        leak_s = ("  LEAKS " + " ".join(f"{k}={v:.3f}" for k, v in sorted(over.items()))
                  if over else "")
        print(f"    {name:<22} recall {rec:>6}  eval-null FPR {fpr_s:>7}  "
              f"{blk['verdict']}{leak_s}")


if __name__ == "__main__":
    main()
