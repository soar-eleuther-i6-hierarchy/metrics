"""CLI driver: one (toy, seed, read) -> one seed-namespaced, provenance-stamped artifact dir.

    python -m scoring.benchmark.run_benchmark --toy only_isa --read oracle --seed 0 --tag TAG

Writes `<out>/<TAG>/seed<N>/<toy>/<read>/{scores.npz,expressions.json}`; `--force` to overwrite.
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
from scoring.benchmark.registry import (DESIGNATED, EXPRESSIONS, GATES, METRICS,
                                        MIN_SCORABLE_SUPPORT, NULL_CLASS, REPORT_SCHEMA, TOYS)
from scoring.core.gates import GATE_CONSTANT_KEYS, GATE_SOURCES
from scoring.core.registry import CONSTANTS, ruleset_stamp
from toygen import labels

GATE_TOL = 1e-5          # the saved trained arrays are stored float32

# The detectors in the checkpoint's saved scoring arrays, named rather than derived from
# `DETECTORS`; each must match at least one saved key or `harness_gate` aborts.
GATE_DETECTORS: tuple[str, ...] = (
    "coverage_R", "asymmetry_R", "joint_child_J", "pmi", "token_freq_survival",
    "recon_2a", "sibling_redundancy", "joint_child_mass", "outdegree",
)
REQUIRED_META = ("toy", "seed", "read", "n_tokens", "s_res_mode", "freeze_tag",
                 "git_sha", "git_dirty", "checkpoint", "checkpoint_weights_sha256")


# --- layout + write guard ---
def artifact_dir(out: Path, tag: str, seed: int, toy: str, read: str) -> Path:
    """`<out>/<tag>/seed<N>/<toy>/<read>`: seed and read in the path, so no two runs share files."""
    return Path(out) / str(tag) / f"seed{int(seed)}" / str(toy) / str(read)


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


# --- provenance ---
def git_provenance(cwd: Path) -> dict:
    """`{git_sha, git_dirty}`, or `"unavailable"` / `None` off a git checkout (e.g. the server)."""
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
    """Hash of the checkpoint's weight file, so a retrained checkpoint at the same path shows."""
    if not ckpt:
        return "n/a"
    d = Path(ckpt)
    for name in ("sae_weights.safetensors", "sae.safetensors", "sae_weights.pt"):
        if (d / name).exists():
            return file_sha256(d / name)
    cands = sorted(list(d.glob("*.safetensors")) + list(d.glob("*.pt")))
    return file_sha256(cands[0]) if cands else "unavailable"


# --- the harness gate (trained read only) ---
def gate_max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a - b|, or inf if the two differ in length or in which entries are finite.

    Comparing only the jointly finite entries would let a different universe pass at 0.0.
    """
    if a.numel() != b.numel():
        return float("inf")
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    if not bool((fa == fb).all()):
        return float("inf")
    return float((a[fa] - b[fb]).abs().max()) if int(fa.sum()) else 0.0


def harness_gate(read: RD.Read, saved_path: Path, tol: float = GATE_TOL) -> dict:
    """Hold `GATE_DETECTORS` to the checkpoint's saved arrays element-wise; raises on mismatch.

    Agreement pins the matcher, recovered universe, held-out draw and pair order. `s_res` is
    left out because the saved run stored it in probe mode.
    """
    saved = np.load(saved_path, allow_pickle=True)
    worst, where = 0.0, ""
    checked = 0
    per_det = {d: 0 for d in GATE_DETECTORS}
    for det in GATE_DETECTORS:
        vals = read.vals[det]
        for name in labels.LABELS:
            key = f"{det}__{name}"
            if key not in saved.files:
                continue
            mine = vals[read.y == labels._index(name)]
            theirs = torch.from_numpy(saved[key].astype(np.float64))
            d = gate_max_diff(mine, theirs)
            checked += 1
            per_det[det] += 1
            if d > worst:
                worst, where = d, key
    # every detector must be compared, or a renamed one drops out while the gate prints OK
    absent = [d for d, n in per_det.items() if n == 0]
    if absent:
        raise AssertionError(
            f"harness gate found no comparable keys for {absent} in {saved_path}. A detector "
            f"the gate cannot see is a detector the gate does not check, and the remaining "
            f"{checked} comparisons would still have printed OK.")
    if not worst <= tol:
        raise AssertionError(
            f"HARNESS GATE FAILED at {where}: max diff {worst:.3e} > {tol:.0e}. The universe, "
            f"matcher or held-out draw does not reproduce the saved run, so every number "
            f"downstream would be computed on a different population.\n"
            f"BEFORE debugging the matcher: an INFINITE diff on coverage_R, asymmetry_R or pmi "
            f"against a reference written before the scorability mask is EXPECTED and is not a "
            f"matcher problem. `detectors.MASKED_DETECTORS` NaNs those three below the support "
            f"floor, so their finite pattern legitimately differs from any pre-mask array.\n"
            f"A regenerated reference does NOT revalidate the mask -- it comes from the same "
            f"masked `compute_all`, so the gate would then be comparing this code with itself, "
            f"which is the one thing it exists to rule out. Regenerating restores the gate for "
            f"FUTURE runs only. That the mask changes exactly these three detectors and no "
            f"others is pinned separately, by "
            f"tests_local/test_detector_mask.py::test_a_pre_mask_reference_aborts_the_harness_gate.")
    return {"gate_max_diff": worst, "gate_worst_key": where, "gate_keys_checked": checked,
            "gate_detectors": list(GATE_DETECTORS), "gate_reference": str(saved_path)}


# --- one read, end to end ---
def run_read(read: RD.Read) -> tuple[dict, dict]:
    """Grade one read and assemble the report and the arrays to persist."""
    vals = {m: read.vals[m] for m in METRICS}
    gate_vals = {g: read.gate_vals[g] for g in GATES}
    null = C.null_mask(read.pairs, read.feats, read.pair_labels, NULL_CLASS)
    eval_null_idx = [i for i in range(len(read.pairs)) if bool(null[i])]

    in_universe = read.recovered
    n_total = RD.class_totals(read.pair_labels)
    n_recovered = RD.class_recovered(read.pair_labels, in_universe)

    # clauses read the gates; the metrics ride along for the diagnostics
    exprs = E.grade_rules(vals | gate_vals, read.y, n_total, eval_null_idx,
                          EXPRESSIONS, probe_available=(read.s_res_mode == "probe"))
    # `class_counts` counts recovered pairs on the scored frame; cross-check the answer key's count
    for name, c in next(iter(exprs.values()))["counts"].items():
        if name in n_recovered and c["N_recovered"] != n_recovered[name]:
            raise RuntimeError(f"recovered-pair count disagrees for {name}: scored frame says "
                               f"{c['N_recovered']}, answer key says {n_recovered[name]}")

    # scorability splits "no rule fired" into rejected-by-all and unscorable-for-all
    overlap = E.rule_overlap({k: v["_mask"] for k, v in exprs.items()}, read.y, eval_null_idx,
                             scorables={k: v["_scorable"] for k, v in exprs.items()})
    expr_masks = {k: v.pop("_mask") for k, v in exprs.items()}      # not JSON-serialisable
    expr_scorable = {k: v.pop("_scorable") for k, v in exprs.items()}
    # Each metric's `constant_target` uses the first designated rule reaching it. Clauses name
    # gates, so metrics inherit through `GATE_SOURCES`; without that every `constant_target`
    # would be None and the degenerate-separation flag would never fire.
    metric_target: dict[str, tuple[str, ...]] = {}
    for ename in DESIGNATED:
        target = EXPRESSIONS[ename]["target"]
        for _pred, g in EXPRESSIONS[ename]["clauses"]:
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
    for g in GATES:
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


# --- CLI ---
def _manifest_main(args) -> None:
    """`--write-manifest` / `--verify-manifest`: the freeze record and its check."""
    from scoring.benchmark import manifest as M

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [0]
    toys = args.toys.split(",") if args.toys else list(TOYS)
    if args.write_manifest:
        m = M.build_manifest(tag=args.tag, seeds=seeds, toys=toys, notes=args.notes)
        path = M.write_manifest(args.out, m, force=args.force)
        print(f"freeze record -> {path}")

    # load, never rebuild: a rebuilt record compares the evaluator with itself
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


def _cross_world_main(args) -> None:
    """Build the benchmark-wide tables from the artifacts on disk, without recomputing anything."""
    from scoring.benchmark import aggregate as AG

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [0]
    toys = tuple(args.toys.split(",")) if args.toys else TOYS
    root = Path(args.out) / args.tag
    wrote = []
    for seed in seeds:
        for read in ("oracle", "trained"):
            worlds = []
            for toy in toys:
                js = root / f"seed{seed}" / toy / read / "expressions.json"
                if not js.exists():
                    continue
                blob = json.loads(js.read_text())
                meta = blob.get("__meta__") or {}
                worlds.append({
                    "toy": blob.get("toy", toy), "seed": blob.get("seed", seed),
                    "read": blob.get("read", read),
                    "freeze_tag": meta.get("freeze_tag"),
                    "git_sha": meta.get("git_sha"),
                    "report_schema": meta.get("report_schema"),
                    # fingerprint of the settings that decide a verdict; must match to pool
                    "settings_sha256": hashlib.sha256(json.dumps(
                        {k: meta.get(k) for k in ("gates", "gate_constants",
                                                  "ruleset_version", "gate_constant_set",
                                                  "min_scorable_support")},
                        sort_keys=True, default=str).encode()).hexdigest()[:16],
                    "expressions": blob.get("expressions") or {},
                })
            if not worlds:
                continue
            rows = AG.combine_all(worlds, list(EXPRESSIONS), expected_toys=tuple(toys))
            d = root / "tables"
            d.mkdir(parents=True, exist_ok=True)
            md = d / f"cross_world_seed{seed}_{read}.md"
            md.write_text(AG.format_cross_world_md(rows, seed, read), encoding="utf-8")
            csvp = AG.write_cross_world_csv(rows, d / f"cross_world_seed{seed}_{read}.csv",
                                            seed, read)
            wrote += [md, csvp]
            print(f"[cross-world] seed {seed} {read}: {len(worlds)}/{len(toys)} worlds -> {md}")
            for name, r in rows.items():
                flag = "  <-- DISAGREES with the within-world verdict" if \
                    r["disagrees_with_within_world"] else ""
                over = ", ".join(f"{k}={v:.3f}" for k, v in
                                 sorted((r.get("leak_exceedances") or {}).items()))
                print(f"    {name:<22} {r['verdict']:<22} "
                      f"worst-FPR {AG._f(r['eval_null_fpr_worst'])} "
                      f"({r['eval_null_fpr_worst_world'] or '--'})"
                      + (f"  LEAKS {over}" if over else "") + flag)
    if not wrote:
        raise SystemExit(f"no artifacts found under {root}; nothing to roll up")


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
    ap.add_argument("--cross-world", action="store_true",
                    help=("build the benchmark-wide verdict table from an existing tag tree and "
                          "write it to <out>/<tag>/tables/. A rule's target lives in one world "
                          "and its leakage in the others, so no per-world verdict can see it."))
    args = ap.parse_args()

    if args.write_manifest or args.verify_manifest:
        _manifest_main(args)
        return
    if args.cross_world:
        _cross_world_main(args)
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
            # `write_artifacts` refuses any other value
            "report_schema": REPORT_SCHEMA,
            # ties the artifact to the code; `git_sha` is "unavailable" on the server tree
            "evaluator_sha256": M.evaluator_sha256(Path(__file__).resolve().parents[2]),
            # lets `verify_manifest` check both reads of a (seed, toy) built the same world
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(read.extra.get("resolved_config", {}), sort_keys=True,
                           default=str).encode()).hexdigest(),
            "freeze_tag": args.tag, "checkpoint": ckpt,
            "checkpoint_weights_sha256": checkpoint_weights_sha256(ckpt),
            # with nothing fitted, the gates and their constants decided every pass
            "gates": list(GATES),
            "gate_constants": {k: CONSTANTS[k] for k in GATE_CONSTANT_KEYS},
            **ruleset_stamp(),
            "min_scorable_support": MIN_SCORABLE_SUPPORT,
            "scoring_precision": "float64",
            "support": report.get("support"),
            # read off the Read, so the recorded draw is the one actually used
            "scoring_sample_seed": read.extra.get("scoring_sample_seed"),
            "matching_sample_seed": read.extra.get("matching_sample_seed"),
            "held_out_draw_rule": "seed + 10000 (scoring.core.grid.held_out_sample_seed)",
            "probe_seed": ("the recovered position of the child "
                           "(scoring/core/detectors.py:fit_probe_directions)"),
            "probe_fit_sample_seed": read.extra.get("probe_fit_sample_seed"),
            "probe_fit_labels": read.extra.get("probe_fit_labels"),
            "probe_fit_rule": "seed + 20000 (scoring.benchmark.registry.probe_fit_sample_seed)",
            # known limits of the probe-draw separation, recorded rather than fixed
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
        # The verdict is within-world: a rule whose target is absent reads UNTESTABLE while it
        # may still accept confound pairs, so the exceedances are printed beside it.
        over = blk.get("leak_exceedances") or {}
        leak_s = ("  LEAKS " + " ".join(f"{k}={v:.3f}" for k, v in sorted(over.items()))
                  if over else "")
        print(f"    {name:<22} recall {rec:>6}  eval-null FPR {fpr_s:>7}  "
              f"{blk['verdict']}{leak_s}")


if __name__ == "__main__":
    main()
