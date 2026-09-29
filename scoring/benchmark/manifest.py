"""The freeze record `<out>/<TAG>/MANIFEST.json`, and the check that an artifact tree matches it.

The record stores the rules, gates and settings as evaluated, plus a content hash of the evaluator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scoring.benchmark.registry import (BAR_CONFOUND_LEAK, BAR_EVAL_NULL_FPR, BAR_RECALL,
                                        CONSTANT_TOL, DESIGNATED, EXPRESSIONS, GATES,
                                        METRICS, MIN_SCORABLE_SUPPORT, NULL_CLASS,
                                        REPORT_SCHEMA)
from scoring.core.gates import GATE_CONSTANT_KEYS
from scoring.core.registry import CONSTANTS, ruleset_stamp

FROZEN_DOC = "PRECOMMIT.md"
MANIFEST_NAME = "MANIFEST.json"

# The source that decides what a number means, hashed by content because `git_sha` is
# "unavailable" on the server's rsync tree. A new top-level dependency must be added here.
EVALUATOR_SOURCES: tuple[str, ...] = (
    "scoring/benchmark", "scoring/core", "toygen", "metrics/sres.py",
    "metrics/rules",
    # single files: the rest of these directories is code the benchmark never imports
    "scoring/oracle/validate_metrics.py",
    "scoring/trained/loaders.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluator_sha256(root: Path | None = None) -> str:
    """SHA-256 over every `.py` in `EVALUATOR_SOURCES`, path and bytes, sorted by path.

    Hashing the path makes a rename a change; `__pycache__` is skipped as machine-dependent.
    """
    r = Path(root) if root is not None else _repo_root()
    h = hashlib.sha256()
    files: list[Path] = []
    for rel in EVALUATOR_SOURCES:
        p = r / rel
        if p.is_dir():
            files += [q for q in p.rglob("*.py") if "__pycache__" not in q.parts]
        elif p.exists():
            files.append(p)
    for q in sorted(files, key=lambda x: str(x.relative_to(r))):
        h.update(str(q.relative_to(r)).encode())
        h.update(b"\0")
        h.update(q.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def checkpoint_pins(toys, seeds, search_roots: tuple[str, ...] = ()) -> dict:
    """Path and weight hash of the checkpoint for each (toy, seed), `None` where none was found."""
    from scoring.benchmark.run_benchmark import checkpoint_weights_sha256

    roots = [Path(x) for x in (search_roots or ("checkpoints", "../toysae/checkpoints"))]
    out: dict[str, dict] = {}
    for toy in toys:
        out[str(toy)] = {}
        for seed in seeds:
            hit = None
            for r in roots:
                if not r.exists():
                    continue
                for d in sorted(r.glob(f"{toy}-*-s{int(seed)}")):
                    hit = {"path": str(d.resolve()),
                           "weights_sha256": checkpoint_weights_sha256(str(d))}
                    break
                if hit:
                    break
            out[str(toy)][str(int(seed))] = hit
    return out


def _sha256_text(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(tag: str, seeds: list[int], toys: list[str],
                   repo_root: Path | None = None, notes: str = "") -> dict:
    """Assemble the freeze record from the registry and the document; writes nothing."""
    from scoring.benchmark.run_benchmark import git_provenance

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    doc = root / FROZEN_DOC
    return {
        "tag": tag,
        # required, not defaulted: older schemas use the same rate keys for other quotients
        "report_schema": REPORT_SCHEMA,
        "evaluator_sha256": evaluator_sha256(root),
        "evaluator_sources": list(EVALUATOR_SOURCES),
        **git_provenance(root),
        "frozen_doc": {"path": FROZEN_DOC, "sha256": _sha256_text(doc)},
        "registry": {name: {"target": list(spec["target"]),
                            "clauses": [list(c) for c in spec["clauses"]],
                            "text": spec["text"]}
                     for name, spec in EXPRESSIONS.items()},
        "designated_rules": list(DESIGNATED),
        "metrics": list(METRICS),
        # With nothing fitted, the gates and their constants are the decision rule.
        "gates": list(GATES),
        "settings": {
            # the same key list the artifacts stamp, so the two records cannot diverge
            "gate_constants": {k: CONSTANTS[k] for k in GATE_CONSTANT_KEYS},
            **ruleset_stamp(),
            "null_population": "the WHOLE null; nothing is fitted, so there is no split",
            "constant_tol": CONSTANT_TOL,
            "null_class": NULL_CLASS,
            "bars": {"recall_given_recovery": BAR_RECALL,
                     "eval_null_fpr": BAR_EVAL_NULL_FPR,
                     "confound_leak": BAR_CONFOUND_LEAK},
            # Asymmetric on purpose: an unscorable target is a miss, an unscorable null or
            # confound pair must not dilute a rate (PRECOMMIT.md s8).
            "recall_denominator": "N_recovered",
            "fpr_denominator": "N_scorable within the whole null",
            "leak_denominator": "N_scorable within each confound class",
            "min_scorable_support": MIN_SCORABLE_SUPPORT,
            "support_floor_applies_to": ["target", "null", "each confound"],
            "scoring_precision": "float64",
        },
        "checkpoints": checkpoint_pins(toys, seeds),
        "seed_plan": {"seeds": list(seeds), "toys": list(toys),
                      "reads": ["oracle", "trained"],
                      "held_out_draw": "train_seed + 10000 (grid.held_out_sample_seed)",
                      "probe_seed": "the recovered position of the child (detectors.s_res_probe)"},
        "notes": notes,
    }


def write_manifest(out: Path, manifest: dict, force: bool = False) -> Path:
    """Write `<out>/<tag>/MANIFEST.json`, refusing to overwrite a freeze record without `force`."""
    d = Path(out) / str(manifest["tag"])
    path = d / MANIFEST_NAME
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite the existing freeze record at {path}. The manifest is the "
            f"record of what was frozen; replacing it in place loses that. Pass force=True.")
    d.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_manifest(out: Path, tag: str) -> dict:
    """Read `<out>/<tag>/MANIFEST.json`; raises when absent, as a default would pass vacuously."""
    path = Path(out) / str(tag) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no freeze record at {path}. `--verify-manifest` checks artifacts against the SAVED "
            f"manifest; rebuilding it from the live code would compare the evaluator with "
            f"itself and pass on an unfrozen tree. Write one with --write-manifest first.")
    return json.loads(path.read_text())


def _artifact_meta(d: Path) -> dict | None:
    js = d / "expressions.json"
    if not js.exists():
        return None
    try:
        return json.loads(js.read_text()).get("__meta__")
    except (json.JSONDecodeError, OSError):
        return None


def verify_manifest(out: Path, manifest: dict) -> dict:
    """Check the artifact tree against the freeze record; returns `{ok, n_artifacts, problems}`.

    `manifest` must come from `load_manifest`: a rebuilt one makes every check compare the code
    with itself.
    """
    tag = str(manifest["tag"])
    root = Path(out) / tag
    seeds = [int(s) for s in manifest["seed_plan"]["seeds"]]
    toys = list(manifest["seed_plan"]["toys"])
    reads = list(manifest["seed_plan"]["reads"])
    problems: list[str] = []
    n = 0

    # ---- the record against the code ----
    live = build_manifest(tag, seeds, toys)
    if manifest.get("registry") != live["registry"]:
        changed = sorted(set(manifest.get("registry", {})) ^ set(live["registry"])) or [
            k for k in live["registry"]
            if manifest.get("registry", {}).get(k) != live["registry"][k]]
        problems.append(f"registry in the freeze record disagrees with the code: {changed}")
    if manifest.get("designated_rules") != live["designated_rules"]:
        problems.append("designated_rules in the freeze record disagrees with the code")
    if manifest.get("metrics") != live["metrics"]:
        problems.append("metrics in the freeze record disagrees with the code")
    if manifest.get("gates") != live["gates"]:
        problems.append("gates in the freeze record disagrees with the code")
    for k, want in live["settings"].items():
        got = manifest.get("settings", {}).get(k, "<absent>")
        if got != want:
            problems.append(f"settings.{k} is {got!r} in the freeze record, {want!r} in the code")
    if manifest.get("report_schema") != REPORT_SCHEMA:
        problems.append(f"report_schema is {manifest.get('report_schema')!r} in the freeze "
                        f"record, {REPORT_SCHEMA} in the code")
    if manifest.get("frozen_doc", {}).get("sha256") != live["frozen_doc"]["sha256"]:
        problems.append(f"{FROZEN_DOC} has changed since the freeze record was written")

    # ---- provenance ----
    # `git_sha` is "unavailable" where the runs happen, so a git check alone passes vacuously.
    code_hash = manifest.get("evaluator_sha256")
    live_hash = live["evaluator_sha256"]
    if manifest.get("git_sha") in (None, "", "unavailable") and not code_hash:
        problems.append(
            "git_sha is unavailable and no evaluator_sha256 was recorded, so NOTHING ties these "
            "artifacts to the tagged commit")
    if code_hash and code_hash != live_hash:
        problems.append(f"evaluator_sha256 {code_hash[:12]} in the freeze record, {live_hash[:12]} "
                        f"for the current source: the code changed since the freeze")
    if manifest.get("git_dirty"):
        problems.append("git_dirty is true: the freeze was taken on uncommitted code")

    # ---- the artifact tree ----
    by_key: dict[tuple[int, str], dict[str, dict]] = {}
    for seed in seeds:
        for toy in toys:
            for read in reads:
                d = root / f"seed{seed}" / toy / read
                meta = _artifact_meta(d)
                if meta is None:
                    problems.append(f"missing artifact: seed{seed}/{toy}/{read}")
                    continue
                if not (d / "scores.npz").exists():
                    problems.append(f"seed{seed}/{toy}/{read}: expressions.json without scores.npz")
                n += 1
                by_key.setdefault((seed, toy), {})[read] = meta
                if meta.get("freeze_tag") != tag:
                    problems.append(
                        f"seed{seed}/{toy}/{read}: freeze_tag is "
                        f"{meta.get('freeze_tag')!r}, manifest tag is {tag!r}")
                # path and metadata are two independent records of the same fact
                for key, want in (("seed", seed), ("toy", toy), ("read", read)):
                    if meta.get(key) != want:
                        problems.append(
                            f"seed{seed}/{toy}/{read}: path says {key}={want!r} but "
                            f"__meta__ says {key}={meta.get(key)!r}")
                if meta.get("report_schema") != REPORT_SCHEMA:
                    problems.append(
                        f"seed{seed}/{toy}/{read}: report_schema is "
                        f"{meta.get('report_schema')!r}, expected {REPORT_SCHEMA}. Its rate keys "
                        f"carry a different arithmetic under the same names.")
                a_hash = meta.get("evaluator_sha256")
                if code_hash and a_hash and a_hash != code_hash:
                    problems.append(
                        f"seed{seed}/{toy}/{read}: evaluator_sha256 {str(a_hash)[:12]} does not "
                        f"match the manifest's {str(code_hash)[:12]}; produced by another revision")
                pin = (manifest.get("checkpoints") or {}).get(toy, {}).get(str(seed))
                if read == "trained" and pin and meta.get("checkpoint"):
                    if str(meta["checkpoint"]) != str(pin.get("path")):
                        problems.append(
                            f"seed{seed}/{toy}/trained: checkpoint {meta['checkpoint']!r} is not "
                            f"the pinned {pin.get('path')!r}")
                    elif (meta.get("checkpoint_weights_sha256") not in (None, "n/a")
                          and meta["checkpoint_weights_sha256"] != pin.get("weights_sha256")):
                        problems.append(
                            f"seed{seed}/{toy}/trained: checkpoint weights hash does not match "
                            f"the pinned one; the checkpoint was retrained at the same path")

    # ---- cross-read agreement ----
    # The only place both reads of a (seed, toy) are compared. `n_tokens` must match too: `h` is
    # not prefix-stable across token counts, so different counts are different draws.
    for (seed, toy), got in sorted(by_key.items()):
        if len(got) < 2:
            continue
        o, t = got.get("oracle"), got.get("trained")
        if not (o and t):
            continue
        for key in ("scoring_sample_seed", "gate_constants", "n_tokens",
                    "probe_fit_sample_seed"):
            if o.get(key) != t.get(key):
                problems.append(
                    f"seed{seed}/{toy}: oracle and trained disagree on {key} "
                    f"({o.get(key)!r} vs {t.get(key)!r}); the two reads are not comparable")
        oc = (o.get("resolved_config_sha256"), t.get("resolved_config_sha256"))
        if all(oc) and oc[0] != oc[1]:
            problems.append(
                f"seed{seed}/{toy}: the two reads used different resolved configs; the world "
                f"drifted between training and scoring")

    # a seed or toy on disk that the plan never declared would silently widen the ensemble
    if root.exists():
        for sd in sorted(p for p in root.iterdir() if p.is_dir()):
            if not sd.name.startswith("seed"):
                continue
            try:
                got = int(sd.name[4:])
            except ValueError:
                problems.append(f"undeclared directory in the tag tree: {sd.name}")
                continue
            if got not in seeds:
                problems.append(f"{sd.name} is on disk but the seed plan declares {seeds}")
            for td in sorted(p for p in sd.iterdir() if p.is_dir()):
                if td.name not in toys:
                    problems.append(f"{sd.name}/{td.name} is on disk but not in the seed plan")

    if n == 0:
        problems.append("no artifact was found under this tag; a check over zero artifacts is "
                        "not a passed check")

    return {"ok": not problems, "n_artifacts": n, "problems": problems}
