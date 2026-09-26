"""Score one dial point through the frozen evaluator and write its artifacts. No matcher.

Has its own CLI because the benchmark's limits `--read` and `--toy` to fixed choices;
`run_read` and `write_artifacts` are reused as is. Artifacts go to
<out>/<tag>/seed<N>/<toy>/<kind>/<dials>/<readout>/ (scores.npz, expressions.json,
run_config.json, census.json), never under a benchmark tag.

  python -m synthdict.run_synth --toy only_isa --kind hedging --gamma-rel 1 --tag T [--no-probe]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scoring.benchmark.manifest import evaluator_sha256
from scoring.benchmark.registry import GATES, REPORT_SCHEMA
from scoring.benchmark.run_benchmark import git_provenance, run_read, write_artifacts
from scoring.core.gates import GATE_CONSTANT_KEYS
from scoring.core.grid import held_out_sample_seed
from scoring.core.registry import CONSTANTS, ruleset_stamp
from scoring.core.world import regenerate_world

from synthdict.census import absorption_classifier_sha256, run_census
from synthdict.corruptions import (SPLIT_ROLES, AbsorptionDials, CompositionDials,
                                   HedgingDials, SplitDials, build_corruption)
from synthdict.planted import READOUTS
from synthdict.read import ACTS_MODELS, resolved_config, synthetic_read

SYNTHDICT_SOURCES = ("synthdict",)
_ROOT = Path(__file__).resolve().parents[1]


def synthdict_sha256(root: Path | None = None) -> str:
    """Content hash of this package's `.py` files, the study-side analogue of
    `evaluator_sha256` for the git-less server copy."""
    root = _ROOT if root is None else Path(root)
    h = hashlib.sha256()
    for rel in SYNTHDICT_SOURCES:
        for p in sorted((root / rel).rglob("*.py")):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def toygen_commit(root: Path = _ROOT) -> str:
    """The last commit touching `toygen/`, or "unavailable" off a git checkout."""
    try:
        return subprocess.run(["git", "log", "-1", "--format=%H", "--", "toygen"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip() \
            or "unavailable"
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return "unavailable"


@dataclasses.dataclass(frozen=True)
class NoDamageDials:
    """No damage: the perfect dictionary, W = g. `acts_mode` picks the oracle column (`true_A`,
    the true coefficients) or the undamaged one (`nnls`, read through the encoder).

    A dials class only to fit the (kind, dials, readout) addressing; it never reaches
    `build_corruption`.
    """

    KIND = "none"

    acts_mode: str

    def __post_init__(self) -> None:
        if self.acts_mode not in ACTS_MODELS:
            raise ValueError(f"unknown acts_mode {self.acts_mode!r}; the encoder is 'nnls' "
                             f"('true_A' is the passthrough anchor)")


def damage_of(dials) -> object | None:
    """The damage `dials` describes, or None for the no-damage point."""
    return None if isinstance(dials, NoDamageDials) else dials


def acts_mode_of(dials) -> str:
    """The encoder a point runs under. Only the no-damage point may choose."""
    return dials.acts_mode if isinstance(dials, NoDamageDials) else "nnls"


def dial_dirname(dials) -> str:
    """The dial point's directory name, one branch per damage."""
    if isinstance(dials, NoDamageDials):
        # the encoder is all that separates oracle from undamaged; without it they share a
        # directory and overwrite each other
        return f"acts_{dials.acts_mode}"
    if isinstance(dials, AbsorptionDials):
        return f"beta{dials.beta:g}-eta{dials.eta:g}-f{dials.edge_fraction:g}"
    if isinstance(dials, HedgingDials):
        return f"gamma{dials.gamma_rel:g}-f{dials.edge_fraction:g}"
    if isinstance(dials, SplitDials):
        return (f"k{dials.k:g}-skew{dials.skew:g}-f{dials.fraction:g}"
                f"-roles_{'+'.join(dials.roles)}")
    if isinstance(dials, CompositionDials):
        return f"pi{dials.pi:g}-f{dials.fraction:g}"
    raise ValueError(f"no directory naming registered for {type(dials).__name__}")


def point_dir(out, tag: str, seed: int, toy: str, kind: str, dials, readout: str) -> Path:
    """The artifact path for one dial point, shared by the driver and the launcher's resume
    check."""
    return (Path(out) / tag / f"seed{int(seed)}" / toy / kind / dial_dirname(dials) / readout)


def provenance_tuple(meta: dict) -> tuple:
    """What one tag must not mix, from an artifact's meta. None of it is in the path, so a
    shrunken or probe-less run could otherwise overwrite, or be resumed as, a full one."""
    return (meta.get("acts_model"), meta.get("readout"),
            meta.get("corruption", "absorption"), int(meta.get("n_tokens", -1)),
            json.dumps(meta.get("cfg_overrides"), sort_keys=True),
            meta.get("s_res_mode"))


def build_run_config(read, toy: str, seed: int, dials, n_tokens: int,
                     census_seed: int | None) -> dict:
    """The full record of one dial point: world, dials, selections, encoder, readout,
    constants and code identity."""
    ex = read.extra
    L, F = int(ex["n_latents"]), int(read.F)
    ftl = ex["feature_to_latents"]
    kind = ex["corruption_kind"]
    return {
        "toy": toy,
        "toy_config": ex["resolved_config"],
        "world_seed": int(seed),
        "sample_seeds": {"scoring": ex["scoring_sample_seed"],
                         "probe_fit": ex["probe_fit_sample_seed"],
                         "in_sample": census_seed},
        "n_tokens": int(n_tokens),
        "kind": kind,
        "dials": ex["dials"],
        "derived": ex["details"],
        "selection": {
            "corrupted_edges": [list(e) for e in ex["corrupted_edges"]],
            "corrupted_features": list(ex["corrupted_features"]),
            "hedged_children": list(ex["hedged_children"]),
            "composition_pairs": [list(p) for p in ex["composition_pairs"]],
            "shard_map": ([[f, list(ftl[f])] for f in ex["corrupted_features"]]
                          if kind == "split" else []),
        },
        "acts_model": {"name": ex["acts_model"], "lambda": ex["nnls_lambda"]},
        "zeroed_rate": ex["zeroed_rate"],
        "zeroed_rate_damaged": ex["zeroed_rate_damaged"],
        "fvu": ex["fvu"],
        "readout": ex["readout"],
        "pair_rule": ex["corrupted_pair_rule"],
        "severity_kind": ex["severity_kind"],
        "realized_severity": [float(x) for x in ex["realized_severity"]],
        "L": L, "F": F, "L_over_F": L / F,
        "n_lost_features": int(ex["n_lost_features"]),
        "detector_constants": dict(CONSTANTS),
        "gates_location": "expressions.json -> __meta__.gate_constants (fixed; nothing fitted)",
        "code": {"evaluator_sha256": evaluator_sha256(),
                 "synthdict_sha256": synthdict_sha256(),
                 "toygen_commit": toygen_commit(),
                 "absorption_classifier_sha256": absorption_classifier_sha256(),
                 **git_provenance(_ROOT)},
    }


def run_dial_point(toy: str, seed: int, dials, n_tokens: int, out: Path, tag: str,
                   readout: str = "identity", with_probe: bool = True,
                   with_census: bool = True, force: bool = False,
                   cfg_overrides: dict | None = None) -> list[Path]:
    """Score one (toy, seed, dials) under the declared readout; write its artifacts."""
    rc = resolved_config(toy, seed, cfg_overrides)
    damage, acts_mode = damage_of(dials), acts_mode_of(dials)
    t0 = time.time()
    read = synthetic_read(toy, seed, damage, readout, n_tokens, with_probe=with_probe,
                          acts_mode=acts_mode, cfg_overrides=cfg_overrides)
    report, arrays = run_read(read)
    report["secs"] = round(time.time() - t0, 1)
    ex = read.extra
    arrays["corrupted_pair"] = ex["corrupted_pair"].numpy()
    arrays["touched_pair"] = ex["touched_pair"].numpy()
    arrays["realized_severity"] = ex["realized_severity"].numpy()

    d = point_dir(out, tag, seed, toy, ex["corruption_kind"], dials, readout)
    npz = d / "scores.npz"
    if npz.exists():
        # The encoder, world shape and probe state are not in the path; one tag holds one
        # construct, even under --force.
        prev = provenance_tuple(json.loads(str(np.load(npz, allow_pickle=False)["__meta__"])))
        now = provenance_tuple({"acts_model": ex["acts_model"], "readout": readout,
                                "corruption": ex["corruption_kind"], "n_tokens": n_tokens,
                                "cfg_overrides": cfg_overrides, "s_res_mode": read.s_res_mode})
        if prev != now:
            raise FileExistsError(
                f"{d} holds an (acts_model, readout, kind, n_tokens, cfg_overrides, s_res_mode)="
                f"{prev} artifact; refusing to overwrite with {now} even under --force - one tag "
                f"holds one construct.")
        if not force:
            raise FileExistsError(f"refusing to overwrite the existing artifact at {d}; pass "
                                  f"--force to overwrite.")

    cen = None
    if with_census:
        # the read's corruption depends on (world seed, dials) only, so any draw rebuilds it
        world = regenerate_world(rc, sample_seed=held_out_sample_seed(int(seed)),
                                 n_tokens=n_tokens)
        corruption = (build_corruption(world, damage, world_seed=int(seed), readout=readout)
                      if damage is not None else None)
        cen = run_census(rc, corruption, seed, n_tokens=n_tokens, readout=readout)
    run_config = build_run_config(read, toy, seed, dials, n_tokens,
                                  census_seed=cen["sample_seed"] if cen is not None else None)

    meta = {
        "toy": toy, "seed": int(seed), "read": "synthetic", "n_tokens": int(n_tokens),
        "s_res_mode": read.s_res_mode, "report_schema": REPORT_SCHEMA,
        "freeze_tag": tag,                      # a study tag, not a benchmark freeze
        "checkpoint": "synthetic:none",
        "checkpoint_weights_sha256": hashlib.sha256(ex["W_raw"].numpy().tobytes()).hexdigest(),
        "evaluator_sha256": run_config["code"]["evaluator_sha256"],
        "synthdict_sha256": run_config["code"]["synthdict_sha256"],
        "corruption": ex["corruption_kind"], "dials": ex["dials"],
        "corrupted_edges_sha256": hashlib.sha256(
            json.dumps(list(ex["corrupted_edges"])).encode()).hexdigest(),
        "n_corrupted_edges": len(ex["corrupted_edges"]),
        "realized_severity_median": ex["realized_severity_median"],
        "zeroed_rate": ex["zeroed_rate"], "zeroed_rate_damaged": ex["zeroed_rate_damaged"],
        "fvu": ex["fvu"], "fvu_true_A": ex["fvu_true_A"], "n_holed_total": ex["n_holed_total"],
        "corrupted_features": list(ex["corrupted_features"]),
        "severity_kind": ex["severity_kind"],
        "corrupted_pair_rule": ex["corrupted_pair_rule"],
        # with nothing fitted, the gate constants alone decide a pass
        "gates": list(GATES),
        "gate_constants": {k: CONSTANTS[k] for k in GATE_CONSTANT_KEYS},
        **ruleset_stamp(),
        "support": report.get("support"),
        "readout": readout, "acts_model": ex["acts_model"],
        "planted_map_sha256": ex["planted_map_sha256"],
        "n_latents": ex["n_latents"],
        "true_l0": ex["true_l0"],
        "latent_l0": ex["latent_l0"], "feature_l0": ex["feature_l0"],
        "scoring_sample_seed": ex["scoring_sample_seed"],
        "probe_fit_sample_seed": ex["probe_fit_sample_seed"],
        "probe_fit_labels": ex["probe_fit_labels"],
        "cfg_overrides": cfg_overrides,
        "run_config": run_config,
    } | git_provenance(_ROOT)

    # A forced rewrite must not leave a previous run's side files beside the new artifact.
    for stale in ("census.json", "run_config.json"):
        (d / stale).unlink(missing_ok=True)
    write_artifacts(d, arrays, report, meta, force=force)
    (d / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if cen is not None:
        (d / "census.json").write_text(json.dumps(cen, indent=2), encoding="utf-8")
    print(f"[{toy} seed{seed} {dial_dirname(dials)} {readout}] wrote {d} "
          f"({report['secs']}s, sev_med={ex['realized_severity_median']:.3f}, "
          f"fvu={ex['fvu']['scoring']:.3f}, zeroed={ex['zeroed_rate']['scoring']:.2e})")
    return [d]


# CLI kind -> (dials class, its knobs); a knob of another kind is refused, not ignored
DIALS_BY_KIND = {
    "absorption": (AbsorptionDials, ("beta", "eta", "edge_fraction")),
    "hedging": (HedgingDials, ("gamma_rel", "edge_fraction")),
    "split": (SplitDials, ("k", "roles", "skew", "fraction")),
    "composition": (CompositionDials, ("pi", "fraction")),
    # the oracle and undamaged columns
    "none": (NoDamageDials, ("acts_mode",)),
}


def dials_from_args(args) -> object:
    """The damage's dials from the CLI, refusing another damage's knobs.

    A dials field with no default must be supplied, so a shared flag cannot inherit another
    damage's default.
    """
    cls, names = DIALS_BY_KIND[args.kind]
    required = {f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING}
    kwargs = {}
    for n in names:
        v = getattr(args, n)
        if v is None:
            if n in required:
                raise SystemExit(f"--kind {args.kind} requires --{n.replace('_', '-')}")
            continue
        kwargs[n] = v
    foreign = [n for k, (_c, ns) in DIALS_BY_KIND.items() if k != args.kind
               for n in ns if n not in names and getattr(args, n, None) is not None]
    if foreign:
        raise SystemExit(f"--kind {args.kind} does not take "
                         f"{', '.join('--' + n.replace('_', '-') for n in dict.fromkeys(foreign))}")
    return cls(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="absorption", choices=tuple(DIALS_BY_KIND))
    # every dial defaults to None so "not supplied" is distinguishable from a default value
    ap.add_argument("--beta", type=float, help="absorption: parent carry into the child row")
    ap.add_argument("--eta", type=float, help="absorption: share of co-fire tokens holed")
    ap.add_argument("--edge-fraction", type=float, help="absorption, hedging (default 1.0)")
    ap.add_argument("--gamma-rel", type=float, help="hedging: mixing as a multiple of gamma*")
    ap.add_argument("--k", type=int, help="split: shards per split feature")
    ap.add_argument("--roles", nargs="+", choices=SPLIT_ROLES, help="split: roles to split")
    ap.add_argument("--skew", type=float, help="split: 0 = equal shard shares (default)")
    ap.add_argument("--fraction", type=float,
                    help="split: share per role; composition: share of partner pairs "
                         "(default 1.0)")
    ap.add_argument("--pi", type=float, help="composition: share of co-firing tokens")
    ap.add_argument("--acts-mode", choices=ACTS_MODELS,
                    help="none: true_A = oracle column, nnls = undamaged column")
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--tag", default="SYNTH-R1")
    ap.add_argument("--out", default="outputs_local/synthdict")
    ap.add_argument("--readout", default="identity", choices=READOUTS)
    ap.add_argument("--n-roots", type=int, default=None,
                    help="shrink the world (cheap local runs); recorded in resolved_config")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    dials = dials_from_args(args)
    overrides = {"n_roots": args.n_roots} if args.n_roots is not None else None
    run_dial_point(args.toy, args.seed, dials, args.n_tokens, Path(args.out), args.tag,
                   readout=args.readout, with_probe=not args.no_probe,
                   with_census=not args.no_census, force=args.force, cfg_overrides=overrides)


if __name__ == "__main__":
    main()
