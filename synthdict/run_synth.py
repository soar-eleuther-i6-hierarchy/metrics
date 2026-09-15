"""CLI driver: score one dial point through the FROZEN evaluator. No matcher anywhere.

Ships its own CLI because the benchmark's is deliberately closed (`--read choices=
("oracle","trained")`, `--toy choices=TOYS`); `run_read` and `write_artifacts` themselves are
name-agnostic and are reused verbatim. Artifacts land under their own root
(`outputs_local/synthdict/<TAG>/...`), NEVER under a benchmark tag — `verify_manifest`
correctly rejects unknown read dirs in a benchmark tree.

Layout:  <out>/<tag>/seed<N>/<toy>/<kind>/<dials>/<readout>/
           scores.npz         run_read arrays + corrupted_pair, realized_severity
           expressions.json   run_read report | __meta__
           census.json        annotation (manipulation check; see synthdict/census.py)

`<kind>` and `<dials>` come from the dials TYPE, so two damages under one tag can never collide
on a path:  absorption/beta0.4-eta0.4-f0.1 · split/k8-skew0-f1 · missing/f0.25 · noise/sigma0.3

Usage (one dial point; the grid launches many of these in parallel on the server):
  python -m synthdict.run_synth --toy only_isa --seed 0 --beta 0.4 --eta 0.4 \
      --edge-fraction 0.1 --tag SYNTH-R1 [--n-tokens 200000] [--no-probe] [--no-census]
      [--readout identity] [--out outputs_local/synthdict] [--force] [--n-roots 12]
  python -m synthdict.run_synth --toy only_isa --kind split --k 8 --readout union ...
  python -m synthdict.run_synth --toy only_isa --kind missing --fraction 0.25 ...
  python -m synthdict.run_synth --toy only_isa --kind noise --sigma 0.3 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scoring.benchmark.manifest import evaluator_sha256
from scoring.benchmark.registry import REPORT_SCHEMA
from scoring.benchmark.run_benchmark import git_provenance, run_read, write_artifacts
from scoring.core.grid import held_out_sample_seed
from scoring.core.world import regenerate_world

from synthdict.census import run_census
from synthdict.corruptions import (AbsorptionDials, MissingDials, NoiseDials, SplitDials,
                                   build_corruption)
from synthdict.planted import READOUTS
from synthdict.read import resolved_config, synthetic_read

SYNTHDICT_SOURCES = ("synthdict",)


def synthdict_sha256(root: Path | None = None) -> str:
    """Content hash of this package's `.py` files — the study-side analogue of
    `evaluator_sha256`, for runs on the git-less server copy."""
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    h = hashlib.sha256()
    for rel in SYNTHDICT_SOURCES:
        for p in sorted((root / rel).rglob("*.py")):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def dial_dirname(dials) -> str:
    """The dial point's directory name, one branch per damage.

    Absorption's form is unchanged from round 1 so its saved artifacts stay addressable.
    """
    if isinstance(dials, AbsorptionDials):
        return f"beta{dials.beta:g}-eta{dials.eta:g}-f{dials.edge_fraction:g}"
    if isinstance(dials, SplitDials):
        return f"k{dials.k:g}-skew{dials.skew:g}-f{dials.fraction:g}"
    if isinstance(dials, MissingDials):
        return f"f{dials.fraction:g}"
    if isinstance(dials, NoiseDials):
        return f"sigma{dials.sigma:g}"
    raise ValueError(f"no directory naming registered for {type(dials).__name__}")


def point_dir(out, tag: str, seed: int, toy: str, kind: str, dials, readout: str) -> Path:
    """THE artifact path for one dial point. One function, because the driver's write path and
    the grid's resume probe were built independently in two files: a desync is silent in both
    directions (the grid reruns everything, or skips points that do not exist and exits 0
    having produced nothing).

    `kind` is a path segment so two damages under one tag cannot collide on a dial name.
    """
    return (Path(out) / tag / f"seed{int(seed)}" / toy / kind / dial_dirname(dials) / readout)


def run_dial_point(toy: str, seed: int, dials: AbsorptionDials, n_tokens: int,
                   out: Path, tag: str, readout: str = "identity",
                   with_probe: bool = True, with_census: bool = True,
                   force: bool = False, cfg_overrides: dict | None = None,
                   acts_mode: str = "ridge") -> list[Path]:
    """Score one (toy, seed, dials) under the declared readout; write its three artifacts."""
    written = []
    rc = resolved_config(toy, seed, cfg_overrides)
    mode = readout
    t0 = time.time()
    read = synthetic_read(toy, seed, dials, mode, n_tokens, with_probe=with_probe,
                          cfg_overrides=cfg_overrides, acts_mode=acts_mode)
    report, arrays = run_read(read)
    report["secs"] = round(time.time() - t0, 1)

    ex = read.extra
    arrays["corrupted_pair"] = ex["corrupted_pair"].numpy()
    arrays["realized_severity"] = ex["realized_severity"].numpy()

    w_bytes = ex["W_raw"].numpy().tobytes()
    meta = {
        "toy": toy, "seed": int(seed), "read": "synthetic", "n_tokens": int(n_tokens),
        "s_res_mode": read.s_res_mode, "report_schema": REPORT_SCHEMA,
        "freeze_tag": tag,                      # a study tag, explicitly NOT a benchmark freeze
        "checkpoint": "synthetic:none",
        "checkpoint_weights_sha256": hashlib.sha256(w_bytes).hexdigest(),
        "evaluator_sha256": evaluator_sha256(),
        "synthdict_sha256": synthdict_sha256(),
        "corruption": ex["corruption_kind"], "dials": ex["dials"],
        "corrupted_edges_sha256": hashlib.sha256(
            json.dumps(list(ex["corrupted_edges"])).encode()).hexdigest(),
        "n_corrupted_edges": len(ex["corrupted_edges"]),
        "realized_severity_median": ex["realized_severity_median"],
        "support_flip_rate": ex["support_flip_rate"], "fvu": ex["fvu"],
        "fvu_true_A": ex["fvu_true_A"], "n_holed_total": ex["n_holed_total"],
        "corrupted_features": list(ex["corrupted_features"]),
        "severity_kind": ex["severity_kind"],
        "corrupted_pair_rule": ex["corrupted_pair_rule"],
        "readout": mode, "acts_mode": ex["acts_mode"],
        "planted_map_sha256": ex["planted_map_sha256"],
        "n_latents": ex["n_latents"],
        "true_l0": ex["true_l0"],
        "latent_l0": ex["latent_l0"], "feature_l0": ex["feature_l0"],
        "scoring_sample_seed": ex["scoring_sample_seed"],
        "probe_fit_sample_seed": ex["probe_fit_sample_seed"],
        "probe_fit_labels": ex["probe_fit_labels"],
        "cfg_overrides": cfg_overrides,
    } | git_provenance(Path(__file__).resolve().parents[1])

    d = point_dir(out, tag, seed, toy, ex["corruption_kind"], dials, mode)
    # acts_mode is meta, not path: a --force overwrite must never silently mix two
    # constructs under one tag (review MED-1).
    npz = d / "scores.npz"
    if npz.exists():
        pm = json.loads(str(np.load(npz, allow_pickle=True)["__meta__"]))
        # The WORLD SHAPE is part of the construct: n_tokens and cfg_overrides are not in the
        # path, so a shrunken `--n-roots` run lands on top of a full one and is indistinguishable
        # afterwards. Compared here as well as in the grid's resume, so the driver refuses even
        # when invoked directly.
        prev = (pm.get("acts_mode", "ridge"), pm.get("readout", pm.get("match_mode")),
                pm.get("corruption", "absorption"), int(pm.get("n_tokens", -1)),
                json.dumps(pm.get("cfg_overrides"), sort_keys=True))
        now = (acts_mode, mode, ex["corruption_kind"], int(n_tokens),
               json.dumps(cfg_overrides, sort_keys=True))
        if prev != now:
            raise FileExistsError(
                f"{d} holds an (acts_mode, readout, kind, n_tokens, cfg_overrides)={prev} "
                f"artifact; refusing to overwrite with {now} even under --force - one tag "
                f"holds one construct.")
    write_artifacts(d, arrays, report, meta, force=force)
    written.append(d)

    if with_census:
        # Rebuild the same corruption the read used — deterministic in (world seed, dials),
        # and geometry is draw-independent, so any draw's g/CONT reproduces it exactly.
        world = regenerate_world(rc, sample_seed=held_out_sample_seed(int(seed)),
                                 n_tokens=n_tokens)
        corruption = build_corruption(world.g, world.CONT, dials, world_seed=int(seed), readout=mode)
        cen = run_census(rc, corruption, seed, n_tokens=n_tokens, readout=mode,
                         acts_mode=acts_mode)
        (d / "census.json").write_text(json.dumps(cen, indent=2), encoding="utf-8")
    print(f"[{toy} seed{seed} {dial_dirname(dials)} {mode}] wrote {d} "
          f"({report['secs']}s, sev_med={ex['realized_severity_median']:.3f}, "
          f"fvu={ex['fvu']['scoring']:.3f}, flips={ex['support_flip_rate']['scoring']:.2e})")
    return written


# CLI kind -> which knobs that damage takes. Each damage reads ONLY its own, so a `--beta`
# passed to a split run is refused rather than silently ignored.
DIALS_BY_KIND = {
    "absorption": (AbsorptionDials, ("beta", "eta", "edge_fraction")),
    "split": (SplitDials, ("k", "skew", "fraction")),
    "missing": (MissingDials, ("fraction",)),
    "noise": (NoiseDials, ("sigma",)),
}


def dials_from_args(args) -> object:
    """Build the damage's dials from the CLI, refusing knobs that belong to another damage.

    Which knobs are REQUIRED is read off the dials dataclass rather than listed here: a knob
    with no dataclass default must be supplied. That matters because `--fraction` is shared,
    and it means opposite things - for `split` it selects which features to shard and 1.0 is
    the sensible default, for `missing` it IS the dose and 1.0 deletes the entire dictionary.
    A hand-maintained list gave it one default for both, so `--kind missing` with no
    `--fraction` silently wrote a complete artifact over an empty scored frame.
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
            continue                       # not supplied and optional: take the dataclass default
        kwargs[n] = v
    foreign = [n for k, (_c, ns) in DIALS_BY_KIND.items() if k != args.kind
               for n in ns if n not in names and getattr(args, n, None) is not None]
    if foreign:
        raise SystemExit(f"--kind {args.kind} does not take "
                         f"{', '.join('--' + n.replace('_', '-') for n in dict.fromkeys(foreign))}")
    return cls(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="absorption", choices=tuple(DIALS_BY_KIND))
    # absorption
    ap.add_argument("--beta", type=float)
    ap.add_argument("--eta", type=float)
    # every dial defaults to None so "not supplied" is distinguishable from "supplied as the
    # default" - that is what makes both the required-knob check and the foreign-knob check work
    ap.add_argument("--edge-fraction", type=float)
    # split / missing
    ap.add_argument("--k", type=int, help="shards per split feature")
    ap.add_argument("--skew", type=float, help="0 = equal shard shares (default)")
    ap.add_argument("--fraction", type=float,
                    help="split: share of features sharded (default 1.0). "
                         "missing: share of features DELETED - required, no default")
    # noise
    ap.add_argument("--sigma", type=float, help="cosine-scale decoder perturbation")
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--tag", default="SYNTH-R1")
    ap.add_argument("--out", default="outputs_local/synthdict")
    ap.add_argument("--readout", default="identity", choices=READOUTS)
    ap.add_argument("--acts-mode", default="ridge", choices=("ridge", "clean"))
    ap.add_argument("--n-roots", type=int, default=None,
                    help="shrink the world (cheap local runs); recorded in resolved_config, so "
                         "a shrunken world can never masquerade as the real toy")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dials = dials_from_args(args)
    overrides = {"n_roots": args.n_roots} if args.n_roots is not None else None
    run_dial_point(args.toy, args.seed, dials, args.n_tokens, Path(args.out), args.tag,
                   readout=args.readout,
                   with_probe=not args.no_probe, with_census=not args.no_census,
                   force=args.force, acts_mode=args.acts_mode, cfg_overrides=overrides)


if __name__ == "__main__":
    main()
