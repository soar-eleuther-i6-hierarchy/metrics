"""CLI driver: score one dial point (both match modes) through the FROZEN evaluator.

Ships its own CLI because the benchmark's is deliberately closed (`--read choices=
("oracle","trained")`, `--toy choices=TOYS`); `run_read` and `write_artifacts` themselves are
name-agnostic and are reused verbatim. Artifacts land under their own root
(`outputs_local/synthdict/<TAG>/...`), NEVER under a benchmark tag — `verify_manifest`
correctly rejects unknown read dirs in a benchmark tree.

Layout:  <out>/<tag>/seed<N>/<toy>/absorption/beta<b>-eta<e>-f<f>/<match_mode>/
           scores.npz         run_read arrays + corrupted_pair, realized_severity, matched_corr
           expressions.json   run_read report | __meta__
           census.json        annotation (manipulation check; see synthdict/census.py)

Usage (one dial point, the grid launches many of these in parallel on the server):
  python -m synthdict.run_synth --toy only_isa --seed 0 --beta 0.4 --eta 0.4 \
      --edge-fraction 0.1 --tag SYNTH-R1 [--n-tokens 200000] [--no-probe] [--no-census]
      [--match-modes identity,hungarian] [--out outputs_local/synthdict] [--force]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scoring.benchmark.manifest import evaluator_sha256
from scoring.benchmark.registry import REPORT_SCHEMA
from scoring.benchmark.run_benchmark import git_provenance, run_read, write_artifacts
from scoring.core.grid import held_out_sample_seed
from scoring.core.world import regenerate_world

from synthdict.census import run_census
from synthdict.corruptions import AbsorptionDials, absorb
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


def dial_dirname(dials: AbsorptionDials) -> str:
    return f"beta{dials.beta:g}-eta{dials.eta:g}-f{dials.edge_fraction:g}"


def run_dial_point(toy: str, seed: int, dials: AbsorptionDials, n_tokens: int,
                   out: Path, tag: str, match_modes=("identity", "hungarian"),
                   with_probe: bool = True, with_census: bool = True,
                   force: bool = False, cfg_overrides: dict | None = None,
                   acts_mode: str = "ridge") -> list[Path]:
    """Score one (toy, seed, dials) under each match mode; write the three artifacts each."""
    written = []
    rc = resolved_config(toy, seed, cfg_overrides)
    for mode in match_modes:
        t0 = time.time()
        read = synthetic_read(toy, seed, dials, mode, n_tokens, with_probe=with_probe,
                              cfg_overrides=cfg_overrides, acts_mode=acts_mode)
        report, arrays = run_read(read)
        report["secs"] = round(time.time() - t0, 1)

        ex = read.extra
        arrays["corrupted_pair"] = ex["corrupted_pair"].numpy()
        arrays["realized_severity"] = ex["realized_severity"].numpy()
        if isinstance(ex.get("matched_corr"), torch.Tensor):
            arrays["matched_corr"] = ex["matched_corr"].numpy()

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
            "match_mode": mode, "acts_mode": ex["acts_mode"],
            "true_l0": ex["true_l0"], "realized_l0": ex["realized_l0"],
            "scoring_sample_seed": ex["scoring_sample_seed"],
            "matching_sample_seed": ex["matching_sample_seed"],
            "probe_fit_sample_seed": ex["probe_fit_sample_seed"],
            "probe_fit_labels": ex["probe_fit_labels"],
            "cfg_overrides": cfg_overrides,
        } | git_provenance(Path(__file__).resolve().parents[1])

        d = Path(out) / tag / f"seed{int(seed)}" / toy / "absorption" / dial_dirname(dials) / mode
        write_artifacts(d, arrays, report, meta, force=force)
        written.append(d)

        if with_census:
            # Rebuild the same corruption the read used — deterministic in (world seed, dials),
            # and geometry is draw-independent, so any draw's g/CONT reproduces it exactly.
            world = regenerate_world(rc, sample_seed=held_out_sample_seed(int(seed)),
                                     n_tokens=n_tokens)
            corruption = absorb(world.g, world.CONT, dials, world_seed=int(seed))
            cen = run_census(rc, corruption, seed, n_tokens=n_tokens, match_mode=mode,
                             acts_mode=acts_mode)
            (d / "census.json").write_text(json.dumps(cen, indent=2), encoding="utf-8")
        print(f"[{toy} seed{seed} {dial_dirname(dials)} {mode}] wrote {d} "
              f"({report['secs']}s, sev_med={ex['realized_severity_median']:.3f}, "
              f"fvu={ex['fvu']['scoring']:.3f}, flips={ex['support_flip_rate']['scoring']:.2e})")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--edge-fraction", type=float, default=1.0)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--tag", default="SYNTH-R1")
    ap.add_argument("--out", default="outputs_local/synthdict")
    ap.add_argument("--match-modes", default="identity,hungarian")
    ap.add_argument("--acts-mode", default="ridge", choices=("ridge", "clean"))
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dials = AbsorptionDials(beta=args.beta, eta=args.eta, edge_fraction=args.edge_fraction)
    run_dial_point(args.toy, args.seed, dials, args.n_tokens, Path(args.out), args.tag,
                   match_modes=tuple(args.match_modes.split(",")),
                   with_probe=not args.no_probe, with_census=not args.no_census,
                   force=args.force, acts_mode=args.acts_mode)


if __name__ == "__main__":
    main()
