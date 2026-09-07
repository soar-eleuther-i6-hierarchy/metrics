"""Round-1 grid launcher: the 29 approved dial points, N parallel subprocesses.

The grid is FROZEN in SYNTH_PRECOMMIT.md; this file only enumerates it:
  * diagonal beta=eta in {0,.2,.4,.6,.8,1.0} x f in {0.1, 1.0} x {only_isa, only_firing}  (24)
  * decoupled controls (beta=.6, eta=0) and (beta=0, eta=.6) at f=.1, both toys           (4)
  * bridge: only_firing, f=11/120, beta=eta=0.547 -> realized severity ~0.48              (1)

Each point runs `python -m synthdict.run_synth` as its own subprocess (crash isolation: one
failed point leaves 28 artifacts, not zero), logging to <out>/<tag>/logs/. `--timed-pilot`
runs ONE representative point first — the plan's "time one dial point before the grid" rule.

  python -m synthdict.run_grid --tag SYNTH-R1 --n-jobs 8 [--dry-run|--timed-pilot]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DIAG = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TOYS = ("only_isa", "only_firing")
BRIDGE_BETA = 0.547            # beta/sqrt(1+beta^2) = 0.48 on an orthogonal edge
BRIDGE_F = 11 / 120            # the trained artifact's absorbed-edge rate (11 of 120 edges)


def grid_points() -> list[tuple[str, float, float, float]]:
    pts = [(toy, b, b, f) for toy in TOYS for f in (0.1, 1.0) for b in DIAG]
    pts += [(toy, 0.6, 0.0, 0.1) for toy in TOYS]          # decoder-only carry (hedging-like)
    pts += [(toy, 0.0, 0.6, 0.1) for toy in TOYS]          # hole-only
    pts += [("only_firing", BRIDGE_BETA, BRIDGE_BETA, BRIDGE_F)]
    return pts


def _artifact_acts_mode(npz_path: Path) -> str:
    """The acts_mode stamped in an artifact's meta ('ridge' for pre-acts-mode artifacts)."""
    import json

    import numpy as np

    meta = json.loads(str(np.load(npz_path, allow_pickle=True)["__meta__"]))
    return meta.get("acts_mode", "ridge")


def point_cmd(toy: str, beta: float, eta: float, f: float, args) -> list[str]:
    cmd = [sys.executable, "-m", "synthdict.run_synth", "--toy", toy,
           "--seed", str(args.seed), "--beta", str(beta), "--eta", str(eta),
           "--edge-fraction", str(f), "--n-tokens", str(args.n_tokens),
           "--tag", args.tag, "--out", args.out]
    cmd += ["--acts-mode", args.acts_mode]
    if args.no_probe:
        cmd.append("--no-probe")
    if args.force:
        cmd.append("--force")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="SYNTH-R1")
    ap.add_argument("--out", default="outputs_local/synthdict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--threads-per-job", type=int, default=8)
    ap.add_argument("--acts-mode", default="ridge", choices=("ridge", "clean"))
    ap.add_argument("--only-f01", action="store_true",
                    help="the f=0.1 subset + the bridge point (the approved clean-mode re-run)")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timed-pilot", action="store_true",
                    help="run ONE representative point (only_firing beta=eta=0.6 f=0.1) and report wall-clock")
    args = ap.parse_args()

    pts = grid_points()
    if args.only_f01:
        pts = [p for p in pts if p[3] != 1.0]
    if args.timed_pilot:
        pts = [("only_firing", 0.6, 0.6, 0.1)]
    if args.dry_run:
        for p in pts:
            print(" ".join(point_cmd(*p, args)))
        print(f"# {len(pts)} points")
        return

    logdir = Path(args.out) / args.tag / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, str, float]] = []
    failed: list[str] = []
    done = 0
    t_start = time.time()

    def reap(block: bool) -> None:
        nonlocal done
        while running and (block or any(p.poll() is not None for p, _, _ in running)):
            for i, (proc, name, t0) in enumerate(running):
                rc = proc.poll()
                if rc is None:
                    continue
                dt = time.time() - t0
                done += 1
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(f"[{done}/{len(pts)}] {name}: {status} ({dt:.0f}s)", flush=True)
                if rc != 0:
                    failed.append(name)
                running.pop(i)
                break
            else:
                time.sleep(2.0)

    for toy, beta, eta, f in pts:
        name = f"{toy}-beta{beta:g}-eta{eta:g}-f{f:g}"
        dial = f"beta{beta:g}-eta{eta:g}-f{f:g}"
        # Idempotent resume: a point whose artifacts exist was produced by this same code
        # (content-hash stamped in its meta); rerunning would only trip the overwrite guard.
        existing = [Path(args.out) / args.tag / f"seed{args.seed}" / toy / "absorption"
                    / dial / m / "scores.npz" for m in ("identity", "hungarian")]
        if not args.force and all(p.exists() for p in existing):
            # Resume only PAST artifacts of the SAME construct: acts_mode is not in the path,
            # so skipping on existence alone would let `--acts-mode clean` against a ridge tag
            # exit 0 having run nothing (review MED-1).
            modes = {_artifact_acts_mode(p) for p in existing}
            if modes == {args.acts_mode}:
                done += 1
                print(f"[{done}/{len(pts)}] {name}: skipped (artifacts exist)", flush=True)
                continue
            raise SystemExit(
                f"{name}: existing artifacts under this tag were produced with acts_mode="
                f"{sorted(modes)}, not {args.acts_mode!r}. One tag holds ONE construct: "
                f"use a different --tag.")
        while len(running) >= args.n_jobs:
            reap(block=True)
        log = open(logdir / f"{name}.log", "w")
        # Cap BLAS threads per child: n_jobs x default-all-cores oversubscribes the box.
        env = dict(os.environ,
                   OMP_NUM_THREADS=str(args.threads_per_job),
                   MKL_NUM_THREADS=str(args.threads_per_job))
        proc = subprocess.Popen(point_cmd(toy, beta, eta, f, args),
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        running.append((proc, name, time.time()))
    while running:
        reap(block=True)

    print(f"\n{len(pts) - len(failed)}/{len(pts)} points ok in {time.time() - t_start:.0f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
