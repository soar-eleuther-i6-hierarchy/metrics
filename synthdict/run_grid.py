"""Run a JSON list of dial points as parallel `synthdict.run_synth` subprocesses.

Each point is {"toy", "kind", "readout", "dials": {...}}. Points whose artifacts already exist
under the same construct are skipped; logs go to <out>/<tag>/logs/.

  python -m synthdict.run_grid --points points.json --tag T --n-jobs 8 [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from synthdict.planted import READOUTS
from synthdict.run_synth import (DIALS_BY_KIND, acts_mode_of, dial_dirname, point_dir,
                                 provenance_tuple)


def point_dials(point: dict):
    """The dials object a point describes. A field the CLI cannot carry must stay at its
    default, or the subprocess would run a different point."""
    kind = point["kind"]
    if kind not in DIALS_BY_KIND:
        raise SystemExit(f"unknown kind {kind!r}; kinds are {tuple(DIALS_BY_KIND)}")
    if point["readout"] not in READOUTS:
        raise SystemExit(f"unknown readout {point['readout']!r}; readouts are {READOUTS}")
    toy = point["toy"]
    if "/" in toy or "\\" in toy or toy.startswith("."):
        raise SystemExit(f"toy {toy!r} is not a plain name")
    cls, names = DIALS_BY_KIND[kind]
    dials = cls(**point["dials"])
    for f in dataclasses.fields(cls):
        if f.name not in names and getattr(dials, f.name) != f.default:
            raise SystemExit(f"{kind} point sets {f.name}={getattr(dials, f.name)!r}, which "
                             f"the run_synth CLI does not carry")
    return dials


def point_cmd(point: dict, args) -> list[str]:
    """The run_synth argv for one point, every dial of its kind spelled out."""
    kind = point["kind"]
    dials = point_dials(point)
    cmd = [sys.executable, "-m", "synthdict.run_synth", "--toy", point["toy"], "--kind", kind,
           "--readout", point["readout"], "--seed", str(args.seed),
           "--n-tokens", str(args.n_tokens), "--tag", args.tag, "--out", str(args.out)]
    for name in DIALS_BY_KIND[kind][1]:
        v = getattr(dials, name)
        flag = "--" + name.replace("_", "-")
        cmd += [flag, *v] if name == "roles" else [flag, str(v)]
    if args.n_roots is not None:
        cmd += ["--n-roots", str(args.n_roots)]
    if args.no_probe:
        cmd.append("--no-probe")
    if args.no_census:
        cmd.append("--no-census")
    if args.force:
        cmd.append("--force")
    return cmd


def resume_npz(out, tag: str, seed: int, toy: str, dials, readout: str) -> Path:
    """The artifact resume checks, from the driver's own `point_dir`."""
    return point_dir(out, tag, seed, toy, type(dials).KIND, dials, readout) / "scores.npz"


POLL_SECS = 2.0


def reap_finished(running: list, block: bool, report) -> int:
    """Drop finished children from `running`, calling `report(name, rc, secs)` for each.

    With `block=True` it waits for one child only; draining the pool would turn the parallel
    slots into sequential waves.
    """
    reaped = 0
    while running and ((block and reaped == 0)
                       or any(p.poll() is not None for p, _, _, _ in running)):
        for i, (proc, name, t0, log) in enumerate(running):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            report(name, rc, time.time() - t0)
            running.pop(i)
            reaped += 1
            break
        else:
            time.sleep(POLL_SECS)
    return reaped


def _artifact_provenance(npz_path: Path) -> tuple:
    import numpy as np

    return provenance_tuple(json.loads(str(np.load(npz_path, allow_pickle=False)["__meta__"])))


def _requested_provenance(args, kind: str, readout: str, acts_mode: str = "nnls") -> tuple:
    """`provenance_tuple` for the run about to happen. `acts_mode` must come from the point:
    it is all that separates the oracle and undamaged columns."""
    overrides = {"n_roots": args.n_roots} if args.n_roots is not None else None
    return provenance_tuple({"acts_model": acts_mode, "readout": readout, "corruption": kind,
                             "n_tokens": int(args.n_tokens), "cfg_overrides": overrides,
                             "s_res_mode": "absent" if args.no_probe else "probe"})


def check_unique_points(pts: list[dict], args) -> None:
    """Two points writing one directory would race; refuse before launching either."""
    seen: dict[Path, int] = {}
    for i, p in enumerate(pts):
        d = point_dir(args.out, args.tag, args.seed, p["toy"], p["kind"], point_dials(p),
                      p["readout"])
        if d in seen:
            raise SystemExit(f"points {seen[d]} and {i} both write {d}")
        seen[d] = i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True, help="JSON file: a list of dial points")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="outputs_local/synthdict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--threads-per-job", type=int, default=8)
    ap.add_argument("--n-roots", type=int, default=None,
                    help="shrink the world (cheap local dry runs)")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pts = json.loads(Path(args.points).read_text())
    cmds = [point_cmd(p, args) for p in pts]            # validates every point before any run
    check_unique_points(pts, args)
    if args.dry_run:
        for c in cmds:
            print(" ".join(c))
        print(f"# {len(pts)} points")
        return

    logdir = Path(args.out) / args.tag / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, str, float, object]] = []
    failed: list[str] = []
    done = 0
    t_start = time.time()

    def on_done(name: str, rc: int, secs: float) -> None:
        nonlocal done
        done += 1
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[{done}/{len(pts)}] {name}: {status} ({secs:.0f}s)", flush=True)
        if rc != 0:
            failed.append(name)

    def reap(block: bool) -> None:
        reap_finished(running, block, on_done)

    for i, (point, cmd) in enumerate(zip(pts, cmds)):
        dials = point_dials(point)
        kind, readout = point["kind"], point["readout"]
        name = f"{i:03d}-{point['toy']}-{kind}-{dial_dirname(dials)}-{readout}"
        npz = resume_npz(args.out, args.tag, args.seed, point["toy"], dials, readout)
        census_missing = not args.no_census and not (npz.parent / "census.json").exists()
        if not args.force and npz.exists() and not census_missing:
            have = _artifact_provenance(npz)
            want = _requested_provenance(args, kind, readout, acts_mode_of(dials))
            if have == want:
                done += 1
                print(f"[{done}/{len(pts)}] {name}: skipped (artifacts exist)", flush=True)
                continue
            raise SystemExit(
                f"{name}: existing artifact under this tag is (acts_model, readout, kind, "
                f"n_tokens, cfg_overrides, s_res_mode)={have}, not {want}. One tag holds ONE "
                f"construct; use a different --tag.")
        while len(running) >= args.n_jobs:
            reap(block=True)
        log = open(logdir / f"{name}.log", "w")
        # Cap BLAS threads per child: n_jobs x all cores oversubscribes the box.
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads_per_job),
                   MKL_NUM_THREADS=str(args.threads_per_job))
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        running.append((proc, name, time.time(), log))
    while running:
        reap(block=True)

    print(f"\n{len(pts) - len(failed)}/{len(pts)} points ok in {time.time() - t_start:.0f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
