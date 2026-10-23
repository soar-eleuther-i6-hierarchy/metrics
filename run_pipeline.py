#!/usr/bin/env python3
"""Run the stages in order, and refuse to run one whose inputs are missing.

The stage numbers already exist -- every script prints `[01]`, `[02]`, `[03]` -- but
they live only in the output, so the order has to be remembered or looked up. Here it
is executable: this file *is* the ordering, and it cannot go stale the way a README
paragraph can, because a wrong order fails.

    python3 run_pipeline.py                 # every stage, current layer
    python3 run_pipeline.py --from 02       # resume after a slow stage 01
    python3 run_pipeline.py --only 02 03
    python3 run_pipeline.py --list          # show the order and what is satisfied
    EXP0_LAYER=12 python3 run_pipeline.py   # any layer

Filenames are deliberately not numbered. A module whose name starts with a digit
cannot be imported, and `collect_statistics.collect` is imported by the adapters in
the umbrella repo -- numbering the files would break them. The name says what a stage
does; this file says when it runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import config as C

HERE = Path(__file__).resolve().parent


class Stage:
    def __init__(self, num, cmd, what, needs=(), produces=(), optional=False):
        self.num, self.cmd, self.what = num, cmd, what
        self.needs, self.produces, self.optional = needs, produces, optional

    def missing(self) -> list[Path]:
        return [p for p in self.needs if not p.exists()]


def stages() -> list[Stage]:
    """Declared inputs and outputs, so a stage can refuse rather than fail obscurely."""
    run = C.RUN_DIR
    return [
        Stage("01", [sys.executable, "collect_statistics.py"],
              "stream the corpus through model+SAE, cache every statistic",
              needs=(), produces=(C.EXP0_STATS_PATH,)),
        Stage("01b", [sys.executable, "fetch_labels.py"],
              "Neuronpedia feature labels for this layer (display only)",
              needs=(), produces=(run / "feature_labels.json",), optional=True),
        # Sits here because it needs 01 and nothing after it: it builds its own
        # candidate set from the within-block co-firing matrix rather than filtering
        # stage 02's, so it answers the same question on a different domain. Optional
        # because nothing downstream reads its output -- `reporting/make_report_figures`
        # draws fig5 when `in_block_edges.json` happens to exist and skips it otherwise.
        Stage("01c", [sys.executable, "in_block_edges.py"],
              "same-level (within-block) edges and duplicates",
              needs=(C.EXP0_STATS_PATH, C.TOKEN_CACHE_DIR),
              produces=(C.IN_BLOCK_PATH,), optional=True),
        Stage("02", [sys.executable, "run_metrics.py"],
              "grade every block pair -> metrics_report.{json,md}",
              needs=(C.EXP0_STATS_PATH,), produces=(C.METRICS_JSON_PATH,)),
        Stage("02b", [sys.executable, "-m", "validation.qualitative_check"],
              "Tier 4: survivor vs rejected edges against Neuronpedia labels",
              needs=(C.EXP0_STATS_PATH, C.METRICS_JSON_PATH),
              produces=(run / "qualitative_check.json",)),
        Stage("03", [sys.executable, "run_token_metrics.py"],
              "S_res probes, parent-conditioned siblings, kept-children union",
              needs=(C.METRICS_JSON_PATH, C.TOKEN_CACHE_DIR),
              produces=(run / "second_pass.json",)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, help="which layer to run (overrides EXP0_LAYER)")
    ap.add_argument("--from", dest="start", metavar="NN", help="start at this stage")
    ap.add_argument("--only", nargs="+", metavar="NN", help="run just these")
    ap.add_argument("--list", action="store_true", help="show the order, run nothing")
    ap.add_argument("--skip-optional", action="store_true")
    args = ap.parse_args()
    # Before stages(), which reads C.RUN_DIR to decide what each stage needs. The
    # re-exec also puts EXP0_LAYER in the environment the child stages inherit, so
    # one flag reaches all of them without any stage growing its own.
    C.use_layer(args.layer)

    all_stages = stages()
    print(f"layer {C.LAYER} -> {C.RUN_DIR}\n")

    if args.list:
        for s in all_stages:
            miss = s.missing()
            mark = "OK  " if not miss else "WAIT"
            tail = "" if not miss else f"   <- needs {', '.join(p.name for p in miss)}"
            opt = " (optional)" if s.optional else ""
            print(f"  [{s.num:<3}] {mark} {s.what}{opt}{tail}")
        return 0

    todo = all_stages
    if args.only:
        todo = [s for s in todo if s.num in args.only]
    elif args.start:
        idx = next((i for i, s in enumerate(todo) if s.num == args.start), None)
        if idx is None:
            raise SystemExit(f"no stage {args.start!r}; known: {[s.num for s in all_stages]}")
        todo = todo[idx:]
    if args.skip_optional:
        todo = [s for s in todo if not s.optional]

    for s in todo:
        missing = s.missing()
        if missing:
            names = ", ".join(str(p) for p in missing)
            if s.optional:
                print(f"[{s.num}] skipped — missing {names}\n")
                continue
            # Refusing beats running: a stage fed a missing input either crashes
            # obscurely later or, worse, reads a stale file from a previous run.
            raise SystemExit(f"[{s.num}] cannot run — missing {names}\n"
                             f"      run the earlier stage first, or --list to see the order")
        print(f"=== [{s.num}] {s.what}")
        r = subprocess.run(s.cmd, cwd=HERE)
        if r.returncode != 0:
            if s.optional:
                print(f"[{s.num}] optional stage failed — continuing\n")
                continue
            raise SystemExit(f"[{s.num}] failed with {r.returncode}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
