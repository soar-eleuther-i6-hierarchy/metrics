#!/usr/bin/env bash
# Seed-0 benchmark under an ARBITRARY tag: 5 toys x {oracle, trained}, ten reads in parallel.
#
#   bash scoring/benchmark/run_seed0_tagged.sh <TAG> [extra args passed to every read]
#
# Generalises `run_seed0.sh`, which hard-codes SEED0-PREFREEZE. B2 needs two more runs and the
# write guard (correctly) refuses to reuse a tag, so the tag has to be an argument:
#
#   SEED0-BRIDGE     --probe-fit-seed 10000   fit draw == scoring draw. Reproduces the
#                                             unseparated probe, which is the only check the
#                                             B2.2 rewiring gets: `harness_gate` deliberately
#                                             excludes s_res, so nothing else would catch it.
#   SEED0-CORRECTED  (no extra args)          the real run, fit draw = seed + 20000.
#
# Never reuses SEED0-PREFREEZE: that tag is the pilot record and must survive.
#
# Runs ON THE SERVER, from ~/exp0-chidaksh, after `./exp0_remote.sh push` from soar/.
# The five seed-0 checkpoints live in two places, which is why the paths are spelled out:
# only_isa is under ~/toysae/checkpoints and the other four under ~/exp0-chidaksh/checkpoints.
set -u
TAG="${1:?usage: run_seed0_tagged.sh <TAG> [extra args]}"
shift || true
EXTRA=("$@")

if [ "$TAG" = "SEED0-PREFREEZE" ]; then
  echo "refusing: SEED0-PREFREEZE is the pilot record. Use a new tag." >&2
  exit 2
fi

cd "$HOME/exp0-chidaksh"
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
UV="$HOME/.local/bin/uv"
CK_ISA="$HOME/toysae/checkpoints/only_isa-matryoshka-k29-x4-s0"
CK="$HOME/exp0-chidaksh/checkpoints"
LOG="logs/$TAG"
mkdir -p "$LOG"
: > "$LOG/DONE"

trained() {
  "$UV" run python -m scoring.benchmark.run_benchmark \
    --toy "$1" --read trained --ckpt "$2" --n-tokens 200000 \
    --tag "$TAG" --out outputs_local/benchmark \
    --gate-against "outputs_local/$1/matryoshka_scores.npz" \
    "${EXTRA[@]}" > "$LOG/tr_$1.log" 2>&1
  echo "trained $1 exit=$?" >> "$LOG/DONE"
}

oracle() {
  "$UV" run python -m scoring.benchmark.run_benchmark \
    --toy "$1" --read oracle --seed 0 --n-tokens 200000 \
    --tag "$TAG" --out outputs_local/benchmark \
    "${EXTRA[@]}" > "$LOG/or_$1.log" 2>&1
  echo "oracle $1 exit=$?" >> "$LOG/DONE"
}

trained only_isa         "$CK_ISA" &
trained only_firing      "$CK/only_firing-matryoshka-k29-x4-s0" &
trained only_frequency   "$CK/only_frequency-matryoshka-k23-x4-s0" &
trained only_superparent "$CK/only_superparent-matryoshka-k24-x4-s0" &
trained only_topical     "$CK/only_topical-matryoshka-k24-x4-s0" &
oracle only_isa &
oracle only_firing &
oracle only_frequency &
oracle only_superparent &
oracle only_topical &
wait
echo ALLDONE >> "$LOG/DONE"
