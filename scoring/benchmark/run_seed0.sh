#!/usr/bin/env bash
# Seed-0 benchmark: 5 toys x {oracle, trained}, with the probe, all ten reads in parallel.
# ~104 s wall on soar-gpu; the slowest single read is only_firing/oracle at ~102 s.
#
# Runs ON THE SERVER, from ~/exp0-chidaksh, after `./exp0_remote.sh push` from soar/.
# The five seed-0 checkpoints live in two places, which is why the paths are spelled out:
# only_isa is under ~/toysae/checkpoints and the other four under ~/exp0-chidaksh/checkpoints.
# Phase D should put all fifteen fresh-seed checkpoints in one location.
#
# Tracked here rather than left on the server: REGRESSION_GATE.md's reproduction instructions
# name this script, and instructions pointing at a file that exists on one machine are not
# reproduction instructions.
set -u
cd "$HOME/exp0-chidaksh"
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
UV="$HOME/.local/bin/uv"
TAG=SEED0-PREFREEZE
CK_ISA="$HOME/toysae/checkpoints/only_isa-matryoshka-k29-x4-s0"
CK="$HOME/exp0-chidaksh/checkpoints"
mkdir -p logs/seed0
: > logs/seed0/DONE

trained() {
  "$UV" run python -m scoring.benchmark.run_benchmark \
    --toy "$1" --read trained --ckpt "$2" --n-tokens 200000 \
    --tag "$TAG" --out outputs_local/benchmark \
    --gate-against "outputs_local/$1/matryoshka_scores.npz" \
    > "logs/seed0/tr_$1.log" 2>&1
  echo "trained $1 exit=$?" >> logs/seed0/DONE
}

oracle() {
  "$UV" run python -m scoring.benchmark.run_benchmark \
    --toy "$1" --read oracle --seed 0 --n-tokens 200000 \
    --tag "$TAG" --out outputs_local/benchmark \
    > "logs/seed0/or_$1.log" 2>&1
  echo "oracle $1 exit=$?" >> logs/seed0/DONE
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
echo ALLDONE >> logs/seed0/DONE
