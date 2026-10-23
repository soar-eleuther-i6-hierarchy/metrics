"""Prove the grading stages are source-agnostic too, not just collect().

tests/test_collect_generic.py guards stage 01: the accumulation reads its block
structure from the cfg it is handed. Stage 03 once did not. It sliced
`config.BLOCK_RANGES` -- gemma's 32768 latents in 5 blocks -- so a dictionary
with a different shape came back wrong in the worst available way: the first
four pairs are in range, so the numbers are plausible and nothing raises until
the fifth pair, if there is one.

This test runs a PCFG-SHAPED file (8 blocks, not 5) through 02 -> 03 and asserts
the reports describe the file they were built from:

    - every block pair is graded, including the ones past gemma's fifth block
    - the report header names the run's own dictionary, not gemma's
    - S_res (stage 03) runs off the run's own decoder, not the released SAE's

Stub model, stub SAE, no network, no GPU.

    python3 -m tests.test_stages_generic
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOOK = "blocks.1.hook_resid_post"
D_MODEL, D_VOCAB = 16, 60
# 8 nested blocks like the PCFG SAE's [224 ... 1792], scaled down. The count is
# the point: pairs 4->5, 5->6 and 6->7 do not exist in gemma's structure.
STEPS = [16, 32, 48, 64, 80, 96, 112, 128]
N_PAIRS = len(STEPS) - 1


class StubModel:
    """The slice of HookedTransformer that collect() actually touches."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cfg = SimpleNamespace(d_vocab=D_VOCAB, d_model=D_MODEL)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self._emb = torch.randn(D_VOCAB, D_MODEL, generator=g)

    def run_with_cache(self, tokens, stop_at_layer=None, names_filter=None):
        return None, {HOOK: self._emb[tokens]}


class StubSAE:
    def __init__(self, d_sae, seed=0):
        g = torch.Generator().manual_seed(seed + 1)
        self.W_enc = torch.randn(D_MODEL, d_sae, generator=g) * 0.5
        self.W_dec = torch.randn(d_sae, D_MODEL, generator=g) * 0.5
        self.b_enc = torch.full((d_sae,), -0.15)

    def encode(self, x):
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec


def block_ranges(steps):
    out, prev = [], 0
    for s in steps:
        out.append((prev, s))
        prev = s
    return out


def make_cfg(run_dir: Path):
    r = block_ranges(STEPS)
    return SimpleNamespace(
        LAYER=1, HOOK_NAME=HOOK, SAE_ID="matryoshka_hook_resid_post_L1",
        SAE_RELEASE="stub-matryoshka", SAE_SOURCE="pcfg",
        MATRYOSHKA_STEPS=STEPS, BLOCK_RANGES=r, N_BLOCKS=len(r), D_SAE=STEPS[-1],
        INCLUDE_B3_B4=True, FIRE_THRESHOLD=1e-3, BATCH_DOCS=8, CONTEXT_SIZE=32,
        SIBLING_BLOCKS=list(range(1, len(r))), IN_BLOCK_BLOCKS=list(range(len(r) - 1)),
        N_FREQ_BUCKETS=3, FREQ_HIGH_MASS=0.50, FREQ_MID_MASS=0.40, MIN_JOINT=5,
        LOCAL_FREQ_BUCKETS=False,
        # On, because stage 03 is half of what this test guards.
        CACHE_RESIDUALS=True,
        TOKEN_CACHE_DIR=run_dir / "token_cache",
        EXP0_STATS_PATH=run_dir / "exp0_stats.pt",
    )


def make_seqs(n_docs=60, length=32, seed=0):
    """Zipf-ish stream so the frequency buckets are not all one bucket.

    Token id 0 is reserved for padding and never sampled -- keep_mask drops every
    position equal to pad_id.
    """
    g = torch.Generator().manual_seed(seed + 2)
    w = 1.0 / torch.arange(1, D_VOCAB).float()
    return [torch.multinomial(w, length, replacement=True, generator=g) + 1
            for _ in range(n_docs)]


def stage(cmd, env, failures):
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        failures.append(f"{' '.join(str(c) for c in cmd[1:3])} failed:\n{p.stderr[-1500:]}")
    return p.returncode == 0


def main() -> int:
    from collect_statistics import collect

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        run_dir = out_dir / "pcfg-matryoshka"
        run_dir.mkdir(parents=True)

        cfg = make_cfg(run_dir)
        sae = StubSAE(cfg.D_SAE)
        collect(StubModel(), sae, make_seqs(), device="cpu", cfg=cfg, pad_id=0,
                extra_config={"source": "pcfg", "n_docs": 60,
                              "base_model": {"n_layers": 2, "d_model": D_MODEL,
                                             "vocab_size": D_VOCAB}})
        # The hand-off stage 03 needs: this dictionary's decoder, beside the stats.
        torch.save(sae.W_dec.detach().cpu(), run_dir / "w_dec.pt")

        env = {**os.environ, "EXP0_OUT": str(out_dir), "EXP0_RUN": "pcfg-matryoshka"}
        failures = []
        ran = stage([sys.executable, "run_metrics.py", "--stats", str(cfg.EXP0_STATS_PATH),
                     "--out-dir", str(run_dir)], env, failures)
        # One pair, on CPU: stage 03 trains a probe per child, and running all
        # seven takes a minute for no extra coverage. 6->7 is the pair that does
        # not exist in gemma's structure at all, so it is the one worth probing.
        ran &= stage([sys.executable, "run_token_metrics.py",
                      "--pairs", "6->7", "--device", "cpu"], env, failures)

        def check(cond, msg):
            if not cond:
                failures.append(msg)

        if ran:
            report = json.loads((run_dir / "metrics_report.json").read_text())
            check(len(report["pairs"]) == N_PAIRS,
                  f"graded {len(report['pairs'])} pairs, not this file's {N_PAIRS}")
            check(report["config"]["block_ranges"] == [list(r) for r in cfg.BLOCK_RANGES],
                  "the report's block_ranges are not the ones the file was built with")

            second = report.get("second_pass") or {}
            check(any("sres" in v for v in second.values()),
                  "stage 03 produced no S_res column (it needs the run's own decoder)")

            md_path = run_dir / "metrics_report.md"
            check(md_path.exists(), "no metrics_report.md")
            if md_path.exists():
                md = md_path.read_text()
                # The scope line renders as `model / sae_source`; it should come
                # from the run's own config, not gemma's defaults.
                check("gemma-2-2b /" not in md, "the report header still says gemma-2-2b")
                check(f"{STEPS[-1]} latents in {len(STEPS)} blocks" in md,
                      "the report header does not state this dictionary's shape")

        for f in failures:
            print(f"FAIL {f}")
        if failures:
            print(f"\n{len(failures)} failure(s)")
            return 1
        print(f"grading is source-agnostic: {N_PAIRS} pairs from a "
              f"{STEPS[-1]}-feature, {len(STEPS)}-block stub, not gemma")
        return 0


if __name__ == "__main__":
    sys.exit(main())
