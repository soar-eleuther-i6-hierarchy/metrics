"""Which signal sees a split feature: firing overlap, or decoder geometry?

    python3 -m metrics_v2.validation.split_detectors [--toy only_isa] [--seeds 0 1 2] [--k 2 3 4 6]

Two constructions of "splitting" exist in this repository, and they are not the same thing:

  copies      one feature carried by latents that fire on the SAME tokens
              (the Tier-1 synthetic world's split parent; validation/synthetic_toy_world.py)
  shards      one feature carried by latents that share its DIRECTION and PARTITION its tokens
              (synthdict.corruptions.split, the collaborator's splitting toys)

The firing-overlap detector (pairwise Jaccard among a parent's children, metric 7) reads copies
at 1.00 and shards at 0.00 by construction. The decoder-cosine detector reads shards at 1.00.
This script runs the collaborator's shard corruption on his own world and reports both, per k,
against the unsplit features as the null.

Outputs: outputs/metrics_v2/split_detectors.{md,json}. Reads the synthdict/toygen code only;
nothing is typed in.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dataclasses  # noqa: E402

from types import SimpleNamespace  # noqa: E402

import config as C  # noqa: E402
from metrics_v2.pathology import sibling_overlap_within_parent  # noqa: E402
from metrics_v2.structure import decoder_cosine  # noqa: E402
from synthdict.corruptions import SplitDials, build_corruption, expand_support  # noqa: E402
from toygen import geometry, sample, spec, strengths  # noqa: E402
from toygen import tree as tree_mod  # noqa: E402
from toygen.world import resolve_config  # noqa: E402


def regenerate_world(rc: dict, sample_seed: int, n_tokens: int):
    """scoring.core.world.regenerate_world without the pair labels: at the current PR tip
    toygen.labels names classes that metrics.rules.classes no longer lists, so labelling
    raises; this experiment needs the tree, the directions and the draw only."""
    known = {f.name for f in dataclasses.fields(spec.ToyConfig)}
    cfg = spec.ToyConfig(**{k: v for k, v in rc.items() if k in known})
    tree = tree_mod.build_tree(cfg)
    strength = strengths.build_strengths(cfg, tree)
    geo = geometry.build_directions(cfg, tree, seed=cfg.seed)
    world = sample.sample_world(cfg, tree, strength, geo, n_tokens=n_tokens, seed=sample_seed)
    cont = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    return SimpleNamespace(A=world.A, h=world.h, g=geo.g, tree=tree, p=strength.p, CONT=cont, cfg=cfg)

COS_CUT = 0.9   # a shard pair shares one direction exactly; 0.9 leaves room for a trained SAE
HELD_OUT_SEED_OFFSET = 10_000   # scoring.core.grid.HELD_OUT_SEED_OFFSET, copied: importing
                                # scoring.core.grid pulls in scoring.core.gates, whose
                                # GATE_SOURCES assertion fails at the current PR tip


def held_out_sample_seed(seed: int) -> int:
    return int(seed) + HELD_OUT_SEED_OFFSET


def resolved_config(toy: str, seed: int) -> dict:
    """synthdict.read.resolved_config without its scoring imports."""
    return dataclasses.asdict(spec.replace(resolve_config(toy), seed=int(seed)))


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else float("nan")


def run_point(toy: str, seed: int, k: int, n_tokens: int, fraction: float) -> dict:
    rc = resolved_config(toy, seed)
    sample_seed = held_out_sample_seed(seed)
    world = regenerate_world(rc, sample_seed=sample_seed, n_tokens=n_tokens)
    dials = SplitDials(k=k, roles=("parent",), fraction=fraction)
    corr = build_corruption(world, dials, world_seed=seed)
    support = expand_support(world.A > 0, corr, seed, sample_seed)      # [n, L] bool
    W = corr.W_raw.double()                                            # [L, D]
    f2l = corr.planted_map.feature_to_latents
    split = list(corr.corrupted_features)
    parents = sorted({p for p, _ in world.CONT})
    unsplit = [p for p in parents if p not in set(split)]

    # (a) among a split feature's shards: firing overlap and decoder cosine
    jac_global, jac_within, cos_shards, cov_shard = [], [], [], []
    for f in split:
        lats = list(f2l[f])
        union = support[:, lats].any(dim=1)
        jac_global += [_jaccard(support[:, i], support[:, j]) for i, j in itertools.combinations(lats, 2)]
        jac_within.append(sibling_overlap_within_parent(union, support[:, lats]))
        cs = decoder_cosine(W[lats], W[lats])
        cos_shards += [float(cs[a, b]) for a, b in itertools.combinations(range(len(lats)), 2)]
        # what his figure plots: P(shard | child) for the feature's child
        for c in world.tree.children.get(f, []):
            cl = f2l[c][0]
            for i in lats:
                cov_shard.append(_jaccard(support[:, i] & support[:, cl], support[:, cl]))

    # (b) the null: pairs of different, unsplit parents (one latent each)
    null_lats = [f2l[p][0] for p in unsplit]
    cos_null = decoder_cosine(W[null_lats], W[null_lats])
    off = ~torch.eye(len(null_lats), dtype=torch.bool)
    cos_null_vals = cos_null[off].abs().tolist()
    jac_null = [_jaccard(support[:, i], support[:, j])
                for i, j in itertools.combinations(null_lats[:60], 2)]

    # (c) a cosine detector over every pair of parent-block latents: precision and recall
    block = sorted({j for p in parents for j in f2l[p]})
    cs = decoder_cosine(W[block], W[block])
    truth = {frozenset((i, j)) for f in split for i, j in itertools.combinations(f2l[f], 2)}
    flagged = {frozenset((block[a], block[b])) for a in range(len(block)) for b in range(a + 1, len(block))
               if float(cs[a, b]) >= COS_CUT}
    tp = len(flagged & truth)
    return {
        "toy": toy, "seed": seed, "k": k, "n_split_parents": len(split), "n_unsplit_parents": len(unsplit),
        "shard_jaccard_global_mean": statistics.mean(jac_global) if jac_global else float("nan"),
        "shard_jaccard_within_parent_mean": statistics.mean(jac_within) if jac_within else float("nan"),
        "shard_cosine_mean": statistics.mean(cos_shards) if cos_shards else float("nan"),
        "shard_cosine_min": min(cos_shards) if cos_shards else float("nan"),
        "null_cosine_abs_mean": statistics.mean(cos_null_vals), "null_cosine_abs_max": max(cos_null_vals),
        "null_jaccard_mean": statistics.mean(jac_null) if jac_null else float("nan"),
        "coverage_shard_given_child_median": statistics.median(cov_shard) if cov_shard else float("nan"),
        "cosine_detector": {"cut": COS_CUT, "n_true_pairs": len(truth), "n_flagged": len(flagged),
                            "precision": tp / len(flagged) if flagged else float("nan"),
                            "recall": tp / len(truth) if truth else float("nan")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", default="only_isa")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--n-tokens", type=int, default=50_000)
    ap.add_argument("--fraction", type=float, default=0.5, help="share of parents split (his runs: 0.5)")
    a = ap.parse_args()
    rows = [run_point(a.toy, s, k, a.n_tokens, a.fraction) for k in a.k for s in a.seeds]
    out = C.OUT_DIR / "metrics_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_detectors.json").write_text(json.dumps(rows, indent=1))

    def agg(k, key):
        v = [r[key] if not isinstance(r[key], dict) else None for r in rows if r["k"] == k]
        v = [x for x in v if x is not None and x == x]
        return f"{statistics.mean(v):.3f} ± {statistics.pstdev(v):.3f}" if v else "n/a"

    def aggd(k, key):
        v = [r["cosine_detector"][key] for r in rows if r["k"] == k]
        return f"{statistics.mean(v):.2f}"

    lines = [f"# Split detectors on the collaborator's shard corruption (toy `{a.toy}`, seeds {a.seeds}, "
             f"fraction {a.fraction}, {a.n_tokens:,} tokens)", "",
             "Shards of one feature share its direction and partition its tokens. Firing-overlap "
             "detectors read them as disjoint siblings; decoder cosine reads them as one direction.", "",
             "| k | shard Jaccard (global) | shard Jaccard (within parent) | shard decoder cosine | "
             "null cosine |abs| (unsplit parents) | null Jaccard | P(shard \\| child), median | "
             f"cosine ≥ {COS_CUT} detector: precision / recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in a.k:
        lines.append(f"| {k} | {agg(k, 'shard_jaccard_global_mean')} | {agg(k, 'shard_jaccard_within_parent_mean')} | "
                     f"{agg(k, 'shard_cosine_mean')} | {agg(k, 'null_cosine_abs_mean')} (max {agg(k, 'null_cosine_abs_max')}) | "
                     f"{agg(k, 'null_jaccard_mean')} | {agg(k, 'coverage_shard_given_child_median')} | "
                     f"{aggd(k, 'precision')} / {aggd(k, 'recall')} |")
    md = "\n".join(lines) + "\n"
    (out / "split_detectors.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
