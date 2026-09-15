"""Does the Matryoshka nesting itself respect the hierarchy?

`calibrate_on_trained_toy.py` asks whether the *metrics* recover the true tree,
and deliberately indexes by ground truth rather than by block: mixing the two
would confound "is the metric right?" with "did Matryoshka order the features
right?". That separation is what lets Tier 2 report its recall as the SAE's
ceiling rather than the metrics'.

This script asks the other half of that question, which nothing else does. The
toy SAE has ten Matryoshka blocks of two latents each, and the true tree is
known, so we can check the architecture's own claim directly: a parent should
land in an earlier block than its children.

On gemma this is unanswerable -- the correct ordering is unknown, so a violation
is indistinguishable from a concept we misunderstood. Here it is measurable.

    python3 -m validation.block_tree_alignment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.calibrate_on_trained_toy import (  # noqa: E402
    build_tree,
    load_sae,
    match_latents,
    n_features,
    true_edges,
)


def block_ranges(latent_sizes):
    """Matryoshka blocks are nested prefixes, so the ranges are contiguous."""
    out, prev = [], 0
    for s in latent_sizes:
        out.append((prev, s))
        prev = s
    return out


def block_of(idx, ranges):
    return next(b for b, (s, e) in enumerate(ranges) if s <= idx < e)


def main():
    tree = build_tree()
    truth = true_edges(tree)
    F = n_features(tree)
    w, cfg = load_sae()
    match = match_latents(w, torch.eye(F))
    ranges = block_ranges(cfg["latent_sizes"])

    # Where each true feature was recovered. A feature can be recovered by more
    # than one latent (that is feature splitting); take the EARLIEST block, which
    # is the reading most favourable to the architecture's claim.
    first_block: dict[int, int] = {}
    all_blocks: dict[int, list[int]] = {}
    for i, t in enumerate(match.tolist()):
        if t < 0:
            continue
        b = block_of(i, ranges)
        all_blocks.setdefault(t, []).append(b)
        first_block[t] = min(first_block.get(t, b), b)

    # --- the actual test: does every true edge run early -> late? -------------
    rows, violations, untestable = [], [], []
    for p, c in sorted(truth):
        bp, bc = first_block.get(p), first_block.get(c)
        if bp is None or bc is None:
            untestable.append((p, c))
            rows.append({"edge": f"{p} -> {c}", "parent_block": bp, "child_block": bc,
                         "verdict": "untestable: feature not recovered"})
            continue
        ok = bp < bc
        if not ok:
            violations.append((p, c))
        rows.append({"edge": f"{p} -> {c}", "parent_block": bp, "child_block": bc,
                     "verdict": "respected" if ok else "VIOLATED: child is not deeper than its parent"})

    testable = len(truth) - len(untestable)
    splits = {t: bs for t, bs in all_blocks.items() if len(bs) > 1}
    unmatched = int((match < 0).sum())

    parents = sorted({p for p, _ in truth})
    children = sorted({c for _, c in truth})
    p_blocks = [first_block[t] for t in parents if t in first_block]
    c_blocks = [first_block[t] for t in children if t in first_block]

    print(f"toy SAE: d_sae={cfg['d_sae']}, {len(ranges)} Matryoshka blocks {cfg['latent_sizes']}")
    print(f"true tree: {len(parents)} parents, {len(children)} children, {len(truth)} edges\n")
    for r in rows:
        print(f"  {r['edge']:<10} parent B{r['parent_block']}  child B{r['child_block']}   {r['verdict']}")

    print(f"\nmean block — parents {sum(p_blocks)/len(p_blocks):.1f}   children {sum(c_blocks)/len(c_blocks):.1f}")
    print(f"edges respecting the nesting: {testable - len(violations)}/{testable} testable "
          f"({len(untestable)} untestable — feature never recovered)")
    if violations:
        print(f"  violations: {violations}")
    if splits:
        print(f"feature splitting: {len(splits)} true feature(s) recovered by >1 latent -> {splits}")
    print(f"latents matching nothing: {unmatched}/{cfg['d_sae']}")

    result = {
        "d_sae": cfg["d_sae"],
        "latent_sizes": cfg["latent_sizes"],
        "block_ranges": ranges,
        "first_block_of_feature": {str(k): v for k, v in sorted(first_block.items())},
        "edge_rows": rows,
        "n_testable": testable,
        "n_respected": testable - len(violations),
        "violations": [list(v) for v in violations],
        "untestable": [list(u) for u in untestable],
        "split_features": {str(k): v for k, v in splits.items()},
        "unmatched_latents": unmatched,
        "mean_parent_block": sum(p_blocks) / len(p_blocks) if p_blocks else None,
        "mean_child_block": sum(c_blocks) / len(c_blocks) if c_blocks else None,
    }
    out = ROOT / "outputs" / "block_tree_alignment.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    return result


if __name__ == "__main__":
    main()
