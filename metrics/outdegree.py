"""
Metric 4 - Out-degree distribution: the superparent / poly-parenting detector.

A healthy hierarchy has parents with a handful of children each and children
with ~1 parent. The two pathologies show up directly in the degree
distributions:

    superparent   : one parent holding most of the next block's in-edges
                    (out-degree ~ |block|, fires on a huge share of tokens)
    poly-parenting: children with many parents, usually THROUGH superparents

We report the raw distributions plus two scalars per block pair that are easy
to track across sweeps (Exp 2): the top-1 parent's share of all edges, and the
Gini coefficient of the out-degree distribution (0 = evenly spread children,
-> 1 = all edges concentrated on few parents).
"""

from __future__ import annotations

import torch


def degree_stats(edge_mask: torch.Tensor) -> dict:
    """Degree statistics for one block pair. edge_mask: [P, C] bool."""
    outdeg = edge_mask.sum(dim=1)          # [P] children per parent
    indeg = edge_mask.sum(dim=0)           # [C] parents per child
    n_edges = int(edge_mask.sum())

    n_parented = int((indeg > 0).sum())
    # Review note: a block pair with no edges gets poly_frac, top1_edge_share and outdeg_gini of
    # 0.0, the values of a healthy spread, though nothing was measured; NaN, or skipping the
    # pair, would keep it out of averages.
    stats = {
        "n_edges": n_edges,
        "outdeg": outdeg,
        "indeg": indeg,
        "n_parents_with_children": int((outdeg > 0).sum()),
        "n_children_with_parent": n_parented,
        "n_multi_parented": int((indeg >= 2).sum()),
        # PolyFrac: share of PARENTED children with >=2 parents
        "poly_frac": int((indeg >= 2).sum()) / n_parented if n_parented else 0.0,
        "top1_edge_share": float(outdeg.max()) / n_edges if n_edges else 0.0,
        "outdeg_gini": gini(outdeg.double()),
    }
    return stats


def kept_outdegree(
    edge_mask: torch.Tensor,   # [P, C] kept edges
    fire_p: torch.Tensor,      # [P]    parent firing counts
    *,
    undefined: float | None = None,
) -> torch.Tensor:
    """[P] kept children per parent, float64. A parent that never fires has out-degree 0, or
    `undefined` when given."""
    outdeg = edge_mask.to(torch.float64).sum(dim=1)
    if undefined is not None:
        outdeg = torch.where(fire_p > 0, outdeg, undefined)
    return outdeg


def either_endpoint_outdegree(
    outdeg_p: torch.Tensor,    # parent out-degree, broadcastable to [P, C] (e.g. [P, 1])
    outdeg_c: torch.Tensor,    # child out-degree (e.g. [1, C])
) -> torch.Tensor:
    """[P, C] the larger out-degree of the pair's two endpoints, so a pair touching a wide
    feature at either end scores wide. NaN unless both are finite."""
    both = torch.isfinite(outdeg_p) & torch.isfinite(outdeg_c)
    wide = torch.maximum(outdeg_p, outdeg_c)
    return torch.where(both, wide, torch.full_like(wide, float("nan")))


def gini(x: torch.Tensor) -> float:
    """Gini coefficient of a non-negative 1-D tensor (0 = equal, 1 = concentrated)."""
    x = x.double().flatten()
    if x.numel() == 0 or float(x.sum()) == 0.0:
        return 0.0
    xs, _ = torch.sort(x)
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2 * (idx * xs).sum() / (n * xs.sum())) - (n + 1) / n)


def find_superparents(
    edge_mask: torch.Tensor,   # [P, C]
    fire_p: torch.Tensor,      # [P]
    total_tokens: int,
    outdeg_frac: float = 0.30,
    fire_frac: float = 0.10,
) -> list[dict]:
    """Parents covering >= outdeg_frac of the child block (the flag is on
    out-degree ALONE). Firing rate is reported as an attribute,
    and the old outdeg-AND-fire gate survives as `strict` — the AND gate let
    L24's feature 14 (fires 41.9%, fan-out 21.9%... and conversely high-fanout
    low-fire parents) slip through. Returns local indices + stats."""
    n_children = edge_mask.shape[1]
    outdeg = edge_mask.sum(dim=1)
    fire_rate = fire_p.double() / max(total_tokens, 1)
    flag = outdeg >= outdeg_frac * n_children
    out = []
    for p in torch.nonzero(flag).flatten().tolist():
        out.append(
            {
                "parent_local": p,
                "outdeg": int(outdeg[p]),
                "outdeg_frac": float(outdeg[p]) / n_children,
                "fire_frac": float(fire_rate[p]),
                "strict": bool(fire_rate[p] >= fire_frac),   # old AND-gate variant
            }
        )
    out.sort(key=lambda d: -d["outdeg"])
    return out
