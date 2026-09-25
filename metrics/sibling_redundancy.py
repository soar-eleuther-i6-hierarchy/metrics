"""
Metric 3 - Child diversity / sibling redundancy, Tree-SAE style.

If a parent's children are a REAL refinement, they should partition the
parent's firing set: each child claims a different slice, so siblings rarely
co-fire. If instead the children are the same feature split into near-copies
("feature splitting in disguise"), they co-fire massively.

For each parent p with kept children S = {c1..ck}, we report the mean pairwise
sibling Jaccard:

    J(ci, cj) = cofire(ci, cj) / (fire(ci) + fire(cj) - cofire(ci, cj))

    redundancy(p) = mean over pairs (ci, cj) in S

~0   -> children are diverse (healthy refinement)
~1   -> children are duplicates (splitting)

The property under test is disjointness WITHIN the
parent's firing set. The global Jaccard below scores co-firing anywhere —
in unconstrained architectures (Matryoshka) siblings can co-fire or fire
solo where the parent is silent, which is irrelevant to this parent's
partition. `parent_conditioned_redundancy` is the corrected form, computed
in the second pass (needs per-token masks, not just count matrices); the
global form is kept alongside for auditability.
"""

from __future__ import annotations

import torch


def parent_conditioned_redundancy(
    fires_p: torch.Tensor,     # [n] bool: tokens where THIS parent fires
    kids: torch.Tensor,        # [n, k] bool: firing masks of its kept children
    *,
    undefined: float | None = None,
) -> float:
    """Mean pairwise sibling Jaccard restricted to the parent's firing tokens
    (same convention as the global form: high = redundant/splitting).

    Fewer than two children gives 0, and so does a parent that never fires; `undefined`, when
    given, is returned for both instead."""
    sub = kids[fires_p].double()                        # [m, k]
    k = sub.shape[1]
    if undefined is not None and (k < 2 or not bool(fires_p.any())):
        return undefined
    if k < 2:
        return 0.0
    cf = sub.T @ sub                                    # [k, k] co-fire within support
    fi = sub.sum(dim=0)                                 # [k]
    union = fi.unsqueeze(0) + fi.unsqueeze(1) - cf
    jac = cf / union.clamp(min=1.0)
    offdiag = ~torch.eye(k, dtype=torch.bool)
    return float(jac[offdiag].mean())


def sibling_redundancy(
    edge_mask: torch.Tensor,     # [P, C] kept edges for one block pair
    child_cofire: torch.Tensor,  # [C, C] within-block co-firing counts of the child block
    fire_c: torch.Tensor,        # [C]    child firing counts
    max_children: int = 512,     # guard: superparents have thousands of children;
                                 # we subsample pairs via the top-firing children
) -> dict[int, dict]:
    """Per-parent redundancy. Returns {parent_local: {redundancy, n_children}}."""
    fire_c = fire_c.double()
    out: dict[int, dict] = {}
    for p in torch.nonzero(edge_mask.sum(dim=1) >= 2).flatten().tolist():
        kids = torch.nonzero(edge_mask[p]).flatten()
        if kids.numel() > max_children:
            top = torch.argsort(fire_c[kids], descending=True)[:max_children]
            kids = kids[top]
        cf = child_cofire[kids][:, kids].double()          # [k, k]
        fi = fire_c[kids]                                   # [k]
        union = fi.unsqueeze(0) + fi.unsqueeze(1) - cf
        jac = cf / union.clamp(min=1.0)
        k = kids.numel()
        offdiag = ~torch.eye(k, dtype=torch.bool)
        out[p] = {
            "redundancy": float(jac[offdiag].mean()),
            "n_children": int(edge_mask[p].sum()),
        }
    return out
