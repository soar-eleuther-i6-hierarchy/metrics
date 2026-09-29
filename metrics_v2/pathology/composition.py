"""Composition signature (new, uncalibrated, used by no pipeline).

Two readings of composition exist in the project, and the metrics see only one.

Merge (Anders et al. 2024, the paper's Section 4.1): two concepts become one
latent. Through coverage this is a small absorption hole of size rho * P(r | p)
and nothing else, so it has no signature of its own here.

Conjunction (the Tier-1 toy): a latent fires on the intersection of two parents
that both remain. Coverage then proposes *two* edges to the same child, from
parents that are neither nested nor duplicates of each other. That is the
symptom this function reads:

    child c is flagged when it has >= 2 kept parents a, b with
        R_pp[a, b] < tau  and  R_pp[b, a] < tau      (a, b unrelated to each other)

where R_pp[i, j] = P(i | j) over the parent block. A child with two parents one
of which contains the other is transitive closure, not composition; a child
with two co-extensive parents is a split parent. Both are excluded.

It is a report, not a cut. Calibration against the composition toy is to do.
"""

from __future__ import annotations

import torch


def composition_signature(
    edge_mask: torch.Tensor,      # [P, C] kept edges
    parent_cofire: torch.Tensor,  # [P, P] co-firing counts within the parent block
    fire_p: torch.Tensor,         # [P]    parent firing counts
    *,
    tau: float,
) -> dict[str, object]:
    """{"flag" [C] bool, "pairs" {child: [(a, b), ...]}}: children under unrelated parents."""
    P, C = edge_mask.shape
    if parent_cofire.shape != (P, P):
        raise ValueError(f"parent_cofire {tuple(parent_cofire.shape)} does not match {P} parents")
    R_pp = parent_cofire.double() / fire_p.double().clamp(min=1.0).unsqueeze(0)   # P(i | j)
    unrelated = (R_pp < float(tau)) & (R_pp.T < float(tau))
    unrelated.fill_diagonal_(False)

    flag = torch.zeros(C, dtype=torch.bool)
    pairs: dict[int, list[tuple[int, int]]] = {}
    for c in range(C):
        parents = torch.nonzero(edge_mask[:, c]).flatten().tolist()
        if len(parents) < 2:
            continue
        found = [(a, b) for i, a in enumerate(parents) for b in parents[i + 1:] if bool(unrelated[a, b])]
        if found:
            flag[c] = True
            pairs[c] = found
    return {"flag": flag, "pairs": pairs}
