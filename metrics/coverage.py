"""
Metric 1 - Activation coverage, three legs.

For a candidate edge parent p -> child c, from co-firing counts:

    reverse  R = cofire(p,c) / fire(c) = P(parent fires | child fires)
        "child contained in parent" - the classic edge criterion.

    forward  F = cofire(p,c) / fire(p) = P(child fires | parent fires)
        "how much of the parent this one child accounts for".

    joint-child J(p) = P(at least one kept child fires | parent fires)
        "children together account for the parent". Exact J needs the union
        count over tokens (computed in the streaming pass); the closed-form
        upper bound min(1, sum_c F) is provided as a fallback.

Known blind spot (why the other metrics exist): coverage only sees
co-OCCURRENCE. A child that happens to fire inside a very frequent parent
gets a high R without any semantic or reconstructive relationship.
"""

from __future__ import annotations

import torch


def coverage_legs(
    cofire: torch.Tensor,      # [P, C] co-firing counts
    fire_p: torch.Tensor,      # [P]    parent firing counts
    fire_c: torch.Tensor,      # [C]    child firing counts
    *,
    undefined: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (R, F), both [P, C]. Zero-fire features get coverage 0, or `undefined` when
    given (e.g. NaN, for a caller that treats them as unmeasurable)."""
    cofire = cofire.double()
    if undefined is not None:
        fc, fp = fire_c.double().unsqueeze(0), fire_p.double().unsqueeze(1)
        return (torch.where(fc > 0, cofire / fc, undefined),
                torch.where(fp > 0, cofire / fp, undefined))
    R = cofire / fire_c.double().clamp(min=1.0).unsqueeze(0)
    F = cofire / fire_p.double().clamp(min=1.0).unsqueeze(1)
    return R, F


def coverage_asymmetry(
    R: torch.Tensor,           # [P, C] reverse coverage P(p fires | c fires)
    F: torch.Tensor,           # [P, C] forward coverage P(c fires | p fires)
) -> torch.Tensor:
    """[P, C] R - F: > 0 when the child sits inside the parent, ~0 for symmetric co-firing."""
    return R - F


def keep_edges(
    R: torch.Tensor,           # [P, C] reverse coverage
    fire_p: torch.Tensor,
    fire_c: torch.Tensor,
    tau: float,
    min_fire: int,
    cofire: torch.Tensor | None = None,
    min_joint: int = 0,
) -> torch.Tensor:
    """Boolean [P, C] edge mask: R >= tau, both endpoints fire often enough,
    and (when cofire is given) at least min_joint co-firing tokens.

    The joint-support guard kills chance edges: a child
    firing min_fire times inside a ~always-on parent reaches R = 1.0 with no
    evidence beyond base rate. Callers report the excluded count.
    """
    keep = R >= tau
    keep[fire_p < min_fire, :] = False
    keep[:, fire_c < min_fire] = False
    if cofire is not None and min_joint > 0:
        keep = keep & (cofire >= min_joint)
    return keep


def joint_child_coverage_upper(
    F: torch.Tensor,           # [P, C] forward coverage
    edge_mask: torch.Tensor,   # [P, C] kept edges
) -> torch.Tensor:
    """Upper bound on J(p): sum of forward coverages of kept children, capped at 1.

    Children can co-fire, so the sum double-counts; the streaming union count
    (see collect_statistics.py) gives the exact value.
    """
    return (F * edge_mask).sum(dim=1).clamp(max=1.0)


def joint_child_coverage_exact(
    union_count: torch.Tensor,  # [P] tokens where parent fires AND >=1 kept child fires
    fire_p: torch.Tensor,       # [P]
) -> torch.Tensor:
    """Exact J(p) = union_count / fire(p)."""
    return union_count.double() / fire_p.double().clamp(min=1.0)
