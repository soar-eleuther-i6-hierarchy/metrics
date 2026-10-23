"""Containment: is the child inside the parent?

    R[p, c] = cofire(p, c) / fire(c) = P(p fires | c fires)
    F[p, c] = cofire(p, c) / fire(p) = P(c fires | p fires)

An edge is a candidate when R >= tau, both endpoints fire at least `min_fire`
times and the pair co-fires at least `min_joint` times. Strict containment adds
F < tau (in-block: R[c, p] < tau), which fixes the direction and forbids cycles.

Same objects as `metrics.coverage` / `metrics.in_block`; see the package README
for the old-to-new table.
"""

from __future__ import annotations

from metrics.coverage import coverage_asymmetry as containment_asymmetry
from metrics.coverage import coverage_legs as containment_ratios
from metrics.coverage import keep_edges as candidate_edges
from metrics.in_block import directed_coverage as in_block_containment

__all__ = [
    "containment_ratios",
    "candidate_edges",
    "containment_asymmetry",
    "in_block_containment",
]
