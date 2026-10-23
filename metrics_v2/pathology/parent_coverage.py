"""Parent coverage: how much of the parent do the next block's latents account for?

    parent_support_covered(p) = |{x : f_p > 0 and >= 1 child fires}| / |{x : f_p > 0}|
    parent_energy_covered(p)  = sum f_p^2 [>= 1 child fires] / sum f_p^2
    child_energy_share(c, p)  = sum f_p^2 [f_c > 0] / sum f_p^2

Superparent signature: R ~ 1 against many children while energy coverage ~ 0.
One child with share ~ 1 is a rename across the block boundary.

`parent_support_covered_exact` is `metrics.coverage.joint_child_coverage_exact`,
the same ratio from a union count; kept as a drift guard. The old upper bound
min(1, sum F) is dropped, not renamed (see `metrics_v2.DROPPED`).

Same objects as `metrics.joint_child` / `metrics.coverage`.
"""

from __future__ import annotations

from metrics.coverage import joint_child_coverage_exact as parent_support_covered_exact
from metrics.joint_child import r_mass as parent_energy_covered
from metrics.joint_child import r_supp as parent_support_covered
from metrics.joint_child import share_energy as child_energy_share

__all__ = [
    "child_energy_share",
    "parent_support_covered",
    "parent_support_covered_exact",
    "parent_energy_covered",
]
