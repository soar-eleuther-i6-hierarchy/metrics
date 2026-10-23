"""Degree: dense parents and multi-parented children.

    dense_parents     out-degree >= 30% of the child block (flag on out-degree alone)
    degree_summary    out/in-degree, poly-parenting share, top-1 edge share, Gini
    parent_outdegree  kept children per parent
    pair_max_outdegree the larger out-degree of a pair's two endpoints

Note for wide blocks (T-SAE's 13,108-latent second block): a 30% cut is
3,932 children and is never reached, so a zero superparent count there is a
block-width effect, not a finding. A width-independent cut is a v2 decision
still open; see the README.

Same objects as `metrics.outdegree`.
"""

from __future__ import annotations

from metrics.outdegree import degree_stats as degree_summary
from metrics.outdegree import either_endpoint_outdegree as pair_max_outdegree
from metrics.outdegree import find_superparents as dense_parents
from metrics.outdegree import kept_outdegree as parent_outdegree

__all__ = ["degree_summary", "parent_outdegree", "pair_max_outdegree", "dense_parents"]
