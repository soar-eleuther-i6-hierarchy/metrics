"""Supporting metrics: each one reads the neighbourhood of an edge, not the edge.

    splitting        do a parent's children overlap instead of partitioning it?
    absorption       does the parent go silent where the child fires, while the
                     decoders stay aligned?                        (new, uncalibrated)
    composition      does a child sit under two parents that are unrelated to
                     each other?                                   (new, uncalibrated)
    parent_coverage  how much of the parent do its children account for?
    degree           dense parents (superparents) and multi-parented children

`parent_coverage` and `degree` read look-alikes (a dense feature, a child with
two parents), not training pathologies; they are here because they read the
neighbourhood, like the rest of this package.
"""

from .absorption import absorption_signature
from .composition import composition_signature
from .degree import dense_parents, degree_summary, pair_max_outdegree, parent_outdegree
from .parent_coverage import (
    child_energy_share,
    parent_energy_covered,
    parent_support_covered,
    parent_support_covered_exact,
)
from .splitting import (coextensive_pairs, direction_duplicates, sibling_overlap,
                        sibling_overlap_within_parent)

__all__ = [
    "sibling_overlap",
    "sibling_overlap_within_parent",
    "coextensive_pairs",
    "direction_duplicates",
    "absorption_signature",
    "composition_signature",
    "child_energy_share",
    "parent_support_covered",
    "parent_support_covered_exact",
    "parent_energy_covered",
    "degree_summary",
    "parent_outdegree",
    "pair_max_outdegree",
    "dense_parents",
]
