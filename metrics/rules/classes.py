"""The pair classes a rule can target, in the index order pair labels are stored in."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "is_a",         # direct containment edge, parent and child directions overlap
    "firing_only",  # direct containment edge, orthogonal directions
    "transitive",   # the child is a strict descendant but not a direct child
    "sibling",      # the two share a direct parent
    "superparent",  # either end is a superparent, or a dense parent with a non-relative
    "frequency",    # both token-bound, on overlapping token ids
    "topical",      # both topical, on the same topic
    "unrelated",    # the null: no declared property holds
    "reversed",     # a directed ancestry pair read the wrong way round
)

NULL_CLASS = "unrelated"
