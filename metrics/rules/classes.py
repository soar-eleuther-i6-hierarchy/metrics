"""The pair classes a rule can target, in the index order pair labels are stored in."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "hierarchy_overlap",     # direct parent -> child edge, the child's direction overlaps the parent's
    "hierarchy_orthogonal",  # direct parent -> child edge, orthogonal directions
    "transitive",            # the child is a strict descendant but not a direct child
    "sibling",               # the two share a direct parent
    "dense_lookalike",       # a dense feature -> a sparse feature outside its lineage
    "dense_other",           # every other ordering of a dense pair: sparse -> dense, dense <-> dense
    "frequency_lookalike",   # a token container -> a member of its group
    "frequency_other",       # every other pair of token-bound features on overlapping token ids
    "topical_lookalike",     # a topic register -> a member of its topic
    "topical_other",         # every other pair of topical features on the same topic
    "unrelated",             # the null: no declared property holds
    "reversed",              # a directed ancestry pair read the wrong way round
)

NULL_CLASS = "unrelated"
