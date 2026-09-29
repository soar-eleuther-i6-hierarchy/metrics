"""Ground-truth pair labels, read off the declared structure and never from co-firing statistics.

Each ordered pair (a, b), a the candidate parent and b the candidate child, gets exactly one class.
The dense, frequency and topical families cover both orderings of a pair; within each, the ordering
that looks like a parent -> child edge is `*_lookalike` and every other is `*_other`.
"""

from __future__ import annotations

import torch

from metrics.rules.classes import LABELS

from .tree import Tree


def _index(name: str) -> int:
    return LABELS.index(name)


def pair_label(tree: Tree) -> torch.Tensor:
    """`[F, F]` int8 answer key: cell (a, b) holds the `LABELS` index of the pair's class.

    The diagonal holds -1, neither `hierarchy_overlap` (index 0) nor `unrelated`, so code that
    forgets to mask self-pairs can't misread one. `assign` raises if two classes claim a cell.
    """
    F = tree.F
    _UNSET = -1
    y = torch.full((F, F), _UNSET, dtype=torch.int8)

    def assign(a: int, b: int, name: str) -> None:
        idx = _index(name)
        cur = int(y[a, b])
        if cur != _UNSET and cur != idx:
            raise ValueError(
                f"pair ({a}, {b}) matches two classes: {LABELS[cur]} and {name} — the "
                f"relationship classes must be non-intersecting"
            )
        y[a, b] = idx

    def _ids(k: int) -> tuple[int, ...]:
        ids = tree.token_ids.get(k)
        if not ids:
            raise ValueError(f"token-bound feature {k} has no token ids; the frequency rule needs its id set")
        return ids

    # coincidence confounds, on both orderings; a register or container -> its member is the look-alike
    for a in range(F):
        for b in range(F):
            if a == b:
                continue
            if ("topical" in tree.tags[a] and "topical" in tree.tags[b]
                    and tree.topic[a] is not None and tree.topic[a] == tree.topic[b]):
                look = "topic_register" in tree.tags[a] and "topic_member" in tree.tags[b]
                assign(a, b, "topical_lookalike" if look else "topical_other")
            if tree.token_bound[a] and tree.token_bound[b] and set(_ids(a)) & set(_ids(b)):
                look = "token_container" in tree.tags[a] and "token_member" in tree.tags[b]
                assign(a, b, "frequency_lookalike" if look else "frequency_other")

    # siblings: any two features sharing a direct parent (exclusive or not)
    for kids in tree.children.values():
        for i, x in enumerate(kids):
            for z in kids[i + 1:]:
                assign(x, z, "sibling")
                assign(z, x, "sibling")

    # dense: both orderings of every pair touching a superparent, and of every pair between a dense
    # true parent and a feature outside its lineage (same base-rate coverage); dense -> sparse is the look-alike
    dense = {k for k in range(F) if tree.tags[k] & {"superparent", "dense_parent"}}
    for a in sorted(dense):
        lineage = set() if "superparent" in tree.tags[a] else tree.ancestors[a] | tree.descendents[a]
        for b in range(F):
            if a != b and b not in lineage:
                assign(a, b, "dense_other" if b in dense else "dense_lookalike")
                assign(b, a, "dense_other")

    # declared ancestry and its reverse; overlap vs orthogonal is per edge (alpha > 0), not per feature
    for b in range(F):
        parent_alpha = {p: alpha for p, _, alpha in tree.parents.get(b, [])}
        for a in tree.ancestors[b]:
            if a in parent_alpha:
                name = "hierarchy_overlap" if float(parent_alpha[a]) > 0.0 else "hierarchy_orthogonal"
            else:
                name = "transitive"
            assign(a, b, name)
            assign(b, a, "reversed")

    # every other off-diagonal pair is unrelated; the diagonal stays -1
    off_diag = ~torch.eye(F, dtype=torch.bool)
    y[(y == _UNSET) & off_diag] = _index("unrelated")
    y.fill_diagonal_(_UNSET)
    return y
