"""Ground-truth pair labels, read off the declared structure and never from co-firing statistics.

Each ordered pair (a, b), a the candidate parent and b the candidate child, gets exactly one class.
sibling, superparent, frequency and topical go on both orderings even where the co-firing is one-way.
"""

from __future__ import annotations

import torch

from metrics.rules.classes import LABELS

from .tree import Tree


def label_name(i: int) -> str:
    return LABELS[i]


def _index(name: str) -> int:
    return LABELS.index(name)


def pair_label(tree: Tree) -> torch.Tensor:
    """`[F, F]` int8 answer key: cell (a, b) holds the `LABELS` index of the pair's class.

    The diagonal holds -1, neither `is_a` (index 0) nor `unrelated`, so code that forgets to mask
    self-pairs can't misread one. `assign` raises if two classes claim a cell.
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

    # coincidence confounds, on both orderings
    for a in range(F):
        for b in range(F):
            if a == b:
                continue
            if ("topical" in tree.tags[a] and "topical" in tree.tags[b]
                    and tree.topic[a] is not None and tree.topic[a] == tree.topic[b]):
                assign(a, b, "topical")
            if tree.token_bound[a] and tree.token_bound[b] and set(_ids(a)) & set(_ids(b)):
                assign(a, b, "frequency")

    # siblings: any two features sharing a direct parent (exclusive or not)
    for kids in tree.children.values():
        for i, x in enumerate(kids):
            for z in kids[i + 1:]:
                assign(x, z, "sibling")
                assign(z, x, "sibling")

    # superparent: both orderings of every pair touching a superparent, and of every pair between a
    # dense true parent and a feature outside its lineage (same base-rate coverage)
    for a in range(F):
        if "superparent" in tree.tags[a]:
            lineage: set[int] = set()
        elif "dense_parent" in tree.tags[a]:
            lineage = tree.ancestors[a] | tree.descendents[a]
        else:
            continue
        for b in range(F):
            if a != b and b not in lineage:
                assign(a, b, "superparent")
                assign(b, a, "superparent")

    # declared ancestry and its reverse; is_a vs firing_only is per edge (alpha > 0), not per feature
    for b in range(F):
        parent_alpha = {p: alpha for p, _, alpha in tree.parents.get(b, [])}
        for a in tree.ancestors[b]:
            if a in parent_alpha:
                name = "is_a" if float(parent_alpha[a]) > 0.0 else "firing_only"
            else:
                name = "transitive"
            assign(a, b, name)
            assign(b, a, "reversed")

    # every other off-diagonal pair is unrelated; the diagonal stays -1
    off_diag = ~torch.eye(F, dtype=torch.bool)
    y[(y == _UNSET) & off_diag] = _index("unrelated")
    y.fill_diagonal_(_UNSET)
    return y
