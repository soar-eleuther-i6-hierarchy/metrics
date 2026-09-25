"""The ground-truth pair-label table.

Labels are per ordered pair `(a, b)`: "a is the candidate parent, b the candidate child".
`is_a`, `firing_only`, `transitive`, `reversed` are directional. `sibling`, `superparent`,
`frequency`, `topical` are assigned to both orderings of a pair, but the co-firing behind them
need not be symmetric: a token container or topic register contains its members one way only.
Each pair carries exactly one class -- classes can't overlap by construction, and `pair_label`
enforces this with a disjointness check.

Only data-side properties appear here (never absorption/splitting/merging, which are
dictionary-side). Every label is read off the generator's declared structure, never
inferred from co-firing statistics, so the answer key is independent of the metrics under test.
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
    """`[F, F]` int8 answer key: the single class index of each ordered pair.

    Cell (a, b) holds `i` where the pair's class is `LABELS[i]`. The diagonal holds the
    `_UNSET` sentinel `-1`, deliberately neither index-0 `is_a` nor `unrelated` -- so code
    that forgets to mask self-pairs can't misread one. `unrelated` is assigned off-diagonal
    wherever nothing else applies; `assign` raises if a cell is claimed by two classes.
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

    # symmetric coincidence confounds
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

    # superparent: both orderings of every pair touching a declared superparent, since its high base rate contaminates the pair either way.
    # A dense true parent has the same base-rate coverage toward every feature outside its own lineage, so those pairs get the same class.
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

    # declared ancestry (forward) and its reverse. is_a vs firing_only is decided per-edge (alpha > 0), not per-feature -- one feature can sit on both kinds of edge.
    for b in range(F):
        parent_alpha = {p: alpha for p, _, alpha in tree.parents.get(b, [])}
        for a in tree.ancestors[b]:
            if a in parent_alpha:
                name = "is_a" if float(parent_alpha[a]) > 0.0 else "firing_only"
            else:
                name = "transitive"
            assign(a, b, name)
            assign(b, a, "reversed")

    # the null: every off-diagonal pair with no declared property becomes `unrelated`. The diagonal stays at `_UNSET` (-1), so code that forgets to mask self-pairs can't mistake one for an is_a positive.
    off_diag = ~torch.eye(F, dtype=torch.bool)
    y[(y == _UNSET) & off_diag] = _index("unrelated")
    y.fill_diagonal_(_UNSET)
    return y
