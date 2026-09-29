"""The null population: which ordered pairs the false-positive rate is measured over.

Keyed on true feature ids, not recovered positions, so a true pair is in the null for every
read of a (toy, seed) (PRECOMMIT.md s7). Nothing is fitted here, despite the module name.
"""

from __future__ import annotations

import torch

from scoring.benchmark.registry import NULL_CLASS
from toygen import labels

Key = tuple[int, int]


def null_unordered_keys(pair_labels: torch.Tensor,
                        null_class: str = NULL_CLASS) -> list[Key]:
    """Sorted `(a, b)`, `a < b`, where both orderings carry the null label.

    A pair whose flip is `reversed` is ancestry seen backwards and stays out under both orderings.
    """
    idx = labels._index(null_class)
    m = pair_labels == idx
    both = m & m.transpose(0, 1)
    F = int(pair_labels.shape[0])
    iu = torch.triu_indices(F, F, offset=1)
    sel = both[iu[0], iu[1]]
    return [(int(a), int(b)) for a, b, keep in zip(iu[0].tolist(), iu[1].tolist(),
                                                   sel.tolist()) if keep]


def null_mask(pairs: list[tuple[int, int]], feats: list[int],
              pair_labels: torch.Tensor,
              null_class: str = NULL_CLASS) -> torch.Tensor:
    """`[n_pairs]` bool: the ordered pair is in the null, keyed on true ids `feats[a], feats[b]`.

    Not `y == null_class`, which would also admit pairs whose flip is `reversed`.
    """
    keys = set(null_unordered_keys(pair_labels, null_class))
    out = torch.zeros(len(pairs), dtype=torch.bool)
    for i, (a, b) in enumerate(pairs):
        fa, fb = int(feats[a]), int(feats[b])
        out[i] = ((fa, fb) if fa < fb else (fb, fa)) in keys
    return out
