"""The null population: which ordered pairs the false-positive rate is measured over.

NOTHING IS CALIBRATED HERE ANY MORE. This module used to fit a Q01/Q99 threshold per metric on
half of the unrelated-pair null and measure the FPR on the other half. Every rule now compares
against a fixed constant (`scoring/core/gates.py`), so there is no threshold to fit, the
calibration half has no job, and the false-positive rate is measured over the WHOLE null.

`null_quantiles`, `fit_thresholds`, `calibration_support`, `boundary_ties`,
`null_split_assignment` and `split_masks` are deleted rather than left unused. What went with
them, recorded because the artifacts on disk still carry it:

  * the `q` / `quantile_interpolation` / `min_cal_support` / `cal_split_seed` manifest settings;
  * the `thresholds` report block and `null_split_sha256`;
  * the three-valued `split` array (0 not-null / 1 calibration / 2 evaluation), now a plain
    null indicator;
  * `fpr_over_half`, which had no other half to name.

Do not re-introduce a split "for holdout". With nothing fitted there is nothing to leak, and
the old split was never a real holdout for the endpoint-broadcast metrics anyway: their value
is a function of ONE endpoint, so both orderings and both halves carry the same underlying
number. A real holdout for that class needs a FEATURE-level split, which is a different
procedure and was never built.

WHAT SURVIVES is the definition of the null population itself, which is a property of the
answer key rather than of the split. `PRECOMMIT.md` s6.4: build it from the generator's `[F, F]`
labels using TRUE feature ids, before recovery is known, so the same true pair is in the null
for every read of that (toy, seed). The pilot evaluator used RECOVERED POSITIONS instead, which
are read-specific -- the oracle read scores all F features and the trained read scores whatever
survived matching -- so one true pair sat at different indices in the two reads.
"""

from __future__ import annotations

import torch

from scoring.benchmark.registry import NULL_CLASS
from toygen import labels

Key = tuple[int, int]


def null_unordered_keys(pair_labels: torch.Tensor,
                        null_class: str = NULL_CLASS) -> list[Key]:
    """Sorted `(a, b)` with `a < b` where BOTH orderings carry the null label.

    Requiring both orderings makes the key unambiguous: a pair whose flip is `reversed` is an
    ancestry pair seen the wrong way round, not a null one, and it must not enter the null
    population under either ordering.
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
    """`[n_pairs]` bool: this ordered pair belongs to the null population.

    `pairs` are POSITIONS into `feats`; the lookup key is built from the TRUE feature ids
    `feats[a], feats[b]`, so the population is the same set of true pairs in every read.

    This is NOT `y == unrelated`. A pair can carry the null label in one ordering while its
    flip is `reversed`, and `null_unordered_keys` excludes exactly those -- an ancestry pair
    seen backwards is not evidence about unrelated features. Reading the per-ordered-pair label
    directly would quietly admit them and lower every FPR.
    """
    keys = set(null_unordered_keys(pair_labels, null_class))
    out = torch.zeros(len(pairs), dtype=torch.bool)
    for i, (a, b) in enumerate(pairs):
        fa, fb = int(feats[a]), int(feats[b])
        out[i] = ((fa, fb) if fa < fb else (fb, fa)) in keys
    return out
