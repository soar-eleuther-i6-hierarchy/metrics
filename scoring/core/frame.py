"""The ordered-pair frame over the scored features, the seeds of the scoring and probe-fitting
draws, and per-class recovery."""

from __future__ import annotations

import torch

from scoring.config import BENCHMARK, BenchmarkSettings
from toygen import labels


def pair_frame(recovered_feats: list[int], pair_labels: torch.Tensor
               ) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """All ordered off-diagonal pairs, as positions into `recovered_feats`, with their class
    index from `pair_labels`."""
    R = len(recovered_feats)
    pairs, ys = [], []
    for a in range(R):
        for b in range(R):
            if a == b:
                continue
            pairs.append((a, b))
            ys.append(int(pair_labels[recovered_feats[a], recovered_feats[b]]))
    return pairs, torch.tensor(ys, dtype=torch.long)


def held_out_sample_seed(train_seed: int, *, benchmark: BenchmarkSettings = BENCHMARK) -> int:
    """The sampling seed of the scoring draw; a zero offset is refused, since it reuses the
    in-sample draw."""
    if benchmark.held_out_seed_offset == 0:
        raise ValueError("held_out offset must be non-zero: a 0 offset reuses the training draw")
    return int(train_seed) + benchmark.held_out_seed_offset


def probe_fit_sample_seed(seed: int, *, benchmark: BenchmarkSettings = BENCHMARK) -> int:
    """The sampling seed of the probe-fitting draw."""
    return int(seed) + benchmark.probe_fit_seed_offset


def per_class_recovery(recovered: torch.Tensor, pair_labels: torch.Tensor, label_names: list[str]) -> dict[str, tuple[int, int]]:
    """Per label, `(ordered pairs with both endpoints recovered, total pairs)`.

    Scoring runs on recovered pairs only, so this shows how much of each class was dropped."""
    F = pair_labels.shape[0]
    eye = torch.eye(F, dtype=torch.bool, device=pair_labels.device)
    both_recovered = recovered[:, None] & recovered[None, :]   # [F, F]
    out: dict[str, tuple[int, int]] = {}
    for name in label_names:
        in_class = (pair_labels == labels._index(name)) & ~eye
        total = int(in_class.sum())
        both = int((in_class & both_recovered).sum())
        out[name] = (both, total)
    return out
