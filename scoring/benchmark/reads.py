"""The `Read` shape a scored dictionary is reduced to, and the metric and gate columns built on it.

synthdict's `synthetic_read` builds one per dial point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metrics.outdegree import either_endpoint_outdegree
from metrics.rules import GATE_NAMES
from scoring.benchmark.registry import METRICS
from scoring.core.detectors import decoder_cosine
from scoring.core.frame import per_class_recovery
from scoring.core.registry import DETECTOR_SIGN, DETECTORS
from toygen import labels

_NAN = float("nan")


@dataclass(frozen=True)
class Read:
    """One (toy, seed, read) scored universe. `pairs` are positions into `feats`, the true ids.

    `vals` holds one float64 score per METRICS name; `gate_vals` one float64 tristate
    {1.0, 0.0, NaN} per GATE_NAMES name, kept apart so a gate is never read as a continuous score.
    """

    toy: str
    seed: int
    read: str                       # "oracle" | "trained"
    n_tokens: int
    s_res_mode: str                 # "probe" | "absent"
    F: int
    feats: list[int]
    pairs: list[tuple[int, int]]
    y: torch.Tensor
    vals: dict[str, torch.Tensor]
    gate_vals: dict[str, torch.Tensor]
    pair_labels: torch.Tensor
    W_unit: torch.Tensor
    recovered: torch.Tensor         # [F] bool, the scored universe
    detector_matrices: dict[str, torch.Tensor]
    extra: dict

    @property
    def n_recovered(self) -> int:
        return len(self.feats)


def wide_matrix(outdegree: torch.Tensor) -> torch.Tensor:
    """`wide` from the oriented out-degree matrix, via `metrics.outdegree.either_endpoint_outdegree`:
    `min(outdegree[p,c], outdegree[c,p])`, NaN unless both orderings are finite (PRECOMMIT s6).

    Orientation negates out-degree, so the larger raw out-degree is the smaller oriented value.
    """
    s = DETECTOR_SIGN["outdegree"]
    assert s == -1, "wide_matrix takes the larger raw out-degree, the negated minimum"
    return s * either_endpoint_outdegree(s * outdegree, s * outdegree.transpose(0, 1))


def _scored(mat: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long)
    return mat[pa, pb].double()


def add_derived(vals: dict[str, torch.Tensor], outdegree_matrix: torch.Tensor | None,
                pairs: list[tuple[int, int]] | None) -> dict[str, torch.Tensor]:
    """Add `abs_asymmetry_R`, the absolute value of `metrics.coverage.coverage_asymmetry`, and,
    when the outdegree matrix is given, `wide`."""
    out = dict(vals)
    if "asymmetry_R" in out:
        out["abs_asymmetry_R"] = out["asymmetry_R"].abs()
    if outdegree_matrix is not None and pairs is not None:
        out["wide"] = _scored(wide_matrix(outdegree_matrix), pairs)
    return out


def class_totals(pair_labels: torch.Tensor) -> dict[str, int]:
    """Generated ordered pairs per class over the full feature set: the `N_total` denominator.

    From the answer key, not the scored pairs, which drop every unrecovered target.
    """
    F = int(pair_labels.shape[0])
    eye = torch.eye(F, dtype=torch.bool)
    return {name: int(((pair_labels == labels._index(name)) & ~eye).sum())
            for name in labels.LABELS}


def class_recovered(pair_labels: torch.Tensor, in_universe: torch.Tensor) -> dict[str, int]:
    """Ordered pairs per class with BOTH endpoints in the scored universe (`N_recovered`)."""
    return {name: n for name, (n, _tot) in
            per_class_recovery(in_universe, pair_labels, list(labels.LABELS)).items()}


def assemble_metrics(dets: dict[str, torch.Tensor], W_unit: torch.Tensor,
                     probe: torch.Tensor | None, pairs: list[tuple[int, int]]
                     ) -> dict[str, torch.Tensor]:
    """Every detector, plus `G` (the decoder cosine), `S_res` (`probe`, the matrix from
    `s_res_from_directions`, or None when no probe ran) and the derived metrics.

    Every read goes through here, so none can fill `G` from the probe or `S_res` from the cosine.
    """
    vals = {d: _scored(dets[d], pairs) for d in DETECTORS}
    vals["G"] = _scored(decoder_cosine(W_unit), pairs)
    vals["S_res"] = (_scored(probe, pairs) if probe is not None
                     else torch.full((len(pairs),), _NAN, dtype=torch.float64))
    vals = add_derived(vals, dets["outdegree"], pairs)
    missing = [m for m in METRICS if m not in vals]
    if missing:
        raise RuntimeError(f"the read did not produce {missing}; a registered expression "
                           f"would go untestable for a harness reason, not a metric one")
    return vals


def assemble_gates(gate_mats: dict[str, torch.Tensor],
                   pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Every registered gate on the ordered-pair frame; raises if the read lacks one.

    A missing gate would otherwise pass for a rule that is untestable on the data.
    """
    missing = [g for g in GATE_NAMES if g not in gate_mats]
    if missing:
        raise RuntimeError(f"the read did not produce {missing}; a registered expression "
                           f"would go untestable for a harness reason, not a metric one")
    return {g: _scored(gate_mats[g], pairs) for g in GATE_NAMES}
