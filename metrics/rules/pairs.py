"""One entry point: every gate and every rule decision for a frame of candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import GateConstants
from .gates import (
    DT,
    coverage_gates,
    either_endpoint,
    freq_survives_gate,
    nan_self_pairs,
    recon_gate,
    sres_rank_gate,
    support_gate,
    supported,
)
from .rules import RULES, evaluate


@dataclass(frozen=True)
class PairStats:
    """What the gates read, for candidate parents P against candidate children C."""

    cofire: torch.Tensor          # [P, C] tokens on which both fire
    fire_p: torch.Tensor          # [P] firing counts
    fire_c: torch.Tensor          # [C]
    R: torch.Tensor               # [P, C] P(p fires | c fires)
    R_rev: torch.Tensor           # [P, C] P(c fires | p fires)
    parent_gain: torch.Tensor     # [P, C] parent's relative reconstruction gain on c's tokens
    child_gain: torch.Tensor      # [C] child's own relative reconstruction gain
    survival: torch.Tensor        # [P, C] coverage ratio once frequent tokens are removed
    survival_scale: str           # "raw" or "squashed"
    high_outdeg_p: torch.Tensor   # [P] tristate from gates.high_outdegree
    high_outdeg_c: torch.Tensor   # [C] tristate, the child's own out-degree as a parent
    probe_corr: torch.Tensor | None = None       # [C, pool] child probe vs each pool decoder
    parent_ids: torch.Tensor | None = None       # [P] pool index of each parent
    child_ids: torch.Tensor | None = None        # [C] pool index of each child
    probe_available: torch.Tensor | None = None  # [C] whether child's probe trained
    self_pairs: bool = False      # square frame of one set against itself: NaN the diagonal


def score_pairs(stats: PairStats, constants: GateConstants) -> dict:
    """`{"gates": {name: [P, C] tristate}, "rules": {name: (mask, scorable)}}`.

    Without probe correlations `gate_sres_rank` is all NaN, so the rules reading it are
    unscorable rather than rejected.
    """
    s, c = stats, constants
    if s.self_pairs and s.R.shape[0] != s.R.shape[1]:
        raise ValueError(f"self_pairs needs a square frame, got {tuple(s.R.shape)}")
    support = supported(s.cofire, s.fire_p, s.fire_c, c.min_fire_count, c.min_joint)
    gates = {"gate_support": support_gate(support)}
    gates |= coverage_gates(s.R, s.R_rev, support, c.edge_tau)
    gates["gate_recon"] = recon_gate(s.parent_gain, s.child_gain, c.recon_rel_gain_min)
    gates["gate_high_outdegree"] = either_endpoint(s.high_outdeg_p, s.high_outdeg_c)
    gates["gate_freq_survives"] = freq_survives_gate(s.survival, c.freq_survival_min,
                                                     s.survival_scale)
    if s.probe_corr is None:
        gates["gate_sres_rank"] = torch.full(support.shape, float("nan"), dtype=DT,
                                             device=support.device)
    else:
        gates["gate_sres_rank"] = sres_rank_gate(s.probe_corr, s.parent_ids, s.child_ids,
                                                 s.probe_available, c.sres_rank_top_k)
    if s.self_pairs:
        gates = {k: nan_self_pairs(v) for k, v in gates.items()}
    rules = {name: evaluate(r["clauses"], gates)[:2] for name, r in RULES.items()}
    return {"gates": gates, "rules": rules}
