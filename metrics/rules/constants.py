"""
Named gate constant sets, one per pipeline, since some values do not transfer across dictionary
sizes. Stamp `name` and `rules.RULESET_VERSION` beside any reported number.

`config.py` reads `GEMMA_MATRYOSHKA` and `scoring/core/registry.py` reads `SYNTHETIC_TOYS`.
Changing a value changes that pipeline's verdicts, so bump `rules.RULESET_VERSION` with it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GateConstants:
    name: str
    fire_threshold: float         # a feature fires when its activation exceeds this
    edge_tau: float               # coverage cut for every containment gate
    min_fire_count: int           # both endpoints fire at least this often (support)
    min_joint: int                # and co-fire at least this often (support)
    recon_rel_gain_min: float     # relative reconstruction gain for gate_recon
    sres_rank_top_k: int          # rank cut for gate_sres_rank
    superparent_outdeg_frac: float  # share of candidate children for gate_high_outdegree
    freq_survival_min: float      # on the raw ratio R_rare / R_all, for gate_freq_survives
    sres_rank_pool: str           # what the probe ranks against: "dictionary" or "scored_frame"
    freq_buckets: str             # how tokens are called frequent: "global" (by id) or "local"

    def as_dict(self) -> dict:
        return asdict(self)


# Her pipeline: the released Gemma-2-2b Matryoshka SAE, ~32k latents.
GEMMA_MATRYOSHKA = GateConstants(
    name="gemma_matryoshka",
    fire_threshold=1e-3,          # post-JumpReLU
    edge_tau=0.5,
    min_fire_count=20,
    min_joint=30,
    recon_rel_gain_min=0.01,
    sres_rank_top_k=5,            # Tree SAE's operational rule
    superparent_outdeg_frac=0.30,
    freq_survival_min=0.5,
    sres_rank_pool="dictionary",
    freq_buckets="global",
)

# Our pipeline: synthetic toys, a few hundred latents.
SYNTHETIC_TOYS = GateConstants(
    name="synthetic_toys",
    fire_threshold=0.0,           # BatchTopK nonzero == top-k
    edge_tau=0.5,
    min_fire_count=20,
    min_joint=30,
    recon_rel_gain_min=0.01,
    # k=2: the probe is fitted on the child's own firing, so the child is rank 1 by
    # construction and k=5 let through any parent in ranks 2-5. PRECOMMIT.md s4.
    sres_rank_top_k=2,
    superparent_outdeg_frac=0.30,
    freq_survival_min=0.5,
    sres_rank_pool="scored_frame",
    freq_buckets="global",
)

CONSTANT_SETS: dict[str, GateConstants] = {s.name: s for s in (GEMMA_MATRYOSHKA, SYNTHETIC_TOYS)}
