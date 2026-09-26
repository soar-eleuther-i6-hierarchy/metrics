"""
Named gate constant sets, one per pipeline, since some values do not transfer across dictionary
sizes. Stamp `name` and `rules.RULESET_VERSION` beside any reported number.

`config.py` reads `GEMMA_MATRYOSHKA` and `scoring/` reads `SYNTHETIC_TOYS`.
Changing a value changes that pipeline's verdicts, so bump `rules.RULESET_VERSION` with it.

`METRIC_SETTINGS` holds the settings that define a metric rather than a decision on it. There is
one set for every pipeline: a probe fitted with other settings gives a different S_res.
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


@dataclass(frozen=True)
class MetricSettings:
    probe_min_pos: int            # a child needs this many firing tokens to get an S_res probe
    probe_neg_ratio: int          # negatives sampled per positive
    probe_max_tokens: int         # cap on positives plus negatives per probe
    probe_min_neg: int            # fewer negatives than this: no probe, the child is untestable
    probe_steps: int              # Adam steps
    probe_lr: float               # Adam learning rate
    freq_high_mass: float         # token bucket 0: the most frequent ids, up to this share of tokens
    freq_mid_mass: float          # bucket 1: the next ids, up to high + mid; bucket 2 is the rest
    n_freq_buckets: int
    freq_min_fire_low: int        # a child firing less than this is untestable for survival

    def as_dict(self) -> dict:
        return asdict(self)


METRIC_SETTINGS = MetricSettings(
    probe_min_pos=50,
    probe_neg_ratio=4,
    probe_max_tokens=20_000,
    probe_min_neg=10,
    probe_steps=300,
    probe_lr=0.05,
    freq_high_mass=0.50,
    freq_mid_mass=0.40,
    n_freq_buckets=3,
    freq_min_fire_low=5,
)

CONSTANT_SETS: dict[str, GateConstants] = {s.name: s for s in (GEMMA_MATRYOSHKA, SYNTHETIC_TOYS)}
