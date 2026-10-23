"""Core metrics: each one tests the candidate edge (parent p, child c) itself.

    containment    is c inside p, and not the reverse?           (defines the candidate set)
    independence   is the co-firing above what base rates give?  (PMI)
    frequency      does the containment hold without the frequent tokens?
    contribution   does p carry reconstruction mass on c's tokens?
    geometry       does p's decoder point toward c's concept?

Everything here is a pure function over cached statistics, same as `metrics/`.
"""

from .containment import (
    candidate_edges,
    containment_asymmetry,
    containment_ratios,
    in_block_containment,
)
from .contribution import contribution_gains, per_token_ablation_gain
from .frequency import frequency_buckets, frequency_survival, local_frequency_buckets
from .geometry import (
    child_probe,
    decoder_cosine,
    probe_negative_parent_share,
    refinement_rank,
    refinement_scores,
)
from .independence import pmi_scores

__all__ = [
    "containment_ratios",
    "candidate_edges",
    "containment_asymmetry",
    "in_block_containment",
    "pmi_scores",
    "frequency_buckets",
    "local_frequency_buckets",
    "frequency_survival",
    "contribution_gains",
    "per_token_ablation_gain",
    "child_probe",
    "refinement_rank",
    "refinement_scores",
    "probe_negative_parent_share",
    "decoder_cosine",
]
