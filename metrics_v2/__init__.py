"""metrics_v2: the metric library, arranged by the question each metric answers.

Two packages:

    structure/   core metrics. Each one tests the candidate edge itself:
                 is the child contained in the parent, above chance, without the
                 frequent tokens, with a causal contribution, in direction space.
    pathology/   supporting metrics. Each one reads the neighbourhood of an edge
                 (the parent's other children, the child's other parents, the
                 decoder geometry) to say which pathology produced it.

Decisions (gates, rules, grading, constant sets) stay in `metrics.rules`, which
both pipelines import. They are re-exported here unchanged.

Phase 1: every function here is the *same object* as its `metrics.*` original,
under a name that says what it measures. No default changed, no formula moved,
so every published number is reproducible from either package. The three new
functions (`decoder_cosine`, `absorption_signature`, `composition_signature`)
are used by no pipeline and are not calibrated; the README says so.

`OLD_TO_NEW` maps every old name to its new dotted name. `tests/test_metrics_v2.py`
asserts each entry resolves to the identical object.
"""

from __future__ import annotations

from metrics import rules  # noqa: F401  (decisions live there; not moved)

from . import pathology, structure  # noqa: F401

OLD_TO_NEW: dict[str, str] = {
    # structure/containment
    "coverage_legs": "structure.containment_ratios",
    "keep_edges": "structure.candidate_edges",
    "coverage_asymmetry": "structure.containment_asymmetry",
    "directed_coverage": "structure.in_block_containment",
    # structure/independence
    "independence_scores": "structure.pmi_scores",
    # structure/frequency
    "frequency_buckets": "structure.frequency_buckets",
    "local_frequency_buckets": "structure.local_frequency_buckets",
    "frequency_controlled_coverage": "structure.frequency_survival",
    # structure/contribution
    "edge_reconstruction_condition": "structure.contribution_gains",
    "per_token_ablation_gain": "structure.per_token_ablation_gain",
    # structure/geometry
    "train_probe": "structure.child_probe",
    "sres_rank_check": "structure.refinement_rank",
    "sres_scores": "structure.refinement_scores",
    "negative_parent_composition": "structure.probe_negative_parent_share",
    # pathology/splitting
    "sibling_redundancy": "pathology.sibling_overlap",
    "parent_conditioned_redundancy": "pathology.sibling_overlap_within_parent",
    "duplicate_pairs": "pathology.coextensive_pairs",
    # pathology/parent_coverage
    "share_energy": "pathology.child_energy_share",
    "r_supp": "pathology.parent_support_covered",
    "r_mass": "pathology.parent_energy_covered",
    "joint_child_coverage_exact": "pathology.parent_support_covered_exact",
    # pathology/degree
    "degree_stats": "pathology.degree_summary",
    "kept_outdegree": "pathology.parent_outdegree",
    "either_endpoint_outdegree": "pathology.pair_max_outdegree",
    "find_superparents": "pathology.dense_parents",
}

# Dropped on purpose, not renamed: `joint_child_coverage_upper` (min(1, sum F))
# double-counts co-firing children and saturates near 1; the exact form replaces it.
DROPPED: tuple[str, ...] = ("joint_child_coverage_upper",)

NEW: tuple[str, ...] = (
    "structure.decoder_cosine",
    "pathology.absorption_signature",
    "pathology.composition_signature",
    "pathology.direction_duplicates",
)

__all__ = ["structure", "pathology", "rules", "OLD_TO_NEW", "DROPPED", "NEW"]
