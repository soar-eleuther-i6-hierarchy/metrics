"""`ToyConfig`, the settings for one synthetic toy world, and the named configs in `CONFIGS`."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ToyConfig:
    name: str
    seed: int = 0

    # --- activation space ---
    D: int = 128                     # activation dimension the SAE sees
    noise_sigma: float = 0.05        # std of the Gaussian noise added to every activation (> 0)
    max_unrelated_cos: float = 0.12  # target cap on |cos| between unrelated residual directions

    # --- tree shape ---
    n_roots: int = 6                 # independent trees in the forest
    branching: int = 3               # children per parent
    depth: int = 3                   # levels below each root
    exclusive_siblings: bool = True  # siblings split the parent's tokens instead of overlapping
    randomize_structure: bool = False  # draw a seed-varied backbone instead of the fixed lattice

    # --- firing ---
    root_p: float = 0.18             # firing rate of a backbone root
    child_p_edge: float = 0.32       # P(child | parent); with exclusive siblings keep <= 1/branching
    eps_p: float = 0.02              # margin: child_p_edge <= 1 - eps_p

    # --- geometry ---
    alpha: float = 0.48              # weight of the parent's residual direction in an overlapping child's direction
    alpha_zero_every: int = 4        # every n-th edge gets alpha = 0 (orthogonal); 0 makes every edge overlap
    eps_alpha: float = 0.02          # margin: alpha <= 1 - eps_alpha, keeps Lam invertible

    # --- strength ---
    strength_spread: float = 0.35    # sd / mean of active strengths, must be in (0, 0.5)
    E0: float = 1.0                  # strength scale: mean squared active strength is E0

    # --- corpus (matters only for the frequency and topical confounds) ---
    vocab: int = 5000                # token-id vocabulary size
    doc_len: int = 128               # tokens per document
    Z: int = 8                       # number of document topics
    zipf_s: float = 1.05             # Zipf exponent of token frequencies

    # --- confounds (built only when confounds=True) ---
    confounds: bool = False          # master switch for every confound family below
    n_superparent: int = 3           # dense childless features at superparent_p (the base-rate confound)
    superparent_p: float = 0.85      # firing rate of superparents and dense true parents
    n_dense_parents: int = 0         # dense true parents at superparent_p, each with orthogonal children
    children_per_dense_parent: int = 1  # children per dense parent; they split the parent's tokens
    n_broad_parent: int = 1          # wide true parents, the superparent's foil
    broad_children: int = 5          # children under each broad parent
    broad_alpha: float = 0.48        # alpha for a broad parent's children
    n_token_bound_pairs: int = 8     # feature pairs that co-fire only through one shared token-id set
    n_topical_pairs: int = 12        # feature pairs lifted by a shared topic, round-robin over Z
    kappa: float = 7.2               # topic-modulation strength of the topical pairs
    n_bind_ids: int = 2              # top-frequency token ids the token-bound pairs fire on
    n_token_groups: int = 0          # token groups: a container and members firing only on the group's ids
    n_token_group_ids: int = 24      # token groups split ids 1..n_token_group_ids
    token_group_members: int = 3     # members per token group
    token_container_rate: float = 0.9  # P(container fires | token in its group)
    token_member_rate: float = 0.32  # P(member fires | token in its group)
    n_topic_registers_per_topic: int = 0  # 0 or 1; 1 builds one register + topic_members per topic
    topic_members: int = 2           # topic-locked members per topic
    topic_register_rate: float = 0.9   # P(register fires | token's document has its topic)
    topic_member_rate: float = 0.32  # P(member fires | token's document has its topic)


# --- seed-varied backbone (read only when randomize_structure=True) ---
# Offset added to the structure RNG seed; see tree.build_tree.
STRUCTURE_SEED_OFFSET: int = 9973
# A randomized draw is redrawn until it clears these floors.
F_MIN: int = 120                     # minimum feature count
MIN_PAIRS_PER_CLASS: int = 5         # minimum ordered pairs per guarded class
STRUCTURE_MAX_ATTEMPTS: int = 64     # redraws before build_tree raises
# Classes held to MIN_PAIRS_PER_CLASS.
GUARDED_STRUCTURE_CLASSES: tuple[str, ...] = ("hierarchy_overlap", "hierarchy_orthogonal", "sibling")


def replace(cfg: ToyConfig, **kw) -> ToyConfig:
    """dataclasses.replace, re-exported so callers need not import dataclasses."""
    return dataclasses.replace(cfg, **kw)


def backbone_config() -> ToyConfig:
    """Backbone tree only (hierarchy edges, siblings, transitive pairs), no confounds."""
    return ToyConfig(name="backbone", confounds=False)


def full_config() -> ToyConfig:
    """Backbone plus superparents, a broad parent, token-bound pairs and topical pairs."""
    return ToyConfig(name="full", confounds=True)


def _hierarchy_config(name: str, alpha_zero_every: int) -> ToyConfig:
    """120 root-child edges and nothing else: no siblings, transitive pairs or confounds."""
    return ToyConfig(name=name, n_roots=120, branching=1, depth=1,
                     alpha_zero_every=alpha_zero_every, confounds=False)


def hierarchy_overlap_config() -> ToyConfig:
    """Hierarchy world with every child's direction overlapping its parent's (alpha > 0)."""
    return _hierarchy_config("hierarchy_overlap", alpha_zero_every=0)


def hierarchy_orthogonal_config() -> ToyConfig:
    """`hierarchy_overlap` with alpha = 0 on every edge.

    Children fire nested in their parents but share no direction with them. It fires exactly as
    `hierarchy_overlap`, since the sampler never reads alpha.
    """
    return _hierarchy_config("hierarchy_orthogonal", alpha_zero_every=1)


def _confound_backbone(**kw) -> ToyConfig:
    """Base for the single-confound toys: 120 unrelated roots, no tree, every confound family off."""
    return ToyConfig(name="_confound", n_roots=120, branching=1, depth=0, confounds=True,
                     n_superparent=0, n_broad_parent=0, broad_children=0,
                     n_token_bound_pairs=0, n_topical_pairs=0, **kw)


def dense_config() -> ToyConfig:
    """Dense world: 6 superparents and 12 dense true parents, all firing at superparent_p.

    Each dense parent has two children, orthogonal to it (alpha = 0), that never fire together.
    """
    return replace(_confound_backbone(), name="dense", n_superparent=6,
                   n_dense_parents=12, children_per_dense_parent=2)


def frequency_config() -> ToyConfig:
    """Frequency world: 24 token groups of one id each (ids 1-24), each a container (rate 0.9) and 2 members (rate 0.32).

    All fire only on the group's token id, independently given the token, so
    P(container | member) = 0.9. Ids 1-24 stay inside the most common half of a sampled corpus.
    """
    return replace(_confound_backbone(), name="frequency", n_token_groups=24,
                   token_group_members=2)


def topical_config() -> ToyConfig:
    """Topical world: 24 topics, each a register (rate 0.9) and 2 members (rate 0.32).

    All fire only in that topic's documents, independently given the topic, so
    P(register | member) = 0.9.
    """
    return replace(_confound_backbone(), name="topical", n_topic_registers_per_topic=1, Z=24)


CONFIGS = {
    "backbone": backbone_config,
    "full": full_config,
    "hierarchy_overlap": hierarchy_overlap_config,
    "hierarchy_orthogonal": hierarchy_orthogonal_config,
    "dense": dense_config,
    "frequency": frequency_config,
    "topical": topical_config,
}
