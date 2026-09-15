"""
Settings for building a synthetic toy world.

`ToyConfig` is the single knob panel: tree shape, firing rates, geometry, strength, and an
optional set of confounds. Two ready-made recipes at the bottom: backbone (clean tree only)
and full (backbone plus realistic confounds).

Each planted property mirrors a real SAE phenomenon:
  is_a / sibling / transitive: hierarchical & categorical concept geometry -- Park et al. 2024
  firing_only: the orthogonal-geometry null / hard negative for is_a
  superparent: always-on high-base-rate distractor -- foil for coverage/out-degree metrics
  dense_parent: a dense TRUE parent at the superparent's density -- dense latents, Sun et al. 2025
  broad_parent: a genuine wide parent, i.e. feature-splitting family -- Bricken et al. 2023
  token_bound: single-token / spurious co-activation features -- Bricken et al. 2023
  token groups: token-caused containment (a container and members on one id set) -- Dooms & Wilhelm 2025
  topical: co-occurring feature clusters ("lobes") -- Li et al. 2024
  topic registers: topic-caused containment (context-binding latents) -- Sun et al. 2025
Dictionary-side properties (absorption, splitting, merging) come from SAE training, not here.
The base activation model follows Elhage et al. 2022.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ToyConfig:
    name: str
    seed: int = 0

    # --- activation space ---------------------------------------------------
    D: int = 128                     # dimension of the activation vectors the SAE sees
    noise_sigma: float = 0.05        # std of the Gaussian noise added to every activation (> 0)
    max_unrelated_cos: float = 0.12  # cap on cosine between unrelated feature directions

    # --- tree shape ---------------------------------------------------------
    n_roots: int = 6                 # independent trees in the forest
    branching: int = 3               # children per parent
    depth: int = 3                   # levels below each root (number of blocks = depth + 1)
    exclusive_siblings: bool = True  # siblings split the parent's tokens instead of overlapping
    randomize_structure: bool = False  # opt-in: draw a seed-varied backbone instead of the fixed lattice; confound counts stay seed-invariant except superparent, which scales with F

    # --- firing -------------------------------------------------------------
    root_p: float = 0.18             # a root fires on this fraction of tokens
    child_p_edge: float = 0.32       # P(child fires | parent fires); with exclusive siblings must stay <= 1/branching
    eps_p: float = 0.02              # keep child_p_edge <= 1 - eps so a parent can fire without a given child

    # --- composition (geometry) ---------------------------------------------
    alpha: float = 0.48              # cosine of a child's direction onto its parent (is_a overlap); siblings share alpha^2 of the parent
    alpha_zero_every: int = 4        # every n-th edge (across the forest) gets alpha = 0: the firing_only cell; 0 disables firing_only entirely (all edges is-a)
    eps_alpha: float = 0.02          # keep alpha <= 1 - eps so the change-of-basis matrix stays invertible

    # --- strength -----------------------------------------------------------
    strength_spread: float = 0.35    # spread of firing strengths (sd / mean); < 0.5 keeps every strength positive
    K: int = 6                       # legacy sparsity hint only; actual k is derived from the tree's true L0 by world.choose_k.

    # --- strength scale -----------------------------------------------------
    E0: float = 1.0                  # base strength scale; every feature's mean active strength is q * sqrt(E0), with no designed ladder

    # --- corpus (only exercised by the frequency / topical confounds) -------
    vocab: int = 5000                # token-id vocabulary size
    doc_len: int = 128               # tokens per document
    freq_high_mass: float = 0.50     # corpus-mass cut point for the high frequency token bucket
    freq_mid_mass: float = 0.40      # cut point for the mid frequency bucket
    Z: int = 8                       # number of topics (used by the topical confound)
    zipf_s: float = 1.05             # Zipf exponent controlling how skewed token frequencies are

    # --- confounds (enabled by confounds=True; off in backbone) ---
    confounds: bool = False          # master switch for all the distractor confounds below
    n_superparent: int = 3           # always-on wide parents -- the base-rate confound; canonical count, balances pair-mass against L0 inflation
    superparent_p: float = 0.85      # firing rate of superparents and of dense true parents (one shared density)
    n_dense_parents: int = 0         # dense TRUE parents at superparent_p, each with one orthogonal child at child_p_edge
    n_broad_parent: int = 1          # genuine wide parents -- the superparent's honest foil
    broad_children: int = 5          # children under each broad parent
    broad_alpha: float = 0.48        # is_a overlap for a broad parent's children; kept equal to `alpha` to stay above the unrelated ceiling
    n_token_bound_pairs: int = 8     # token-bound pairs co-firing via one shared token-id set; 8 pairs = 16 features = 240 ordered frequency pairs
    n_topical_pairs: int = 12        # feature pairs lifted by a shared topic, round-robin over Z; 12 pairs = 24 features across 8 topic groups
    kappa: float = 7.2               # topic-modulation strength; higher lifts same-topic co-firing (bounded so per-topic rates stay in [0, 1])
    n_bind_ids: int = 2              # top-frequency token ids shared by token-bound features; must stay under the id set's Zipf mass
    n_token_groups: int = 0          # token groups: one container + members, all firing only on the group's token ids
    token_ids_per_group: int = 5     # average ids per group; ids 1..n_token_groups*this are split into equal design-Zipf mass groups of uneven size
    token_group_members: int = 3     # members per token group
    token_container_rate: float = 0.9  # P(container fires | token in its group)
    token_member_rate: float = 0.32  # P(member fires | token in its group)
    n_topic_registers_per_topic: int = 0  # 0 or 1; 1 builds one register + topic_members per topic
    topic_members: int = 2           # topic-locked members per topic
    topic_register_rate: float = 0.9   # P(register fires | token's document has its topic)
    topic_member_rate: float = 0.32  # P(member fires | token's document has its topic)


# The hierarchy-candidate cut P(parent|child) >= tau; mirrors scoring.core.registry CONSTANTS["edge_tau"].
EDGE_TAU_REFERENCE: float = 0.5


# --- seed-varied backbone knobs (only read when randomize_structure=True) -------------
# Structure RNG is seeded from cfg.seed + offset + attempt, so draws are reproducible and never collide with the geometry/sampling streams.
STRUCTURE_SEED_OFFSET: int = 9973
# A randomized draw is retried until it clears these floors, so it can't starve a scored class or shrink the dictionary:
F_MIN: int = 120                     # minimum total feature count
MIN_PAIRS_PER_CLASS: int = 5         # minimum ordered pairs per guarded scored class
STRUCTURE_MAX_ATTEMPTS: int = 64     # deterministic retries before giving up (then raises)
# Backbone-derived classes whose pair count moves with structure and must stay well-populated.
GUARDED_STRUCTURE_CLASSES: tuple[str, ...] = ("is_a", "firing_only", "sibling")


def replace(cfg: ToyConfig, **kw) -> ToyConfig:
    """dataclasses.replace, re-exported so callers need not import dataclasses."""
    return dataclasses.replace(cfg, **kw)


def backbone_config() -> ToyConfig:
    """Clean stratum: the backbone tree only (is_a, firing_only, sibling, transitive), no confounds."""
    return ToyConfig(name="backbone", confounds=False)


def full_config() -> ToyConfig:
    """Backbone plus the full set of confounds -- the main validation world."""
    return ToyConfig(name="full", confounds=True)


def only_isa_config() -> ToyConfig:
    """Pure is-a world for single-property metric characterization (Stage-1 oracle read).

    Isolation is by tree shape, not by injecting negatives: branching=1 removes siblings,
    depth=1 removes transitive, alpha_zero_every=0 disables the firing_only edge (so every
    edge is a real alpha>0 is-a edge), and confounds=False removes superparent/frequency/
    topical. The only pair classes left are is_a, its reversed flip (child->parent, the
    asymmetry test), and the unrelated null. n_roots=120 gives F=240 (matching the full
    world's backbone size) and ~120 is-a edges, well clear of the N>=10 reporting floor.
    """
    return ToyConfig(name="only_isa", n_roots=120, branching=1, depth=1,
                     alpha_zero_every=0, confounds=False)


def only_firing_config() -> ToyConfig:
    """Pure firing_only world: co-firing with NO geometry (the is_a null / hard negative).

    Identical shape to `only_isa` (branching=1 removes siblings, depth=1 removes transitive,
    confounds=False removes the distractors), but `alpha_zero_every=1` sets alpha=0 on EVERY edge,
    so a child's direction is orthogonal to its parent (cos=0) while still firing nested inside it
    (child => parent). The only pair classes are firing_only, its reversed flip, and the unrelated
    null. n_roots=120 gives F=240 and ~120 firing_only edges, well clear of the N>=10 floor. Paired
    against `only_isa`: same firing structure, geometry switched off — so any is_a detector that
    scores high here (rather than only on `only_isa`) is responding to co-firing, not to hierarchy.
    """
    return ToyConfig(name="only_firing", n_roots=120, branching=1, depth=1,
                     alpha_zero_every=1, confounds=False)


def _confound_backbone(**kw) -> ToyConfig:
    """Shared base for the single-confound toys: `depth=0` (roots only, so no is_a/firing_only
    backbone tree), `confounds=True`, and EVERY confound family zeroed. Each toy then enables
    exactly one family. 120 independent roots at root_p supply the `unrelated` null population.
    """
    return ToyConfig(name="_confound", n_roots=120, branching=1, depth=0, confounds=True,
                     n_superparent=0, n_broad_parent=0, broad_children=0,
                     n_token_bound_pairs=0, n_topical_pairs=0, **kw)


def only_superparent_config() -> ToyConfig:
    """Dense world: dense TRUE parents next to dense unrelated superparents at the same density.

    6 childless superparents and 12 dense true parents all fire at `superparent_p=0.85`; each
    dense parent has one child (P(child|parent)=0.32, alpha=0, so exactly orthogonal). The true
    edges have P(parent|child)=1; every dense feature paired with a non-relative has coverage
    0.85 from base rate alone and carries the `superparent` label in both orderings. F=150:
    superparent=5034 pairs, firing_only=12, reversed=12, unrelated=17292; true L0 ~ 40.2.
    """
    return replace(_confound_backbone(), name="only_superparent", n_superparent=6,
                   n_dense_parents=12)


def only_frequency_config() -> ToyConfig:
    """Frequency world: token-caused containment against the unrelated null.

    5 token groups split design-Zipf ids 1..25 into equal mass (~0.069 each, id 0 excluded, all
    inside the top-50%-mass bucket). Per group a container fires on the group's tokens with
    rate 0.9 and 3 members with rate 0.32, independently given the token, so
    P(container|member)=0.9 and P(member|container)=0.32. F=140: frequency=60 pairs (within a
    group), unrelated=19400.
    """
    return replace(_confound_backbone(), name="only_frequency", n_token_groups=5)


def only_topical_config() -> ToyConfig:
    """Topical world: topic-caused containment against the unrelated null.

    For each of Z=8 topics a register fires on tokens of that topic's documents with rate 0.9
    and 2 members with rate 0.32, independently given the topic, so P(register|member)=0.9 and
    P(member|register)=0.32. No detector sees the topic. F=144: topical=48 pairs (within a
    topic), unrelated=20544.
    """
    return replace(_confound_backbone(), name="only_topical", n_topic_registers_per_topic=1)


CONFIGS = {
    "backbone": backbone_config,
    "full": full_config,
    "only_isa": only_isa_config,
    "only_firing": only_firing_config,
    "only_superparent": only_superparent_config,
    "only_frequency": only_frequency_config,
    "only_topical": only_topical_config,
}
