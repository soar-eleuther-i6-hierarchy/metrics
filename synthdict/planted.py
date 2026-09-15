"""The planted feature->latent correspondence — an INPUT to the synthetic path, never an inference.

Matching exists to solve an INVERSE problem: someone else built the dictionary, so which latent
corresponds to which feature has to be inferred (activation correlation, a Hungarian assignment,
a rho threshold). Synthesis has no inverse problem — we build the dictionary, so the
correspondence is known at construction time and simply recorded here.

Two things live in this record, and keeping them apart is the point:

  the MAP        feature f -> the latent ids carrying it. Identity for every damage that
                 preserves one-latent-per-feature (absorption, hedging, noise). Round-2 damages
                 that change the latent count (splitting, merging, missing) are the only reason
                 an entry is ever not a single id.
  the READOUT    when a feature lives in several latents, WHICH single column the metrics score.
                 That is a measurement decision, declared per run, never inferred — which is
                 what keeps this matcher-free rather than a matcher by another name.

The three policies, and what each one measures:

  identity         one latent per feature; a multi-latent feature is REFUSED rather than
                   silently reduced. Every round-1 damage.
  strongest_shard  the feature is read on its strongest DECLARED shard — what a one-to-one
                   pipeline effectively sees once a feature has been split.
  union            the feature IS its whole shard set: activations sum, directions average.
                   What the dictionary actually contains.

For a split cell the gap between `strongest_shard` and `union` is the measurement of what
splitting costs the metrics, with no matcher noise in it.

The ordering of `feature_to_latents[f]` is what keeps "strongest" declarative: shards are listed
by DESCENDING PLANTED SHARE, so `lats[0]` is strongest by construction and nothing here inspects
an activation to decide. At an equal-share split (`skew = 0`) that ordering is a tie-break on
lowest latent id — recorded here rather than left to be discovered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

# Floor on ||mean of a union group's unit rows||. Shards share a direction by construction, so
# this sits at ~1.0 in every reachable case; anything near 0 means the group is cancelling and
# the averaged direction is meaningless. Loose on purpose — it is a sanity floor, not a dial.
UNION_MIN_MEAN_NORM = 0.5

# The registered readout policies. Membership is checked at construction, so a typo fails at the
# call site rather than silently selecting a default.
READOUTS = ("identity", "strongest_shard", "union")


@dataclass(frozen=True)
class PlantedMap:
    """`feature_to_latents[f]` = the latent ids carrying true feature f, in the dictionary the
    corruption built. `n_latents` is that dictionary's width (== F only while the map is 1-1)."""

    feature_to_latents: tuple[tuple[int, ...], ...]
    readout: str
    n_latents: int

    def __post_init__(self) -> None:
        if self.readout not in READOUTS:
            raise ValueError(
                f"unknown readout {self.readout!r}; registered policies are {READOUTS}")
        if self.n_latents < 0:
            raise ValueError(f"n_latents must be non-negative, got {self.n_latents}")
        seen: dict[int, int] = {}
        for f, lats in enumerate(self.feature_to_latents):
            for j in lats:
                if not (0 <= int(j) < self.n_latents):
                    raise ValueError(
                        f"feature {f} names latent {j}, outside a dictionary of "
                        f"{self.n_latents}; the planted map must index the dictionary it "
                        f"was built with")
                # A latent shared by two features is a MERGE, which needs its own readout
                # policy; silently allowing it here would score both features on one column
                # with every metric finite and both wrong.
                if int(j) in seen:
                    raise ValueError(
                        f"latent {j} is claimed by features {seen[int(j)]} and {f}; a shared "
                        f"latent is a merge and needs a readout policy that declares how it "
                        f"is scored, not an implicit one")
                seen[int(j)] = f

    # --------------------------------------------------------------------
    @classmethod
    def identity(cls, F: int, readout: str = "identity") -> "PlantedMap":
        """One latent per feature, latent id == feature id — every round-1 damage."""
        return cls(feature_to_latents=tuple((f,) for f in range(int(F))),
                   readout=readout, n_latents=int(F))

    @property
    def F(self) -> int:
        return len(self.feature_to_latents)

    def is_identity(self) -> bool:
        """Is latent id == feature id, with no row unclaimed?

        Strict on purpose. The `clean` activation model indexes `bundle.A` and `bundle.g` by
        FEATURE id, so a merely one-to-one map (a permutation, a narrower dictionary) would
        read one feature's magnitudes onto another feature's row with nothing to notice it.
        """
        return (self.n_latents == self.F
                and all(lats == (f,) for f, lats in enumerate(self.feature_to_latents)))

    def recovered(self) -> torch.Tensor:
        """`[F]` bool: does this feature have any latent at all.

        DECLARATIVE — a planted feature is present because we planted it, not because a matcher
        cleared a correlation threshold. Only a round-2 missing-latent damage empties an entry.
        Shape is fixed at `[F]` because `scoring.core.recovery.per_class_recovery` broadcasts it
        against `pair_labels`, and that stays feature-indexed even when the latent count does not.
        """
        return torch.tensor([len(lats) > 0 for lats in self.feature_to_latents],
                            dtype=torch.bool)

    def feats(self) -> list[int]:
        """True feature ids in the scored frame: every feature that has a latent."""
        return [f for f, lats in enumerate(self.feature_to_latents) if lats]

    def groups(self) -> tuple[tuple[int, ...], ...]:
        """The latent ids per SCORED feature, aligned with `feats()`.

        The one shared helper behind all three reductions. `feats()` skips features with no
        latent, so this must skip exactly the same ones or every reduction sits one position
        away from the feature it claims to describe.
        """
        return tuple(self.feature_to_latents[f] for f in self.feats())

    def _check_latent_width(self, width: int) -> None:
        if int(width) != self.n_latents:
            raise ValueError(
                f"expected a latent-wide array ({self.n_latents} latents), got {int(width)}; "
                f"feature-space arrays must be expanded through the map before reduction")

    def columns(self) -> torch.Tensor:
        """The latent column each scored feature is read on, aligned with `feats()`.

        Single-column readouts only. Under `identity` a feature carried by several latents has
        no single column and choosing one here (the first, the strongest) would be exactly the
        silent reduction this module exists to prevent; under `union` no single column exists
        at all, by definition of the policy.
        """
        if self.readout == "union":
            raise ValueError(
                "the 'union' readout scores a feature on its whole shard set, so it has no "
                "single column; reduce with reduce_acts/reduce_unit/reduce_raw instead")
        cols = []
        for f in self.feats():
            lats = self.feature_to_latents[f]
            if self.readout == "identity" and len(lats) != 1:
                raise ValueError(
                    f"feature {f} is carried by {len(lats)} latents, which the 'identity' "
                    f"readout cannot reduce to one column; declare a multi-latent readout "
                    f"policy instead of letting the scorer pick")
            cols.append(int(lats[0]))
        return torch.tensor(cols, dtype=torch.long)

    def feature_lookup(self) -> torch.Tensor:
        """`[F]` long, indexed by TRUE FEATURE id: the latent carrying feature f, or -1.

        Distinct from `columns()`, which is indexed by POSITION in the scored frame. Consumers
        that index by feature id — `scoring.trained.absorption.classify_dictionary` does
        (`match[c]` for a true child id) — must use this one. The two coincide only while every
        feature is present, which is exactly the case that hides the bug.

        Readout-keyed, like `columns()`: under `union` there is no single latent id for a
        feature, and returning `lats[0]` here would report a union run's numbers against one
        shard. Consumers that must name a latent anyway take `representative_lookup()` and
        stamp the approximation.
        """
        if self.readout == "union":
            raise ValueError(
                "the 'union' readout has no single latent per feature; use "
                "representative_lookup() and record the policy beside the output")
        out = torch.full((self.F,), -1, dtype=torch.long)
        for f, lats in enumerate(self.feature_to_latents):
            if not lats:
                continue
            # `> 1` rather than `!= 1`: the empty case already went to `continue` above.
            if self.readout == "identity" and len(lats) > 1:
                raise ValueError(
                    f"feature {f} is carried by {len(lats)} latents, which the 'identity' "
                    f"readout cannot reduce to one latent id")
            out[f] = int(lats[0])
        return out

    def representative_lookup(self) -> torch.Tensor:
        """`[F]` long, feature-indexed: ONE latent standing in for each feature, or -1.

        The strongest declared shard, defined under every readout including `union`. This is an
        APPROXIMATION wherever a feature has more than one shard, so every consumer records the
        policy beside its output (`census.py` stamps `census_latent_policy`) — an approximation
        that is visible on disk is a caveat; one that is not is a wrong number.
        """
        out = torch.full((self.F,), -1, dtype=torch.long)
        for f, lats in enumerate(self.feature_to_latents):
            if lats:
                out[f] = int(lats[0])
        return out

    # -- the three reductions: [.., L] -> [.., R], one per channel --------
    def reduce_acts(self, acts_L: torch.Tensor) -> torch.Tensor:
        """`[n, L]` -> `[n, R]` activations, one column per scored feature.

        Under `union` the shards SUM. Planted shards fire disjointly, so at most one term is
        nonzero per token and the sum recovers the unsplit magnitude exactly (0 + x == x in
        IEEE754) — which is what makes the union anchor bit-exact in the firing channel.
        """
        self._check_latent_width(acts_L.shape[1])
        if self.readout != "union":
            return acts_L[:, self.columns()]
        groups = self.groups()
        out = torch.zeros(acts_L.shape[0], len(groups),
                          dtype=acts_L.dtype, device=acts_L.device)
        for i, lats in enumerate(groups):
            out[:, i] = acts_L[:, list(lats)].sum(dim=1)
        return out

    def reduce_unit(self, unit_L: torch.Tensor) -> torch.Tensor:
        """`[L, D]` -> `[R, D]` geometry-channel rows: under `union`, the unit-normalized MEAN.

        A ONE-SHARD group is the row itself, not the normalized mean of one row. Normalizing
        exists to undo the averaging, and there is no averaging over a single element — while
        re-normalizing an already-unit row moves it by ~6e-17, which would leave the readout
        seam inert only to a tolerance instead of exactly (measured; it is what G1 caught).

        REQUIRES the shards of one feature to point the same way. They share a direction by
        construction, but `signed_normalized_decoder` orients each decoder row independently,
        on that shard's own tokens — so a shard starved to one or two tokens can be oriented
        against its siblings. Checked rather than assumed: cancelling rows average to nearly
        zero and normalize to a meaningless direction, and a zero row sends `G` to 0 for every
        pair touching that feature. That is an O(1) wrong number, not float noise, so it fails
        loudly instead.
        """
        self._check_latent_width(unit_L.shape[0])
        if self.readout != "union":
            return unit_L[self.columns()]
        rows = []
        for lats in self.groups():
            if len(lats) == 1:
                rows.append(unit_L[lats[0]])
                continue
            m = unit_L[list(lats)].mean(dim=0)
            n = float(m.norm())
            if n < UNION_MIN_MEAN_NORM:
                raise ValueError(
                    f"the {len(lats)} shards reduced here do not point the same way (their "
                    f"mean has norm {n:.3g}); the 'union' readout averages directions, so a "
                    f"cancelling group would score a meaningless direction with every metric "
                    f"still finite")
            rows.append(m / m.norm())
        return torch.stack(rows)

    def reduce_raw(self, raw_L: torch.Tensor) -> torch.Tensor:
        """`[L, D]` -> `[R, D]` reconstruction-channel rows: under `union`, a PLAIN mean.

        Not renormalized, unlike `reduce_unit`. With disjoint shard firing, (sum of acts) x
        (mean of rows) reproduces the unsplit reconstruction term exactly; renormalizing would
        rescale every reconstruction contribution by 1/||mean||.
        """
        self._check_latent_width(raw_L.shape[0])
        if self.readout != "union":
            return raw_L[self.columns()]
        return torch.stack([raw_L[list(lats)].mean(dim=0) for lats in self.groups()])

    def sha256(self) -> str:
        """Content hash of the correspondence, stamped into every artifact so two runs whose
        maps differ are distinguishable on disk."""
        payload = json.dumps({"map": [list(l) for l in self.feature_to_latents],
                              "readout": self.readout, "n_latents": self.n_latents},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def resolve_map(corruption, F: int, readout: str = "identity") -> PlantedMap:
    """The planted map for this dictionary: the corruption's own when it declares one (a
    round-2 damage that changes the latent count), else the identity map over `F` features."""
    declared = getattr(corruption, "planted_map", None) if corruption is not None else None
    if declared is not None:
        if declared.readout != readout:
            raise ValueError(
                f"the corruption declares readout {declared.readout!r} but the run requested "
                f"{readout!r}; the readout is a registered setting, not a per-call override")
        if declared.F != int(F):
            raise ValueError(
                f"the corruption declares a map over {declared.F} features but this world has "
                f"{F}; a map built for a different world would silently score a different "
                f"universe")
        return declared
    return PlantedMap.identity(F, readout=readout)
