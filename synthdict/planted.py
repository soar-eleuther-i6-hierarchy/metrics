"""The planted feature->latent correspondence — an INPUT to the synthetic path, never an inference.

Matching exists to solve an INVERSE problem: someone else built the dictionary, so which latent
corresponds to which feature has to be inferred (activation correlation, a Hungarian assignment,
a rho threshold). Synthesis has no inverse problem — we build the dictionary, so the
correspondence is known at construction time and simply recorded here.

Two things live in this record, and keeping them apart is the point:

  the MAP        feature f -> the latent ids carrying it. Identity for absorption. Splitting
                 gives a feature several shards; hedging leaves a child with none; composition
                 gives two features a shared combination latent after their own ones.
  the READOUT    when a feature lives in several latents, WHICH single column the metrics score.
                 That is a measurement decision, declared per run, never inferred — which is
                 what keeps this matcher-free rather than a matcher by another name.

The four policies, and what each one measures:

  identity         one latent per feature; a multi-latent feature is REFUSED rather than
                   silently reduced.
  strongest_shard  the feature is read on its strongest DECLARED shard — what a one-to-one
                   pipeline effectively sees once a feature has been split.
  own              a composed feature is read on its own latent only, ignoring the
                   combination latent it shares.
  union            the feature IS its whole latent group: activations sum, directions and
                   reconstruction rows are averaged by activation mass. A shared combination
                   latent counts in both features' groups.

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
READOUTS = ("identity", "strongest_shard", "own", "union")


@dataclass(frozen=True)
class PlantedMap:
    """`feature_to_latents[f]` = the latent ids carrying true feature f, in the dictionary the
    corruption built. `n_latents` is that dictionary's width (== F only while the map is 1-1).

    `composition` declares each shared latent as (feature_a, feature_b, latent). A latent may be
    claimed by two features only through such an entry, and it must follow both features' own
    latent, so `lats[0]` is always a feature's own latent or strongest shard.
    """

    feature_to_latents: tuple[tuple[int, ...], ...]
    readout: str
    n_latents: int
    composition: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.readout not in READOUTS:
            raise ValueError(
                f"unknown readout {self.readout!r}; registered policies are {READOUTS}")
        if self.n_latents < 0:
            raise ValueError(f"n_latents must be non-negative, got {self.n_latents}")
        claims: dict[int, set[int]] = {}
        for f, lats in enumerate(self.feature_to_latents):
            for j in lats:
                if not (0 <= int(j) < self.n_latents):
                    raise ValueError(
                        f"feature {f} names latent {j}, outside a dictionary of "
                        f"{self.n_latents}; the planted map must index the dictionary it "
                        f"was built with")
                claims.setdefault(int(j), set()).add(f)
        declared = {int(lat): {int(a), int(b)} for a, b, lat in self.composition}
        if len(declared) != len(self.composition):
            raise ValueError("a composition latent is declared twice")
        # A latent shared by two features scores both on one column; undeclared, every metric
        # stays finite and both are wrong.
        for j, fs in claims.items():
            if len(fs) > 1 and declared.get(j) != fs:
                raise ValueError(
                    f"latent {j} is claimed by features {sorted(fs)} without a matching "
                    f"composition entry; a shared latent must be declared")
        for lat, fs in declared.items():
            if claims.get(lat) != fs or len(fs) != 2:
                raise ValueError(
                    f"composition entry for latent {lat} names features {sorted(fs)}, but the "
                    f"map gives it to {sorted(claims.get(lat, set()))}")
            for f in fs:
                if self.feature_to_latents[f][0] == lat:
                    raise ValueError(
                        f"feature {f} lists combination latent {lat} first; its own latent "
                        f"must come first, or the 'own' readout would read the combination")

    # --------------------------------------------------------------------
    @classmethod
    def identity(cls, F: int, readout: str = "identity") -> "PlantedMap":
        """One latent per feature, latent id == feature id — every round-1 damage."""
        return cls(feature_to_latents=tuple((f,) for f in range(int(F))),
                   readout=readout, n_latents=int(F))

    @property
    def F(self) -> int:
        return len(self.feature_to_latents)

    def combination_latents(self) -> set[int]:
        return {int(lat) for _, _, lat in self.composition}

    def _single_latent(self, f: int) -> int:
        """The one latent a single-column readout scores feature f on, or a refusal."""
        lats = self.feature_to_latents[f]
        comb = self.combination_latents()
        if self.readout == "identity" and len(lats) != 1:
            raise ValueError(
                f"feature {f} is carried by {len(lats)} latents, which the 'identity' readout "
                f"cannot reduce to one column; declare a multi-latent readout policy instead "
                f"of letting the scorer pick")
        if self.readout == "own" and any(j not in comb for j in lats[1:]):
            raise ValueError(
                f"feature {f} is split across {len(lats)} latents; the 'own' readout reads a "
                f"composed feature's own latent and has no meaning for shards")
        if self.readout == "strongest_shard" and any(j in comb for j in lats):
            raise ValueError(
                f"feature {f} holds a combination latent, which is not a shard and has no "
                f"declared share; 'strongest_shard' is undefined here")
        return int(lats[0])

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
        return torch.tensor([self._single_latent(f) for f in self.feats()], dtype=torch.long)

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
        for f in self.feats():
            out[f] = self._single_latent(f)
        return out

    def representative_lookup(self) -> torch.Tensor:
        """`[F]` long, feature-indexed: ONE latent standing in for each feature, or -1.

        The feature's own latent or strongest declared shard, defined under every readout. This is an
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

        Under `union` the group SUMS. Planted shards fire disjointly, and a composed feature's
        own latent is off wherever its combination latent fires, so at most one term is nonzero
        per token (0 + x == x in IEEE754): the union anchor is bit-exact in the firing channel.
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

    def _live_weights(self, lats: tuple[int, ...], mass: torch.Tensor
                      ) -> tuple[list[int], torch.Tensor]:
        """The group's latents that carry activation mass, with that mass as weights.

        A latent with zero mass on the draw contributes nothing to the feature's reconstruction,
        so it is dropped rather than averaged in: a combination latent that never fires (pi = 0)
        must leave its features' rows untouched. A group with no mass at all falls back to
        equal weights over every latent.
        """
        live = [j for j in lats if float(mass[j]) > 0.0]
        if not live:
            return list(lats), torch.ones(len(lats), dtype=mass.dtype, device=mass.device)
        return live, mass[live]

    def reduce_unit(self, unit_L: torch.Tensor, acts_L: torch.Tensor) -> torch.Tensor:
        """`[L, D]` -> `[R, D]` geometry-channel rows: under `union`, the unit-normalized mean
        weighted by each latent's activation mass on the draw (`acts_L` [n, L]).

        Mass weighting is the direction of the feature's total reconstruction contribution.
        For split shards, which share one direction, it equals the plain mean; for a composed
        feature it weights the combination row by how much it actually fires.

        A group with ONE live latent is that row itself, not a renormalized copy (renormalizing
        an already-unit row moves it by ~6e-17, so the seam would be inert only to a tolerance).

        REQUIRES the live latents to point roughly the same way. `signed_normalized_decoder`
        orients each row on its own tokens, so a starved shard can be oriented against its
        siblings; a cancelling group would normalize to a meaningless direction, so it fails.
        """
        self._check_latent_width(unit_L.shape[0])
        self._check_latent_width(acts_L.shape[1])
        if self.readout != "union":
            return unit_L[self.columns()]
        mass = acts_L.sum(dim=0)
        rows = []
        for lats in self.groups():
            live, w = self._live_weights(lats, mass)
            if len(live) == 1:
                rows.append(unit_L[live[0]])
                continue
            m = (unit_L[live] * w[:, None]).sum(dim=0) / w.sum()
            n = float(m.norm())
            if n < UNION_MIN_MEAN_NORM:
                raise ValueError(
                    f"the {len(live)} latents reduced here do not point the same way (their "
                    f"weighted mean has norm {n:.3g}); the 'union' readout averages directions, "
                    f"so a cancelling group would score a meaningless direction with every "
                    f"metric still finite")
            rows.append(m / m.norm())
        return torch.stack(rows)

    def reduce_raw(self, raw_L: torch.Tensor, acts_L: torch.Tensor) -> torch.Tensor:
        """`[L, D]` -> `[R, D]` reconstruction-channel rows: under `union`, the activation-mass
        weighted mean, NOT renormalized.

        (sum of a group's acts) x (mass-weighted mean row) has the same total reconstruction
        over the draw as the group's latents; renormalizing would rescale every reconstruction
        contribution by 1/||mean||.
        """
        self._check_latent_width(raw_L.shape[0])
        self._check_latent_width(acts_L.shape[1])
        if self.readout != "union":
            return raw_L[self.columns()]
        mass = acts_L.sum(dim=0)
        rows = []
        for lats in self.groups():
            live, w = self._live_weights(lats, mass)
            if len(live) == 1:
                rows.append(raw_L[live[0]])
                continue
            rows.append((raw_L[live] * w[:, None]).sum(dim=0) / w.sum())
        return torch.stack(rows)

    def sha256(self) -> str:
        """Content hash of the correspondence, stamped into every artifact so two runs whose
        maps differ are distinguishable on disk."""
        payload = {"map": [list(l) for l in self.feature_to_latents],
                   "readout": self.readout, "n_latents": self.n_latents}
        if self.composition:
            payload["composition"] = [list(e) for e in self.composition]
        payload = json.dumps(payload, sort_keys=True)
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
