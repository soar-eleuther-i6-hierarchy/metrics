"""The planted feature->latent map: an input to the synthetic path, never inferred by a matcher.

The map lists the latents carrying each feature; the readout picks which single column the
metrics score when there are several:

  identity         one latent per feature; a multi-latent feature is refused
  strongest_shard  the strongest declared shard, what a one-to-one pipeline sees after a split
  own              a composed feature's own latent, ignoring the shared combination latent
  union            the whole group: activations sum, rows are activation-mass-weighted means

Shards are listed by descending planted share, so `lats[0]` is the strongest by construction
(lowest latent id on a tie).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

# Floor on the norm of a union group's mean unit row: a loose check that the group is not
# cancelling, not a dial.
UNION_MIN_MEAN_NORM = 0.5

# PlantedMap checks membership, so a readout typo fails at construction
READOUTS = ("identity", "strongest_shard", "own", "union")


@dataclass(frozen=True)
class PlantedMap:
    """`feature_to_latents[f]` = the latent ids carrying feature f; `n_latents` is the width.

    `composition` declares each shared latent as (feature_a, feature_b, latent), listed after
    both features' own latents, so `lats[0]` is always a feature's own latent or strongest shard.
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

    @classmethod
    def identity(cls, F: int, readout: str = "identity") -> "PlantedMap":
        """One latent per feature, latent id == feature id (absorption, and every no-op dial)."""
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
        """`[F]` bool: whether feature f has any latent, by declaration; only hedging empties one.

        Feature-indexed because `per_class_recovery` broadcasts it against `pair_labels`.
        """
        return torch.tensor([len(lats) > 0 for lats in self.feature_to_latents],
                            dtype=torch.bool)

    def feats(self) -> list[int]:
        """True feature ids in the scored frame: every feature that has a latent."""
        return [f for f, lats in enumerate(self.feature_to_latents) if lats]

    def groups(self) -> tuple[tuple[int, ...], ...]:
        """The latent ids per scored feature, aligned with `feats()`: it must skip exactly the
        features `feats()` skips, or every reduction is off by a position."""
        return tuple(self.feature_to_latents[f] for f in self.feats())

    def _check_latent_width(self, width: int) -> None:
        if int(width) != self.n_latents:
            raise ValueError(
                f"expected a latent-wide array ({self.n_latents} latents), got {int(width)}; "
                f"feature-space arrays must be expanded through the map before reduction")

    def columns(self) -> torch.Tensor:
        """The latent column each scored feature is read on, aligned with `feats()`.

        Single-column readouts only: `identity` refuses a multi-latent feature rather than pick
        one, and `union` has no single column.
        """
        if self.readout == "union":
            raise ValueError(
                "the 'union' readout scores a feature on its whole shard set, so it has no "
                "single column; reduce with reduce_acts/reduce_unit/reduce_raw instead")
        return torch.tensor([self._single_latent(f) for f in self.feats()], dtype=torch.long)

    def feature_lookup(self) -> torch.Tensor:
        """`[F]` long, indexed by feature id: the latent carrying feature f, or -1.

        Not `columns()`, which is indexed by position in the scored frame; the two agree only
        while every feature is present. Refuses `union`: use `representative_lookup()` there.
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
        """`[F]` long, feature-indexed: the own latent or strongest shard, or -1, under any readout.

        An approximation for multi-latent features, so consumers record the policy beside their
        output (`census.py` stamps `census_latent_policy`).
        """
        out = torch.full((self.F,), -1, dtype=torch.long)
        for f, lats in enumerate(self.feature_to_latents):
            if lats:
                out[f] = int(lats[0])
        return out

    # --- the three reductions: [.., L] -> [.., R], one per channel ---
    def reduce_acts(self, acts_L: torch.Tensor) -> torch.Tensor:
        """`[n, L]` -> `[n, R]` activations, one column per scored feature.

        Under `union` the group sums. At most one term is nonzero per token, so the union
        readout is bit-exact in the firing channel.
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
        """The group's latents with activation mass, weighted by it; equal weights if none has any.

        Dropping zero-mass latents keeps a never-firing combination latent (pi = 0) out of its
        features' rows.
        """
        live = [j for j in lats if float(mass[j]) > 0.0]
        if not live:
            return list(lats), torch.ones(len(lats), dtype=mass.dtype, device=mass.device)
        return live, mass[live]

    def reduce_unit(self, unit_L: torch.Tensor, acts_L: torch.Tensor) -> torch.Tensor:
        """`[L, D]` -> `[R, D]` geometry rows: under `union`, the unit-normalized mean weighted
        by each latent's activation mass on the draw.

        A group with one live latent is that row itself, not a renormalized copy, so the seam
        stays bit-exact. A cancelling group raises: `signed_normalized_decoder` can orient a
        starved shard against its siblings.
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
        """`[L, D]` -> `[R, D]` reconstruction rows: under `union`, the activation-mass weighted
        mean, not renormalized, so the group's total reconstruction over the draw is kept."""
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
        """Content hash of the map, stamped into every artifact."""
        payload = {"map": [list(l) for l in self.feature_to_latents],
                   "readout": self.readout, "n_latents": self.n_latents}
        if self.composition:
            payload["composition"] = [list(e) for e in self.composition]
        payload = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def resolve_map(corruption, F: int, readout: str = "identity") -> PlantedMap:
    """The planted map for this dictionary: the corruption's own when it declares one (a
    damage that changes the latent count), else the identity map over `F` features."""
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
