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

Only `identity` is implemented. `strongest_shard` and `union` are the registered names of the
round-2 policies and raise until `SYNTH_PRECOMMIT` registers their bars; a readout that quietly
fell back to identity would publish numbers under a policy nobody approved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

# The registered readout policies. Membership is checked at construction, so a typo fails at the
# call site rather than silently selecting a default.
READOUTS = ("identity", "strongest_shard", "union")
_IMPLEMENTED = ("identity",)


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

    def columns(self) -> torch.Tensor:
        """The latent column each scored feature is read on, aligned with `feats()`.

        Under the `identity` readout a feature carried by several latents has no single column,
        and choosing one here (the first, the strongest) would be exactly the silent reduction
        this module exists to prevent — so it raises instead.
        """
        if self.readout not in _IMPLEMENTED:
            raise NotImplementedError(
                f"readout {self.readout!r} is a registered round-2 policy with no implementation "
                f"yet; SYNTH_PRECOMMIT must register its bars before it can produce numbers")
        cols = []
        for f in self.feats():
            lats = self.feature_to_latents[f]
            if len(lats) != 1:
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
        """
        out = torch.full((self.F,), -1, dtype=torch.long)
        for f, lats in enumerate(self.feature_to_latents):
            if not lats:
                continue
            if len(lats) != 1 and self.readout not in _IMPLEMENTED:
                raise NotImplementedError(
                    f"feature {f} has {len(lats)} latents and readout {self.readout!r} is "
                    f"not implemented; no single latent id can be reported for it")
            out[f] = int(lats[0])
        return out

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
