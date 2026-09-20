"""
World-building, config resolution, and SAE-sizing helpers for the toy generator.

Shared by trainer and scorer: the trainer builds a world to train on, the scorer rebuilds
the same world from a checkpoint's `resolved_config` to score against. Pure toygen + torch,
no GPU dependency.

Matryoshka prefixes follow a fixed geometric schedule, independent of the toy's hierarchy --
sizing SAE nesting from the toy's own depth would hand it the answer.
"""

from __future__ import annotations

import torch

from . import geometry, sample, spec, strengths
from . import tree as tree_mod


def geometric_prefixes(F: int, expansion: int, n_steps: int = 4) -> tuple[list[int], int]:
    """Matryoshka prefix cutoffs for a dictionary of width `expansion * F`.

    Cutoffs halve down from the full width: e.g. F=240, expansion=4 -> d_sae=960 and steps
    [120, 240, 480, 960]. The last step is always the full width (the outermost doll). Depends
    only on (F, expansion, n_steps), never on the tree's depth.
    """
    if F <= 0:
        raise ValueError(f"F must be positive; got {F}")
    if expansion <= 0:
        raise ValueError(f"expansion must be positive; got {expansion}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    d_sae = expansion * F
    steps = sorted({max(1, round(d_sae / (2 ** i))) for i in range(n_steps)})
    if steps[-1] != d_sae:                       # rounding guard; keep full width last
        steps.append(d_sae)
    return steps, d_sae


# Confound counts a "powered" run may raise (each needs >= 5 independent nodes for valid inference); the only overridable knobs, read by resolve_config and train_toy.
CONFOUND_OVERRIDES: tuple[str, ...] = (
    "n_superparent", "n_token_bound_pairs", "n_topical_pairs", "n_bind_ids",
)

# Each override applies only to a config that already builds that family; otherwise it would add a second confound to a pure toy or tune a knob nothing reads.
_OVERRIDE_FAMILY: dict[str, str] = {
    "n_superparent": "n_superparent", "n_token_bound_pairs": "n_token_bound_pairs",
    "n_topical_pairs": "n_topical_pairs", "n_bind_ids": "n_token_bound_pairs",
}

# Short, stable abbreviations for the checkpoint-dir suffix (e.g. sp3-tp8).
_OVERRIDE_ABBR: dict[str, str] = {
    "n_superparent": "sp", "n_token_bound_pairs": "tb",
    "n_topical_pairs": "tp", "n_bind_ids": "bi",
}


def choose_k(tree: tree_mod.Tree, k_override: int | None = None) -> int:
    """The SAE's BatchTopK `k`, derived from the world's true L0 (never a starved value).

    Defaults to `round(target_l0(tree))`. An override below the derived value would starve
    the dictionary on an average token, so it raises rather than training silently starved;
    an override >= derived is returned unchanged.
    """
    l0 = strengths.target_l0(tree)
    # round() is round-half-to-even; safe here since L0 is a sum of continuous rate products (never exactly x.5).
    derived = round(l0)
    if k_override is None:
        return derived
    if k_override < derived:
        raise ValueError(
            f"k={k_override} starves a world with true L0 ~= {l0:.1f} (derived k={derived}); "
            f"the SAE could not represent an average token. Raise k to >= {derived}, or omit "
            f"--k to derive it from the world.")
    return k_override


def resolve_config(cfg_name: str, **overrides: int) -> spec.ToyConfig:
    """Return the named base config with the given confound counts overridden.

    `overrides` may name only `CONFOUND_OVERRIDES` knobs (typos raise); only non-None values
    are applied. Refuses loudly rather than building the wrong world: overriding a config
    with `confounds=False` (would be powered in name only), a negative count (silently
    un-powers), a bool (slips past `< 0` and resolves True -> 1), or a knob whose confound family
    the base config does not build. Feasibility is still checked by `build_tree`.
    """
    if cfg_name not in spec.CONFIGS:
        raise ValueError(
            f"unknown config {cfg_name!r}; known: {sorted(spec.CONFIGS)}")
    unknown = set(overrides) - set(CONFOUND_OVERRIDES)
    if unknown:
        raise ValueError(
            f"unknown override(s) {sorted(unknown)}; overridable: "
            f"{list(CONFOUND_OVERRIDES)}")
    overrides = {name: val for name, val in overrides.items() if val is not None}

    base = spec.CONFIGS[cfg_name]()
    if not overrides:
        return base
    if not base.confounds:
        raise ValueError(
            f"config {cfg_name!r} has confounds disabled, so the overrides "
            f"{sorted(overrides)} would be silently ignored (the generator reads the "
            f"confound counts only when confounds=True); refusing to build a 'powered' "
            f"world that is powered in name only")
    bad = {name: val for name, val in overrides.items()
           if isinstance(val, bool) or val < 0}
    if bad:
        raise ValueError(f"confound counts must be non-negative ints (not bool); got {bad}")
    idle = sorted(name for name in overrides if getattr(base, _OVERRIDE_FAMILY[name]) <= 0)
    if idle:
        raise ValueError(
            f"override {idle} does not apply to config {cfg_name!r}: it builds no "
            f"{sorted({_OVERRIDE_FAMILY[n] for n in idle})} family, so the override would add a "
            f"second confound or tune a knob the world never reads")
    return spec.replace(base, **overrides)


def checkpoint_dirname(config_name: str, variant: str, k: int, expansion: int,
                       overrides: dict[str, int], seed: int | None = None,
                       randomize_structure: bool = False) -> str:
    """Checkpoint directory name, kept distinct by any active confound overrides and the seed.

    Base stem is `{config_name}-{variant}-k{k}-x{expansion}`. Overrides append a sorted
    `-pow-<tag>` suffix so differently-powered runs don't collide; `randomize_structure`
    appends `-rand` so a seed-varied and fixed-lattice run don't clobber each other; `seed`
    appends `-s{seed}`. Defaults reproduce the original name, so old callers stay resolvable.
    """
    stem = f"{config_name}-{variant}-k{k}-x{expansion}"
    if overrides:
        tag = "-".join(f"{_OVERRIDE_ABBR.get(key, key)}{overrides[key]}"
                       for key in sorted(overrides))
        stem = f"{stem}-pow-{tag}"
    if randomize_structure:
        stem = f"{stem}-rand"
    if seed is not None:
        stem = f"{stem}-s{int(seed)}"
    return stem


# A legacy confound only counts as a real distractor if it clears the scorer's edge cut -- validate_config checks its Zipf mass but not this, so an approved override can silently un-power it.
_EDGE_TAU_REFERENCE = spec.EDGE_TAU_REFERENCE
# Population-level check needs a real-scale draw; tiny smoke/test builds (n_tokens ~ hundreds) skip it since training/scoring draws are far above this floor.
_MIN_TOKENS_FOR_CONFOUND_CHECK = 100_000


def _assert_confounds_powered(A: torch.Tensor, tree: tree_mod.Tree) -> None:
    """Refuse a powered world whose frequency or topical confound sits below edge_tau (forms no
    inferred edges, leaving that negative class empty and the detector unchallenged). A low
    `kappa` or high `n_bind_ids` can quietly trigger this even though `validate_config` passes.

    Only pairs of the legacy token-bound / kappa-modulated archetypes are checked. Token groups and
    topic registers fire at a fixed rate inside their cause, so `validate_config` checks their
    containment exactly; their member pairs sit below edge_tau by design."""
    from toygen import labels
    pl = labels.pair_label(tree)
    firing = A > 0
    fire = firing.double().sum(0)
    legacy = torch.tensor([tree.cause_rate.get(k) is None for k in range(tree.F)])
    legacy_pair = legacy.reshape(-1, 1) & legacy.reshape(1, -1)

    def _median_reverse_coverage(cls_name: str) -> float | None:
        fp = ((pl == labels._index(cls_name)) & legacy_pair).nonzero()
        if fp.numel() == 0:
            return None
        rs = [float((firing[:, p] & firing[:, c]).double().sum()) / max(float(fire[c]), 1.0)
              for p, c in fp.tolist()]
        return float(torch.tensor(rs).median())

    knob_hint = {
        "frequency": "LOWER n_bind_ids (a larger id set dilutes R)",
        "topical": "RAISE kappa (stronger topic modulation lifts R)",
    }
    for cls_name in ("frequency", "topical"):
        med = _median_reverse_coverage(cls_name)
        if med is not None and med <= _EDGE_TAU_REFERENCE:
            raise ValueError(
                f"{cls_name} confound is un-powered: median reverse coverage {med:.3f} <= edge_tau "
                f"{_EDGE_TAU_REFERENCE} -> the {cls_name} pairs form no inferred edges, so the "
                f"'{cls_name}' negative class is empty and the detector is not actually challenged. "
                f"{knob_hint[cls_name]} so the {cls_name} pairs clear edge_tau.")


def build_world(cfg_name: str, n_tokens: int, seed: int, device: str,
                config: spec.ToyConfig | None = None,
                ) -> tuple[torch.Tensor, tree_mod.Tree, spec.ToyConfig]:
    """Generate one toy and return (activations, tree, config).

    Returns float32 on `device` (the generator works in float64 internally; an 8M x D double
    buffer would be pure overhead). Geometry uses the config's own seed; sampling uses `seed`,
    so a different `seed` gives an independent world over the same design.

    `config` lets a caller pass a pre-resolved `ToyConfig` (e.g. from `resolve_config`); it
    must match `cfg_name` so a caller can't build one world under another world's label.
    """
    if config is not None and config.name != cfg_name:
        raise ValueError(
            f"config.name ({config.name!r}) != cfg_name ({cfg_name!r}); a passed config "
            f"must match the name it is built under, or the world and its label diverge")
    cfg = config if config is not None else spec.CONFIGS[cfg_name]()
    tree = tree_mod.build_tree(cfg)
    strength = strengths.build_strengths(cfg, tree)
    geo = geometry.build_directions(cfg, tree, seed=cfg.seed)
    world = sample.sample_world(cfg, tree, strength, geo, n_tokens=n_tokens, seed=seed)
    if cfg.confounds and n_tokens >= _MIN_TOKENS_FOR_CONFOUND_CHECK:
        _assert_confounds_powered(world.A, tree)
    return world.h.to(device=device, dtype=torch.float32), tree, cfg
