"""Config resolution: a named toy config, optionally with its confound counts overridden."""

from __future__ import annotations

from . import spec

# The only confound counts a "powered" run may override via resolve_config.
CONFOUND_OVERRIDES: tuple[str, ...] = (
    "n_superparent", "n_token_bound_pairs", "n_topical_pairs", "n_bind_ids",
)

# override -> the family count the base config must already build, so an override never adds a new confound
_OVERRIDE_FAMILY: dict[str, str] = {
    "n_superparent": "n_superparent", "n_token_bound_pairs": "n_token_bound_pairs",
    "n_topical_pairs": "n_topical_pairs", "n_bind_ids": "n_token_bound_pairs",
}


def resolve_config(cfg_name: str, **overrides: int) -> spec.ToyConfig:
    """Return the named base config with `CONFOUND_OVERRIDES` counts overridden; None values are skipped.

    Raises on an unknown knob, a base with confounds=False, a negative or bool count (True
    would pass as 1), or a family the base config does not build.
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
