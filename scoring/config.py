"""Settings only our pipeline uses: the benchmark's draws and solver, and the census thresholds.

Gate thresholds and the settings shared with `config.py` live in `metrics/rules/constants.py`.
Artifacts stamp each set whole with `as_dict()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkSettings:
    # The three draws of one experiment seed, kept apart so no two share sampling noise
    # (PRECOMMIT.md s6 step 2): seed (the census's in-sample draw), seed + held_out (scoring),
    # seed + probe_fit (probe fitting).
    held_out_seed_offset: int
    probe_fit_seed_offset: int
    ridge_lambda: float           # ridge penalty of the synthetic dictionary's per-token NNLS
    constant_tol: float           # finite range below this flags a metric as constant
    union_min_mean_norm: float    # a union readout's weighted mean unit row must be this long

    def as_dict(self) -> dict:
        return asdict(self)


BENCHMARK = BenchmarkSettings(
    held_out_seed_offset=10_000,
    probe_fit_seed_offset=20_000,
    ridge_lambda=1e-4,
    constant_tol=1e-9,
    union_min_mean_norm=0.5,
)


@dataclass(frozen=True)
class PathologySettings:
    hole_min: float               # parent recall must drop this far below 1 to count as a hole
    solo_min: float               # min parent recall on parent-solo tokens; near 0 means merging
    conj_min: float               # min conjunction cos K above baseline to count as composition
    null_target_exceedances: float  # Bonferroni target on expected chance latents per dictionary
    n_null_perm: int              # random in-span directions for a stable tail quantile
    mult_prec_min: float          # a candidate shard needs P(feature | latent) >= this
    mult_recall_min: float        # a shard recalls at least this much of the exclusive support
    dup_recall_min: float         # one shard recalling this much is a duplicate, not a split
    split_union_min: float        # >= 2 shards must jointly recall this much to be a split

    def as_dict(self) -> dict:
        return asdict(self)


PATHOLOGY = PathologySettings(
    hole_min=0.15,
    solo_min=0.10,
    conj_min=0.10,
    null_target_exceedances=0.01,
    n_null_perm=1000,
    mult_prec_min=0.5,
    mult_recall_min=0.25,
    dup_recall_min=0.8,
    split_union_min=0.8,
)
