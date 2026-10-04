"""Detectors for SAE pathologies, run on any dictionary.

Each detector reimplements a published method, so the same code runs on a synthetic dictionary
and on a trained SAE's cached activations. Unlike the census classifier (`scoring/core/
pathologies.py`), these share no definitions with the damage the toy plants. The package is kept
out of the benchmark's hashed sources and has its own content hash.

  absorption_splitting  feature absorption and feature splitting, as SAEBench measures them
                        (independent reimplementation, no code copied: Karvonen et al. 2025,
                        https://github.com/adamkarvonen/SAEBench, commit 8042bb3,
                        `sae_bench/evals/absorption/`, which adapts Chanin et al. 2409.14507,
                        `sae-spelling`, MIT). Built from `probes`, `splitting` and `absorption`.

sklearn is imported only by `splitting.k_sparse_curve`.
"""

from .absorption import absorption_fraction, full_absorption, latent_probe_cos
from .absorption_splitting import evaluate_absorption_splitting
from .probes import train_probes
from .splitting import k_sparse_curve, l1_rank, split_set
from .settings import (ABSORPTION_SPLITTING, CODE_THRESHOLDS, FULL_ABSORPTION, TABLE8_THRESHOLDS,
                       AbsorptionSplittingSettings, FullAbsorptionThresholds, ThresholdSet)

__all__ = [
    "ABSORPTION_SPLITTING", "CODE_THRESHOLDS", "FULL_ABSORPTION", "TABLE8_THRESHOLDS",
    "AbsorptionSplittingSettings", "FullAbsorptionThresholds", "ThresholdSet",
    "absorption_fraction", "full_absorption", "k_sparse_curve", "l1_rank", "latent_probe_cos",
    "evaluate_absorption_splitting", "split_set", "train_probes",
]
