"""Settings of the absorption and splitting detector (`absorption_splitting`).

Values are SAEBench's at commit 8042bb3: `eval_config.py` (k_max, F1 jump, probe F1 gate, L1 probe),
`common.py::load_or_train_probe` (ground-truth probe) and `feature_absorption.py:35-44` (absorption
thresholds). The thresholds come in two sets: the ones SAEBench's code uses and the ones its paper's
Table 8 lists (arXiv 2503.09532). The code set is primary; both are reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ThresholdSet:
    name: str
    probe_cos_min: float          # an absorbing latent's decoder needs this cosine with the probe
    projection_share_min: float   # absorbers must carry this share of the probe projection
    max_absorbers: int | None     # at most this many absorbing latents; None = all


CODE_THRESHOLDS = ThresholdSet("saebench_code", 0.1, 0.4, 3)
TABLE8_THRESHOLDS = ThresholdSet("saebench_table8", -1.0, 0.0, None)


@dataclass(frozen=True)
class FullAbsorptionThresholds:
    top_cos_min: float            # the top projecting latent's cosine with the probe
    projection_share_min: float   # its share of the probe projection
    main_off_below: float         # every main latent below this counts as off


FULL_ABSORPTION = FullAbsorptionThresholds(0.025, 0.4, 1e-8)


@dataclass(frozen=True)
class AbsorptionSplittingSettings:
    # ground-truth probe (SAEBench `load_or_train_probe`)
    gt_epochs: int
    gt_batch: int
    gt_lr: float
    gt_end_lr: float
    gt_weight_decay: float
    # L1 multi-probe that ranks latents (SAEBench `train_sparse_multi_probe`)
    l1_epochs: int
    l1_batch: int
    l1_lr: float
    l1_end_lr: float
    l1_weight_decay: float
    l1_coef: float
    # split rule and gates
    k_max: int
    f1_jump: float                # a k-sparse probe must beat k - 1 by more than this
    probe_f1_min: float           # a concept is scored only if its probe F1 exceeds this
    stability_epoch_factor: int   # the ranking is rerun this many times longer
    threshold_sets: tuple[ThresholdSet, ...]
    full_absorption: FullAbsorptionThresholds
    reference: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["threshold_sets"] = [asdict(t) for t in self.threshold_sets]
        return d


ABSORPTION_SPLITTING = AbsorptionSplittingSettings(
    gt_epochs=50, gt_batch=64, gt_lr=1e-2, gt_end_lr=1e-5, gt_weight_decay=1e-4,
    l1_epochs=50, l1_batch=4096, l1_lr=1e-2, l1_end_lr=1e-5, l1_weight_decay=1e-6, l1_coef=0.01,
    k_max=10, f1_jump=0.03, probe_f1_min=0.6, stability_epoch_factor=4,
    threshold_sets=(CODE_THRESHOLDS, TABLE8_THRESHOLDS),
    full_absorption=FULL_ABSORPTION,
    reference="SAEBench main @ 8042bb3",
)
