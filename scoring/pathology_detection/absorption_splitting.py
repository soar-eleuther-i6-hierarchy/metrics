"""Feature absorption and feature splitting on one dictionary, per labeled concept.

Reimplements SAEBench's absorption evaluation (commit 8042bb3, `main.py` and
`feature_absorption.py`), built from `probes`, `splitting` and `absorption`.
Inputs are plain tensors, so the same call serves a synthetic dictionary and a trained SAE's cached
activations: inputs h and labels Y on a fit draw and an eval draw, the latent activations z on both,
and the decoder W_dec. Two modes share the ground-truth probe direction:

  ksparse_main  main latents = the k-sparse split set, as SAEBench runs on a real SAE
  planted_main  main latents = the latents the caller declares for the concept
"""

from __future__ import annotations

import math

import torch

from .absorption import absorption_fraction, full_absorption, latent_probe_cos
from .splitting import k_sparse_curve, l1_rank, split_set
from .probes import train_probes
from .settings import AbsorptionSplittingSettings


def _f1(y: torch.Tensor, pred: torch.Tensor) -> float:
    """Binary F1, 0 when there is nothing to score (sklearn's zero_division=0)."""
    tp = int((y & pred).sum())
    denom = 2 * tp + int((~y & pred).sum()) + int((y & ~pred).sum())
    return 2 * tp / denom if denom else 0.0


def _mean(x: torch.Tensor) -> float | None:
    if x.numel() == 0:
        return None
    m = float(x.double().mean())
    return None if math.isnan(m) else m


def _absorption(h: torch.Tensor, z: torch.Tensor, cos: torch.Tensor, p: torch.Tensor,
                main: list[int], settings: AbsorptionSplittingSettings) -> dict:
    return {
        "absorption_fraction": {ts.name: _mean(absorption_fraction(h, z, cos, p, main, ts))
                                for ts in settings.threshold_sets},
        "full_absorption_rate": _mean(
            full_absorption(h, z, cos, p, main, settings.full_absorption)),
    }


def evaluate_absorption_splitting(h_fit: torch.Tensor, z_fit: torch.Tensor, Y_fit: torch.Tensor,
               h_eval: torch.Tensor, z_eval: torch.Tensor, Y_eval: torch.Tensor,
               W_dec: torch.Tensor, concepts: list[int], planted_main: dict[int, list[int]],
               settings: AbsorptionSplittingSettings, seed: int) -> dict:
    """Per-concept split set, probe F1 and absorption scores. Y columns follow `concepts`.

    Absorption means are over eval tokens where the concept is present and its probe fires.
    """
    concepts = [int(c) for c in concepts]
    Yf, Ye = torch.as_tensor(Y_fit).bool(), torch.as_tensor(Y_eval).bool()
    if Yf.shape[1] != len(concepts) or Ye.shape[1] != len(concepts):
        raise ValueError(f"{len(concepts)} concepts but label widths {Yf.shape[1]} and "
                         f"{Ye.shape[1]}")
    out: list[dict] = []
    testable = []
    for i, c in enumerate(concepts):
        counts = (int(Yf[:, i].sum()), int(Ye[:, i].sum()))
        if 0 < counts[0] < Yf.shape[0] and 0 < counts[1] < Ye.shape[0]:
            testable.append(i)
            continue
        out.append({"concept": c, "testable": False,
                    "reason": f"positives (fit, eval) = {counts}: a probe needs both classes"})

    if testable:
        Ytf = Yf[:, testable]
        s = settings
        W_gt, b_gt = train_probes(h_fit, Ytf, epochs=s.gt_epochs, batch=s.gt_batch, lr=s.gt_lr,
                                  end_lr=s.gt_end_lr, weight_decay=s.gt_weight_decay, seed=seed)
        logits = h_eval.float() @ W_gt.T + b_gt
        l1 = dict(batch=s.l1_batch, lr=s.l1_lr, end_lr=s.l1_end_lr,
                  weight_decay=s.l1_weight_decay, l1_coef=s.l1_coef, seed=seed + 1)
        ranked = l1_rank(train_probes(z_fit, Ytf, epochs=s.l1_epochs, **l1)[0], s.k_max)
        ranked_long = l1_rank(train_probes(
            z_fit, Ytf, epochs=s.l1_epochs * s.stability_epoch_factor, **l1)[0], s.k_max)

        for j, i in enumerate(testable):
            c = concepts[i]
            y_fit, y_eval = Yf[:, i], Ye[:, i]
            fires = logits[:, j] > 0
            probe_f1 = _f1(y_eval, fires)
            cache: dict = {}
            curve = k_sparse_curve(z_fit, y_fit, z_eval, y_eval, ranked[j], s.k_max, cache=cache)
            split = split_set(curve, ranked[j], s.f1_jump)
            curve_long = k_sparse_curve(z_fit, y_fit, z_eval, y_eval, ranked_long[j], s.k_max,
                                        cache=cache)
            split_long = split_set(curve_long, ranked_long[j], s.f1_jump)

            scored = y_eval & fires
            p = W_gt[j].double()
            cos = latent_probe_cos(W_dec, p)
            hs, zs = h_eval[scored], z_eval[scored]
            ksparse_main = {
                "split_latents": split,
                "num_split_features": len(split) - 1,
                "n_main_latents": len(split),
                "f1_curve": [float(x) for x in curve],
                "stable": set(split) == set(split_long),
                "split_latents_long": split_long,
            } | _absorption(hs, zs, cos, p, split, s)
            main = planted_main.get(c)
            planted = (None if main is None else
                       {"main_latents": [int(x) for x in main]}
                       | _absorption(hs, zs, cos, p, list(main), s))
            out.append({"concept": c, "testable": True, "probe_f1": probe_f1,
                        "gated_in": probe_f1 > s.probe_f1_min,
                        "n_eval_pos": int(y_eval.sum()), "n_scored_tokens": int(scored.sum()),
                        "ksparse_main": ksparse_main, "planted_main": planted})

    out.sort(key=lambda r: concepts.index(r["concept"]))
    return {"reference": settings.reference, "settings": settings.as_dict(),
            "concepts": out, "n_concepts": len(concepts), "n_testable": len(testable),
            "n_gated_in": sum(1 for r in out if r.get("gated_in"))}
