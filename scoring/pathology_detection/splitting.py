"""k-sparse probing on latent activations and the feature-splitting rule (Chanin et al. Sec. 4).

Reimplements SAEBench's `k_sparse_probing.py` (commit 8042bb3): `train_k_sparse_probes` and
`add_feature_splits_to_metrics_df`.
"""

from __future__ import annotations

import torch


def l1_rank(W_l1: torch.Tensor, k_max: int) -> torch.Tensor:
    """[C, min(k_max, L)] latent ids per probe, largest weight first. Signed, as in SAEBench: only
    latents that predict the label directly are ranked."""
    k = min(int(k_max), int(W_l1.shape[1]))
    return torch.topk(W_l1, k, dim=1).indices


def k_sparse_curve(z_fit: torch.Tensor, y_fit: torch.Tensor, z_eval: torch.Tensor,
                   y_eval: torch.Tensor, ranked_row, k_max: int,
                   cache: dict | None = None) -> torch.Tensor:
    """Eval F1 of a logistic probe on the top-k ranked latents, for k = 1..min(k_max, K).

    `cache` maps a column tuple to its F1, so a rerun on an equal prefix is not refitted.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    ranked = [int(j) for j in ranked_row][:int(k_max)]
    zf = z_fit[:, ranked].double().cpu().numpy()
    ze = z_eval[:, ranked].double().cpu().numpy()
    yf = torch.as_tensor(y_fit).bool().cpu().numpy().astype("int64")
    ye = torch.as_tensor(y_eval).bool().cpu().numpy()
    out = []
    for k in range(1, len(ranked) + 1):
        key = tuple(ranked[:k])
        if cache is not None and key in cache:
            out.append(cache[key])
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(zf[:, :k], yf)
        score = ze[:, :k] @ clf.coef_[0] + clf.intercept_[0]
        f1 = float(f1_score(ye, score > 0, zero_division=0))
        if cache is not None:
            cache[key] = f1
        out.append(f1)
    return torch.tensor(out, dtype=torch.float64)


def split_set(f1: torch.Tensor, ranked_row, f1_jump: float) -> list[int]:
    """The latents at the last k whose F1 beat k - 1 by more than f1_jump, stopping at the first k
    that does not. k = 1 is always kept."""
    prev, kept = -100.0, []
    for k in range(1, len(f1) + 1):
        score = float(f1[k - 1])
        if score > prev + f1_jump:
            prev = score
            kept = [int(j) for j in ranked_row[:k]]
        else:
            break
    return kept
