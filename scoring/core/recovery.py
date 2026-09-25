"""Match learned latents to true features; measure recovery, splitting, dispersion and dead latents.

Shapes: true_coeff [n, F], learned acts [n, S], g [F, D], W_dec [S, D].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toygen import labels


_TIEBREAK_EPS = 1e-6
_TINY = 1e-12


def activation_corr(true_coeff: torch.Tensor, learned_acts: torch.Tensor) -> torch.Tensor:
    """`[F, S]` Pearson correlation of each true feature with each latent, clipped to [-1, 1].

    A zero-variance column gets 0.0, not NaN, so its match cost is 1 and it is never matched."""
    tc = true_coeff - true_coeff.mean(dim=0, keepdim=True) # [n, F]
    lc = learned_acts - learned_acts.mean(dim=0, keepdim=True) # [n, S]
    std_t = tc.pow(2).sum(dim=0).sqrt()          # [F]
    std_l = lc.pow(2).sum(dim=0).sqrt()          # [S]
    cov = tc.transpose(0, 1) @ lc                 # [F, S]
    denom = std_t[:, None] * std_l[None, :]       # [F, S]
    corr = torch.zeros_like(cov)
    nz = denom > 0                                # zero variance on either side -> stays 0.0
    corr[nz] = cov[nz] / denom[nz]
    return corr.clamp(-1.0, 1.0)


@dataclass(frozen=True)
class MatchResult:
    """One-to-one assignment of learned latents to true features."""

    match: torch.Tensor           # [F] long, latent per true feature, -1 if unassigned
    matched_corr: torch.Tensor    # [F] corr at the matched latent, -inf if unassigned
    recovered: torch.Tensor       # [F] bool, matched_corr >= rho


def match_features(corr: torch.Tensor, g: torch.Tensor, W_dec: torch.Tensor,
                   rho: float = 0.5) -> MatchResult:
    """Hungarian match of true features to latents on cost `1 - corr`; recovered means matched corr >= rho.

    Ties break toward higher decoder cosine with `g`. When S < F the surplus features get match -1."""
    from scipy.optimize import linear_sum_assignment

    F, S = corr.shape
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(_TINY) # [F, D]
    wn = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY) # [S, D]
    geom = gn @ wn.transpose(0, 1)                # [F, S] cos(g_i, W_dec[j])
    cost = (1.0 - corr) - _TIEBREAK_EPS * geom    # tiny tiebreak, so corr still dominates
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())

    match = torch.full((F,), -1, dtype=torch.long, device=corr.device)
    matched_corr = torch.full((F,), float("-inf"), dtype=corr.dtype, device=corr.device)
    for r, c in zip(rows.tolist(), cols.tolist()):
        match[r] = c
        matched_corr[r] = corr[r, c]
    recovered = matched_corr >= rho
    return MatchResult(match=match, matched_corr=matched_corr, recovered=recovered)


def recovery_rate_curve(corr: torch.Tensor, rhos: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict[float, float]:
    """Fraction of true features whose best latent clears each rho.

    Uses the best latent per feature, not the Hungarian match, so it upper-bounds the recovery rate."""
    best = corr.max(dim=1).values  # [F]
    return {rho: float((best >= rho).double().mean()) for rho in rhos}


def match_one_to_many(corr: torch.Tensor, rho: float = 0.5) -> dict[str, int | dict[int, list[int]]]:
    """Every latent with corr >= rho for each true feature, best first.

    `split_count` is the number of features claimed by more than one latent."""
    per_feature: dict[int, list[int]] = {}
    split_count = 0
    for i in range(corr.shape[0]):
        row = corr[i]  # [S]
        qualifying = (row >= rho).nonzero(as_tuple=True)[0]
        if qualifying.numel() == 0:
            per_feature[i] = []
            continue
        # stable sort so exact ties resolve by latent index
        order = torch.argsort(row[qualifying], descending=True, stable=True)
        latents = qualifying[order].tolist()
        per_feature[i] = latents
        if len(latents) > 1:
            split_count += 1
    return {"per_feature": per_feature, "split_count": split_count}


def per_class_recovery(recovered: torch.Tensor, pair_labels: torch.Tensor, label_names: list[str]) -> dict[str, tuple[int, int]]:
    """Per label, `(ordered pairs with both endpoints recovered, total pairs)`.

    Scoring runs on recovered pairs only, so this shows how much of each class was dropped."""
    F = pair_labels.shape[0]
    eye = torch.eye(F, dtype=torch.bool, device=pair_labels.device)
    both_recovered = recovered[:, None] & recovered[None, :]   # [F, F]
    out: dict[str, tuple[int, int]] = {}
    for name in label_names:
        in_class = (pair_labels == labels._index(name)) & ~eye
        total = int(in_class.sum())
        both = int((in_class & both_recovered).sum())
        out[name] = (both, total)
    return out


def child_direction_dispersion(W_dec: torch.Tensor, g: torch.Tensor, match: torch.Tensor, children: list[int], m: int = 5) -> torch.Tensor:
    """Per child: over the top-m latents by |cos| with g_c, the share of positive cos^2 off its own match.

    Mixes absorption and splitting; a zero denominator gives 0.0, not NaN."""
    wn = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(_TINY)   # [S, D]
    gn = g / g.norm(dim=1, keepdim=True).clamp_min(_TINY)           # [F, D]
    out = torch.zeros(len(children), dtype=W_dec.dtype, device=W_dec.device)
    for pos, c in enumerate(children):
        cos = wn @ gn[c]  # [S]
        # stable sort, not topk, so |cos| ties resolve deterministically
        top = torch.sort(cos.abs(), descending=True, stable=True).indices[:min(m, cos.numel())]
        pos_energy = cos[top].clamp_min(0.0).pow(2)
        den = float(pos_energy.sum())
        if den <= 0.0:
            continue
        mc = int(match[c])
        own = torch.zeros_like(pos_energy)
        if mc >= 0:
            own[top == mc] = pos_energy[top == mc]
        out[pos] = (pos_energy.sum() - own.sum()) / den
    return out


def reconstruction_fvu(h: torch.Tensor, acts: torch.Tensor, W_dec: torch.Tensor, b_dec: torch.Tensor | None = None) -> float:
    """Fraction of variance unexplained by `acts @ W_dec (+ b_dec)`: 0 is perfect, 1 is the mean baseline."""
    h_hat = acts @ W_dec # [n, D]
    if b_dec is not None:
        h_hat = h_hat + b_dec
    resid = (h - h_hat).pow(2).sum()
    total = (h - h.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(resid / total.clamp_min(_TINY))


def count_dead(acts: torch.Tensor, thresh: float = 0.0) -> int:
    """Number of latents that never exceed `thresh` on any token (never fire)."""
    return int((acts.max(dim=0).values <= thresh).sum())


def realized_l0(acts: torch.Tensor) -> float:
    """Mean number of firing latents per token.

    Pass the SAE's real activations: a randn probe input overstates L0."""
    return float((acts > 0).double().sum(dim=1).mean())
