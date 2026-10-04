"""One-vs-rest logistic probes.

Reimplements SAEBench's `probing.py::train_multi_probe` and `_run_probe_training` (commit 8042bb3),
with initialisation and shuffling drawn from a local generator.
"""

from __future__ import annotations

import math

import torch


def train_probes(X: torch.Tensor, Y: torch.Tensor, *, epochs: int, batch: int, lr: float,
                 end_lr: float, weight_decay: float, l1_coef: float = 0.0, seed: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """(W [C, d], b [C]) float32 for labels Y [n, C] in {0, 1} on inputs X [n, d].

    BCE with pos_weight = n_neg / n_pos per column, plus l1_coef * mean over probes of sum |w|.
    Adam; the learning rate decays exponentially from lr to end_lr, stepped once per epoch.
    """
    X = torch.as_tensor(X).detach().float()
    Y = torch.as_tensor(Y).detach().float()
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"X {tuple(X.shape)} and Y {tuple(Y.shape)} must be [n, d] and [n, C]")
    n, d = X.shape
    n_pos = Y.sum(dim=0)
    n_neg = n - n_pos
    bad = ((n_pos == 0) | (n_neg == 0)).nonzero(as_tuple=True)[0].tolist()
    if bad:
        # pos_weight would be inf or 0, and the probe would learn nothing for that column
        raise ValueError(f"label columns {bad} have no positive or no negative row")

    gen = torch.Generator().manual_seed(int(seed))
    bound = 1.0 / math.sqrt(d)        # nn.Linear's default init for weights and bias
    W = ((torch.rand(Y.shape[1], d, generator=gen) * 2 - 1) * bound).requires_grad_()
    b = ((torch.rand(Y.shape[1], generator=gen) * 2 - 1) * bound).requires_grad_()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=n_neg / n_pos)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=math.exp(math.log(end_lr / lr) / epochs))
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, int(batch)):
            idx = perm[start:start + int(batch)]
            opt.zero_grad()
            loss = loss_fn(X[idx] @ W.T + b, Y[idx])
            if l1_coef:
                loss = loss + l1_coef * W.abs().sum(dim=-1).mean()
            loss.backward()
            opt.step()
        sched.step()
    return W.detach().clone(), b.detach().clone()
