"""activations.ridge_acts — honest magnitudes on a planted support, for an ARBITRARY dictionary.

Failure modes anchored: solving off the support (silently densifying the SAE), dropping the
ridge term, wrong reconstruction target, and the W1 support-flip accounting.
"""

from __future__ import annotations

import torch

from synthdict.activations import ridge_acts, support_flip_rate


def test_acts_zero_off_support(isa_world):
    support = isa_world.A > 0
    acts = ridge_acts(isa_world.h, isa_world.g, support)
    assert float(acts[~support].abs().max()) == 0.0


def test_single_latent_closed_form():
    # One token, one latent on support: a = <w, h> / (<w, w> + lam) exactly.
    torch.manual_seed(0)
    D = 16
    w = torch.randn(1, D, dtype=torch.float64)
    h = torch.randn(1, D, dtype=torch.float64)
    lam = 1e-4
    acts = ridge_acts(h, w, torch.ones(1, 1, dtype=torch.bool), lam=lam)
    expect = float((w @ h.T) / (w @ w.T + lam))
    assert abs(float(acts[0, 0]) - expect) < 1e-12


def test_lambda_actually_regularizes():
    # Same setup; lam=0 differs from lam=1.0 by the closed form. Catches a dropped ridge term.
    torch.manual_seed(1)
    D = 8
    w = torch.randn(1, D, dtype=torch.float64)
    h = torch.randn(1, D, dtype=torch.float64)
    a0 = float(ridge_acts(h, w, torch.ones(1, 1, dtype=torch.bool), lam=0.0)[0, 0])
    a1 = float(ridge_acts(h, w, torch.ones(1, 1, dtype=torch.bool), lam=1.0)[0, 0])
    assert abs(a0) > abs(a1)
    assert abs(a1 - float((w @ h.T) / (w @ w.T + 1.0))) < 1e-12


def test_ridge_reconstructs_the_true_world(isa_world):
    # W = g, support = A>0: reconstruction must sit at the noise floor, and acts must track A.
    from scoring.oracle.validate_metrics import reconstruction_fvu

    support = isa_world.A > 0
    acts = ridge_acts(isa_world.h, isa_world.g, support)
    fvu = reconstruction_fvu(isa_world.h, acts, isa_world.g)
    # true-A FVU on this draw is the noise floor; the ridge solve should land essentially there
    fvu_true = reconstruction_fvu(isa_world.h, isa_world.A, isa_world.g)
    assert fvu < fvu_true + 0.02

    on = support & (acts != 0)
    rel = ((acts - isa_world.A).abs()[on] / isa_world.A[on].clamp_min(1e-9))
    assert float(rel.median()) < 0.15   # magnitudes near truth token-by-token, not just in aggregate


def test_support_flip_rate_counts_nonpositive_on_support():
    acts = torch.tensor([[1.0, -0.5, 0.0], [2.0, 3.0, 0.0]], dtype=torch.float64)
    support = torch.tensor([[True, True, False], [True, True, True]])
    # on-support entries: 5; nonpositive among them: -0.5 and the planted-but-zero cell
    assert abs(support_flip_rate(acts, support) - 2 / 5) < 1e-12


def test_flip_rate_low_on_the_real_world(isa_world):
    # The W1 gate, measured on a real (small) draw: planted-support ridge magnitudes must
    # almost never go nonpositive when W = g. The grid re-measures this per dial point.
    support = isa_world.A > 0
    acts = ridge_acts(isa_world.h, isa_world.g, support)
    assert support_flip_rate(acts, support) < 0.005
