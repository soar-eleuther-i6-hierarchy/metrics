"""Contribution: does the parent carry reconstruction mass on the child's tokens?

    g_f = 2 a_f <d_f, x - x_hat> + a_f^2 ||d_f||^2        (closed-form ablation)
    parent_gain[p, c] = sum g_p / sum err   over c's tokens
    child_gain[c]     = sum g_c / sum err

A contribution filter, not Tree SAE's S_res: two strong unrelated co-firing
features pass it. Kept because it is the one causal condition in the set.

Same objects as `metrics.reconstruction`.
"""

from __future__ import annotations

from metrics.reconstruction import edge_reconstruction_condition as contribution_gains
from metrics.reconstruction import per_token_ablation_gain

__all__ = ["contribution_gains", "per_token_ablation_gain"]
