"""Independence: is the co-firing above what the two base rates give?

    PMI(p, c) = log[ n_joint * N / (n_p * n_c) ]
    Dev(p, c) = R(p, c) - rho_p

A parent that fires on most tokens reaches R ~ 1 against every child by base
rate alone; PMI ~ 0 there. Pairs under the support guard are NaN and counted.

Same object as `metrics.independence_null.independence_scores`.
"""

from __future__ import annotations

from metrics.independence_null import independence_scores as pmi_scores

__all__ = ["pmi_scores"]
