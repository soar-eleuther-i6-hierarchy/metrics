"""Frequency: does the containment hold once the frequent tokens are removed?

Token ids are bucketed by corpus mass (bucket 0 = the ids covering the top 50%
of tokens, 1 = the next 40%, 2 = the tail). Survival is R over buckets 1+2
divided by R over all buckets; ~1 the edge holds on rare tokens, ~0 it lives on
frequent tokens only.

Same objects as `metrics.token_control`.
"""

from __future__ import annotations

from metrics.token_control import frequency_buckets, local_frequency_buckets
from metrics.token_control import frequency_controlled_coverage as frequency_survival

__all__ = ["frequency_buckets", "local_frequency_buckets", "frequency_survival"]
