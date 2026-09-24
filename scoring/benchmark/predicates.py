"""The two gate predicates and conjunction over them, now shared from `metrics.rules`.

Every predicate returns `(mask, scorable)`: `scorable` is where the gate has a finite value,
`mask` where the predicate holds, always a subset of `scorable`.

Both pilot defects stay guarded in the shared copy: `FAILS` is written literally, never as
`~PASSES` (which passes on NaN), and a conjunction's scorability is the intersection of its
clauses' masks, not one gate's finiteness.

The quantile predicates that used to live here (HIGH, NOT-HIGH, LOW, ...) were deleted when
every rule moved to fixed constants; PRECOMMIT.md s4 records why.
"""

from __future__ import annotations

from metrics.rules import GATE_TRUE, PREDICATES, evaluate, fails, passes

__all__ = ["GATE_TRUE", "PREDICATES", "evaluate", "fails", "passes"]

# `wide_matrix` deliberately lives in `reads.py` and only there. It was briefly duplicated
# here, which is how a NaN-propagation rule ends up correct in one copy and wrong in the other.
