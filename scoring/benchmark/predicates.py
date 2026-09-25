"""The PASSES/FAILS gate predicates and the clause evaluator, re-exported from `metrics.rules`.

Each predicate returns `(mask, scorable)`; `FAILS` is not `~PASSES`, since NaN must fail both.
"""

from __future__ import annotations

from metrics.rules import GATE_TRUE, PREDICATES, evaluate, fails, passes

__all__ = ["GATE_TRUE", "PREDICATES", "evaluate", "fails", "passes"]

# `wide_matrix` lives only in `reads.py`; a second copy can drift on NaN handling.
