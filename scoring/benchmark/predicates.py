"""The eight literal predicates from `PRECOMMIT.md` s6, and conjunction over them.

Every predicate returns `(mask, scorable)`:

  `scorable`  the pair has a finite value AND the thresholds this predicate reads are usable.
  `mask`      the predicate is TRUE, and the pair is scorable. `mask` is always a subset of
              `scorable`, so `N_pass <= N_scorable` is an invariant, not a hope.

Both defects the pilot evaluator had live here.

1. NOT-HIGH was implemented as `~HIGH`. In IEEE arithmetic `NaN > q` is False, so `~(NaN > q)`
   is TRUE: a pair with a MISSING score satisfied the negative clause. `superparent_v5` is
   `LOW(wide) AND NOT-HIGH(pmi)`, so this was live on a registered rule. Every predicate here
   is written literally and gated on finiteness instead.

2. Scorability was taken from ONE metric for every expression. A conjunction is scorable only
   where every clause it contains is, which is why `evaluate` intersects the per-clause masks
   rather than taking a single finiteness vector from the caller.

An unusable threshold (NaN, from calibration support below the floor) makes every pair
UNSCORABLE. It must not make every comparison False and be reported as a rejection: "0/N
rejected" and "no measurement exists" are different results.
"""

from __future__ import annotations

import math

import torch

Thresholds = tuple[float | None, float | None]


def _usable(x: float | None) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def _finite(v: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(v)


def _gate(v: torch.Tensor, ok: bool) -> torch.Tensor:
    """Scorable mask: finite value AND a usable threshold."""
    f = _finite(v)
    return f if ok else torch.zeros_like(f)


# --------------------------------------------------------------------------
# the predicates (PRECOMMIT.md s6). Strict/inclusive comparisons are frozen.
# --------------------------------------------------------------------------
def high(v: torch.Tensor, th: Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """`m > Q99`. Needs only the upper threshold."""
    s = _gate(v, _usable(th[1]))
    return (s & (v > (th[1] if _usable(th[1]) else 0.0)), s)


def not_high(v: torch.Tensor, th: Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """`m <= Q99`, written literally. NOT `~high`: that form passes on NaN."""
    s = _gate(v, _usable(th[1]))
    return (s & (v <= (th[1] if _usable(th[1]) else 0.0)), s)


def low(v: torch.Tensor, th: Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """`m < Q01`. Needs only the lower threshold."""
    s = _gate(v, _usable(th[0]))
    return (s & (v < (th[0] if _usable(th[0]) else 0.0)), s)


def not_low(v: torch.Tensor, th: Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """`m >= Q01`, written literally. Registered for completeness; no frozen rule uses it."""
    s = _gate(v, _usable(th[0]))
    return (s & (v >= (th[0] if _usable(th[0]) else 0.0)), s)


def in_band(v: torch.Tensor, th: Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """`Q01 <= m <= Q99`, inclusive at both ends. Needs BOTH thresholds."""
    ok = _usable(th[0]) and _usable(th[1])
    s = _gate(v, ok)
    lo, hi = (th[0], th[1]) if ok else (0.0, 0.0)
    return (s & (v >= lo) & (v <= hi), s)


def sym(v: torch.Tensor, th: Thresholds) -> tuple[torch.Tensor, torch.Tensor]:
    """`|A| <= Q99(|A|)`.

    The threshold passed in is Q99 of the ABSOLUTE asymmetry (see `registry.abs_asymmetry_R`).
    Comparing the SIGNED score against it would accept every strongly negative asymmetry, i.e.
    every reversed containment pair -- the opposite of a symmetry test.
    """
    s = _gate(v, _usable(th[1]))
    return (s & (v.abs() <= (th[1] if _usable(th[1]) else 0.0)), s)


def freq_local(v: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`S < tau_surv`. A FIXED boundary, never a fitted quantile."""
    s = _finite(v)
    return (s & (v < tau), s)


def survives(v: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`S >= tau_surv`. Equality is assigned here, not to FREQ-LOCAL (PRECOMMIT s6)."""
    s = _finite(v)
    return (s & (v >= tau), s)


# Predicates keyed by their PRECOMMIT name. `basis` names the metric whose THRESHOLD the
# predicate reads, given the metric whose VALUE it reads; `None` means it reads `tau_surv`
# instead of a calibrated threshold.
PREDICATES: dict[str, dict] = {
    "HIGH": {"fn": high, "basis": lambda m: m},
    "NOT-HIGH": {"fn": not_high, "basis": lambda m: m},
    "LOW": {"fn": low, "basis": lambda m: m},
    "NOT-LOW": {"fn": not_low, "basis": lambda m: m},
    "IN-BAND": {"fn": in_band, "basis": lambda m: m},
    "SYM": {"fn": sym, "basis": lambda m: f"abs_{m}"},
    "FREQ-LOCAL": {"fn": freq_local, "basis": None},
    "SURVIVES": {"fn": survives, "basis": None},
}


# --------------------------------------------------------------------------
# conjunction
# --------------------------------------------------------------------------
def evaluate(clauses, vals: dict[str, torch.Tensor], thresholds: dict[str, Thresholds],
             tau_surv: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Evaluate a conjunction of `(PREDICATE, metric)` clauses.

    Returns `(mask, scorable, per_clause)`. `scorable` is the intersection of the clauses'
    scorable masks -- CLAUSE-SPECIFIC, which is the point: a pair with a finite `G` but a
    missing PMI is not scorable for `C AND HIGH(G)` and must not be counted as a rejection.

    An unknown predicate or metric raises `KeyError` rather than being skipped: a typo in the
    frozen registry would otherwise loosen a rule after the freeze with no visible change.
    """
    if not clauses:
        raise ValueError("an expression needs at least one clause")
    n = len(next(iter(vals.values())))
    mask = torch.ones(n, dtype=torch.bool)
    scorable = torch.ones(n, dtype=torch.bool)
    per: dict[str, dict] = {}
    for pred_name, metric in clauses:
        if pred_name not in PREDICATES:
            raise KeyError(f"unknown predicate {pred_name!r} in a registered expression")
        if metric not in vals:
            raise KeyError(f"unknown metric {metric!r} in a registered expression")
        spec = PREDICATES[pred_name]
        v = vals[metric]
        if spec["basis"] is None:
            m, s = spec["fn"](v, tau_surv)
        else:
            basis = spec["basis"](metric)
            if basis not in thresholds:
                raise KeyError(f"no threshold fitted for {basis!r} (needed by "
                               f"{pred_name}({metric}))")
            m, s = spec["fn"](v, thresholds[basis])
        key = f"{pred_name}({metric})"
        per[key] = {"n_scorable": int(s.sum()), "n_pass": int(m.sum())}
        mask = mask & m
        scorable = scorable & s
    # mask is already a subset of every clause's scorable mask, but state the invariant.
    mask = mask & scorable
    return mask, scorable, per

# `wide_matrix` deliberately lives in `reads.py` and only there. It was briefly duplicated
# here, which is how a NaN-propagation rule ends up correct in one copy and wrong in the other.
