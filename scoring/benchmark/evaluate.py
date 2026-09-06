"""The reporting contract from `PRECOMMIT.md` s6: four nested counts, distribution flags,
four verdict labels.

Counts, per (expression, class):

  N_total      generated ordered pairs of that class, over the FULL feature set.
  N_recovered  those with both endpoints in the scored universe.
  N_scorable   recovered pairs where every score and threshold the expression reads is usable.
  N_pass       complete-expression positives among those.

Four rates fall out, and all four are reported because they answer different questions:

  recall_given_recovery    N_pass / N_recovered  -- THE BAR
  pass_rate_given_scorable N_pass / N_scorable   -- the rate among decisions actually made
  recall_end_to_end        N_pass / N_total      -- including features recovery never found
  scorable_fraction        N_scorable / N_recovered -- how much recovered evidence was usable

The first is the product of the other two, which is the arithmetic check that they agree.

CONSERVATIVE DENOMINATORS, AND THEY ARE DELIBERATELY ASYMMETRIC. Every bar takes the
denominator that is HARDER to clear:

  * recall rides on N_recovered, so an unscorable TARGET pair counts as a miss. A rule that
    cannot measure a pair has not found it.
  * an FPR or a leak rides on N_scorable, so an unscorable NULL or CONFOUND pair does not
    dilute the rate.

Do not "fix" this into symmetry. Putting the leak bar on N_recovered would let a rule clear the
leak budget by rendering confound pairs unmeasurable -- the rule gets credit for the pairs it
failed to evaluate. Both denominators are reported for all three quantities, under names that
say which is which (`leakage` vs `leakage_over_recovered`, `fpr_given_scorable` vs
`fpr_over_half`), so a reader can always see the one the bar did not use.

This is also why `leakage` reads `pass_rate_given_scorable` and NOT `recall_given_recovery`.
The two keys held the same arithmetic before B2.3; moving the recall bar to N_recovered while
`leakage` still read that key would have moved the leak bar with it, silently, in the same
edit that declared the asymmetry.

Unrecovered targets are end-to-end MISSES; a zero denominator is UNTESTABLE, never 0.0. That
distinction is load-bearing here -- the trained `superparent` class had zero recovered pairs,
and "0/0" rendered as a rate reads as "the rule rejected every superparent pair" when in fact
no measurement exists. `recall_given_recovery` extends the same guard to a NON-zero denominator:
with N_recovered > 0 and N_scorable == 0, N_pass is necessarily 0, so the rate would render
0.000 off zero evidence. It is `None` there too.

Null FPR is taken over the EVALUATION half only. The thresholds were fitted on the calibration
half, so a rate over every unrelated pair is fitted-on. The full-null count is kept beside it
under a different name, as a labelled diagnostic (PRECOMMIT s6).

The support floor guards the NEGATIVE evidence as well as the target: an FPR over two pairs and
a leak over three pairs are arithmetic, not measurements. An under-supported negative row blocks
a PASS without rescuing a FAILURE -- see `verdict` for why that ordering is not symmetric.

Any change to a rate's arithmetic under an existing NAME must bump `registry.REPORT_SCHEMA`,
because artifacts on disk carry the old arithmetic under the same key.
"""

from __future__ import annotations

import math

import torch

from scoring.benchmark.calibrate import boundary_ties
from scoring.benchmark.registry import (BAR_CONFOUND_LEAK, BAR_EVAL_NULL_FPR, BAR_RECALL,
                                        CONSTANT_TOL, MIN_SCORABLE_SUPPORT, NULL_CLASS,
                                        REPORT_SCHEMA)
from toygen import labels

VERDICTS: tuple[str, ...] = ("MET CRITERIA", "DID NOT MEET CRITERIA",
                             "UNTESTABLE", "INVALID MEASUREMENT")

# A verdict is a statement about ONE world. `containment_baseline` reads MET CRITERIA in
# only_isa and only_firing while accepting 25.8% of frequency pairs and 32.1% of topical pairs
# in the other two worlds -- where its own target is absent, so it is UNTESTABLE there and the
# verdict never sees the leak. PRECOMMIT s3 forbids declaring C successful from its two
# positive worlds alone, so the scope is labelled and `leak_exceedances` is recorded whether or
# not the target is testable.
VERDICT_SCOPE = "within-world"


def _rate(num: int, den: int) -> float | None:
    """`None`, never 0.0, on a zero denominator: an absent measurement is not a rejection."""
    return (num / den) if den > 0 else None


def _recall(n_pass: int, n_recovered: int, n_scorable: int) -> float | None:
    """`N_pass / N_recovered`, the recall bar -- but `None` when nothing was scorable.

    The zero-denominator guard is not enough here. With `n_recovered > 0` and `n_scorable == 0`,
    `n_pass` is necessarily 0, so this would render 0.000: "the rule rejected every target pair",
    over pairs no rule ever evaluated. That is the same failure `_rate` exists to prevent,
    reached through a non-zero denominator instead of a zero one.
    """
    if n_scorable <= 0:
        return None
    return _rate(n_pass, n_recovered)


def class_counts(y: torch.Tensor, mask: torch.Tensor, scorable: torch.Tensor,
                 n_total: dict[str, int], eval_null_idx: list[int],
                 null_class: str = NULL_CLASS, cal_null_idx: list[int] | None = None,
                 min_support: int = MIN_SCORABLE_SUPPORT) -> dict[str, dict]:
    """The four counts per ground-truth class, plus evaluation- and calibration-null rows.

    Every class in `labels.LABELS` gets a row even when the toy has none of it: an omitted row
    and a zero row read very differently to someone scanning the table, and a pure toy is
    missing most classes by construction.

    A class with fewer than `min_support` scorable pairs is `UNDER_SUPPORTED`, not `MEASURED`.
    Without that floor a verdict can rest on a single pair: an adversarial review constructed a
    target with `N_scorable=1, N_pass=1` and obtained recall 1.000 and MET CRITERIA. The rate is
    still reported -- it is the VERDICT that must not rest on it.
    """
    out: dict[str, dict] = {}
    for name in labels.LABELS:
        sel = y == labels._index(name)
        n_rec = int(sel.sum())
        n_sc = int((sel & scorable).sum())
        n_pass = int((sel & mask).sum())
        tot = int(n_total.get(name, 0))
        out[name] = {
            "N_total": tot, "N_recovered": n_rec, "N_scorable": n_sc, "N_pass": n_pass,
            # THE BAR. An unscorable target pair is a miss (see the module docstring).
            "recall_given_recovery": _recall(n_pass, n_rec, n_sc),
            # The rate among decisions actually made. This is what a LEAK bar reads, so that an
            # unscorable confound pair cannot dilute it into compliance.
            "pass_rate_given_scorable": _rate(n_pass, n_sc),
            # An unrecovered target is an end-to-end MISS, so it stays in this denominator.
            "recall_end_to_end": _rate(n_pass, tot),
            # The gap between the two rates above, reported rather than left to be inferred.
            "scorable_fraction": _rate(n_sc, n_rec),
            "status": ("UNTESTABLE" if n_sc == 0 else
                       "UNDER_SUPPORTED" if n_sc < min_support else "MEASURED"),
            "min_scorable_support": min_support,
        }
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    n_ev = int(idx.numel())
    n_ev_sc = int(scorable[idx].sum()) if n_ev else 0
    n_ev_pass = int(mask[idx].sum()) if n_ev else 0
    # NOT `N_total`: on every other row that means "generated pairs over the full feature set",
    # and there is no such quantity for a half of the null. Naming it `N_in_half` keeps the two
    # denominators from being read as the same thing.
    out["unrelated_eval"] = {
        "N_in_half": n_ev, "N_scorable": n_ev_sc, "N_pass": n_ev_pass,
        # THE BAR: positives over the SCORABLE evaluation-half null, so an unscorable null pair
        # does not dilute the rate into compliance.
        "fpr_given_scorable": _rate(n_ev_pass, n_ev_sc),
        # The population rate, as a labelled diagnostic. Never the bar.
        "fpr_over_half": _rate(n_ev_pass, n_ev),
        "scorable_fraction": _rate(n_ev_sc, n_ev),
        # The floor guards the NEGATIVE evidence too: an FPR over two pairs clears 0.01
        # arithmetically without establishing anything. Reproduced by an adversarial review,
        # which reached MET CRITERIA off a one-pair evaluation null.
        "status": ("UNTESTABLE" if n_ev_sc == 0 else
                   "UNDER_SUPPORTED" if n_ev_sc < min_support else "MEASURED"),
        "min_scorable_support": min_support,
        "note": (f"evaluation half only; the full-null count is the `{null_class}` row above "
                 "and is fitted-on, not a held-out FPR"),
    }
    # The calibration half as a SEPARATELY LABELLED diagnostic (PRECOMMIT s6), not something a
    # reader has to recover by subtracting two other rows -- and never quotable as an FPR.
    if cal_null_idx is not None:
        cidx = torch.tensor(cal_null_idx, dtype=torch.long)
        n_c = int(cidx.numel())
        n_c_sc = int(scorable[cidx].sum()) if n_c else 0
        n_c_pass = int(mask[cidx].sum()) if n_c else 0
        out["unrelated_cal"] = {
            "N_in_half": n_c, "N_scorable": n_c_sc, "N_pass": n_c_pass,
            "fpr_given_scorable": _rate(n_c_pass, n_c_sc),
            "fpr_over_half": _rate(n_c_pass, n_c),
            "scorable_fraction": _rate(n_c_sc, n_c),
            "status": ("UNTESTABLE" if n_c_sc == 0 else
                       "UNDER_SUPPORTED" if n_c_sc < min_support else "MEASURED"),
            "min_scorable_support": min_support,
            "note": "calibration half -- fitted-on, a DIAGNOSTIC; never quote as an FPR",
        }
    return out


def _finite(v: torch.Tensor) -> torch.Tensor:
    return v[torch.isfinite(v)]


def _is_constant(v: torch.Tensor, tol: float) -> bool | None:
    """`None` below two finite values: one point cannot establish constancy, and zero points is
    an absent measurement rather than a perfectly stable one."""
    f = _finite(v)
    if f.numel() < 2:
        return None
    return bool(float(f.max() - f.min()) <= tol)


def constant_flags(vals: torch.Tensor, y: torch.Tensor, eval_null_idx: list[int],
                   tol: float = CONSTANT_TOL, null_class: str = NULL_CLASS,
                   target: tuple[str, ...] | None = None) -> dict:
    """`constant_null` / `constant_target` / `constant_overall` with the tolerance and the
    finite counts beside them (PRECOMMIT s6).

    A point-mass null and a different point-mass target separate perfectly, which is a
    structurally simple separation rather than a strong detector. The oracle `superparent`
    result is exactly that shape -- null `wide` at 0 against a constant target at -122 -- so
    this flag has to travel with the verdict or the number reads as a clean success.
    """
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    # The evaluation half is the right null population. Falling back to the full null when the
    # evaluation half is empty uses a different, fitted-on population, so the fallback is
    # LABELLED rather than silent.
    if idx.numel():
        null_vals, null_pop = vals[idx], "evaluation half"
    else:
        null_vals, null_pop = (vals[y == labels._index(null_class)],
                               "full-null (no evaluation half)")
    by_class = {name: _is_constant(vals[y == labels._index(name)], tol)
                for name in labels.LABELS}
    n_by_class = {name: int(torch.isfinite(vals[y == labels._index(name)]).sum())
                  for name in labels.LABELS}
    # `constant_target` is named in PRECOMMIT s6, so it is emitted under that name rather than
    # left to be read out of `constant_by_class` by someone who already knows the target. A
    # multi-class target (the containment baseline's union) is constant only if the classes
    # TOGETHER are, so the values are pooled rather than checked one class at a time.
    constant_target = None
    if target:
        sel = torch.zeros(vals.numel(), dtype=torch.bool)
        for name in target:
            sel |= (y == labels._index(name))
        constant_target = _is_constant(vals[sel], tol)
    return {
        "tol": tol,
        "null_population": null_pop,
        "n_finite_overall": int(torch.isfinite(vals).sum()),
        "n_finite_null": int(torch.isfinite(null_vals).sum()),
        "n_finite_by_class": n_by_class,
        "constant_overall": _is_constant(vals, tol),
        "constant_null": _is_constant(null_vals, tol),
        "constant_target": constant_target,
        "constant_by_class": by_class,
    }


def verdict(recall: float | None, fpr: float | None, leaks: dict[str, float],
            testable: bool, supported: bool = True,
            unmeasurable_confounds: tuple[str, ...] = (),
            null_supported: bool = True,
            under_supported_confounds: tuple[str, ...] = ()) -> str:
    """One of the four `PRECOMMIT.md` s8 labels. No fifth label exists.

    A rule meets criteria only when it has a MEASURED recall that clears the bar, an
    evaluation-null FPR within budget, and every named confound leak within budget. All three
    bars are inclusive at their stated values. A missing recall is UNTESTABLE -- a rule cannot
    pass on specificity alone, which is the shape of "superparent rejected 0/0 targets, so its
    specificity is perfect".

    THE ORDER OF THE THREE STAGES BELOW IS LOAD-BEARING AND IS NOT SYMMETRIC.

    An ESTABLISHED failure is stated before any unmeasurable or under-supported evidence can
    convert the verdict to UNTESTABLE. A rule with a well-supported recall of 0.018 against a
    0.80 bar DID NOT MEET CRITERIA, and a third class nobody could measure does not make that
    result unknown -- the thing it failed on was measured.

    A PASS gets the opposite treatment: it cannot be certified over evidence nobody could
    measure, because "no confound pair was scorable" is not "no confound leaked".

    So unmeasurability blocks a pass without rescuing a failure, and that asymmetry is the
    whole point. Collapsing the two stages into one order -- either order -- loses one of them.
    """
    # Stage 1: is there a measurement at all? A rule with no recall and no FPR has no result,
    # in either direction.
    if not testable or recall is None:
        return "UNTESTABLE"
    if fpr is None:
        return "UNTESTABLE"

    # Stage 2: ESTABLISHED failures, each on evidence that meets the support floor. A rate over
    # a handful of pairs is arithmetic, not a basis for a verdict -- in EITHER direction, which
    # is why an under-supported leak over the bar is excluded here and caught by stage 3.
    if supported and recall < BAR_RECALL:
        return "DID NOT MEET CRITERIA"
    if null_supported and fpr > BAR_EVAL_NULL_FPR:
        return "DID NOT MEET CRITERIA"
    if any(v is not None and v > BAR_CONFOUND_LEAK
           for name, v in leaks.items() if name not in under_supported_confounds):
        return "DID NOT MEET CRITERIA"

    # Stage 3: nothing failed on evidence we trust, so the only remaining question is whether
    # the evidence was good enough to CERTIFY a pass.
    if not supported or not null_supported:
        return "UNTESTABLE"
    # A confound class that exists but is wholly unscorable, or scorable on too few pairs,
    # cannot be cleared -- and a rule must not pass by default on a leak nobody could measure.
    if unmeasurable_confounds or under_supported_confounds:
        return "UNTESTABLE"

    return "MET CRITERIA"


def target_rollup(counts: dict[str, dict], target: tuple[str, ...]) -> dict:
    """Collapse a multi-class target (the containment baseline's `is_a UNION firing_only`) into
    one row, summing the counts rather than averaging the rates.

    The primary class rows are unchanged and still reported separately; PRECOMMIT s4 requires
    the two code labels to stay distinct everywhere except this baseline's own target row.
    """
    tot = sum(counts[c]["N_total"] for c in target)
    rec = sum(counts[c]["N_recovered"] for c in target)
    sc = sum(counts[c]["N_scorable"] for c in target)
    ps = sum(counts[c]["N_pass"] for c in target)
    floor = max((counts[c].get("min_scorable_support", 0) for c in target), default=0)
    return {"classes": list(target), "N_total": tot, "N_recovered": rec,
            "N_scorable": sc, "N_pass": ps,
            # Same four rates, same denominators, as a class row. If these ever diverge, the
            # containment baseline's verdict stops being comparable to every other rule's.
            "recall_given_recovery": _recall(ps, rec, sc),
            "pass_rate_given_scorable": _rate(ps, sc),
            "recall_end_to_end": _rate(ps, tot),
            "scorable_fraction": _rate(sc, rec),
            "status": ("UNTESTABLE" if sc == 0 else
                       "UNDER_SUPPORTED" if sc < floor else "MEASURED"),
            "min_scorable_support": floor}


def leakage(counts: dict[str, dict], target: tuple[str, ...],
            null_class: str = NULL_CLASS) -> dict[str, float | None]:
    """Acceptance rate on every class that is NOT this expression's target and not the null.

    Reported for every rule in every world. C's two positive worlds looked like a success until
    its 25.8% frequency and 32.1% topical acceptance were put in the same table.

    THE DENOMINATOR IS `N_scorable`, read from `pass_rate_given_scorable` and deliberately NOT
    from `recall_given_recovery`. The two keys carried the same arithmetic until B2.3 moved the
    recall bar to `N_recovered`; had this function still read that key, the leak bar would have
    moved with it in the same edit that declared the denominators asymmetric. An unscorable
    confound pair must not dilute a leak rate, or a rule clears the leak budget by rendering
    confound pairs unmeasurable. `leakage_over_recovered` reports the other denominator.
    """
    return {name: counts[name]["pass_rate_given_scorable"]
            for name in labels.LABELS
            if name not in target and name != null_class
            and counts[name]["N_scorable"] > 0}


def leakage_over_recovered(counts: dict[str, dict], target: tuple[str, ...],
                           null_class: str = NULL_CLASS) -> dict[str, float | None]:
    """The same leaks over `N_recovered`, as a labelled diagnostic. NEVER the bar.

    Reported so a reader can see the denominator the bar did not use, and see how much of each
    confound row was unscorable, without recomputing it from the counts.
    """
    return {name: counts[name]["recall_given_recovery"]
            for name in labels.LABELS
            if name not in target and name != null_class
            and counts[name]["N_scorable"] > 0}


def under_supported_confounds(counts: dict[str, dict], target: tuple[str, ...],
                              null_class: str = NULL_CLASS) -> list[str]:
    """Confound classes with at least one scorable pair but fewer than the support floor.

    `unmeasurable_confounds` catches the wholly-unscorable case; this catches the one just above
    it, where a leak rate exists but rests on too few decisions to carry a verdict in either
    direction. `verdict` lets these block a pass without letting them establish a failure.
    """
    return [name for name in labels.LABELS
            if name not in target and name != null_class
            and counts[name]["status"] == "UNDER_SUPPORTED"]


def unmeasurable_confounds(counts: dict[str, dict], target: tuple[str, ...],
                           null_class: str = NULL_CLASS) -> list[str]:
    """Confound classes that are PRESENT in this world but wholly unscorable for this rule.

    `leakage` can only report classes with at least one scorable pair, so a confound that exists
    and cannot be measured simply vanishes from the leak table -- and a rule then appears to have
    cleared a bar nobody could evaluate. Naming them keeps that from reading as a pass.
    """
    return [name for name in labels.LABELS
            if name not in target and name != null_class
            and counts[name]["N_recovered"] > 0 and counts[name]["N_scorable"] == 0]


def leak_exceedances(leaks: dict[str, float | None]) -> dict[str, float]:
    """The named confounds this expression accepts above the leakage bar.

    Computed whether or not the expression's own target is testable in this world, which is the
    whole point: an expression whose target is absent here reads UNTESTABLE, and its verdict
    never looks at the confound pairs it is happily accepting.
    """
    return {k: v for k, v in leaks.items() if v is not None and v > BAR_CONFOUND_LEAK}


def rule_overlap(masks: dict[str, torch.Tensor], y: torch.Tensor,
                 eval_null_idx: list[int], null_class: str = NULL_CLASS,
                 scorables: dict[str, torch.Tensor] | None = None) -> dict:
    """Multiple-rule and no-rule outcomes over the five DESIGNATED rules (PRECOMMIT s6).

    Per-rule recall says nothing about how the rules interact, and two of the interactions
    matter here. `frequency_v6` and `topical_v6` are exact complements on one fitted `tau_surv`
    boundary given high PMI, so given high PMI a pair is FORCED into one of them rather than
    abstained on -- they partition rather than detect, and only a no-rule count makes that
    visible. And two rules firing on one pair means the property assignment is ambiguous, which
    is a different failure from either rule missing.

    The baseline and the two historical comparators are excluded: `containment_baseline` is a
    sub-expression of both G rules and the probe comparators overlap them by construction, so
    counting them would manufacture ambiguity that is not there.
    """
    from scoring.benchmark.registry import DESIGNATED

    names = [n for n in DESIGNATED if n in masks]
    n_pairs = int(next(iter(masks.values())).numel()) if masks else 0
    if not names:
        empty = {"n": 0, "no_rule": None, "rejected_by_all": None,
                 "unscorable_for_all": None, "exactly_one": None, "multiple": None}
        by = {c: dict(empty, n=int((y == labels._index(c)).sum())) for c in labels.LABELS}
        by["unrelated_eval"] = dict(empty, n=len(eval_null_idx))
        return {"designated_rules": [], "by_class": by, "collisions": {}}

    stack = torch.stack([masks[n] for n in names])          # [R, n_pairs]
    fired = stack.sum(dim=0)
    # "No rule fired" merges two different facts unless scorability is supplied: every rule
    # REJECTED the pair, versus no rule could MEASURE it. A pass mask is already gated on
    # scorability, so the two are indistinguishable from `masks` alone -- and the distinction is
    # the one this whole package is built to preserve. When `scorables` is given they are split.
    any_scorable = (torch.stack([scorables[n] for n in names]).any(dim=0)
                    if scorables else None)

    def block(sel: torch.Tensor) -> dict:
        n = int(sel.sum())
        f = fired[sel]
        none = f == 0
        out = {"n": n, "no_rule": int(none.sum()), "exactly_one": int((f == 1).sum()),
               "multiple": int((f > 1).sum())}
        if any_scorable is None:
            out["rejected_by_all"] = None
            out["unscorable_for_all"] = None
        else:
            sc = any_scorable[sel]
            out["rejected_by_all"] = int((none & sc).sum())
            out["unscorable_for_all"] = int((none & ~sc).sum())
        return out

    by = {c: block(y == labels._index(c)) for c in labels.LABELS}
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    ev = torch.zeros(n_pairs, dtype=torch.bool)
    if idx.numel():
        ev[idx] = True
    by["unrelated_eval"] = block(ev)

    # Which rules collide, not merely that some did: overlap_v6 + orthogonal_v6 colliding means
    # the subtype split failed; frequency_v6 + topical_v6 colliding would mean the tau boundary
    # is not the partition it is claimed to be.
    collisions: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = int((masks[a] & masks[b]).sum())
            if n:
                collisions[f"{a}+{b}"] = n
    return {"designated_rules": names, "by_class": by, "collisions": collisions}


def evaluate_read(vals: dict[str, torch.Tensor], y: torch.Tensor, thresholds: dict,
                  n_total: dict[str, int], eval_null_idx: list[int],
                  expressions: dict, tau_surv: float,
                  probe_available: bool = True,
                  cal_null_idx: list[int] | None = None) -> dict:
    """Every registered expression against every class in this read.

    `probe_available=False` marks the two probe comparators INVALID MEASUREMENT rather than
    letting an all-NaN `S_res` column render as a rejection: a rule that was never evaluated is
    not a rule that failed.
    """
    from scoring.benchmark import predicates as P
    from scoring.benchmark.registry import PROBE_EXPRESSIONS

    out: dict[str, dict] = {}
    for name, spec in expressions.items():
        mask, scorable, per_clause = P.evaluate(spec["clauses"], vals, thresholds, tau_surv)
        counts = class_counts(y, mask, scorable, n_total, eval_null_idx,
                              cal_null_idx=cal_null_idx)
        roll = target_rollup(counts, spec["target"])
        leaks = leakage(counts, spec["target"])
        leaks_rec = leakage_over_recovered(counts, spec["target"])
        unmeasurable = unmeasurable_confounds(counts, spec["target"])
        under_supported = under_supported_confounds(counts, spec["target"])
        ev = counts["unrelated_eval"]
        if name in PROBE_EXPRESSIONS and not probe_available:
            v = "INVALID MEASUREMENT"
        else:
            v = verdict(roll["recall_given_recovery"], ev["fpr_given_scorable"],
                        leaks, roll["status"] in ("MEASURED", "UNDER_SUPPORTED"),
                        supported=(roll["status"] == "MEASURED"),
                        unmeasurable_confounds=tuple(unmeasurable),
                        null_supported=(ev["status"] == "MEASURED"),
                        under_supported_confounds=tuple(under_supported))
        out[name] = {"text": spec["text"], "target": list(spec["target"]),
                     "clauses": [f"{p}({m})" for p, m in spec["clauses"]],
                     "per_clause": per_clause, "counts": counts, "target_rollup": roll,
                     "leakage": leaks, "leakage_over_recovered": leaks_rec,
                     "leak_exceedances": leak_exceedances(leaks),
                     "unmeasurable_confounds": unmeasurable,
                     "under_supported_confounds": under_supported,
                     "verdict": v, "verdict_scope": VERDICT_SCOPE,
                     "report_schema": REPORT_SCHEMA}
        out[name]["_mask"] = mask
        out[name]["_scorable"] = scorable
    return out


# Detectors whose value is a function of ONE endpoint, broadcast across a row
# (`scoring/core/detectors.py:_broadcast_parent`), plus `wide`, which is a function of the
# unordered pair's two endpoints. A pair-level calibration split holds out essentially nothing
# for these: on the seed-0 worlds 100% of evaluation-half pairs share both endpoints with some
# calibration-half pair. Recorded per metric so an FPR on one of them is not read as held out.
ENDPOINT_BROADCAST: tuple[str, ...] = ("outdegree", "joint_child_J", "joint_child_mass",
                                       "sibling_redundancy", "wide")

# Detectors symmetric in (parent, child). Both orderings of a pair share a calibration half by
# design AND take the same value, so the two decisions are one decision and the reported
# denominator is twice the number of independent ones.
# Metrics that take the SAME value on (p, c) and (c, p), so both orderings of a pair are one
# decision, not two -- which is what `n_effective_eval_null` halves the denominator for.
#
# `S_res` is deliberately NOT here. The repo's doctrine is at scoring/oracle/score_dump.py's
# `symmetric_detectors`: `s_res` is symmetric only in COSINE mode, where it is the Gram matrix
# `W @ W.T`. In this package `S_res` is the PROBE on both reads (reads.py:145) -- a directional
# margin -- and `G` is the cosine one. Listing `S_res` here halved its effective null denominator
# on a metric that never had the symmetry that justifies halving.
SYMMETRIC_METRICS: tuple[str, ...] = ("pmi", "G", "wide", "abs_asymmetry_R")


def metric_diagnostics(vals: dict[str, torch.Tensor], y: torch.Tensor,
                       eval_null_idx: list[int], thresholds: dict,
                       cal_support: dict[str, int], tol: float = CONSTANT_TOL,
                       targets: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Per-metric distributions, thresholds, calibration support, constant-distribution flags,
    boundary ties, and how much of a holdout the split actually is for this metric.

    The last two are the ones easy to omit and easy to be misled by. PRECOMMIT s7 box 5 asks for
    tied boundaries because the predicates fix strict-vs-inclusive precisely so ties decide
    cases. And `split_kind` / `n_effective_eval_null` say when a reported FPR rests on fewer
    independent decisions than its denominator suggests.
    """
    out: dict[str, dict] = {}
    n_ev = len(eval_null_idx)
    for name, v in vals.items():
        fin = _finite(v)
        q = thresholds.get(name, (float("nan"), float("nan")))
        broadcast = name in ENDPOINT_BROADCAST
        symmetric = name in SYMMETRIC_METRICS
        out[name] = {
            "q01": q[0], "q99": q[1],
            "n_cal_finite": cal_support.get(name, 0),
            "threshold_usable": bool(math.isfinite(q[0]) or math.isfinite(q[1])),
            "n": int(v.numel()), "n_finite": int(fin.numel()),
            "median": float(fin.median()) if fin.numel() else float("nan"),
            "boundary_ties": boundary_ties(v, q),
            "split_kind": ("endpoint-broadcast: the pair-level split holds out little or "
                           "nothing for this metric" if broadcast else "pair-level"),
            "symmetric": symmetric,
            # Both orderings of a pair take the same value for a symmetric metric, so the
            # independent-decision count is half the ordered-pair denominator.
            "n_effective_eval_null": (n_ev // 2) if symmetric else n_ev,
            "flags": constant_flags(v, y, eval_null_idx, tol,
                                    target=(targets or {}).get(name)),
        }
    return out
