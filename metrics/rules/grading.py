"""
Grading: rule decisions over labelled pairs, turned into counts, rates and a verdict.

Per (rule, class) there are four nested counts: N_total (generated pairs), N_recovered (both
endpoints in the scored frame), N_scorable (every gate the rule reads is defined) and N_pass.
Recall rides on N_recovered, so an unscorable target is a miss; FPR and leaks ride on
N_scorable, so an unscorable null or confound pair cannot dilute them. PRECOMMIT.md s6 and s8.
"""

from __future__ import annotations

import torch

from .classes import LABELS, NULL_CLASS
from .rules import RULES, evaluate

# Version of the keys and arithmetic `grade_rules` writes. History in PRECOMMIT.md s8.
REPORT_SCHEMA = 4

BAR_RECALL = 0.80            # recall given recovery, >=
BAR_EVAL_NULL_FPR = 0.01     # null FPR in each world, <=
BAR_CONFOUND_LEAK = 0.05     # each confound class's pass rate, <=
MIN_SCORABLE_SUPPORT = 10    # fewer scorable pairs than this cannot carry a verdict

# One rule per planted class. The baseline and the no-recon variants overlap these by
# construction, so counting them in `rule_overlap` would manufacture ambiguity.
DESIGNATED: tuple[str, ...] = ("rule_is_a", "rule_firing_only", "rule_superparent",
                               "rule_frequency", "rule_topical")

# Rules that read the probe: INVALID MEASUREMENT when no probe was fitted.
PROBE_RULES: tuple[str, ...] = tuple(
    name for name, rule in RULES.items() if any(g == "gate_sres_rank" for _, g in rule["clauses"]))

VERDICTS: tuple[str, ...] = ("MET CRITERIA", "DID NOT MEET CRITERIA",
                             "UNTESTABLE", "INVALID MEASUREMENT")
VERDICT_SCOPE = "within-world"


def _index(name: str) -> int:
    return LABELS.index(name)


def _rate(num: int, den: int) -> float | None:
    """`None` on a zero denominator: an absent measurement is not a rejection."""
    return (num / den) if den > 0 else None


def _recall(n_pass: int, n_recovered: int, n_scorable: int) -> float | None:
    """`N_pass / N_recovered`, or `None` when nothing was scorable (not a recall of 0)."""
    if n_scorable <= 0:
        return None
    return _rate(n_pass, n_recovered)


def _status(n_scorable: int, floor: int) -> str:
    return "UNTESTABLE" if n_scorable == 0 else "UNDER_SUPPORTED" if n_scorable < floor else "MEASURED"


def class_counts(y: torch.Tensor, mask: torch.Tensor, scorable: torch.Tensor,
                 n_total: dict[str, int], eval_null_idx: list[int],
                 null_class: str = NULL_CLASS,
                 min_support: int = MIN_SCORABLE_SUPPORT) -> dict[str, dict]:
    """The four counts and rates for every class, including absent ones, plus the null row."""
    out: dict[str, dict] = {}
    for name in LABELS:
        sel = y == _index(name)
        n_rec = int(sel.sum())
        n_sc = int((sel & scorable).sum())
        n_pass = int((sel & mask).sum())
        tot = int(n_total.get(name, 0))
        out[name] = {
            "N_total": tot, "N_recovered": n_rec, "N_scorable": n_sc, "N_pass": n_pass,
            "recall_given_recovery": _recall(n_pass, n_rec, n_sc),
            "pass_rate_given_scorable": _rate(n_pass, n_sc),
            "recall_end_to_end": _rate(n_pass, tot),
            "scorable_fraction": _rate(n_sc, n_rec),
            "status": _status(n_sc, min_support),
            "min_scorable_support": min_support,
        }
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    n_ev = int(idx.numel())
    n_ev_sc = int(scorable[idx].sum()) if n_ev else 0
    n_ev_pass = int(mask[idx].sum()) if n_ev else 0
    # keyed `unrelated_eval` for the artifacts on disk, though the null is no longer split
    out["unrelated_eval"] = {
        "N_null": n_ev, "N_scorable": n_ev_sc, "N_pass": n_ev_pass,
        "fpr_given_scorable": _rate(n_ev_pass, n_ev_sc),
        "scorable_fraction": _rate(n_ev_sc, n_ev),
        "status": _status(n_ev_sc, min_support),
        "min_scorable_support": min_support,
        "note": (f"the WHOLE null population, both orderings of every pair labelled "
                 f"`{null_class}` in both directions. Nothing is fitted, so there is no "
                 "calibration half to hold out from and no fitted-on rate to distinguish."),
    }
    return out


def verdict(recall: float | None, fpr: float | None, leaks: dict[str, float],
            testable: bool, supported: bool = True,
            unmeasurable_confounds: tuple[str, ...] = (),
            null_supported: bool = True,
            under_supported_confounds: tuple[str, ...] = ()) -> str:
    """One of the four PRECOMMIT.md s8 labels. All three bars are inclusive.

    A failure on supported evidence is decided before thin or missing evidence is looked at:
    missing evidence blocks a pass but never rescues a failure.
    """
    if not testable or recall is None:
        return "UNTESTABLE"
    if fpr is None:
        return "UNTESTABLE"

    if supported and recall < BAR_RECALL:
        return "DID NOT MEET CRITERIA"
    if null_supported and fpr > BAR_EVAL_NULL_FPR:
        return "DID NOT MEET CRITERIA"
    if any(v is not None and v > BAR_CONFOUND_LEAK
           for name, v in leaks.items() if name not in under_supported_confounds):
        return "DID NOT MEET CRITERIA"

    if not supported or not null_supported:
        return "UNTESTABLE"
    if unmeasurable_confounds or under_supported_confounds:
        return "UNTESTABLE"
    return "MET CRITERIA"


def target_rollup(counts: dict[str, dict], target: tuple[str, ...]) -> dict:
    """One row for a multi-class target, summing counts rather than averaging rates."""
    tot = sum(counts[c]["N_total"] for c in target)
    rec = sum(counts[c]["N_recovered"] for c in target)
    sc = sum(counts[c]["N_scorable"] for c in target)
    ps = sum(counts[c]["N_pass"] for c in target)
    floor = max((counts[c].get("min_scorable_support", 0) for c in target), default=0)
    return {"classes": list(target), "N_total": tot, "N_recovered": rec,
            "N_scorable": sc, "N_pass": ps,
            "recall_given_recovery": _recall(ps, rec, sc),
            "pass_rate_given_scorable": _rate(ps, sc),
            "recall_end_to_end": _rate(ps, tot),
            "scorable_fraction": _rate(sc, rec),
            "status": _status(sc, floor),
            "min_scorable_support": floor}


def _confounds(target: tuple[str, ...], null_class: str) -> list[str]:
    return [name for name in LABELS if name not in target and name != null_class]


def leakage(counts: dict[str, dict], target: tuple[str, ...],
            null_class: str = NULL_CLASS) -> dict[str, float | None]:
    """Pass rate over N_scorable on each confound class with a scorable pair. The leak bar."""
    return {name: counts[name]["pass_rate_given_scorable"]
            for name in _confounds(target, null_class) if counts[name]["N_scorable"] > 0}


def leakage_over_recovered(counts: dict[str, dict], target: tuple[str, ...],
                           null_class: str = NULL_CLASS) -> dict[str, float | None]:
    """The same leaks over N_recovered. A diagnostic, never the bar."""
    return {name: counts[name]["recall_given_recovery"]
            for name in _confounds(target, null_class) if counts[name]["N_scorable"] > 0}


def under_supported_confounds(counts: dict[str, dict], target: tuple[str, ...],
                              null_class: str = NULL_CLASS) -> list[str]:
    """Confound classes scorable on fewer pairs than the support floor."""
    return [name for name in _confounds(target, null_class)
            if counts[name]["status"] == "UNDER_SUPPORTED"]


def unmeasurable_confounds(counts: dict[str, dict], target: tuple[str, ...],
                           null_class: str = NULL_CLASS) -> list[str]:
    """Confound classes present in the world but wholly unscorable for this rule."""
    return [name for name in _confounds(target, null_class)
            if counts[name]["N_recovered"] > 0 and counts[name]["N_scorable"] == 0]


def leak_exceedances(leaks: dict[str, float | None]) -> dict[str, float]:
    """Confounds accepted above the leak bar, recorded even when the target is untestable."""
    return {k: v for k, v in leaks.items() if v is not None and v > BAR_CONFOUND_LEAK}


def rule_overlap(masks: dict[str, torch.Tensor], y: torch.Tensor,
                 eval_null_idx: list[int], null_class: str = NULL_CLASS,
                 scorables: dict[str, torch.Tensor] | None = None) -> dict:
    """How many DESIGNATED rules fire per pair, per class, and which pairs of rules collide.

    With `scorables`, a pair no rule fired on is split into rejected-by-all and
    unscorable-for-all.
    """
    names = [n for n in DESIGNATED if n in masks]
    n_pairs = int(next(iter(masks.values())).numel()) if masks else 0
    if not names:
        empty = {"n": 0, "no_rule": None, "rejected_by_all": None,
                 "unscorable_for_all": None, "exactly_one": None, "multiple": None}
        by = {c: dict(empty, n=int((y == _index(c)).sum())) for c in LABELS}
        by["unrelated_eval"] = dict(empty, n=len(eval_null_idx))
        return {"designated_rules": [], "by_class": by, "collisions": {}}

    fired = torch.stack([masks[n] for n in names]).sum(dim=0)
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

    by = {c: block(y == _index(c)) for c in LABELS}
    idx = torch.tensor(eval_null_idx, dtype=torch.long)
    ev = torch.zeros(n_pairs, dtype=torch.bool)
    if idx.numel():
        ev[idx] = True
    by["unrelated_eval"] = block(ev)

    collisions: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = int((masks[a] & masks[b]).sum())
            if n:
                collisions[f"{a}+{b}"] = n
    return {"designated_rules": names, "by_class": by, "collisions": collisions}


def grade_rules(vals: dict[str, torch.Tensor], y: torch.Tensor, n_total: dict[str, int],
                eval_null_idx: list[int], rules: dict = RULES,
                probe_available: bool = True) -> dict:
    """Grade every rule against every class of one world.

    `vals` holds one gate vector per gate, aligned with the pair labels `y` (class indices);
    `eval_null_idx` indexes the null pairs. Without a probe, rules that read `gate_sres_rank`
    are INVALID MEASUREMENT rather than rejections. Each entry carries its pass and scorable
    masks under `_mask` and `_scorable` for the caller to persist.
    """
    out: dict[str, dict] = {}
    for name, spec in rules.items():
        mask, scorable, per_clause = evaluate(spec["clauses"], vals)
        counts = class_counts(y, mask, scorable, n_total, eval_null_idx)
        roll = target_rollup(counts, spec["target"])
        leaks = leakage(counts, spec["target"])
        unmeasurable = unmeasurable_confounds(counts, spec["target"])
        under_supported = under_supported_confounds(counts, spec["target"])
        ev = counts["unrelated_eval"]
        reads_probe = any(g == "gate_sres_rank" for _, g in spec["clauses"])
        if reads_probe and not probe_available:
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
                     "leakage": leaks,
                     "leakage_over_recovered": leakage_over_recovered(counts, spec["target"]),
                     "leak_exceedances": leak_exceedances(leaks),
                     "unmeasurable_confounds": unmeasurable,
                     "under_supported_confounds": under_supported,
                     "verdict": v, "verdict_scope": VERDICT_SCOPE,
                     "report_schema": REPORT_SCHEMA,
                     "_mask": mask, "_scorable": scorable}
    return out
