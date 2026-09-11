"""Putting two base models of different depth on one axis, once.

`make_report_figures` draws the gemma/PCFG comparison and `make_report_tables`
prints it. Both need the same three decisions -- which runs pair with which, how
big the gap between a pair is, and where the line between "these agree" and
"these do not" falls -- and if each file made them itself the figure and the
table would eventually state different numbers for the same claim. That is the
failure this repo already logs about hand-built pages, one level up: two places
holding the same truth, one of them stale. So the decisions live here and both
importers read them.

This module is deliberately free of matplotlib. The tables generator has no other
reason to import it, and a LaTeX file should not need a plotting backend.

Scope. Everything here is about block pair B0->B1 and nothing else. The PCFG runs
carry 0-3 candidate edges in each of their deeper block pairs, so a comparison
drawn there would be one or two edges wide on one side and thousands on the
other; `beyond_b0b1` exposes those counts so a caller can say so with the actual
numbers rather than assert it.
"""

from __future__ import annotations

import json
from pathlib import Path

import config as C


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pair(report, name="0->1"):
    return next((p for p in report["pairs"] if p["pair"] == name), None)


# The six quantities that are shares of something and therefore land on a common
# 0-100 axis, which is what lets a gap in one be compared with a gap in another.
#
# Candidate-edge DENSITY is deliberately absent. It is a rate per feature pair
# running from 0.6% to 6%, so on a 0-100 scale it would sit at the floor and read
# as agreement that its 3-5x ratio does not support. It belongs in the table,
# where its denominator can be printed beside it.
PANELS = [
    ("children with ≥ 2 parents", "poly"),
    ("sibling redundancy (mean Jaccard)", "sib"),
    ("edges carried by frequent tokens", "freq"),
    ("fan-out concentration (Gini)", "gini"),
    ("edges improving reconstruction", "recon"),
    ("edges at chance for the base rate", "chance"),
]


def rows():
    """Every graded run of every source, with both ways of placing it on one axis.

    There are exactly two defensible alignments and this returns the ingredients
    for both, because they are not interchangeable and the choice is usually made
    silently in an axis label.

    `layer` is the raw block index. Both sources have an SAE trained on layers 1
    and 3, so a same-index comparison is available with no interpolation and no
    matching argument at all -- the strongest form the comparison can take, and
    why it is what the figure draws.

    `depth` is (L + 1) / N, the fraction of the network that has run when the SAE
    reads its input: the hook is `hook_resid_post` on block L, the output of that
    block, so L + 1 of the N blocks have executed. (L + 1) / N and not L / N,
    because the latter puts layer 3 of a 4-block model at 0.75 when it is in fact
    the last block there is. Under this rule the PCFG runs land at 0.50 and 1.00,
    so PCFG layer 1 sits on gemma layer 12 to three decimal places.

    The two rules send PCFG's layers to opposite ends of gemma, which is why
    `alignment_gaps` measures which one the data prefers rather than this
    docstring picking one.

    N comes from the run's own report where the report carries it -- the PCFG
    stats file records `config.base_model.n_layers` -- and from `config` only for
    gemma, whose report does not. Everything else is read from the report: block
    sizes give the density denominator, so a dictionary of a different shape is
    normalised by its own geometry and not by gemma's.
    """
    out = []
    # `layer_*` also matches a variant run sitting beside the layers, such as
    # `layer_12_b3b4`, which grades the same layer with an extra block pair. Reading one
    # as a separate layer counts layer 12 twice, and the duplicate is invisible: the
    # figure gains a point, the caption's run count goes up by one, and both look like
    # more evidence rather than the same evidence repeated.
    #
    # The name is the contract. A layer run is `layer_` plus digits and nothing else;
    # anything with a suffix is a variant and belongs to whoever asked for it.
    import re as _re
    _LAYER_DIR = _re.compile(r"layer_\d+$")
    for src, fallback_n in [(C.SOURCE_NAME, C.BASE_N_LAYERS), ("pcfg-matryoshka", None)]:
        for d in sorted(q for q in (C.OUT_DIR / src).glob("layer_*")
                        if _LAYER_DIR.fullmatch(q.name)):
            rep = _json(d / "metrics_report.json")
            pr = _pair(rep) if rep else None
            if pr is None:
                continue
            cfg = rep.get("config") or {}
            n_layers = (cfg.get("base_model") or {}).get("n_layers") or fallback_n
            L, br = cfg.get("layer"), rep.get("block_ranges") or []
            if n_layers is None or L is None or len(br) < 2:
                continue
            (b0s, b0e), (b1s, b1e) = br[0], br[1]
            out.append({
                "src": src,
                "is_ref": src == C.SOURCE_NAME,
                "label": ("gemma" if src == C.SOURCE_NAME else "PCFG SAE") + f" L{L}",
                "layer": L,
                "n_layers": n_layers,
                "depth": (L + 1) / n_layers,
                "tokens": rep["total_tokens"],
                "d_sae": br[-1][-1],
                "n_blocks": len(br),
                "n_pairs": (b0e - b0s) * (b1e - b1s),
                "n_cand": pr["n_candidate_edges"],
                "poly": pr["degree"]["poly_frac"],
                "sib": pr["sibling_redundancy"]["mean_redundancy"],
                "freq": pr["freq_control"]["frac_freq_driven"],
                "gini": pr["degree"]["outdeg_gini"],
                "recon": pr["reconstruction"]["frac_pass"],
                "chance": pr["independence_null"]["frac_chance_level"],
                # what a reader needs to check the "B0->B1 only" restriction
                # instead of taking it on trust
                "beyond_b0b1": sum(q["n_candidate_edges"] for q in rep["pairs"]
                                   if q["pair"] != "0->1"),
                "n_pairs_computed": len(rep["pairs"]),
                # Which point of Exp 2's three-axis sweep this run is, or None for
                # a source that has no grammar. Carried because "PCFG" names a
                # sweep and a run is one point in it: the published pair are both
                # zipf 1.5 with EOS as the only delimiter, which is rung 2 of the
                # 4 in the PCFG repo's configs/sweep/formatting.yaml. A caption
                # saying "on PCFG" where it means "at one grammar config" is the
                # same overclaim as quoting a layer-6 number as a global fact.
                "grammar": cfg.get("grammar"),
            })
    return sorted(out, key=lambda r: (not r["is_ref"], r["depth"]))


def grammar_line(rs) -> str:
    """The grammar config the non-reference runs share, or how they differ.

    Returns "" when no run carries a grammar block, so a source without one costs
    a clause rather than an exception.
    """
    gs = [r["grammar"] for r in rs if r.get("grammar")]
    if not gs:
        return ""
    if any(g != gs[0] for g in gs[1:]):
        return "several grammar configurations"
    on = [k.replace("_delim", "") for k, v in sorted(gs[0].get("formatting", {}).items()) if v]
    return (f"zipf exponent {gs[0].get('zipf_exponent')}, "
            + ("delimiters: " + "/".join(on) if on else "no delimiters"))


def density(row) -> float:
    """Candidate edges as a share of every ordered B0 x B1 feature pair, in %.

    The denominator is the run's own block geometry. It happens to be within 2%
    between the two sources here -- gemma's 128 x 384 against PCFG's 224 x 224 --
    which is luck, not design, and is what makes the two densities directly
    comparable rather than merely both normalised.
    """
    return 100 * row["n_cand"] / row["n_pairs"]


def matched(rs, by="layer"):
    """Pair each non-gemma run with a gemma run, under one alignment rule.

    `by="layer"` pairs on the raw block index and returns only exact matches, so
    a PCFG layer gemma has not graded is dropped rather than snapped to a
    neighbour. `by="depth"` pairs on (L+1)/N and takes the nearest, which always
    yields a pair and therefore always needs its residual reported.

    Returns (other, reference, residual): 0.0 for an exact index match, and
    |Δdepth| otherwise, so a caller can print the mismatch instead of implying
    the two runs were measured at the same place.
    """
    ref = [r for r in rs if r["is_ref"]]
    out = []
    for o in rs:
        if o["is_ref"] or not ref:
            continue
        if by == "layer":
            hit = next((r for r in ref if r["layer"] == o["layer"]), None)
            if hit:
                out.append((o, hit, 0.0))
        else:
            near = min(ref, key=lambda r: abs(r["depth"] - o["depth"]))
            out.append((o, near, abs(near["depth"] - o["depth"])))
    return out


def gap(pairs, key) -> float:
    """Mean |difference| between the paired runs, in points of a 0-100 share."""
    return sum(abs(100 * o[key] - 100 * g[key]) for o, g, _ in pairs) / len(pairs)


def tex(title: str) -> str:
    """A measure's name as LaTeX. The names carry maths that a figure can set as
    a glyph and a .tex file cannot: a bare U+2265 compiles only under XeLaTeX or
    with inputenc coaxed into it, and the ICLR template is neither."""
    return title.replace("≥", r"$\geq$")


def spread(rs, key) -> float:
    """How far a measure moves across ALL graded runs, in points of a 0-100 share."""
    vals = [100 * r[key] for r in rs]
    return max(vals) - min(vals)


def gap_ratio(rs, pairs, key) -> float:
    """The cross-source gap as a fraction of how much the measure varies at all.

    The correction that keeps the raw gap honest. Three of the six measures sit
    against a boundary on both sources -- multi-parenting is at 89-100%, sibling
    redundancy at 3-6%, the frequency control at 0-2% -- and a measure pinned to
    its ceiling cannot disagree, so its small gap is partly an artefact of having
    nowhere to go. Dividing by the measure's own observed range asks the harder
    question: is the two sources' disagreement small compared with the variation
    the measure shows anyway?

    The two readings do not rank the six the same way, and the disagreement is
    not cosmetic -- on raw points the reconstruction filter is among the worst
    and on this ratio it is among the best. Both are therefore reported wherever
    either is, and neither is called the correct one.
    """
    s = spread(rs, key)
    return gap(pairs, key) / s if s > 0 else float("nan")


def ranked(pairs):
    """The six measures sorted by their own gap, ascending, as (title, key, gap).

    Sorted rather than listed in a fixed order so that the layout of anything
    built from it is a result and not an arrangement that flatters one metric.
    """
    return sorted(((t, k, gap(pairs, k)) for t, k in PANELS), key=lambda x: x[2])


def agree_split(gaps) -> int:
    """How many of the sorted gaps belong to the "these agree" group.

    Cut at the largest RELATIVE jump in the sorted list rather than at a round
    number. A fixed cut is a thumb on the scale: max/4 would call a 5.8-point gap
    agreement purely because some other metric happened to differ by 35, and
    adding a seventh metric would move the line without any measurement changing.
    The biggest ratio between neighbours is a property of the data, so "n of six
    agree to within x points" is measured on both counts.
    """
    if len(gaps) < 2:
        return len(gaps)
    return max((gaps[i + 1] / gaps[i] if gaps[i] > 0 else float("inf"), i + 1)
               for i in range(len(gaps) - 1))[1]


def alignment_gaps(rs):
    """Mean gap over all six measures under each alignment rule.

    Returns (by_layer_mean, by_depth_mean), or None when the two rules pair the
    same runs and the comparison would be vacuous -- which is the case whenever
    the reference source has graded nothing outside the shared layers.
    """
    bl, bd = matched(rs, "layer"), matched(rs, "depth")
    if not bl or not bd:
        return None
    if all(o["layer"] == g["layer"] for o, g, _ in bd):
        return None
    return (sum(gap(bl, k) for _, k in PANELS) / len(PANELS),
            sum(gap(bd, k) for _, k in PANELS) / len(PANELS))
