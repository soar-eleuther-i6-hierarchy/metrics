"""
Calibrate the hierarchy metrics on the synthetic ground-truth toy.

For each metric we know, by construction, which edges it SHOULD keep and which
pathology it SHOULD catch (see validation/synthetic_toy_world.py). We run the production metric
functions with the production thresholds from config.py and check:

    Metric 1  coverage          recovers 100% of the genuine tree edges.
    Metric 2a reconstruction    rejects the superparent's edges, keeps genuine.
    Metric 2b probe S_res       accepts every true parent from a self-labeled probe.
    Metric 3  sibling redundancy flags the feature-split parent, spares healthy.
    Metric 3' conditioned form   the same, inside the parent's own firing set.
    Metric 4  out-degree        identifies the superparent, spares genuine.
    Metric 5  frequency control rejects the frequency-coincidence edge, keeps
                                genuine.
    Metric 6  independence null ranks genuine above base-rate co-firing.
    Metrics 7-9 joint-child / energy concentration.
    Metric 7 (in-block)         directs a containment pair, calls a co-extensive
                                pair a duplicate.

Plus two NEGATIVE controls, which pass when the battery does *not* do something:
an absorbed edge coverage can never propose, and a shared-topic pair every filter
here accepts. They are the two open columns in the properties matrix, and they
are scored so that a regression turns them from a demonstrated limitation into a
visible failure.

Each metric gets a pass/fail plus a decision MARGIN (how decisively it separated
the two classes); the scorecard is ranked by margin. Run directly for the
printed report, or under pytest for the assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# allow `python3 validation/calibrate_on_synthetic_toy.py` from the experiment_0 root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from metrics import (  # noqa: E402
    coverage_legs,
    degree_stats,
    edge_reconstruction_condition,
    find_superparents,
    frequency_controlled_coverage,
    independence_scores,
    joint_child_coverage_upper,
    keep_edges,
    r_mass,
    r_supp,
    share_energy,
    sibling_redundancy,
)
from metrics.coverage import joint_child_coverage_exact  # noqa: E402  (not in metrics.__all__)
from metrics.sres import (  # noqa: E402
    negative_parent_composition,
    sres_rank_check,
    train_probe,
)
from metrics.sibling_redundancy import parent_conditioned_redundancy  # noqa: E402
from metrics import directed_coverage, duplicate_pairs  # noqa: E402
from validation.synthetic_toy_world import (  # noqa: E402
    GENUINE_TREE,
    IN_BLOCK_DUP,
    SPLIT_CHILDREN,
    build_world,
)

# One child holding >= this fraction of the parent's energy flags a feature split
# (same constant as run_metrics.py's n_share_energy_ge_09 gate).
SPLIT_SHARE_MIN = C.SHARE_ENERGY_SPLIT

# Deleting the superparent's out-edges must drop both the out-degree Gini and the
# top-1 edge share by at least this much; otherwise degree_stats is not
# registering the concentration the superparent causes (the Metric 4
# counterfactual). Observed collapse on the toy is ~0.21 (Gini) and ~0.39 (top-1).
DEGREE_COLLAPSE_MARGIN = 0.10


def _run_metrics(stats):
    """Mirror run_metrics.analyse_pair's calls, on the toy's single block pair."""
    fire_p = stats["fire_p"]
    fire_c = stats["fire_c"]
    cofire = stats["cofire"]

    R, F = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT)

    recon = edge_reconstruction_condition(
        stats["err_sum_c"], stats["g_parent_sum"], stats["g_child_sum"], C.RECON_REL_GAIN_MIN
    )
    fcov = frequency_controlled_coverage(
        stats["cofire_by_bucket"], stats["fire_c_by_bucket"], edge_mask
    )
    deg = degree_stats(edge_mask)
    superparents = find_superparents(
        edge_mask, fire_p, stats["total_tokens"],
        C.SUPERPARENT_OUTDEG_FRAC, C.SUPERPARENT_FIRE_FRAC,
    )
    sib = sibling_redundancy(edge_mask, stats["within_cofire"], fire_c)

    # --- metrics reading the schema-v2 energy / joint-child-union accumulators
    null = independence_scores(cofire, fire_p, fire_c, stats["total_tokens"], C.MIN_JOINT)
    joint_upper = joint_child_coverage_upper(F, edge_mask)
    rs = r_supp(stats["union_count"], fire_p)
    jce = joint_child_coverage_exact(stats["union_count"], fire_p)
    rm = r_mass(stats["union_energy"], stats["energy_total"])
    share = share_energy(stats["energy_cofire"], stats["energy_total"])
    return {
        "R": R, "F": F, "edge_mask": edge_mask,
        "recon": recon, "fcov": fcov, "deg": deg,
        "superparents": superparents, "sib": sib,
        "pmi": null["pmi"], "pmi_valid": null["valid"],
        "joint_upper": joint_upper, "r_supp": rs, "joint_exact": jce,
        "r_mass": rm, "share": share,
    }


def _score(stats, labels, m) -> list[dict]:
    """One scorecard row per metric: pass/fail on its designated job + margin."""
    edge_mask = m["edge_mask"]
    kept = {(int(p), int(c)) for p, c in torch.nonzero(edge_mask).tolist()}
    genuine = labels.genuine
    rows: list[dict] = []

    # --- Metric 1: coverage recovers the genuine tree -----------------------
    recovered = genuine & kept
    rows.append({
        "metric": "1. coverage (edge set)",
        "job": "recover the true tree edges",
        "pass": recovered == genuine,
        "detail": f"{len(recovered)}/{len(genuine)} genuine edges kept; "
                  f"edge set also holds {len(kept) - len(recovered)} non-genuine "
                  f"(that is what metrics 2-5 must prune)",
        "margin": len(recovered) / max(len(genuine), 1),
        "margin_kind": "categorical",
    })

    # --- Metric 2: reconstruction rejects superparent, keeps genuine --------
    passes = m["recon"]["passes"] & edge_mask
    pgain = m["recon"]["parent_gain"]
    sp_edges = [(p, c) for (p, c) in kept if p in labels.superparent_parents]
    sp_rejected = sum(not bool(passes[p, c]) for p, c in sp_edges)
    gen_kept = sum(bool(passes[p, c]) for p, c in (genuine & kept))
    sp_max_gain = max((float(pgain[p, c]) for p, c in sp_edges), default=0.0)
    gen_min_gain = min((float(pgain[p, c]) for p, c in (genuine & kept)), default=0.0)
    rows.append({
        "metric": "2. reconstruction",
        "job": "reject superparent edges, keep the true ones",
        "pass": sp_rejected == len(sp_edges) and gen_kept == len(genuine & kept),
        "detail": f"{sp_rejected}/{len(sp_edges)} superparent edges rejected, "
                  f"{gen_kept}/{len(genuine & kept)} genuine kept "
                  f"(parent-gain: genuine>={gen_min_gain:.2f}, superparent<={sp_max_gain:.4f}, "
                  f"thr={C.RECON_REL_GAIN_MIN})",
        "margin": gen_min_gain / max(sp_max_gain, 1e-9),
    })

    # --- Metric 3: sibling redundancy flags the split parent ----------------
    sib = m["sib"]
    split_red = sib.get(labels.split_parents and next(iter(labels.split_parents)), {}).get("redundancy", 0.0)
    healthy_reds = [v["redundancy"] for p, v in sib.items() if p not in labels.split_parents]
    healthy_max = max(healthy_reds, default=0.0)
    flagged = split_red >= C.SIBLING_REDUNDANCY_FLAG
    healthy_ok = healthy_max < C.SIBLING_REDUNDANCY_FLAG
    rows.append({
        "metric": "3. sibling redundancy",
        "job": "flag feature-split parent, spare healthy",
        "pass": flagged and healthy_ok,
        "detail": f"split parent redundancy={split_red:.2f} "
                  f"({'flagged' if flagged else 'MISSED'}); "
                  f"healthy parents max={healthy_max:.2f} (thr={C.SIBLING_REDUNDANCY_FLAG})",
        "margin": split_red / max(healthy_max, 1e-9),
    })

    # --- Metric 4: out-degree finds the superparent -------------------------
    # find_superparents names the culprit; degree_stats (and the gini it calls)
    # must independently register the concentration that culprit causes. Grade
    # both: deleting the superparent's out-edges has to collapse the out-degree
    # Gini and the top-1 edge share, or degree_stats is not measuring what it
    # claims (a broken gini that returns 0 makes the two blocks equal and fails).
    found = {sp["parent_local"] for sp in m["superparents"]}
    sp_local = next(iter(labels.superparent_parents))
    deg_all = m["deg"]
    mask_no_sp = edge_mask.clone()
    mask_no_sp[sp_local] = False
    deg_no_sp = degree_stats(mask_no_sp)
    gini_collapses = deg_all["outdeg_gini"] > deg_no_sp["outdeg_gini"] + DEGREE_COLLAPSE_MARGIN
    top1_collapses = deg_all["top1_edge_share"] > deg_no_sp["top1_edge_share"] + DEGREE_COLLAPSE_MARGIN
    rows.append({
        "metric": "4. out-degree / superparent",
        "job": "identify superparent, spare the true parents",
        "pass": (found == labels.superparent_parents
                 and gini_collapses and top1_collapses),
        "detail": f"detected superparents {sorted(found)} "
                  f"(truth {sorted(labels.superparent_parents)}); removing the "
                  f"superparent collapses Gini {deg_all['outdeg_gini']:.3f}->"
                  f"{deg_no_sp['outdeg_gini']:.3f} and top-1 share "
                  f"{100 * deg_all['top1_edge_share']:.0f}%->"
                  f"{100 * deg_no_sp['top1_edge_share']:.0f}% "
                  f"(covers degree_stats, gini)",
        "margin": deg_all["outdeg_gini"] / max(deg_no_sp["outdeg_gini"], 1e-9),
    })

    # --- Metric 5: frequency control rejects the coincidence edge -----------
    survival = m["fcov"]["survival"]
    freq_surv = [float(survival[p, c]) for (p, c) in labels.freq_edges if (p, c) in kept]
    freq_rejected = sum(s < C.FREQ_SURVIVAL_MIN for s in freq_surv)
    gen_surv = [float(survival[p, c]) for (p, c) in (genuine & kept)
                if not torch.isnan(survival[p, c])]
    gen_kept_freq = sum(s >= C.FREQ_SURVIVAL_MIN for s in gen_surv)
    rows.append({
        "metric": "5. frequency control",
        "job": "reject frequency-coincidence edge, keep the true one",
        "pass": freq_rejected == len(freq_surv) and gen_kept_freq == len(gen_surv),
        "detail": f"{freq_rejected}/{len(freq_surv)} freq edges rejected "
                  f"(survival={[round(s, 2) for s in freq_surv]}); "
                  f"{gen_kept_freq}/{len(gen_surv)} genuine survive "
                  f"(min genuine survival={min(gen_surv, default=float('nan')):.2f}, "
                  f"thr={C.FREQ_SURVIVAL_MIN})",
        "margin": (min(gen_surv, default=0.0) / max(max(freq_surv, default=1e-9), 1e-9)),
    })

    sp_parent = next(iter(labels.superparent_parents))
    gen_parents = sorted({p for (p, _) in genuine})

    # --- Metric 6: independence null (PMI) ranks genuine above base rate -----
    pmi, pmi_valid = m["pmi"], m["pmi_valid"]
    gen_pmi = [float(pmi[p, c]) for (p, c) in genuine if bool(pmi_valid[p, c])]
    sp_pmi = [float(pmi[sp_parent, c]) for c in range(stats["C"])
              if bool(pmi_valid[sp_parent, c])]
    gen_min_pmi = min(gen_pmi, default=float("nan"))
    sp_max_pmi = max(sp_pmi, default=0.0)
    rows.append({
        "metric": "6. independence null (PMI)",
        "job": "rank true edges above base-rate and superparent co-firing",
        "pass": bool(gen_pmi) and bool(sp_pmi) and gen_min_pmi > sp_max_pmi,
        "detail": f"min genuine PMI={gen_min_pmi:.2f} > max superparent PMI={sp_max_pmi:.2f}; "
                  f"base-rate confound only - topical co-occurrence is not in this toy "
                  f"(needs a model-based null)",
        "margin": gen_min_pmi / max(sp_max_pmi, 1e-9),
        "series": {"name": "PMI against the independence null",
                   "keep": gen_pmi, "reject": sp_pmi, "threshold": 0.0},
    })

    # --- Metric 7: joint-child coverage (support) - genuine covered, not SP --
    rs, jce, ju = m["r_supp"], m["joint_exact"], m["joint_upper"]
    gen_rs = [float(rs[p]) for p in gen_parents]
    sp_rs = float(rs[sp_parent])
    exact_eq = bool(torch.allclose(rs, jce))
    upper_ok = bool((ju >= jce - 1e-9).all())
    rows.append({
        "metric": "7. joint-child coverage (support)",
        "job": "children cover a true parent's firing, not a superparent's",
        "pass": bool(gen_rs) and min(gen_rs) > sp_rs and exact_eq and upper_ok,
        "detail": f"genuine R_supp>={min(gen_rs, default=float('nan')):.2f} vs "
                  f"superparent={sp_rs:.2f}; r_supp and joint_child_coverage_exact "
                  f"agree ({exact_eq}) - a same-formula drift guard, not independent "
                  f"grading; upper>=exact for all parents: {upper_ok} "
                  f"(covers r_supp, joint_child_coverage_upper; "
                  f"joint_child_coverage_exact is proved numerically in "
                  f"tests/test_metric_math.py)",
        "margin": min(gen_rs, default=0.0) / max(sp_rs, 1e-9),
    })

    # --- Metric 8: joint-child coverage (energy-weighted) -------------------
    rm = m["r_mass"]
    gen_rm = [float(rm[p]) for p in gen_parents]
    sp_rm = float(rm[sp_parent])
    rows.append({
        "metric": "8. joint-child coverage (energy)",
        "job": "energy-weighted child coverage separates true from superparent",
        "pass": bool(gen_rm) and min(gen_rm) > sp_rm,
        "detail": f"genuine R_mass>={min(gen_rm, default=float('nan')):.2f} vs "
                  f"superparent={sp_rm:.2f}",
        "margin": min(gen_rm, default=0.0) / max(sp_rm, 1e-9),
    })

    # --- Metric 9: energy concentration flags the feature-split parent ------
    split_parent = next(iter(labels.split_parents))
    max_share = m["share"].max(dim=1).values          # [P]
    split_share = float(max_share[split_parent])
    gen_share_max = max((float(max_share[p]) for p in gen_parents), default=0.0)
    rows.append({
        "metric": "9. energy concentration (share_energy)",
        "job": "flag feature-split parent (a child holds >=90% of its energy)",
        "pass": split_share >= SPLIT_SHARE_MIN and gen_share_max < split_share,
        "detail": f"split parent max child share={split_share:.2f} (thr={SPLIT_SHARE_MIN}); "
                  f"genuine parents max={gen_share_max:.2f}",
        "margin": split_share / max(gen_share_max, 1e-9),
    })

    rows += _score_per_token(stats, labels, m, kept)
    rows += _score_blind_spots(stats, labels, m, kept)
    return rows


# ---------------------------------------------------------------------------
# The four per-token functions. They read residuals and firing masks rather than
# the reduced statistics, which is why they sat outside this file -- and why they
# had no ground-truth calibration anywhere: Tier 2 does not call them either.
# ---------------------------------------------------------------------------
def _score_per_token(stats, labels, m, kept) -> list[dict]:
    resid, fired, W_dec = stats["resid"], stats["fired"], stats["W_dec"]
    P_ = stats["P"]
    sp_parent = next(iter(labels.superparent_parents))
    rows: list[dict] = []

    # --- Metric 2b: probe S_res, rank-scored --------------------------------
    # One probe per child, then the SAME probe scores its true parent and the
    # superparent. Identical input, so the rank rule is the only thing deciding:
    # a difference cannot be an artefact of how the two probes were trained.
    gen_pass = gen_total = sp_pass = sp_total = 0
    gen_ranks, sp_ranks = [], []
    for c in sorted({c for (_, c) in labels.genuine}):
        gc = P_ + c
        probe = train_probe(resid.float(), fired[:, gc], seed=gc,
                            neg_ratio=C.SRES_NEG_RATIO,
                            max_tokens=C.SRES_MAX_PROBE_TOKENS,
                            min_neg=C.SRES_MIN_NEG)
        if probe is None:
            continue
        corr = probe.double() @ W_dec.T                    # [D] over ALL features
        for p in [p for (p, cc) in labels.genuine if cc == c]:
            ok, det = sres_rank_check(corr, p, gc, C.SRES_RANK_TOP_K)
            gen_total += 1
            gen_pass += int(ok)
            gen_ranks.append(det["parent_rank"])
        if (sp_parent, c) in kept:
            ok, det = sres_rank_check(corr, sp_parent, gc, C.SRES_RANK_TOP_K)
            sp_total += 1
            sp_pass += int(ok)
            sp_ranks.append(det["parent_rank"])

    neg_share = negative_parent_composition(
        ~fired[:, P_ + sorted({c for (_, c) in labels.genuine})[0]], fired[:, sp_parent]
    )
    # The rank rule is a geometry test on decoder directions, so an unrelated
    # parent passes exactly when chance puts it in the top k of D -- there is no
    # mechanism by which it detects a superparent. Its null rate is therefore
    # k/D, and asserting zero here would be asserting something the rule does not
    # claim. D = 42 in this toy gives 11.9%; gemma's 32768 gives 0.015%. That
    # dependence is the finding, and it is why an S_res pass rate is only
    # comparable across sources of similar dictionary size.
    D_toy = int(W_dec.shape[0])
    chance = C.SRES_RANK_TOP_K / D_toy
    expected = chance * sp_total
    tol = expected + 3.0 * max(expected, 1.0) ** 0.5          # Poisson slack
    rows.append({
        "metric": "2b. probe S_res (rank rule)",
        "job": "accept every true parent; carry no signal against an unrelated one",
        # neg_share is graded, not just printed: the rank rule's blindness to the
        # superparent is explained by the superparent saturating the negative
        # class, so a broken negative_parent_composition (e.g. returning 0) must
        # break this row rather than pass silently.
        "pass": (gen_total > 0 and gen_pass == gen_total and sp_pass <= tol
                 and neg_share > 0.5),
        "detail": f"{gen_pass}/{gen_total} genuine edges pass the top-{C.SRES_RANK_TOP_K} "
                  f"rank rule (median true-parent rank "
                  f"{sorted(gen_ranks)[len(gen_ranks) // 2] if gen_ranks else float('nan')}); "
                  f"superparent {sp_pass}/{sp_total} (median rank "
                  f"{sorted(sp_ranks)[len(sp_ranks) // 2] if sp_ranks else float('nan')}) "
                  f"against {expected:.1f} expected by chance at k/D={chance:.1%}. "
                  f"The superparent holds {100 * neg_share:.0f}% of the negative class, so "
                  f"it cannot enter the probe direction; what passes is coincidence, not "
                  f"detection. **The rule's strictness is set by dictionary size** — the "
                  f"same k is 0.015% on gemma's 32768 and 0.28% on PCFG's 1792 "
                  f"(covers train_probe, sres_rank_check, negative_parent_composition)",
        "margin": (min(sp_ranks, default=D_toy)
                   / max(max(gen_ranks, default=0) + 1, 1)),
        # the two classes as numbers, so a page can plot the separation instead
        # of quoting the sentence about it -- and so nothing retrains 20 probes
        # to redraw a chart.
        "series": {"name": "probe rank of the parent (0 = top of the dictionary)",
                   "keep": gen_ranks, "reject": sp_ranks,
                   "threshold": C.SRES_RANK_TOP_K, "reject_above": True},
    })

    # --- Metric 3': sibling redundancy conditioned on the parent ------------
    # The global form measures co-firing anywhere; this one measures it inside
    # the parent's firing set, which is the quantity feature splitting is about.
    split_parent = next(iter(labels.split_parents))
    gen_parent = sorted(GENUINE_TREE)[0]
    red_split = parent_conditioned_redundancy(
        fired[:, split_parent], fired[:, [P_ + c for c in SPLIT_CHILDREN]])
    red_gen = parent_conditioned_redundancy(
        fired[:, gen_parent], fired[:, [P_ + c for c in GENUINE_TREE[gen_parent]]])
    rows.append({
        "metric": "3'. parent-conditioned redundancy",
        "job": "flag the split parent inside its own firing set, spare a true one",
        "pass": red_split >= C.SIBLING_REDUNDANCY_FLAG and red_gen < C.SIBLING_REDUNDANCY_FLAG,
        "detail": f"split parent={red_split:.2f}, genuine parent={red_gen:.2f} "
                  f"(thr={C.SIBLING_REDUNDANCY_FLAG}); the conditioned form the global "
                  f"Jaccard defers to (covers parent_conditioned_redundancy)",
        "margin": red_split / max(red_gen, 1e-9),
        "series": {"name": "redundancy inside the parent's firing set",
                   "keep": [red_gen], "reject": [red_split],
                   "threshold": C.SIBLING_REDUNDANCY_FLAG, "reject_above": True},
    })

    # --- Metric 7: within-block directed edges and duplicates ---------------
    # No block ordering exists inside a block, so direction has to fall out of
    # the coverage asymmetry: 28 fires only inside 27, while 29 and 30 fire on
    # identical tokens and must be reported as co-extensive, never as an edge.
    d = directed_coverage(stats["within_cofire"], stats["fire_c"],
                          C.EDGE_TAU, C.MIN_FIRE_COUNT, C.MIN_JOINT)
    ib_edges = {(int(i), int(j)) for i, j in torch.nonzero(d["parent_of"]).tolist()}
    ib_dups = set(duplicate_pairs(d["duplicate"]))
    want_edge = next(iter(labels.in_block_edges))
    antisym = not any((j, i) in ib_edges for (i, j) in ib_edges)
    split_locals = set(SPLIT_CHILDREN)
    split_dups = {(a, b) for (a, b) in ib_dups if a in split_locals and b in split_locals}
    rows.append({
        "metric": "7. in-block directed edges",
        "job": "direct the containment pair, call the co-extensive pair a duplicate",
        "pass": (want_edge in ib_edges and IN_BLOCK_DUP in ib_dups
                 and want_edge not in ib_dups and antisym),
        "detail": f"containment {want_edge} recovered as a directed edge; "
                  f"co-extensive {IN_BLOCK_DUP} reported as a duplicate and not an edge; "
                  f"parent_of antisymmetric: {antisym} (so the in-block graph is acyclic); "
                  f"the {len(split_dups)} split-child pairs also surface as duplicates, "
                  f"which is the same pathology seen from inside one block "
                  f"(covers directed_coverage, duplicate_pairs)",
        "margin": 1.0 if want_edge in ib_edges and IN_BLOCK_DUP in ib_dups else 0.0,
        "margin_kind": "categorical",
    })
    return rows


# ---------------------------------------------------------------------------
# Negative controls and flags. The control rows pass when the battery does NOT
# do something: the properties matrix claims the absorption, topic and
# composition columns are open, and a claim that no metric catches X is worth
# exactly as much as a demonstration. The flag rows check the two structures
# whose edges survive every gate on purpose (multi-parenting, cross-block
# siblings): detection there is a report, not a cut.
# ---------------------------------------------------------------------------
def _score_blind_spots(stats, labels, m, kept) -> list[dict]:
    R, edge_mask = m["R"], m["edge_mask"]
    rows: list[dict] = []

    # --- Absorption is unreachable through coverage -------------------------
    ap, ac = next(iter(labels.absorbed_edges))
    absorbed_R = float(R[ap, ac])
    rows.append({
        "metric": "— absorption (negative control)",
        "job": "confirm coverage cannot propose an absorbed edge at all",
        "pass": (ap, ac) not in kept and absorbed_R < C.EDGE_TAU,
        "detail": f"true edge ({ap}->{ac}) has R={absorbed_R:.2f} < tau={C.EDGE_TAU} and is "
                  f"absent from the candidate set, so metrics 2-9 never see it. "
                  f"Absorption is not measurable in this design — a blind spot that is "
                  f"now demonstrated rather than argued. Fixing it needs a different "
                  f"candidate generator, not a better grader",
        "margin": C.EDGE_TAU / max(absorbed_R, 1e-9),
    })

    # --- Topical co-occurrence passes the whole battery ---------------------
    tp, tc = next(iter(labels.topical_edges))
    survives = {
        "coverage": (tp, tc) in kept,
        "reconstruction": bool((m["recon"]["passes"] & edge_mask)[tp, tc]),
        "frequency": bool(m["fcov"]["survival"][tp, tc] >= C.FREQ_SURVIVAL_MIN),
        "PMI": bool(m["pmi_valid"][tp, tc]) and float(m["pmi"][tp, tc]) > 0,
    }
    rows.append({
        "metric": "— topical co-occurrence (negative control)",
        "job": "confirm no metric here rejects a shared-topic pair",
        "pass": all(survives.values()),
        "detail": "a non-edge (conditionally independent given a shared topic) survives "
                  + ", ".join(k for k, v in survives.items() if v)
                  + (f"; rejected by {', '.join(k for k, v in survives.items() if not v)}"
                     if not all(survives.values()) else "")
                  + f" — R={float(R[tp, tc]):.2f}, PMI={float(m['pmi'][tp, tc]):.2f}. "
                    "This is the open column in the properties matrix; closing it needs a "
                    "model-based topic null, not another threshold",
        "margin": 1.0 if all(survives.values()) else 0.0,
        "margin_kind": "categorical",
    })

    # --- Composition passes the whole battery (third negative control) ------
    comp_ok = {}
    for (p, c) in sorted(labels.composition_edges):
        comp_ok[(p, c)] = ((p, c) in kept
                           and bool((m["recon"]["passes"] & edge_mask)[p, c])
                           and bool(m["fcov"]["survival"][p, c] >= C.FREQ_SURVIVAL_MIN))
    rows.append({
        "metric": "— composition (negative control)",
        "job": "confirm no metric rejects a component -> composed-child edge",
        "pass": all(comp_ok.values()),
        "detail": "the composed child fires exactly where both components fire, so each "
                  "component contains it; "
                  + ", ".join(f"({p}->{c}) " + ("survives" if v else "REJECTED")
                              for (p, c), v in comp_ok.items())
                  + ". Nothing in the battery tests atomicity — the third open column; "
                    "closing it needs a decomposition test, not another threshold",
        "margin": 1.0 if all(comp_ok.values()) else 0.0,
        "margin_kind": "categorical",
    })

    # --- Multi-parenting: the intruder survives, the in-degree reports it ---
    ip, ic = next(iter(labels.multiparent_edges))
    intruder_survives = ((ip, ic) in kept
                         and bool((m["recon"]["passes"] & edge_mask)[ip, ic])
                         and bool(m["fcov"]["survival"][ip, ic] >= C.FREQ_SURVIVAL_MIN))
    in_degree = int(edge_mask[:, ic].sum())
    rows.append({
        "metric": "— multi-parenting (flag)",
        "job": "intruding parent passes every gate; the child's in-degree reports it",
        "pass": intruder_survives and in_degree >= 2,
        "detail": f"intruder edge ({ip}->{ic}) survives coverage, reconstruction and the "
                  f"frequency control; child {ic} has in-degree {in_degree} in the candidate "
                  f"set (true parent + intruder + the superparent overlay), the symptom only "
                  f"the out-degree metric "
                  f"reads. Detection is a report, not a cut",
        "margin": float(in_degree) / 2 if intruder_survives else 0.0,
    })

    # --- Cross-block siblings: kept as an edge, flagged by the energy share --
    cp, cc = next(iter(labels.coextensive_edges))
    pair_kept = (cp, cc) in kept
    pair_share = float(m["share"][cp, cc])
    rows.append({
        "metric": "— cross-block siblings (flag)",
        "job": "co-extensive pair proposed as an edge; energy share ~ 1 flags the rename",
        "pass": pair_kept and pair_share >= SPLIT_SHARE_MIN,
        "detail": f"identical firing sets straddling the block boundary: coverage keeps "
                  f"({cp}->{cc}) with R={float(R[cp, cc]):.2f}, and the child holds "
                  f"{pair_share:.2f} of the parent's energy (thr={SPLIT_SHARE_MIN}) — the "
                  f"rename/duplicate signature from joint-child coverage",
        "margin": pair_share / SPLIT_SHARE_MIN if pair_kept else 0.0,
    })
    return rows


def calibrate(seed: int = 0):
    stats, labels = build_world(seed=seed)
    m = _run_metrics(stats)
    rows = _score(stats, labels, m)
    return stats, labels, rows


def _render(rows) -> str:
    ranked = sorted(rows, key=lambda r: (-int(r["pass"]), -r["margin"]))
    # Site-wide page (no layer): one level down from the site root.
    L = [C.nav_html(depth=1, current="outputs/synthetic_toy_calibration.html"), "",
         "# Synthetic toy calibration — every metric against a known tree", ""]
    L.append("Each metric is graded on the pathology it is meant to catch, using "
             "the production thresholds in `config.py`. Margin = how decisively the "
             "metric separated the two classes (higher is better).")
    L.append("")
    L.append("| rank | metric | job | verdict | margin | detail |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        v = "PASS" if r["pass"] else "**FAIL**"
        mg = ">1000x" if r["margin"] >= 1000 else f"{r['margin']:.1f}x"
        L.append(f"| {i} | {r['metric']} | {r['job']} | {v} | {mg} | {r['detail']} |")
    L.append("")
    n_pass = sum(r["pass"] for r in rows)
    L.append(f"**{n_pass}/{len(rows)} metrics calibrated.** "
             + ("Every scorecard row recovers the genuine tree and rejects its "
                "injected pathology on this toy." if n_pass == len(rows)
                else "Some metrics did not separate cleanly - see FAIL rows."))
    L.append("")
    L.append("These rows grade every metric function on the pathology it targets, "
             "including the four that read per-token residuals and masks "
             "(`train_probe`, `sres_rank_check`, `negative_parent_composition`, "
             "`parent_conditioned_redundancy`) and the two within-block ones "
             "(`directed_coverage`, `duplicate_pairs`). Two pure-formula helpers are "
             "proved against their definitions in `tests/test_metric_math.py` rather "
             "than by class separation here: `joint_child_coverage_exact` (identical to "
             "`r_supp`, kept as a drift guard) and `per_token_ablation_gain` (the "
             "closed-form ablation identity).")
    L.append("")
    L.append("Until 7 August this page claimed the four per-token functions were "
             "*calibrated in Tier 2*. They were not: Tier 2 imports coverage, "
             "reconstruction and the frequency control and nothing else, so the strict "
             "test — the one that rejects most surviving edges on gemma — had no "
             "ground-truth calibration anywhere. The toy now carries the per-token view "
             "(`resid`, `fired`, `W_dec`) that made it testable.")
    L.append("")
    L.append("The dash-named rows are **negative controls and flags**: the controls pass "
             "when the battery does *not* do something (absorption is unreachable because "
             "coverage gates the candidate set; a shared-topic pair and both composition "
             "edges survive every filter — the open columns of the properties matrix, and "
             "a claim that nothing catches them is worth what a demonstration is worth), "
             "and the flags check the two structures whose edges survive on purpose: the "
             "multi-parenting intruder is reported by the child's in-degree, the "
             "cross-block sibling pair by its energy share.")
    return "\n".join(L)


def main():
    _, _, rows = calibrate()
    md = _render(rows)
    print("\n" + md + "\n")
    out_md = C.OUT_DIR / "synthetic_toy_calibration.md"
    out_json = C.OUT_DIR / "synthetic_toy_calibration.json"
    out_md.write_text(md)
    out_json.write_text(json.dumps(
        [{k: (v if not isinstance(v, float) or v != float("inf") else "inf")
          for k, v in r.items()} for r in rows], indent=2))
    print(f"[calib] wrote {out_md}")
    print(f"[calib] wrote {out_json}")
    if not all(r["pass"] for r in rows):
        raise SystemExit("[calib] FAILED: not all metrics recovered ground truth")


# ---- pytest entry points ---------------------------------------------------
def test_all_metrics_calibrate():
    _, _, rows = calibrate()
    failed = [r["metric"] for r in rows if not r["pass"]]
    assert not failed, f"metrics failed calibration: {failed}"


def test_coverage_recovers_tree():
    _, _, rows = calibrate()
    assert rows[0]["pass"], rows[0]["detail"]


def test_reconstruction_rejects_superparent():
    _, _, rows = calibrate()
    assert rows[1]["pass"], rows[1]["detail"]


if __name__ == "__main__":
    main()
