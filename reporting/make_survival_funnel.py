"""
Survival funnel: of the edges activation coverage proposes, how many are confirmed by every
core criterion? Plus what the supporting criteria read on the survivors, and the same funnel
under the shared pre-commit rule.

    python3 -m reporting.make_survival_funnel \
        --toy outputs/toy-temporal/pipeline/matryoshka_toy \
        --tsae-toy outputs/toy-temporal/pipeline/tsaeip_s0 ... \
        [--tsae-stats PATH]     # exp0_stats.pt for the gemma T-SAE run, if available
        [--tsae-json PATH]      # or the same three entries computed on the node from that file

Outputs, under outputs/paper_figuers/:
    survival_funnel.png               left: share of candidate edges still confirmed after each
                                      core criterion, in the order of the hierarchy rule;
                                      right: the supporting criteria read on the edges that
                                      survive all five
    survival_funnel_pcfg_density.png  the PCFG line split by formatting density
    survival_funnel_rules.png         the same funnel under `metrics.rules.rule_hierarchy`
    survival_funnel.{md,json}         every count behind the figures

Stages, cumulative and strict: an edge that a criterion cannot measure (survival with too few
rare-token firings, a child with too few probe positives) is not confirmed, so it leaves the
funnel at that stage; the JSON records how many did.

    1  candidate edges       reverse coverage >= tau, both endpoints and the pair supported
    2  above chance          PMI >= 0.5, the table's chance-level cut
    3  survives on rare      survival >= 0.5
    4  reconstruction        parent and child gains >= 0.01 on the child's tokens
    5  probe rank            both decoders in the child probe's top k

The rule funnel follows `rule_hierarchy` clause by clause: support and strict containment
(R >= tau and R_rev < tau), PMI > 0, survival, probe rank. It has no reconstruction clause and
its PMI cut is 0, not 0.5, so its counts are not the core funnel's.

Every mask is recomputed from the run's cached statistics with the same functions
run_metrics.py uses; the probe verdicts come from second_pass.json. The gemma T-SAE statistics
file (3.5 GB) lives on the compute node: `--tsae-stats` reads it directly, `--tsae-json` reads
the three entries that `funnel_from_stats` and `rules_funnel_from_stats` produced there (the
default points at `outputs/gemma-2-2b-tsae/layer_12/funnel_from_node.json`). With neither,
stages 1 and 2 are exact from the reports and stages 3 to 5 are drawn as the interval the
reports bound, and the supporting panel and the rule funnel are left empty for that setting.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

import config as C
from metrics import (coverage_legs, edge_reconstruction_condition, frequency_controlled_coverage,
                     independence_scores, keep_edges, r_supp, sibling_redundancy)
from metrics.rules import GEMMA_MATRYOSHKA, RULES, RULESET_VERSION, PairStats, score_pairs
from metrics.rules.gates import DT, high_outdegree, tristate
from metrics.rules.rules import evaluate
from reporting.make_pure_metrics import PAPER_DIR, SOURCE_SHORT, SOURCES, _json
from run_metrics import block_selector

STAGES = ["1 candidates", "2 PMI ≥ 0.5", "3 survival ≥ 0.5", "4 reconstruction", "5 probe rank"]
TICKS = {"1 candidates": "1\ncandidates", "2 PMI ≥ 0.5": "2\nPMI ≥ 0.5", "3 survival ≥ 0.5": "3\nsurvival\n≥ 0.5",
         "4 reconstruction": "4\nrecon-\nstruction", "5 probe rank": "5\nprobe\nrank"}
RULE_STAGES = ["1 support ∧ strict\ncontainment", "2 PMI > 0", "3 survival ≥ 0.5", "4 probe rank"]
CHANCE_PMI = 0.5     # the table's "at chance level" cut
PCFG_DENSITY = {"0000": "0.00", "1667": "0.17", "2308": "0.23", "2400": "0.24"}   # fmt_<code>_s<seed> -> share of delimiter tokens

# the supporting criteria, read on the surviving edge set: (key, label)
SUPPORTING = [
    ("multi_parented", "children with ≥ 2 parents"),
    ("dense_parents", "dense parents (≥ 30% of block)"),
    ("sibling_overlap", "sibling overlap (mean Jaccard)"),
    ("parent_coverage", "parent coverage R_supp (mean)"),
]


def _local_index(sel, n_total: int) -> torch.Tensor:
    """global id -> local column for one block, -1 outside it."""
    idx = torch.full((n_total,), -1, dtype=torch.long)
    if isinstance(sel, slice):
        lo, hi = sel.start, sel.stop
        idx[lo:hi] = torch.arange(hi - lo)
    else:
        idx[sel] = torch.arange(len(sel))
    return idx


def _probe_tristate(probe_edges, sel, n_total: int, shape) -> torch.Tensor:
    """[P, C] tristate of the probe rank rule from second_pass.json: 1 pass, 0 scored and
    failed, NaN not scored."""
    lp, lc = _local_index(sel(0), n_total), _local_index(sel(1), n_total)
    passed = torch.zeros(shape, dtype=torch.bool)
    scored = torch.zeros(shape, dtype=torch.bool)
    for e in probe_edges or []:
        i, j = int(lp[e["parent"]]), int(lc[e["child"]])
        if i >= 0 and j >= 0:
            scored[i, j] = True
            if e["pass"]:
                passed[i, j] = True
    return tristate(passed, scored), scored


def _frame(stats):
    """The [P, C] quantities every funnel reads, for block pair 0->1."""
    key = "0->1"
    fire = stats["fire_count"].double()
    total = int(stats["total_tokens"])
    sel = block_selector(stats)
    fire_p, fire_c = fire[sel(0)], fire[sel(1)]
    cofire = stats["cofire"][key].double()
    R, F = coverage_legs(cofire, fire_p, fire_c)
    m1 = keep_edges(R, fire_p, fire_c, C.EDGE_TAU, C.MIN_FIRE_COUNT, cofire=cofire, min_joint=C.MIN_JOINT)
    pmi = independence_scores(cofire, fire_p, fire_c, total, C.MIN_JOINT)["pmi"]
    recon = edge_reconstruction_condition(stats["err_sum_c"][1].double(), stats["g_parent_sum"][key].double(),
                                          stats["g_child_sum"][1].double(), C.RECON_REL_GAIN_MIN)
    return dict(key=key, fire=fire, total=total, sel=sel, fire_p=fire_p, fire_c=fire_c, cofire=cofire,
                R=R, F=F, m1=m1, pmi=pmi, recon=recon)


def _survival(stats, fr, mask):
    return frequency_controlled_coverage(stats["cofire_by_bucket"][fr["key"]].double(),
                                         stats["fire_c_by_bucket"][1].double(), mask,
                                         min_fire_low=C.FREQ_MIN_FIRE_LOW)["survival"]


def funnel_from_stats(stats, probe_edges, *, probe_all_pass=False, has_probe=True) -> dict:
    """Cumulative counts for block pair 0->1 from one cached-statistics file, plus the
    supporting criteria on the survivors."""
    fr = _frame(stats)
    m1, pmi = fr["m1"], fr["pmi"]
    m2 = m1 & (pmi >= CHANCE_PMI)                       # NaN compares False: unmeasurable leaves
    surv = _survival(stats, fr, m1)
    unm3 = int((m2 & torch.isnan(surv)).sum())
    m3 = m2 & (surv >= C.FREQ_SURVIVAL_MIN)
    m4 = m3 & fr["recon"]["passes"]
    out = {"counts": [int(m.sum()) for m in (m1, m2, m3, m4)],
           "unmeasurable": {STAGES[2]: unm3}}
    if not has_probe:
        out["counts"].append(None)
        out["supporting"] = supporting_on(stats, fr, m4)
        return out
    if probe_all_pass:
        m5, unm5 = m4, 0
    else:
        pt, scored = _probe_tristate(probe_edges, fr["sel"], fr["fire"].numel(), m4.shape)
        unm5 = int((m4 & ~scored).sum())
        m5 = m4 & (pt > 0.5)
    out["counts"].append(int(m5.sum()))
    out["unmeasurable"][STAGES[4]] = unm5
    out["supporting"] = supporting_on(stats, fr, m5)
    return out


def supporting_on(stats, fr, m: torch.Tensor) -> dict:
    """The supporting criteria read on one edge set `m` [P, C]: what the survivors look like
    from the neighbourhood, with no further cut."""
    n = int(m.sum())
    if n == 0:
        return {k: None for k, _ in SUPPORTING} | {"n_edges": 0}
    indeg = m.sum(dim=0)                                       # [C] parents per child
    outdeg = m.sum(dim=1)                                      # [P] children per parent
    kids = indeg > 0
    parents = outdeg > 0
    multi = float((indeg[kids] >= 2).double().mean())
    dense = float((outdeg[parents] >= C.SUPERPARENT_OUTDEG_FRAC * m.shape[1]).double().mean())
    sib = sibling_redundancy(m, stats["within_cofire"][1].double(), fr["fire_c"])
    sib_mean = statistics.mean(v["redundancy"] for v in sib.values()) if sib else None
    rs = r_supp(stats["union_count"][fr["key"]].double(), fr["fire_p"])
    rs_vals = rs[parents]
    rs_mean = float(rs_vals[~torch.isnan(rs_vals)].mean()) if rs_vals.numel() else None
    return {"n_edges": n, "multi_parented": multi, "dense_parents": dense,
            "sibling_overlap": sib_mean, "parent_coverage": rs_mean,
            "n_parents_with_2_children": len(sib)}


def rules_funnel_from_stats(stats, probe_edges, *, probe_all_pass=False, has_probe=True) -> dict:
    """`rule_hierarchy` clause by clause on block pair 0->1, through `metrics.rules`."""
    fr = _frame(stats)
    m1, R, F = fr["m1"], fr["R"], fr["F"]
    gains = fr["recon"]
    surv = _survival(stats, fr, torch.ones_like(m1))          # on every pair; the gate reads NaN as unmeasurable
    hp = high_outdegree(m1, fr["fire_p"], m1.shape[1], C.SUPERPARENT_OUTDEG_FRAC, C.MIN_FIRE_COUNT)
    hc = tristate(torch.zeros(m1.shape[1], dtype=torch.bool), torch.ones(m1.shape[1], dtype=torch.bool))
    ps = PairStats(cofire=fr["cofire"], fire_p=fr["fire_p"], fire_c=fr["fire_c"], R=R, R_rev=F,
                   parent_gain=gains["parent_gain"], child_gain=gains["child_gain"],
                   survival=surv, survival_scale="raw", high_outdeg_p=hp, high_outdeg_c=hc,
                   n_tokens=fr["total"])
    gates = score_pairs(ps, GEMMA_MATRYOSHKA)["gates"]
    if has_probe:
        if probe_all_pass:
            gates["gate_sres_rank"] = tristate(torch.ones_like(m1), torch.ones_like(m1))
        else:
            gates["gate_sres_rank"], _ = _probe_tristate(probe_edges, fr["sel"], fr["fire"].numel(), m1.shape)
    clauses = list(RULES["rule_hierarchy"]["clauses"])
    counts = []
    for k in range(1, len(clauses) + 1):
        mask, scorable, _ = evaluate(clauses[:k], gates)
        if not has_probe and clauses[k - 1][1] == "gate_sres_rank":
            counts.append(None)
        else:
            counts.append(int(mask.sum()))
    # baseline: `rule_containment` is clause 1 of the above; `keep_edges` is the core funnel's stage 1
    return {"counts": counts, "n_keep_edges": int(m1.sum()),
            "n_containment_rule": int(evaluate(RULES["rule_containment"]["clauses"], gates)[0].sum())}


def funnel_from_reports(report: dict, second: dict) -> dict:
    """Exact stages 1 and 2, an interval for 3 to 5, from the report files alone."""
    p = next(q for q in report["pairs"] if q["pair"] == "0->1")
    s = second["0->1"]["sres"]
    n1 = p["n_candidate_edges"]
    n2 = n1 - p["independence_null"]["n_chance_level"]
    fq, rc = p["freq_control"], p["reconstruction"]
    # strict: untestable and frequency-driven edges leave at stage 3; where they sit among
    # the stage-2 survivors is unknown, so stage 3 is an interval
    n3_hi = min(n2, n1 - fq["n_freq_driven"] - (n1 - fq["n_testable"]))
    n3_lo = max(0, n2 - fq["n_freq_driven"] - (n1 - fq["n_testable"]))
    n4_hi, n4_lo = min(n3_hi, rc["n_pass"]), max(0, n3_lo - (n1 - rc["n_pass"]))
    # the probe passes are counted over the shortlist; how many of them sit among the
    # stage-4 survivors is unknown, so the last stage is an interval as well
    n5_hi, n5_lo = min(s["n_pass"], n4_hi), max(0, s["n_pass"] - (n1 - n4_lo))
    return {"counts": [n1, n2, None, None, None],
            "interval": {STAGES[2]: [n3_lo, n3_hi], STAGES[3]: [n4_lo, n4_hi],
                         STAGES[4]: [n5_lo, n5_hi]},
            "unmeasurable": {STAGES[2]: n1 - fq["n_testable"],
                             STAGES[4]: s["n_shortlist_edges"] - s["n_edges_scored"]},
            "note": "stages 3 to 5 bounded from the report; statistics file not on disk"}


def _shares(f: dict, n_stages: int | None = None) -> list:
    n1 = f["counts"][0]
    return [None if c is None else c / n1 for c in f["counts"]]


def collect(toy_dir, tsae_toy_dirs, tsae_stats, tsae_json=None) -> dict:
    out, rules = {}, {}
    g = C.OUT_DIR / "gemma-2-2b" / "layer_12"
    st = torch.load(g / "exp0_stats.pt", weights_only=False)
    pe = _json(g / "second_pass.json")["0->1"]["sres"]["edges"]
    out["matryoshka_gemma"] = funnel_from_stats(st, pe)
    rules["matryoshka_gemma"] = rules_funnel_from_stats(st, pe)

    t = C.OUT_DIR / "gemma-2-2b-tsae" / "layer_12"
    if tsae_stats and Path(tsae_stats).exists():
        st = torch.load(tsae_stats, weights_only=False)
        pe = _json(t / "second_pass.json")["0->1"]["sres"]["edges"]
        out["tsae_gemma"] = funnel_from_stats(st, pe)
        rules["tsae_gemma"] = rules_funnel_from_stats(st, pe)
    elif tsae_json and Path(tsae_json).exists():
        # the same two functions, run on the node where the 3.5 GB statistics file lives
        j = _json(tsae_json)
        out["tsae_gemma"] = j["funnel"] | {"note": j.get("source", "computed on the node")}
        rules["tsae_gemma"] = j["rules"]
    else:
        out["tsae_gemma"] = funnel_from_reports(_json(t / "metrics_report.json"), _json(t / "second_pass.json"))

    if toy_dir:
        st = torch.load(Path(toy_dir) / "toy_stats.pt", weights_only=False)
        # the toy's probe pass comes from the Tier-2 calibration: 9 of 9 testable edges pass
        cal = (_json(C.OUT_DIR / "trained_toy_calibration.json") or {}).get("per_token") or {}
        all_pass = bool(cal) and cal.get("n_pass") == cal.get("n_testable")
        out["matryoshka_toy"] = funnel_from_stats(st, None, probe_all_pass=all_pass)
        rules["matryoshka_toy"] = rules_funnel_from_stats(st, None, probe_all_pass=all_pass)
    if tsae_toy_dirs:
        sts = [torch.load(Path(d) / "toy_stats.pt", weights_only=False) for d in tsae_toy_dirs]
        out["tsae_toy"] = {"runs": [funnel_from_stats(s, None, has_probe=False) for s in sts]}
        rules["tsae_toy"] = {"runs": [rules_funnel_from_stats(s, None, has_probe=False) for s in sts]}
    runs, rruns, by_density = [], [], {}
    for d in sorted((C.OUT_DIR / "pcfg-matryoshka").glob("fmt_*")):
        if (d / "exp0_stats.pt").exists() and (d / "second_pass.json").exists():
            st = torch.load(d / "exp0_stats.pt", weights_only=False)
            pe = _json(d / "second_pass.json")["0->1"]["sres"]["edges"]
            f = funnel_from_stats(st, pe)
            runs.append(f)
            rruns.append(rules_funnel_from_stats(st, pe))
            code = d.name.split("_")[1]                       # fmt_2400_s0 -> 2400
            by_density.setdefault(PCFG_DENSITY.get(code, f"0.{code}"), []).append(f)
    if runs:
        out["matryoshka_pcfg"] = {"runs": runs}
        rules["matryoshka_pcfg"] = {"runs": rruns}
        out["_pcfg_by_density"] = {k: {"runs": v} for k, v in sorted(by_density.items())}
    out["_rules"] = rules
    return out


def _series(entry: dict):
    """(mean shares, sd shares or None, interval shares or None) for one source."""
    if "runs" in entry:
        cols = list(zip(*[_shares(r) for r in entry["runs"]]))
        mean = [None if any(v is None for v in c) else statistics.mean(c) for c in cols]
        sd = [None if any(v is None for v in c) else (statistics.stdev(c) if len(c) > 1 else 0.0) for c in cols]
        return mean, sd, None
    sh = _shares(entry)
    iv = None
    if "interval" in entry:
        n1 = entry["counts"][0]
        iv = {STAGES.index(k): (lo / n1, hi / n1) for k, (lo, hi) in entry["interval"].items()}
        for i, (lo, hi) in iv.items():           # the dashed path runs through the midpoints
            sh[i] = (lo + hi) / 2
    return sh, None, iv


def _supporting_series(entry: dict):
    """{key: (mean, sd or None)} of the supporting criteria on the survivors."""
    if "runs" in entry:
        out = {}
        for k, _ in SUPPORTING:
            vals = [r["supporting"].get(k) for r in entry["runs"] if r.get("supporting")]
            vals = [v for v in vals if v is not None]
            out[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0) if vals else (None, None)
        return out
    s = entry.get("supporting") or {}
    return {k: (s.get(k), None) for k, _ in SUPPORTING}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#2B2B33", "#5A6B7B", "#E6E8EB"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#D8DBE0", "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    return plt, GRID


def _draw_funnel(ax, data, stages, GRID, *, key_filter=None):
    """One line per setting on `ax`. A multi-run setting is its mean with a ± sd band; a
    bounded setting is dashed with an interval bar at each bounded stage."""
    x = list(range(len(stages)))
    for key, lab, col in SOURCES:
        if key not in data or (key_filter and not key_filter(key)):
            continue
        mean, sd, iv = _series(data[key])
        pts = [(i, v) for i, v in enumerate(mean) if v is not None]
        if not pts:
            continue
        if iv:
            for i, (lo, hi) in iv.items():
                ax.plot([i, i], [lo, hi], color=col, lw=5, alpha=0.3, solid_capstyle="butt", zorder=2)
            exact = [p for p in pts if p[0] not in iv]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", color=col, lw=1.6, zorder=3)
            ax.scatter([p[0] for p in exact], [p[1] for p in exact], s=34, color=col, edgecolor="white",
                       linewidth=1.0, zorder=4)
        else:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if sd is not None:                      # mean ± sd over runs as a band
                lo = [max(0.0, m - s) for m, s in zip(ys, [sd[i] for i in xs])]
                hi = [min(1.0, m + s) for m, s in zip(ys, [sd[i] for i in xs])]
                ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0, zorder=1)
            ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=3)
            ax.scatter(xs, ys, s=34, color=col, edgecolor="white", linewidth=1.0, zorder=4)
            if len(pts) < len(stages):              # stops early: no probe for this setting
                ax.scatter([xs[-1]], [ys[-1]], s=60, facecolor="white", edgecolor=col, linewidth=1.6, zorder=5)
        xe, ye = pts[-1]
        label = f"{ye:.2f}" if not (iv and xe in iv) else f"{iv[xe][0]:.2f}–{iv[xe][1]:.2f}"
        dy = 9 if len(pts) < len(stages) else 0
        ax.annotate(label, (xe, ye), textcoords="offset points", xytext=(7, dy), va="center", fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([TICKS.get(s, s) for s in stages], fontsize=8.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.3, len(stages) - 0.4)
    ax.set_ylabel("share of candidate edges still confirmed", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _legend_label(key: str, lab: str, data: dict) -> str:
    base = SOURCE_SHORT.get(key, lab)
    e = data.get(key, {})
    if "interval" in e:
        return base + " (stages 3–5: interval)"
    if "runs" in e:
        return base + f" (mean ± sd, {len(e['runs'])} runs)"
    if e.get("counts") and e["counts"][-1] is None:
        return base + " (no probe)"
    return base


def write_figure(data: dict, out: Path):
    """Left: the core funnel. Right: the supporting criteria on the survivors."""
    plt, GRID = _style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [3.0, 1.9], "wspace": 0.55})
    _draw_funnel(ax, data, STAGES, GRID)
    handles = [plt.Line2D([], [], color=col, lw=1.8, marker="o", ms=5, label=_legend_label(k, lab, data))
               for k, lab, col in SOURCES if k in data]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    # right panel: one row per supporting criterion, one dot per setting; a multi-run setting
    # shows its mean with a ± sd whisker
    rows = list(range(len(SUPPORTING)))
    srcs = [(k, lab, col) for k, lab, col in SOURCES if k in data and (data[k].get("supporting") or "runs" in data[k])]
    n = max(1, len(srcs))
    for si, (key, lab, col) in enumerate(srcs):
        vals = _supporting_series(data[key])
        off = (si - (n - 1) / 2) * 0.13
        for r, (k, _) in enumerate(SUPPORTING):
            m, s = vals.get(k, (None, None))
            if m is None:
                continue
            y = r + off
            if s:
                ax2.plot([max(0, m - s), min(1, m + s)], [y, y], color=col, lw=1.2, alpha=0.6, zorder=2)
            ax2.scatter([m], [y], s=30, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    missing = [SOURCE_SHORT.get(k, lab) for k, lab, _ in SOURCES if k in data and k not in [s[0] for s in srcs]]
    ax2.set_yticks(rows)
    ax2.set_yticklabels([lab for _, lab in SUPPORTING], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_xlabel("value on the surviving edges", fontsize=8.5)
    ax2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("supporting criteria, read on\nthe edges that survive all five", fontsize=8.5, loc="left")
    if missing:
        ax2.text(0.0, -0.22, "not computed: " + ", ".join(missing) + " (statistics file not on disk)",
                 transform=ax2.transAxes, fontsize=7, color="#5A6B7B")
    path = out / "survival_funnel.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_pcfg_density_figure(data: dict, out: Path):
    """The PCFG funnel, one line per formatting density, mean ± sd over its seeds as a band."""
    bd = data.get("_pcfg_by_density")
    if not bd:
        return None
    plt, GRID = _style()
    RAMP = ["#F3B98C", "#E68A4F", "#C9601E", "#8A3D0B"]          # one hue, light -> dark with density
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = list(range(len(STAGES)))
    finals = []
    for col, (dens, entry) in zip(RAMP, bd.items()):
        mean, sd, _ = _series(entry)
        lo = [max(0.0, m - s) for m, s in zip(mean, sd)]
        hi = [min(1.0, m + s) for m, s in zip(mean, sd)]
        ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(x, mean, "-", color=col, lw=1.8, zorder=3, label=f"density {dens} (mean ± sd, {len(entry['runs'])} seeds)")
        ax.scatter(x, mean, s=34, color=col, edgecolor="white", linewidth=1.0, zorder=4)
        finals.append((mean[-1], col))
    step, ys = 0.04, []
    for v, col in sorted(finals):
        y = v if not ys else max(v, ys[-1] + step)
        ys.append(y)
        ax.annotate(f"{v:.2f}", (x[-1], v), xytext=(x[-1] + 0.12, y), textcoords="data", va="center", fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([TICKS.get(s, s) for s in STAGES], fontsize=8.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.3, len(STAGES) - 0.4)
    ax.set_ylabel("share of candidate edges still confirmed", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path = out / "survival_funnel_pcfg_density.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_rules_figure(data: dict, out: Path):
    """The funnel under `rule_hierarchy`, clause by clause."""
    rules = data.get("_rules") or {}
    if not rules:
        return None
    plt, GRID = _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _draw_funnel(ax, rules, RULE_STAGES, GRID)
    handles = [plt.Line2D([], [], color=col, lw=1.8, marker="o", ms=5, label=_legend_label(k, lab, rules))
               for k, lab, col in SOURCES if k in rules]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(f"rule_hierarchy, ruleset version {RULESET_VERSION}", fontsize=8.5, loc="left")
    fig.tight_layout()
    path = out / "survival_funnel_rules.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _count_cells(e: dict, stages: list) -> list[str]:
    if "runs" in e:
        cols = list(zip(*[r["counts"] for r in e["runs"]]))
        return ["n/a" if any(v is None for v in c) else f"{statistics.mean(c):.1f} ± {statistics.stdev(c) if len(c) > 1 else 0:.1f}" for c in cols]
    cells = []
    for i, c in enumerate(e["counts"]):
        if c is not None:
            cells.append(f"{c:,}")
        elif "interval" in e and stages[i] in e["interval"]:
            lo, hi = e["interval"][stages[i]]
            cells.append(f"[{lo:,}, {hi:,}]")
        else:
            cells.append("n/a")
    return cells


def _final(e: dict) -> str:
    mean, _, iv = _series(e)
    last = max(i for i, v in enumerate(mean) if v is not None)
    return f"{iv[last][0]:.3f}–{iv[last][1]:.3f}" if iv and last in iv else f"{mean[last]:.3f}"


def write_tables(data: dict, out: Path):
    lines = ["| Setting | " + " | ".join(STAGES) + " | final share |",
             "| --- | " + " | ".join("---:" for _ in STAGES) + " | ---: |"]
    for key, lab, _ in SOURCES:
        if key in data:
            lines.append(f"| {lab} | " + " | ".join(_count_cells(data[key], STAGES)) + f" | {_final(data[key])} |")
    for dens, e in (data.get("_pcfg_by_density") or {}).items():
        lines.append(f"| PCFG density {dens} ({len(e['runs'])} seeds) | " + " | ".join(_count_cells(e, STAGES)) + f" | {_final(e)} |")

    sup = ["| Setting | edges | " + " | ".join(lab for _, lab in SUPPORTING) + " |",
           "| --- | ---: | " + " | ".join("---:" for _ in SUPPORTING) + " |"]
    for key, lab, _ in SOURCES:
        if key not in data or not (data[key].get("supporting") or "runs" in data[key]):
            continue
        vals = _supporting_series(data[key])
        e = data[key]
        n = (f"{statistics.mean(r['supporting']['n_edges'] for r in e['runs']):.1f}" if "runs" in e
             else f"{e['supporting']['n_edges']:,}")
        cells = []
        for k, _ in SUPPORTING:
            m, s = vals[k]
            cells.append("n/a" if m is None else (f"{m:.3f} ± {s:.3f}" if s is not None else f"{m:.3f}"))
        sup.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")

    rl = ["| Setting | " + " | ".join(s.replace("\n", " ") for s in RULE_STAGES) + " | final share | keep_edges | rule_containment |",
          "| --- | " + " | ".join("---:" for _ in RULE_STAGES) + " | ---: | ---: | ---: |"]
    for key, lab, _ in SOURCES:
        e = (data.get("_rules") or {}).get(key)
        if not e:
            continue
        if "runs" in e:
            ke = f"{statistics.mean(r['n_keep_edges'] for r in e['runs']):.1f}"
            rc = f"{statistics.mean(r['n_containment_rule'] for r in e['runs']):.1f}"
        else:
            ke, rc = f"{e['n_keep_edges']:,}", f"{e['n_containment_rule']:,}"
        rl.append(f"| {lab} | " + " | ".join(_count_cells(e, RULE_STAGES)) + f" | {_final(e)} | {ke} | {rc} |")

    md = ("# Survival funnel, block pair 0->1\n\n"
          "Cumulative counts: edges confirmed by every core criterion up to that stage, in the order of the "
          "hierarchy rule. Unmeasurable edges leave the funnel at the stage that cannot measure them (counts in "
          "survival_funnel.json). A bracketed cell is an interval the reports bound when the statistics file is "
          "not on disk.\n\n" + "\n".join(lines) +
          "\n\n## Supporting criteria on the surviving edges\n\nRead on the edges that pass every core criterion "
          "(stage 5, or stage 4 where no probe was run). None of these is a cut.\n\n" + "\n".join(sup) +
          f"\n\n## The same funnel under `rule_hierarchy` (ruleset version {RULESET_VERSION})\n\nClause by clause: "
          "support and strict containment (R ≥ τ, R_rev < τ), PMI > 0, survival ≥ 0.5, probe rank. No "
          "reconstruction clause; PMI cut 0 rather than 0.5. `keep_edges` is the core funnel's stage 1 (R ≥ τ "
          "only); `rule_containment` is the rule's first clause.\n\n" + "\n".join(rl) + "\n")
    (out / "survival_funnel.md").write_text(md)
    (out / "survival_funnel.json").write_text(json.dumps(
        {"stages": STAGES, "rule_stages": [s.replace("\n", " ") for s in RULE_STAGES], "chance_pmi": CHANCE_PMI,
         "ruleset_version": RULESET_VERSION, "supporting": [k for k, _ in SUPPORTING], "data": data}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", type=Path, default=None, help="pipeline dir holding toy_stats.pt (Matryoshka toy)")
    ap.add_argument("--tsae-toy", type=Path, nargs="+", default=None, help="pipeline dirs, one per T-SAE toy seed")
    ap.add_argument("--tsae-stats", type=Path, default=None, help="exp0_stats.pt of the gemma T-SAE run")
    ap.add_argument("--tsae-json", type=Path, default=C.OUT_DIR / "gemma-2-2b-tsae" / "layer_12" / "funnel_from_node.json",
                    help="funnel entries computed on the node from that file (used when the file is absent)")
    ap.add_argument("--out", type=Path, default=PAPER_DIR)
    a = ap.parse_args()
    data = collect(a.toy, a.tsae_toy, a.tsae_stats, a.tsae_json)
    a.out.mkdir(parents=True, exist_ok=True)
    write_tables(data, a.out)
    pngs = [write_figure(data, a.out), write_pcfg_density_figure(data, a.out), write_rules_figure(data, a.out)]
    print((a.out / "survival_funnel.md").read_text())
    for p in pngs:
        if p:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
