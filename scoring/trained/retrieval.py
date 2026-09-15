"""
Score one trained checkpoint's property-vs-rest AUROC grid.

Matches features on the in-sample draw; scores detectors on a separate held-out draw so
reported numbers aren't scored on the matching data. Uses the `scoring.core.grid` toolkit;
this module wires a loaded checkpoint through it and writes the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from scoring.core.detectors import compute_all, s_res_variants
from scoring.core.grid import (
    auroc_matrix,
    class_members,
    dispersion_split,
    held_out_sample_seed,
    pair_frame,
    property_vs_rest_grid,
    random_scalar_control,
    redundancy_map,
    reduce_to_recovered,
    shuffled_label_control,
)
from scoring.core.registry import (
    CONSTANTS,
    LATENT_COLUMNS,
    POSITIVE_LABEL,
    SCORED_COLUMNS,
)
from scoring.trained.cascade import cascade_grid
from toygen import labels

# Only s_res mode allowed into a reported cell; the cosine geometry oracle is diagnostic-only.
_REPORT_S_RES_MODE = "probe"


def run_retrieval(ckpt_dir: str | Path, n_tokens: int = 200_000, rho: float | None = None,
                  out: str | Path | None = None, s_res_diagnostics: bool = False) -> dict:
    """Score one checkpoint's retrieval grid (matching on the in-sample draw, detectors on the held-out draw).

    Not unit-tested (loads a real checkpoint via `scoring.trained.loaders.load_sae`);
    validated on the server.
    """
    # Imports kept local: load_sae pulls in the heavy sae_training backend; keep this file
    # importable in the CPU test env.
    from scoring.trained.loaders import load_sae
    from scoring.core.world import regenerate_world, signed_normalized_decoder
    from scoring.core.recovery import (activation_corr, child_direction_dispersion,
                                       match_features, per_class_recovery, realized_l0)

    rho = CONSTANTS["rho_star"] if rho is None else rho
    loaded = load_sae(ckpt_dir)
    meta = loaded.meta
    W_raw = loaded.W_dec

    # 1. Match features on the in-sample draw.
    in_world = regenerate_world(meta["resolved_config"], sample_seed=meta["train_seed"],
                                n_tokens=n_tokens)
    acts_in = loaded.encode(in_world.h)
    oriented_in = signed_normalized_decoder(W_raw, acts_in, in_world.h)
    res = match_features(activation_corr(in_world.A, acts_in), in_world.g, oriented_in, rho=rho)

    # 2. HELD-OUT draw for the detector statistics (disjoint seed, same geometry).
    ho_seed = held_out_sample_seed(int(meta["train_seed"]))
    ho = regenerate_world(meta["resolved_config"], sample_seed=ho_seed, n_tokens=n_tokens)
    acts_ho = loaded.encode(ho.h)
    oriented_ho = signed_normalized_decoder(W_raw, acts_ho, ho.h)
    # Realized L0 on the HELD-OUT draw — reported provenance for the sparsity the detectors see.
    ho_realized_l0 = realized_l0(acts_ho)

    feats, di, _ = reduce_to_recovered(
        acts_ho, oriented_ho, W_raw, res.match, res.recovered,
        h=ho.h, b_dec=loaded.b_dec, tokens=ho.tokens, vocab=ho.cfg.vocab)
    # Trained detector uses the real probe s_res (one probe per recovered child); cosine oracle is diagnostic-only.
    detectors = compute_all(di, CONSTANTS, s_res_mode=_REPORT_S_RES_MODE)
    pairs, y_label = pair_frame(feats, ho.pair_labels)

    # Latent-side labels (absorbed/merged/split) from the dictionary-damage classification on the
    # in-sample match; scoring-only, firewalled from the detectors.
    from scoring.trained.absorption import (classify_dictionary, latent_pair_masks, split_readout,
                                            tree_edges_and_siblings)
    cont_edges, sibling_pairs, isa_child, descendants = tree_edges_and_siblings(in_world.tree)
    classification = classify_dictionary(in_world.g, oriented_in, in_world.A, acts_in,
                                         res.match, res.matched_corr, res.recovered,
                                         cont_edges, sibling_pairs, isa_child, descendants)
    latent_masks = latent_pair_masks(classification, feats, pairs)

    # Property-vs-rest grid over every generative class + latent columns, scored by the firewalled
    # detectors; orientation stays frozen (AUROC<0.5 is a reported inverted isolator).
    grid_columns = tuple(labels.LABELS) + LATENT_COLUMNS
    grid = property_vs_rest_grid(detectors, pairs, y_label, grid_columns, label_masks=latent_masks)
    per_class_rec = per_class_recovery(res.recovered, ho.pair_labels, list(labels.LABELS))

    # Nuisance floor: a pure firing-count proxy already clears chance on some columns, so the bar a
    # detector must beat there is this floor, not 0.5.
    _fire = (di.acts_rec > 0).double().sum(0)                        # [R] firing count per latent
    _R = int(_fire.numel())

    def _row_const(v: torch.Tensor) -> torch.Tensor:                 # child_fire[p,c] = fire[c]
        m = v.reshape(1, _R).expand(_R, _R).clone(); m.fill_diagonal_(float("nan")); return m

    def _col_const(v: torch.Tensor) -> torch.Tensor:                 # parent_fire[p,c] = fire[p]
        m = v.reshape(_R, 1).expand(_R, _R).clone(); m.fill_diagonal_(float("nan")); return m

    _fire_grid = property_vs_rest_grid(
        {"child_fire": _row_const(_fire), "parent_fire": _col_const(_fire)},
        pairs, y_label, grid_columns, label_masks=latent_masks)

    def _floor(a) -> float:
        return max(a, 1.0 - a) if isinstance(a, float) and math.isfinite(a) else float("nan")

    nuisance_baselines = {
        col: {b: _floor(_fire_grid[b][col].get("auroc")) for b in ("child_fire", "parent_fire")}
        for col in grid_columns}

    # Greedy both-tails Boolean cascade per column; is_a's rule is the deployment cascade. 0-positive
    # columns are skipped, not crashed.
    cascade = cascade_grid(detectors, pairs, y_label, grid_columns, label_masks=latent_masks)

    # Dispersion-conditioned is-a readout over the held-out dictionary; a truth-g diagnostic, never a detector.
    r_disp_vec = child_direction_dispersion(oriented_ho, ho.g, res.match, feats,
                                            m=CONSTANTS["r_disp_m"])
    isa_m = class_members(y_label, POSITIVE_LABEL).tolist()
    fo_m = class_members(y_label, "firing_only").tolist()
    isa_pairs = [pr for pr, inside in zip(pairs, isa_m) if inside]
    fo_pairs = [pr for pr, inside in zip(pairs, fo_m) if inside]
    r_disp = {b: float(r_disp_vec[b]) for _, b in isa_pairs}   # b == child position == feats index
    disp = (dispersion_split(isa_pairs, r_disp, detectors["s_res"], neg_pairs=fo_pairs)
            if isa_pairs and fo_pairs else None)
    # Mean over is-a children only, not all recovered features, matching the recovery driver's dispersion_mean.
    dispersion_mean = float(sum(r_disp.values()) / len(r_disp)) if r_disp else float("nan")

    rng = torch.Generator().manual_seed(0)
    controls = {
        "random_scalar_firing_only": random_scalar_control(pairs, y_label, "firing_only", rng),
        "shuffled_s_res_firing_only": shuffled_label_control(
            detectors["s_res"], pairs, y_label, "firing_only", rng),
    }
    report = {
        "checkpoint": str(ckpt_dir),
        "meta": {k: meta[k] for k in ("config", "variant", "k", "train_seed", "overrides")
                 if k in meta},
        "n_recovered": len(feats), "n_pairs": len(pairs),
        "ci_note": ("Assumes independent pairs, which is optimistic since pairs come from only ~F "
                    "features — these intervals are too narrow. Use the across-seed Student-t from "
                    "aggregate_seeds instead."),
        "grid_note": ("Property-vs-rest: each column's class is positives, every other off-diagonal "
                      "pair is negatives, scored by the 10 firewalled detectors. Orientation is "
                      "frozen, so AUROC below 0.5 means a reversed isolator, reported as-is."),
        "grid": grid, "per_class_recovery": per_class_rec,
        "nuisance_baselines": nuisance_baselines,
        "nuisance_note": ("The AUROC a pure firing-count baseline already reaches per column. Some "
                          "columns floor at 0.85-0.90, so a detector must beat this floor, not 0.5, "
                          "to count as informative."),
        "cascade": cascade,
        "cascade_note": ("Greedy forward-selection of percentile filters that maximize F1 for the "
                         "column among survivors. The is_a rule is the deployment cascade, with "
                         "enrichment and hard-negative precision added. Columns with no positives "
                         "are skipped."),
        "latent_columns": list(LATENT_COLUMNS),
        # absorption `counts` not re-reported here (run_absorption is the source of truth);
        # split_readout is reported since it's scoped to the recovered `feats`.
        "split_readout": split_readout(classification, feats),
        "latent_firewall_note": ("absorbed/merged/split are labels from ground truth and the trained "
                                 "dictionary, used only for scoring — the 10 detectors never see "
                                 "them."),
        "realized_l0": ho_realized_l0, "architecture": loaded.arch,
        "dispersion_mean": dispersion_mean,
        "dispersion_split": disp,
        "redundancy": redundancy_map(detectors, pairs, y_label, SCORED_COLUMNS, grid=grid),
        "redundancy_note": ("marginal_auroc is each detector's added column-vs-rest AUROC from the "
                            "grid; rank_corr is a separate detector-to-detector percentile "
                            "correlation."),
        "controls": controls,
    }
    # J(p) fallback-vs-exact diagnostic on joint_child_J's accuracy; not a scored detector.
    from scoring.trained.diagnostics import j_fallback_vs_exact
    report["j_supp_diagnostic"] = j_fallback_vs_exact(di, CONSTANTS)

    # Opt-in s_res 3-variant diagnostic (trains extra probes); uses ground-truth g/A, so it's diagnostic-only.
    if s_res_diagnostics:
        _TINY = 1e-12
        idx = torch.tensor(feats, dtype=torch.long)
        g_sel = ho.g[idx]
        g_unit = g_sel / g_sel.norm(dim=1, keepdim=True).clamp_min(_TINY)
        variants = s_res_variants(di.acts_rec, ho.h, di.W_unit, g_unit, ho.A[:, idx], CONSTANTS)

        def _isa_row(mat: torch.Tensor) -> dict:
            # is_a-vs-column diagnostic (auroc_matrix is is_a-locked), over the confound columns.
            row = auroc_matrix({"s": mat}, pairs, y_label, SCORED_COLUMNS)["s"]
            return {c: row.get(c, {}).get("auroc") for c in SCORED_COLUMNS}

        aurocs = {name: _isa_row(mat) for name, mat in variants.items()}

        def _fin(x) -> bool:
            return isinstance(x, float) and math.isfinite(x)
        bias = {c: (aurocs["probe_self_W"][c] - aurocs["probe_true_W"][c])
                if (_fin(aurocs["probe_self_W"][c]) and _fin(aurocs["probe_true_W"][c]))
                else float("nan") for c in SCORED_COLUMNS}
        report["s_res_diagnostics"] = {
            "auroc": aurocs, "self_label_bias": bias,
            "note": ("cosine_g is the analytic geometry oracle; probe_true_g calibrates the probe; "
                     "probe_self_W is the deployed detector; self_label_bias isolates the "
                     "self-label circularity."),
        }
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Retrieval-grid scorer for a toy SAE checkpoint.")
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--s-res-diagnostics", action="store_true",
                    help="also emit the s_res 3-variant diagnostic (trains extra probes)")
    args = ap.parse_args()
    report = run_retrieval(args.ckpt, n_tokens=args.n_tokens, out=args.out,
                           s_res_diagnostics=args.s_res_diagnostics)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
