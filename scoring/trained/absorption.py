"""Classify a dictionary's damage to the true features: absorption, split or duplicate firing, composition.

Reads ground truth (`g`, `A`, the match, the tree) but never `pair_labels`.

    python -m scoring.trained.absorption <ckpt_dir> [--n-tokens N] [--out report.json]
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from scoring.core.recovery import activation_corr

DT = torch.float64
_TINY = 1e-12

ABSORPTION_CONSTANTS: dict[str, float] = {
    "hole_min": 0.15,        # parent recall must drop this far below 1 to count as a hole
    "solo_min": 0.10,        # min parent recall on parent-solo tokens; near 0 means merging
    "conj_min": 0.10,        # min conjunction cos K above baseline to count as composition
    "null_target_exceedances": 0.01,  # Bonferroni target on expected chance latents dictionary-wide
    "n_null_perm": 1000,     # random in-span directions for a stable tail quantile
    "multiplicity_excess_min": 1.5,   # flag threshold for parent_multiplicity_excess; nothing here reads it
    # --- firing multiplicity (split vs duplicate on exclusive support) ---
    "mult_prec_min": 0.5,    # candidate latent needs P(feature | latent) >= this
    "mult_recall_min": 0.25, # a shard recalls at least this much of the exclusive support
    "dup_recall_min": 0.8,   # one shard recalling this much is a duplicate, not a split
    "split_union_min": 0.8,  # >= 2 shards must jointly recall this much to be a split
}


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(_TINY)


def _decoder_span_basis(W_dec: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis [r, D] of the decoders' row space.

    Null directions are drawn here, not in all of R^D, since decoders concentrate in this span."""
    Wu = _unit(W_dec.double())
    _, S, Vh = torch.linalg.svd(Wu, full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum()) if S.numel() else 0
    return Vh[:max(r, 1)]


def null_cos_threshold(W_dec: torch.Tensor, g: torch.Tensor, n_perm: int = 200,
                       q: float = 0.95, seed: int = 0) -> float:
    """eps: the q-quantile of |cos| between random in-span unit directions and the decoders.

    Seeded, so deterministic. `g` is unused."""
    basis = _decoder_span_basis(W_dec)                     # [r, D]
    Wu = _unit(W_dec.double())
    r = basis.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n_perm, r, generator=gen, dtype=DT)    # [n_perm, r]
    V = _unit(z @ basis)                                   # [n_perm, D]
    null_cos = (V @ Wu.transpose(0, 1)).abs()              # [n_perm, d_sae]
    return float(torch.quantile(null_cos.flatten(), q))


def _expected_null_count(W_dec: torch.Tensor, eps: float, n_perm: int = 200,
                         seed: int = 1) -> float:
    """Expected number of decoders with |cos| > eps against a random in-span direction."""
    basis = _decoder_span_basis(W_dec)
    Wu = _unit(W_dec.double())
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n_perm, basis.shape[0], generator=gen, dtype=DT)
    V = _unit(z @ basis)
    counts = ((V @ Wu.transpose(0, 1)).abs() > eps).double().sum(dim=1)   # [n_perm]
    return float(counts.mean())


# --- firing multiplicity (split vs duplicate on exclusive support) ---
def firing_precision(A: torch.Tensor, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(fire_lat [n, S] bool, prec_all [S, F])` with `prec_all[L, f] = P(f fires | L fires)`."""
    fire_lat = acts > 0
    fld = fire_lat.double()
    lat_fire = fld.sum(0).clamp_min(1.0)                                       # [S]
    prec_all = (fld.transpose(0, 1) @ (A > 0).double()) / lat_fire[:, None]    # [S, F] = P(f | L)
    return fire_lat, prec_all


def latent_owner(A: torch.Tensor, acts: torch.Tensor, descendants: dict[int, set[int]] | None,
                 constants: dict, prec_all: torch.Tensor | None = None) -> torch.Tensor:
    """Per latent, the most specific feature it fires mostly within (`P(f | L) >= mult_prec_min`).

    Nested firing gives ancestors the higher precision, so a feature loses to a qualifying descendant."""
    if prec_all is None:
        _, prec_all = firing_precision(A, acts)
    F = int(A.shape[1])
    cand = (prec_all >= constants["mult_prec_min"]).double()                   # [S, F]
    desc_adj = torch.zeros(F, F, dtype=prec_all.dtype)                         # desc_adj[f, d]=1 iff d descends f
    for fparent, ds in (descendants or {}).items():
        for d in ds:
            if 0 <= int(fparent) < F and 0 <= int(d) < F:
                desc_adj[int(fparent), int(d)] = 1.0
    dominated = (cand @ desc_adj.transpose(0, 1)) > 0                          # [S, F]: f has a candidate descendant
    eligible = (cand > 0) & (~dominated)
    scored = torch.where(eligible, prec_all, torch.full_like(prec_all, -1.0))
    return scored.argmax(dim=1)                                               # [S]


def firing_multiplicity(A: torch.Tensor, acts: torch.Tensor,
                        descendants: dict[int, set[int]] | None,
                        f: int, constants: dict, owner: torch.Tensor | None = None,
                        prec_all: torch.Tensor | None = None,
                        fire_lat: torch.Tensor | None = None) -> dict:
    """Label feature `f` split, duplicate, clean or unclassified by shard recall on its exclusive support.

    Exclusive support is where `f` fires and no descendant does; when it is empty, `f` is unclassified."""
    if fire_lat is None or prec_all is None:
        fire_lat, prec_all = firing_precision(A, acts)
    if owner is None:
        owner = latent_owner(A, acts, descendants, constants, prec_all=prec_all)
    fire = A > 0
    fire_f = fire[:, f]
    desc = [d for d in descendants.get(f, set()) if d != f] if descendants else []
    excl = (fire_f & ~fire[:, desc].any(dim=1)) if desc else fire_f
    n_excl = int(excl.sum())
    if n_excl == 0:
        return {"feature": int(f), "kind": "unclassified", "shards": [], "n_shards": 0,
                "best_recall": float("nan"), "union_recall": float("nan"), "n_excl": 0}
    cand = ((prec_all[:, f] >= constants["mult_prec_min"]) & (owner == f)).nonzero(as_tuple=True)[0].tolist()
    recalls = {L: float(fire_lat[excl, L].double().mean()) for L in cand}
    shards = [L for L, r in recalls.items() if r >= constants["mult_recall_min"]]
    if len(shards) < 2:
        return {"feature": int(f), "kind": "clean", "shards": shards, "n_shards": len(shards),
                "best_recall": (max(recalls[L] for L in shards) if shards else float("nan")),
                "union_recall": float("nan"), "n_excl": n_excl}
    best = max(recalls[L] for L in shards)
    union = torch.zeros(A.shape[0], dtype=torch.bool, device=A.device)
    for L in shards:
        union = union | fire_lat[:, L]
    union_recall = float(union[excl].double().mean())
    kind = ("duplicate" if best >= constants["dup_recall_min"]
            else "split" if union_recall >= constants["split_union_min"] else "clean")
    return {"feature": int(f), "kind": kind, "shards": [int(L) for L in shards],
            "n_shards": len(shards), "best_recall": best, "union_recall": union_recall,
            "n_excl": n_excl}


# --- decoder multiplicity over null (geometry diagnostic, not used by classify_dictionary) ---
def parent_multiplicity_excess(g: torch.Tensor, W_dec: torch.Tensor, f: int, eps: float) -> dict:
    """`M_P` counts latents with |cos| > eps to `g[f]` whose best feature is `f`; `excess` subtracts the null.

    The best-match condition keeps a parent's latent out of every correlated child's count."""
    Wu = _unit(W_dec.double())
    cosmat = (_unit(g.double()) @ Wu.transpose(0, 1)).abs()   # [F, d_sae]
    best_f = cosmat.argmax(dim=0)                             # [d_sae]
    M_P = int(((cosmat[f] > eps) & (best_f == f)).sum())
    en = _expected_null_count(W_dec, eps)
    return {"M_P": M_P, "expected_null": en, "excess": M_P - en}


# --- Chanin absorption ---
def absorption_signals(g: torch.Tensor, W_dec: torch.Tensor, acts: torch.Tensor, A: torch.Tensor,
                       match: torch.Tensor, p: int, c: int, eps: float, constants: dict) -> dict:
    """Chanin absorption on edge p -> c: `resid_parent > eps`, a parent firing hole, and `R_solo > solo_min`.

    `R_solo` near 0 means a merging latent, not absorption. An unmatched endpoint gives `absorbed=False`."""
    mc, mp = int(match[c]), int(match[p])
    if mc < 0 or mp < 0:
        return {"parent_component": float("nan"), "child_component": float("nan"),
                "resid_parent": float("nan"), "theta_hat": float("nan"), "R_P": float("nan"),
                "R_solo": float("nan"), "hole": False, "absorbed": False}
    gp_u = _unit(g[p].double())
    gc_u = _unit(g[c].double())
    dec_c = _unit(W_dec[mc].double())
    parent_component = float(dec_c @ gp_u)
    child_component = float(dec_c @ gc_u)
    theta_hat = math.atan2(parent_component, child_component)
    # parent direction with the child's designed overlap removed, so a clean is-a child reads 0
    resid_dir = gp_u - (gp_u @ gc_u) * gc_u
    if float(resid_dir.norm()) < _TINY:
        resid_parent = 0.0
    else:
        resid_parent = float(dec_c @ (resid_dir / resid_dir.norm().clamp_min(_TINY)))
    child_fires = A[:, c].double() > 0
    n_cf = int(child_fires.sum())
    if n_cf == 0:
        R_P = float("nan"); hole = False
    else:
        R_P = float((acts[child_fires, mp].double() > 0).double().mean())   # parent recall
        hole = R_P < (1.0 - constants["hole_min"])
    parent_solo = (A[:, p].double() > 0) & (~child_fires)
    n_solo = int(parent_solo.sum())
    R_solo = (float((acts[parent_solo, mp].double() > 0).double().mean())
              if n_solo > 0 else float("nan"))
    solo_ok = math.isfinite(R_solo) and R_solo > constants["solo_min"]
    absorbed = bool(resid_parent > eps and hole and solo_ok)
    return {"parent_component": parent_component, "child_component": child_component,
            "resid_parent": resid_parent, "theta_hat": theta_hat, "R_P": R_P,
            "R_solo": R_solo, "hole": hole, "absorbed": absorbed}


# --- composition (conjunction latent) ---
def conjunction_strength(g: torch.Tensor, W_dec: torch.Tensor, a: int, b: int,
                         match: torch.Tensor | None = None, conj_min: float | None = None) -> dict:
    """`K = cos(W_dec[j*], unit(g_a + g_b))` minus the larger single-feature cos; composed if `K > conj_min`.

    `j*` must sit closer to the sum than to either feature and, given `match`, not be either one's own latent."""
    Wu = _unit(W_dec.double())
    ga, gb = _unit(g[a].double()), _unit(g[b].double())
    conj = _unit(g[a].double() + g[b].double())
    cos_conj = Wu @ conj                                   # [d_sae], signed
    baseline = max(float(conj @ ga), float(conj @ gb))
    cmin = ABSORPTION_CONSTANTS["conj_min"] if conj_min is None else float(conj_min)

    eligible = (cos_conj > (Wu @ ga)) & (cos_conj > (Wu @ gb))   # sum-aligned, not single-aligned
    if match is not None:
        for x in (int(match[a]), int(match[b])):
            if 0 <= x < eligible.numel():
                eligible[x] = False
    if not bool(eligible.any()):
        return {"K": float("-inf"), "latent": -1, "baseline": baseline, "composed": False}
    cand = torch.where(eligible, cos_conj, torch.full_like(cos_conj, float("-inf")))
    jstar = int(torch.argmax(cand))
    K = float(cos_conj[jstar]) - baseline
    return {"K": K, "latent": jstar, "baseline": baseline, "composed": bool(K > cmin)}


# --- orchestration ---
def classify_dictionary(g: torch.Tensor, W_dec: torch.Tensor, A: torch.Tensor, acts: torch.Tensor,
                        match: torch.Tensor, matched_corr: torch.Tensor, recovered: torch.Tensor,
                        cont_edges: Sequence[tuple[int, int]],
                        sibling_pairs: Sequence[tuple[int, int]],
                        isa_child: dict[int, bool],
                        descendants: dict[int, set[int]] | None = None,
                        constants: dict | None = None) -> dict:
    """Absorption per containment edge, split or duplicate firing per feature, composition per sibling pair.

    Mechanisms can coexist on one feature. Structure comes from the tree; `pair_labels` is not an input."""
    constants = ABSORPTION_CONSTANTS if constants is None else constants
    F = int(g.shape[0])
    d_sae = int(W_dec.shape[0])
    # eps is a Bonferroni quantile over the whole dictionary. Known limit: conservative for one
    # edge, so some shallow absorbed edges read clean.
    q_eff = 1.0 - constants["null_target_exceedances"] / max(d_sae, 1)
    eps = null_cos_threshold(W_dec, g, n_perm=int(constants["n_null_perm"]), q=q_eff)
    expected_null = _expected_null_count(W_dec, eps, n_perm=int(constants["n_null_perm"]))

    absorbed_edges: list[dict] = []
    absorbed_children: set[int] = set()
    # edges whose absorption gates could not be evaluated; kept out of clean
    unclassified_edges: list[dict] = []
    unclassified_children: set[int] = set()
    # edges with an unrecovered endpoint; kept out of every bucket and reported separately
    below_rho_edges: list[dict] = []
    below_rho_children: set[int] = set()
    for (p, c) in cont_edges:
        if not (bool(recovered[int(p)]) and bool(recovered[int(c)])):
            below_rho_children.add(int(c))
            below_rho_edges.append({"parent": int(p), "child": int(c),
                                    "is_a": bool(isa_child.get(int(c), False))})
            continue
        sig = absorption_signals(g, W_dec, acts, A, match, int(p), int(c), eps, constants)
        if sig["absorbed"]:
            absorbed_children.add(int(c))
            absorbed_edges.append({"parent": int(p), "child": int(c),
                                   "theta_hat": sig["theta_hat"], "R_P": sig["R_P"],
                                   "is_a": bool(isa_child.get(int(c), False))})
        elif not math.isfinite(sig["R_P"]) or not math.isfinite(sig["R_solo"]):
            reason = ("no_child_fire" if not math.isfinite(sig["R_P"]) else "no_parent_solo")
            unclassified_children.add(int(c))
            unclassified_edges.append({"parent": int(p), "child": int(c), "reason": reason,
                                       "is_a": bool(isa_child.get(int(c), False))})
    # an edge absorbed via one parent overrides an unclassified/below-rho edge via another
    unclassified_children -= absorbed_children
    below_rho_children -= (absorbed_children | unclassified_children)

    descendants = {} if descendants is None else descendants
    fire_lat, prec_all = firing_precision(A, acts)
    owner = latent_owner(A, acts, descendants, constants, prec_all=prec_all)
    multiplicity_features: list[dict] = []      # splits (fragmented firing)
    duplicate_features: list[dict] = []         # redundant copies
    # recovered features with no exclusive support; kept out of clean
    multiplicity_unclassified: set[int] = set()
    for f in range(F):
        if not bool(recovered[f]):
            continue
        fm = firing_multiplicity(A, acts, descendants, f, constants,
                                 owner=owner, prec_all=prec_all, fire_lat=fire_lat)
        if fm["kind"] == "split":
            multiplicity_features.append(fm)
        elif fm["kind"] == "duplicate":
            duplicate_features.append(fm)
        elif fm["kind"] == "unclassified":
            multiplicity_unclassified.add(f)

    composed_pairs: list[dict] = []
    for (a, b) in sibling_pairs:
        if not (bool(recovered[int(a)]) and bool(recovered[int(b)])):
            continue
        cs = conjunction_strength(g, W_dec, int(a), int(b), match=match,
                                  conj_min=constants["conj_min"])
        if cs["composed"]:
            composed_pairs.append({"a": int(a), "b": int(b), "K": cs["K"], "latent": cs["latent"]})

    # A feature counted as split, duplicate, or not-evaluable for multiplicity is not clean.
    multiplicity_set = ({d["feature"] for d in multiplicity_features}
                        | {d["feature"] for d in duplicate_features}
                        | multiplicity_unclassified)
    clean = int(sum(1 for f in range(F)
                    if bool(recovered[f]) and f not in absorbed_children
                    and f not in multiplicity_set and f not in unclassified_children))
    return {
        "eps": eps, "expected_null": expected_null,
        "absorbed_edges": absorbed_edges,
        "decoder_multiplicity": multiplicity_features,
        "duplicate_features": duplicate_features,
        "multiplicity_unclassified": sorted(multiplicity_unclassified),
        "composed_pairs": composed_pairs,
        "unclassified_edges": unclassified_edges,
        "below_rho_edges": below_rho_edges,
        "counts": {"absorbed": len(absorbed_children),
                   "decoder_multiplicity": len(multiplicity_features),
                   "duplicate": len(duplicate_features),
                   "multiplicity_unclassified": len(multiplicity_unclassified),
                   "composed": len(composed_pairs), "clean": clean,
                   "unclassified": len(unclassified_children),
                   "below_rho": len(below_rho_children)},
        "absorbed_by_relation": {
            "is_a": sum(1 for e in absorbed_edges if e["is_a"]),
            "firing_only": sum(1 for e in absorbed_edges if not e["is_a"]),
        },
        "unclassified_by_relation": {
            "is_a": sum(1 for e in unclassified_edges if e["is_a"]),
            "firing_only": sum(1 for e in unclassified_edges if not e["is_a"]),
        },
    }


def latent_pair_masks(classification: dict, feats: list[int],
                      pairs: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Bool masks over `pairs` (positions into `feats`): `absorbed` per ordered edge, `merged` in both orders."""
    absorbed_set = {(int(e["parent"]), int(e["child"])) for e in classification["absorbed_edges"]}
    merged_set: set[tuple[int, int]] = set()
    for e in classification["composed_pairs"]:
        a, b = int(e["a"]), int(e["b"])
        merged_set.add((a, b))
        merged_set.add((b, a))
    ids = [(feats[a], feats[b]) for (a, b) in pairs]
    # CPU masks; a CUDA caller must move them
    absorbed = torch.tensor([pid in absorbed_set for pid in ids], dtype=torch.bool)
    merged = torch.tensor([pid in merged_set for pid in ids], dtype=torch.bool)
    return {"absorbed": absorbed, "merged": merged}


def split_readout(classification: dict, feats: list[int]) -> dict:
    """The features in `feats` whose firing splits across >= 2 shards; per feature, so not a pair column."""
    recovered_ids = {int(f) for f in feats}
    out = [{"feature": int(d["feature"]), "n_shards": int(d["n_shards"]),
            "union_recall": float(d["union_recall"]), "n_excl": int(d["n_excl"])}
           for d in classification["decoder_multiplicity"] if int(d["feature"]) in recovered_ids]
    return {"n_split": len(out), "features": out}


def tree_edges_and_siblings(tree) -> tuple[list[tuple[int, int]], list[tuple[int, int]],
                                           dict[int, bool], dict[int, set[int]]]:
    """`(cont_edges, sibling_pairs, isa_child, descendants)` from the tree.

    `cont_edges` are ordered (parent, child); `isa_child[c]` means alpha > 0; `sibling_pairs` are unordered."""
    cont_edges = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa_child = {c: (tree.alpha_of(c) > 0.0) for _, c in cont_edges}
    sibling_pairs = [(kids[i], kids[j]) for kids in tree.children.values()
                     for i in range(len(kids)) for j in range(i + 1, len(kids))]
    return cont_edges, sibling_pairs, isa_child, tree.descendents


def run_absorption(ckpt_dir, n_tokens: int = 200_000, rho: float | None = None) -> dict:
    """Load a checkpoint (needs sae_training), match it on training-seed tokens, and classify its damage."""
    from scoring.trained.loaders import load_sae
    from scoring.core.world import regenerate_world, signed_normalized_decoder
    from scoring.core.registry import CONSTANTS
    from scoring.core.recovery import match_features

    rho = CONSTANTS["rho_star"] if rho is None else rho
    loaded = load_sae(ckpt_dir)
    meta = loaded.meta
    world = regenerate_world(meta["resolved_config"], sample_seed=meta["train_seed"],
                             n_tokens=n_tokens)
    acts = loaded.encode(world.h)
    oriented = signed_normalized_decoder(loaded.W_dec, acts, world.h)
    res = match_features(activation_corr(world.A, acts), world.g, oriented, rho=rho)

    cont_edges, sibling_pairs, isa_child, descendants = tree_edges_and_siblings(world.tree)
    report = classify_dictionary(world.g, oriented, world.A, acts, res.match, res.matched_corr,
                                 res.recovered, cont_edges, sibling_pairs, isa_child, descendants)
    report["meta"] = {k: meta[k] for k in ("config", "variant", "k", "train_seed", "overrides")
                      if k in meta}
    report["n_recovered"] = int(res.recovered.sum())
    return report


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Chanin absorption / splitting / composition decomposition.")
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--n-tokens", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = run_absorption(args.ckpt, n_tokens=args.n_tokens)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
