"""Classify a dictionary's pathologies against the true features: absorption, split or duplicate
firing, and composition. Hedging is not classified.

Reads ground truth (`g`, `A`, the match, the tree) but never `pair_labels`. The synthdict census
runs it.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from metrics.rules import SYNTHETIC_TOYS
from scoring.config import PATHOLOGY, PathologySettings

DT = torch.float64
_TINY = 1e-12


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(_TINY)


def _decoder_span_basis(W_dec: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis [r, D] of the decoders' row space.

    Null directions are drawn here, not in all of R^D, since decoders concentrate in this span."""
    Wu = _unit(W_dec.double())
    _, S, Vh = torch.linalg.svd(Wu, full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum()) if S.numel() else 0
    return Vh[:max(r, 1)]


def null_cos_threshold(W_dec: torch.Tensor, g: torch.Tensor, n_perm: int, q: float,
                       seed: int = 0) -> float:
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


def _expected_null_count(W_dec: torch.Tensor, eps: float, n_perm: int, seed: int = 1) -> float:
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
    fire_lat = acts > SYNTHETIC_TOYS.fire_threshold
    fld = fire_lat.double()
    lat_fire = fld.sum(0).clamp_min(1.0)                                       # [S]
    prec_all = (fld.transpose(0, 1) @ (A > 0).double()) / lat_fire[:, None]    # [S, F] = P(f | L)
    return fire_lat, prec_all


def latent_owner(A: torch.Tensor, acts: torch.Tensor, descendants: dict[int, set[int]] | None,
                 *, pathology: PathologySettings = PATHOLOGY,
                 prec_all: torch.Tensor | None = None) -> torch.Tensor:
    """Per latent, the most specific feature it fires mostly within (`P(f | L) >= mult_prec_min`).

    Nested firing gives ancestors the higher precision, so a feature loses to a qualifying descendant."""
    if prec_all is None:
        _, prec_all = firing_precision(A, acts)
    F = int(A.shape[1])
    cand = (prec_all >= pathology.mult_prec_min).double()                      # [S, F]
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
                        f: int, *, pathology: PathologySettings = PATHOLOGY,
                        owner: torch.Tensor | None = None,
                        prec_all: torch.Tensor | None = None,
                        fire_lat: torch.Tensor | None = None) -> dict:
    """Label feature `f` split, duplicate, clean or unclassified by shard recall on its exclusive support.

    Exclusive support is where `f` fires and no descendant does; when it is empty, `f` is unclassified."""
    if fire_lat is None or prec_all is None:
        fire_lat, prec_all = firing_precision(A, acts)
    if owner is None:
        owner = latent_owner(A, acts, descendants, pathology=pathology, prec_all=prec_all)
    fire = A > 0
    fire_f = fire[:, f]
    desc = [d for d in descendants.get(f, set()) if d != f] if descendants else []
    excl = (fire_f & ~fire[:, desc].any(dim=1)) if desc else fire_f
    n_excl = int(excl.sum())
    if n_excl == 0:
        return {"feature": int(f), "kind": "unclassified", "shards": [], "n_shards": 0,
                "best_recall": float("nan"), "union_recall": float("nan"), "n_excl": 0}
    cand = ((prec_all[:, f] >= pathology.mult_prec_min) & (owner == f)).nonzero(as_tuple=True)[0].tolist()
    recalls = {L: float(fire_lat[excl, L].double().mean()) for L in cand}
    shards = [L for L, r in recalls.items() if r >= pathology.mult_recall_min]
    if len(shards) < 2:
        return {"feature": int(f), "kind": "clean", "shards": shards, "n_shards": len(shards),
                "best_recall": (max(recalls[L] for L in shards) if shards else float("nan")),
                "union_recall": float("nan"), "n_excl": n_excl}
    best = max(recalls[L] for L in shards)
    union = torch.zeros(A.shape[0], dtype=torch.bool, device=A.device)
    for L in shards:
        union = union | fire_lat[:, L]
    union_recall = float(union[excl].double().mean())
    kind = ("duplicate" if best >= pathology.dup_recall_min
            else "split" if union_recall >= pathology.split_union_min else "clean")
    return {"feature": int(f), "kind": kind, "shards": [int(L) for L in shards],
            "n_shards": len(shards), "best_recall": best, "union_recall": union_recall,
            "n_excl": n_excl}


# --- Chanin absorption ---
def absorption_signals(g: torch.Tensor, W_dec: torch.Tensor, acts: torch.Tensor, A: torch.Tensor,
                       match: torch.Tensor, p: int, c: int, eps: float, *,
                       pathology: PathologySettings = PATHOLOGY) -> dict:
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
        # parent recall
        R_P = float((acts[child_fires, mp].double() > SYNTHETIC_TOYS.fire_threshold).double().mean())
        hole = R_P < (1.0 - pathology.hole_min)
    parent_solo = (A[:, p].double() > 0) & (~child_fires)
    n_solo = int(parent_solo.sum())
    R_solo = (float((acts[parent_solo, mp].double() > SYNTHETIC_TOYS.fire_threshold)
                    .double().mean()) if n_solo > 0 else float("nan"))
    solo_ok = math.isfinite(R_solo) and R_solo > pathology.solo_min
    absorbed = bool(resid_parent > eps and hole and solo_ok)
    return {"parent_component": parent_component, "child_component": child_component,
            "resid_parent": resid_parent, "theta_hat": theta_hat, "R_P": R_P,
            "R_solo": R_solo, "hole": hole, "absorbed": absorbed}


# --- composition (conjunction latent) ---
def conjunction_strength(g: torch.Tensor, W_dec: torch.Tensor, a: int, b: int,
                         match: torch.Tensor | None = None, *,
                         pathology: PathologySettings = PATHOLOGY) -> dict:
    """`K = cos(W_dec[j*], unit(g_a + g_b))` minus the larger single-feature cos; composed if `K > conj_min`.

    `j*` must sit closer to the sum than to either feature and, given `match`, not be either one's own latent."""
    Wu = _unit(W_dec.double())
    ga, gb = _unit(g[a].double()), _unit(g[b].double())
    conj = _unit(g[a].double() + g[b].double())
    cos_conj = Wu @ conj                                   # [d_sae], signed
    baseline = max(float(conj @ ga), float(conj @ gb))

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
    return {"K": K, "latent": jstar, "baseline": baseline, "composed": bool(K > pathology.conj_min)}


# --- orchestration ---
def classify_dictionary(g: torch.Tensor, W_dec: torch.Tensor, A: torch.Tensor, acts: torch.Tensor,
                        match: torch.Tensor, matched_corr: torch.Tensor, recovered: torch.Tensor,
                        cont_edges: Sequence[tuple[int, int]],
                        sibling_pairs: Sequence[tuple[int, int]],
                        isa_child: dict[int, bool],
                        descendants: dict[int, set[int]] | None = None, *,
                        pathology: PathologySettings = PATHOLOGY) -> dict:
    """Absorption per containment edge, split or duplicate firing per feature, composition per sibling pair.

    Mechanisms can coexist on one feature. Structure comes from the tree; `pair_labels` is not an input."""
    F = int(g.shape[0])
    d_sae = int(W_dec.shape[0])
    # eps is a Bonferroni quantile over the whole dictionary. Known limit: conservative for one
    # edge, so some shallow absorbed edges read clean.
    q_eff = 1.0 - pathology.null_target_exceedances / max(d_sae, 1)
    eps = null_cos_threshold(W_dec, g, n_perm=pathology.n_null_perm, q=q_eff)
    expected_null = _expected_null_count(W_dec, eps, n_perm=pathology.n_null_perm)

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
        sig = absorption_signals(g, W_dec, acts, A, match, int(p), int(c), eps,
                                 pathology=pathology)
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
    owner = latent_owner(A, acts, descendants, pathology=pathology, prec_all=prec_all)
    multiplicity_features: list[dict] = []      # splits (fragmented firing)
    duplicate_features: list[dict] = []         # redundant copies
    # recovered features with no exclusive support; kept out of clean
    multiplicity_unclassified: set[int] = set()
    for f in range(F):
        if not bool(recovered[f]):
            continue
        fm = firing_multiplicity(A, acts, descendants, f, pathology=pathology,
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
        cs = conjunction_strength(g, W_dec, int(a), int(b), match=match, pathology=pathology)
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
            "hierarchy_overlap": sum(1 for e in absorbed_edges if e["is_a"]),
            "hierarchy_orthogonal": sum(1 for e in absorbed_edges if not e["is_a"]),
        },
        "unclassified_by_relation": {
            "hierarchy_overlap": sum(1 for e in unclassified_edges if e["is_a"]),
            "hierarchy_orthogonal": sum(1 for e in unclassified_edges if not e["is_a"]),
        },
    }


def tree_edges_and_siblings(tree) -> tuple[list[tuple[int, int]], list[tuple[int, int]],
                                           dict[int, bool], dict[int, set[int]]]:
    """`(cont_edges, sibling_pairs, isa_child, descendants)` from the tree.

    `cont_edges` are ordered (parent, child); `isa_child[c]` means alpha > 0; `sibling_pairs` are unordered."""
    cont_edges = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa_child = {c: (tree.alpha_of(c) > 0.0) for _, c in cont_edges}
    sibling_pairs = [(kids[i], kids[j]) for kids in tree.children.values()
                     for i in range(len(kids)) for j in range(i + 1, len(kids))]
    return cont_edges, sibling_pairs, isa_child, tree.descendents
