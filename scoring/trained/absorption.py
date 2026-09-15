"""
Decompose the damage a trained dictionary does to the true features — absorption, splitting,
and composition (merging).

Uses standard, null-calibrated definitions measured against the toy's ground-truth match:

  - ABSORPTION: a specific child latent soaks up its general parent's direction (child decoder
    carries the parent direction) AND the parent develops a firing hole. Reported as the signed
    absorbed angle `theta_hat`.
  - DECODER MULTIPLICITY ("splitting"): one feature's direction tracked by several latents,
    reported as EXCESS over the null count, never a raw count.
  - COMPOSITION / MERGING: a single conjunction latent for a co-hyponym pair (the "red triangle"
    case).

`eps` is null-calibrated: the q-quantile of |cos| between random directions in the decoder
signal span and the trained decoders. Always report the EXCESS, not the raw count.

FIREWALL: this is a fidelity diagnostic, not a detector. It uses ground truth (`g`/`A`/match)
plus the containment/sibling tree structure; it never touches `pair_labels` and is never a
detector or an AUROC input.
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
    "solo_min": 0.10,        # min parent recall on parent-solo tokens for absorption (vs ~0 for merging)
    "conj_min": 0.10,        # excess conjunction cos (K) above baseline to count as composition; raised from 0.05 (defense-in-depth alongside the sum-alignment + own-latent gates in conjunction_strength)
    "null_target_exceedances": 0.01,  # Bonferroni target on expected chance latents dictionary-wide
    "n_null_perm": 1000,     # random in-span directions for a stable tail quantile
    "multiplicity_excess_min": 1.5,   # excess needed to flag genuine multiplicity over a clean latent (geometry diagnostic `parent_multiplicity_excess` only)
    # --- firing multiplicity (exclusive-support split/duplicate; the deployed orchestration test) ---
    "mult_prec_min": 0.5,    # a candidate latent must fire mostly within the feature: P(feat | latent) >= this
    "mult_recall_min": 0.25, # a candidate counts as a shard if it recalls this fraction of the feature's EXCLUSIVE support
    "dup_recall_min": 0.8,   # a single shard recalling this much of the exclusive support is a duplicate (redundant), not a split
    "split_union_min": 0.8,  # >=2 shards must jointly recall this much of the exclusive support to count as a genuine split
}


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(_TINY)


def _decoder_span_basis(W_dec: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis [r, D] of the trained decoders' row space (the signal span).

    Computed here, not in R^D: decoders concentrate in the signal span where random |cos| is heavier.
    """
    Wu = _unit(W_dec.double())
    _, S, Vh = torch.linalg.svd(Wu, full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum()) if S.numel() else 0
    return Vh[:max(r, 1)]


def null_cos_threshold(W_dec: torch.Tensor, g: torch.Tensor, n_perm: int = 200,
                       q: float = 0.95, seed: int = 0) -> float:
    """eps = the q-quantile of |cos| between random in-span unit directions and the decoders.

    Deterministic (fixed generator). `g` is accepted for signature parity only.
    """
    basis = _decoder_span_basis(W_dec)                     # [r, D]
    Wu = _unit(W_dec.double())
    r = basis.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n_perm, r, generator=gen, dtype=DT)    # [n_perm, r]
    V = _unit(z @ basis)                                   # [n_perm, D] random unit vecs in span
    null_cos = (V @ Wu.transpose(0, 1)).abs()              # [n_perm, d_sae]
    return float(torch.quantile(null_cos.flatten(), q))


def _expected_null_count(W_dec: torch.Tensor, eps: float, n_perm: int = 200,
                         seed: int = 1) -> float:
    """E[ #{j : |cos(v, W_dec[j])| > eps} ] for a random in-span direction v — the chance count
    that `parent_multiplicity_excess` subtracts off to recover the genuine count."""
    basis = _decoder_span_basis(W_dec)
    Wu = _unit(W_dec.double())
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(n_perm, basis.shape[0], generator=gen, dtype=DT)
    V = _unit(z @ basis)
    counts = ((V @ Wu.transpose(0, 1)).abs() > eps).double().sum(dim=1)   # [n_perm]
    return float(counts.mean())


# --------------------------------------------------------------------------
# firing multiplicity (exclusive-support split vs duplicate) -- the deployed detector
# --------------------------------------------------------------------------
def firing_precision(A: torch.Tensor, acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(fire_lat, prec_all)` computed once: `fire_lat` = boolean latent-firing mask `[n, S]`;
    `prec_all[L, f] = P(f fires | L fires)` `[S, F]`. Hoisted out of the per-feature loop so the
    large `[n, S]` cast and the `[S, F]` matmul are done a single time per dictionary.
    """
    fire_lat = acts > 0
    fld = fire_lat.double()
    lat_fire = fld.sum(0).clamp_min(1.0)                                       # [S] latent firing counts
    prec_all = (fld.transpose(0, 1) @ (A > 0).double()) / lat_fire[:, None]    # [S, F] = P(f | L)
    return fire_lat, prec_all


def latent_owner(A: torch.Tensor, acts: torch.Tensor, descendants: dict[int, set[int]] | None,
                 constants: dict, prec_all: torch.Tensor | None = None) -> torch.Tensor:
    """`owner[L]` = the MOST SPECIFIC feature latent `L` fires mostly within.

    Firing is strictly nested in the tree (child => parent), so an ancestor's precision
    `P(ancestor | L)` is always >= a contained feature's `P(f | L)`. A plain `argmax P(f | L)` would
    therefore attribute a child's own latent to its ancestor (worse, to the topmost root, since ties
    break to the lowest id) and hide every genuine split of a non-root feature. Instead: among the
    features `L` fires mostly within (`P(f | L) >= mult_prec_min`), drop any feature that has a
    descendant also in that set (an ancestor is superseded by its more-specific descendant), then take
    the argmax precision over what remains. This keeps a latent attributed to the child it belongs to,
    while still sending an unrelated but higher-precision feature its own latents.

    `descendants` is the tree's transitive-closure `descendents` map; with None/empty it degrades to a
    plain argmax over the high-precision features (no tree to resolve specificity).
    """
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
    """Split vs duplicate by firing recall on feature `f`'s EXCLUSIVE support.

    Exclusive support = tokens where `f` fires AND every ground-truth descendant of `f` is silent.
    A descendant's own latent is silent there, so the is-a containment confound (child => parent, so
    a clean child's latent fires *within* the parent) is removed exactly for firing that tracks the
    ground truth: a clean parent's child-latent scores recall ~0 on the parent's exclusive support and
    is dropped (a leaky trained latent that fires on the parent's solo tokens can still land there).

    Candidates are latents that fire mostly within `f` (`P(f | latent) >= mult_prec_min`) AND are
    attributed to `f` by `latent_owner` (their most-specific containing feature is `f`); attribution
    stops both an unrelated latent that incidentally co-fires inside `f` and an ancestor's latent from
    being miscounted as a shard of `f`. The exclusive-support recall then filters: a candidate counts
    as a shard if its recall on the exclusive support >= `mult_recall_min`. With >=2 shards:
      - duplicate: a single shard already recalls the whole support (`best >= dup_recall_min`) --
        redundant copies;
      - split: no single shard covers it but the shards jointly do (`union >= split_union_min`) --
        fragmented firing;
      - clean: otherwise.
    A feature with no exclusive support (`n_excl == 0`, i.e. it never fires without a descendant) is
    NOT evaluable and returns `kind="unclassified"` -- never silently folded into clean.

    `descendants` is the tree's transitive-closure `descendents` map; a feature with no descendants (a
    leaf, or an empty/None map) uses its full firing set as the exclusive support. `owner`/`prec_all`/
    `fire_lat` are computed once here if not supplied (the caller passes them to avoid recomputation).
    """
    if fire_lat is None or prec_all is None:
        fire_lat, prec_all = firing_precision(A, acts)
    if owner is None:
        owner = latent_owner(A, acts, descendants, constants, prec_all=prec_all)
    fire = A > 0
    fire_f = fire[:, f]
    desc = [d for d in descendants.get(f, set()) if d != f] if descendants else []
    excl = (fire_f & ~fire[:, desc].any(dim=1)) if desc else fire_f
    n_excl = int(excl.sum())
    if n_excl == 0:                                                            # no exclusive support: not evaluable
        return {"feature": int(f), "kind": "unclassified", "shards": [], "n_shards": 0,
                "best_recall": float("nan"), "union_recall": float("nan"), "n_excl": 0}
    cand = ((prec_all[:, f] >= constants["mult_prec_min"]) & (owner == f)).nonzero(as_tuple=True)[0].tolist()
    recalls = {L: float(fire_lat[excl, L].double().mean()) for L in cand}      # recall on the EXCLUSIVE support
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


# --------------------------------------------------------------------------
# splitting (parent multiplicity, excess over null) -- standalone geometry diagnostic
# --------------------------------------------------------------------------
def parent_multiplicity_excess(g: torch.Tensor, W_dec: torch.Tensor, f: int, eps: float) -> dict:
    """`M_P = #{j : |cos(W_dec[j], g[f])| > eps AND f = argmax_f' |cos(W_dec[j], g[f'])|}`, and
    `excess = M_P - E[M_P|null]`.

    A latent counts toward feature f only if f is that latent's best-matching true feature — needed
    on this overcomplete, non-orthogonal toy so a parent's latent isn't miscounted into every
    correlated child's multiplicity. A clean feature scores M_P=1, a real split scores >=2.
    """
    Wu = _unit(W_dec.double())
    cosmat = (_unit(g.double()) @ Wu.transpose(0, 1)).abs()   # [F, d_sae]
    best_f = cosmat.argmax(dim=0)                             # [d_sae] each latent's best feature
    M_P = int(((cosmat[f] > eps) & (best_f == f)).sum())
    en = _expected_null_count(W_dec, eps)
    return {"M_P": M_P, "expected_null": en, "excess": M_P - en}


# --------------------------------------------------------------------------
# Chanin absorption (child latent carries parent + parent hole)
# --------------------------------------------------------------------------
def absorption_signals(g: torch.Tensor, W_dec: torch.Tensor, acts: torch.Tensor, A: torch.Tensor,
                       match: torch.Tensor, p: int, c: int, eps: float, constants: dict) -> dict:
    """Chanin absorption for a containment edge parent=p, child=c.

    `absorbed = (child latent carries the residual parent direction beyond the null) AND (parent has
    a firing hole on the child's tokens) AND (a standalone parent latent survives on parent-solo
    tokens)`. Three gates:

      - `resid_parent`: child decoder's overlap with the parent direction orthogonal to the child's
        own direction (removes the toy's designed is-a overlap so a clean child scores exactly 0).
        `parent_component`/`child_component`/`theta_hat` are also reported as the raw decomposition.
      - `R_P`: parent recall on child-firing tokens; a hole (`R_P` well below 1) means absorption,
        not hedging (which keeps the parent firing).
      - `R_solo`: parent recall on parent-solo tokens; near-zero means a merging latent, not
        absorption, so `R_solo > solo_min` vetoes merging.

    A `match[c] < 0` or `match[p] < 0` (child/parent unrecovered) gives `absorbed=False`.
    """
    mc, mp = int(match[c]), int(match[p])
    if mc < 0 or mp < 0:
        return {"parent_component": float("nan"), "child_component": float("nan"),
                "resid_parent": float("nan"), "theta_hat": float("nan"), "R_P": float("nan"),
                "R_solo": float("nan"), "hole": False, "absorbed": False}
    gp_u = _unit(g[p].double())
    gc_u = _unit(g[c].double())
    dec_c = _unit(W_dec[mc].double())
    parent_component = float(dec_c @ gp_u)                         # absorbed-angle numerator (signed, raw)
    child_component = float(dec_c @ gc_u)                          # absorbed-angle denominator (signed)
    theta_hat = math.atan2(parent_component, child_component)
    # Parent direction with the child's designed overlap removed; a clean is-a child reads 0 here.
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
    parent_solo = (A[:, p].double() > 0) & (~child_fires)         # parent active, child absent
    n_solo = int(parent_solo.sum())
    R_solo = (float((acts[parent_solo, mp].double() > 0).double().mean())
              if n_solo > 0 else float("nan"))
    solo_ok = math.isfinite(R_solo) and R_solo > constants["solo_min"]
    absorbed = bool(resid_parent > eps and hole and solo_ok)
    return {"parent_component": parent_component, "child_component": child_component,
            "resid_parent": resid_parent, "theta_hat": theta_hat, "R_P": R_P,
            "R_solo": R_solo, "hole": hole, "absorbed": absorbed}


# --------------------------------------------------------------------------
# composition / merging (conjunction latent)
# --------------------------------------------------------------------------
def conjunction_strength(g: torch.Tensor, W_dec: torch.Tensor, a: int, b: int,
                         match: torch.Tensor | None = None, conj_min: float | None = None) -> dict:
    """`K = cos(W_dec[j*], unit(g_a+g_b)) - baseline` over ELIGIBLE latents; composed iff `K > conj_min`.

    A real conjunction latent is a THIRD latent that points along the normalized sum of the two true
    directions. Two guards make `j*` that latent rather than a masquerade:
      * **sum-alignment gate** — `j*` must be closer to the sum than to either single feature
        (`cos(W_j, conj) > cos(W_j, g_a)` and `> cos(W_j, g_b)`). A latent that merely recovers one
        feature (its own entanglement-contaminated decoder) is single-aligned, so it is excluded.
        This is geometric and needs no `match`.
      * **own-latent exclusion** — when `match` is given, the pair's own matched latents
        (`match[a]`, `match[b]`) are excluded outright; a feature's own latent can never be the
        pair's conjunction.
    Without these, ~90% of flagged "composites" were a feature's own latent clearing a low `conj_min`.
    The baseline is the larger single-feature cosine onto the sum (actual geometry, not `1/sqrt(2)`,
    since the dictionary is overcomplete/non-orthogonal); composition is the excess over it.
    """
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


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def classify_dictionary(g: torch.Tensor, W_dec: torch.Tensor, A: torch.Tensor, acts: torch.Tensor,
                        match: torch.Tensor, matched_corr: torch.Tensor, recovered: torch.Tensor,
                        cont_edges: Sequence[tuple[int, int]],
                        sibling_pairs: Sequence[tuple[int, int]],
                        isa_child: dict[int, bool],
                        descendants: dict[int, set[int]] | None = None,
                        constants: dict | None = None) -> dict:
    """Per-mechanism tallies (coexistence expected — not a forced single label per feature).

    Absorption per containment edge, firing multiplicity (split / duplicate) per feature, composition
    per sibling pair — all null-calibrated. `cont_edges`/`sibling_pairs`/`isa_child`/`descendants`
    come from the tree (identity truth); `pair_labels` is deliberately not a parameter (firewall).
    `descendants` is the tree's transitive-closure `descendents` map, used to remove the is-a
    containment confound from the multiplicity read (see `firing_multiplicity`).
    """
    constants = ABSORPTION_CONSTANTS if constants is None else constants
    F = int(g.shape[0])
    d_sae = int(W_dec.shape[0])
    # Bonferroni-adaptive quantile targeting a small expected chance count dictionary-wide (a fixed
    # p95 is too noisy at large d_sae); gates both the multiplicity count and the absorption resid gate.
    # KNOWN LIMIT: this dictionary-wide eps is over-conservative for a targeted per-edge test, so some
    # shallow absorbed cells can read clean; a principled per-checkpoint floor is not yet derivable
    # for the deployed BatchTopK activation.
    q_eff = 1.0 - constants["null_target_exceedances"] / max(d_sae, 1)
    eps = null_cos_threshold(W_dec, g, n_perm=int(constants["n_null_perm"]), q=q_eff)
    expected_null = _expected_null_count(W_dec, eps, n_perm=int(constants["n_null_perm"]))

    absorbed_edges: list[dict] = []
    absorbed_children: set[int] = set()
    # Unclassified (catch-all): an edge whose absorption gates couldn't be evaluated (unrecovered,
    # child never fires, or no parent-solo tokens); must not be silently folded into `clean`.
    unclassified_edges: list[dict] = []
    unclassified_children: set[int] = set()
    # Below-rho: an endpoint assigned by the Hungarian match but not recovered (matched_corr < rho)
    # is an arbitrary weak match, not a real mechanism — excluded from every bucket, reported separately.
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

    # Firing multiplicity on each recovered feature's EXCLUSIVE support: split (fragmented firing) vs
    # duplicate (redundant firing). Exclusive support removes the is-a containment confound exactly,
    # so a clean parent is not miscounted from its child's latent (see `firing_multiplicity`).
    descendants = {} if descendants is None else descendants
    fire_lat, prec_all = firing_precision(A, acts)       # hoisted once: [n, S] mask + [S, F] precision
    owner = latent_owner(A, acts, descendants, constants, prec_all=prec_all)
    multiplicity_features: list[dict] = []      # genuine splits (fragmented firing)
    duplicate_features: list[dict] = []         # redundant copies (full-recall shards)
    # A recovered feature with no exclusive support can't be assessed for multiplicity; it is held
    # out of `clean` in its own not-evaluable bucket (parity with the absorption `unclassified` path).
    multiplicity_unclassified: set[int] = set()
    for f in range(F):
        if not bool(recovered[f]):        # recovered-gated (parity with absorbed/clean)
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
        if not (bool(recovered[int(a)]) and bool(recovered[int(b)])):   # recovered-gated
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
    """Boolean masks over `pairs` (positions into `feats`) for the latent-side scoring columns.

      absorbed — per ordered recovered (parent, child) pair that is a Chanin-absorbed edge.
      merged   — per symmetric recovered pair that is a composed sibling conjunction.

    `feats[pos]` maps a pair position back to its true feature id. FIREWALL: these are scoring-only
    labels — the 10 detectors never see them.
    """
    absorbed_set = {(int(e["parent"]), int(e["child"])) for e in classification["absorbed_edges"]}
    merged_set: set[tuple[int, int]] = set()
    for e in classification["composed_pairs"]:
        a, b = int(e["a"]), int(e["b"])
        merged_set.add((a, b))
        merged_set.add((b, a))                         # symmetric: both orderings are merged
    ids = [(feats[a], feats[b]) for (a, b) in pairs]
    # CPU masks (the whole toy scoring path is CPU); would need `.to(device)` for a real CUDA checkpoint.
    absorbed = torch.tensor([pid in absorbed_set for pid in ids], dtype=torch.bool)
    merged = torch.tensor([pid in merged_set for pid in ids], dtype=torch.bool)
    return {"absorbed": absorbed, "merged": merged}


def split_readout(classification: dict, feats: list[int]) -> dict:
    """Per-feature split readout (firing multiplicity) — not a pair column.

    A split is per-feature, not a relation over a pair, so it's reported as a side readout: the
    recovered features whose firing fragments across >=2 shards on their exclusive support (genuine
    splits; duplicates are reported separately). Scoped to `feats` for grid parity.
    """
    recovered_ids = {int(f) for f in feats}
    out = [{"feature": int(d["feature"]), "n_shards": int(d["n_shards"]),
            "union_recall": float(d["union_recall"]), "n_excl": int(d["n_excl"])}
           for d in classification["decoder_multiplicity"] if int(d["feature"]) in recovered_ids]
    return {"n_split": len(out), "features": out}


def tree_edges_and_siblings(tree) -> tuple[list[tuple[int, int]], list[tuple[int, int]],
                                           dict[int, bool], dict[int, set[int]]]:
    """`(cont_edges, sibling_pairs, isa_child, descendants)` derived from the tree (identity truth).

    Shared by `run_absorption` and `run_retrieval` so the two reports can't drift on schema changes.
    `cont_edges` are ordered (parent, child) pairs; `isa_child[c]` is True iff non-zero overlap;
    `sibling_pairs` are unordered co-hyponym pairs; `descendants` is the transitive-closure
    `descendents` map (used by the firing-multiplicity confound removal).
    """
    cont_edges = [(p, c) for c in range(tree.F) for p, _, _ in tree.parents.get(c, [])]
    isa_child = {c: (tree.alpha_of(c) > 0.0) for _, c in cont_edges}
    sibling_pairs = [(kids[i], kids[j]) for kids in tree.children.values()
                     for i in range(len(kids)) for j in range(i + 1, len(kids))]
    return cont_edges, sibling_pairs, isa_child, tree.descendents


def run_absorption(ckpt_dir, n_tokens: int = 200_000, rho: float | None = None) -> dict:
    """Orchestrator (server; needs sae_training): decompose one checkpoint's dictionary damage.

    Not unit-tested (loads a real checkpoint); the pure functions above carry the coverage.
    Uses the SAME in-sample match as run_recovery, then classifies against truth.
    """
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
