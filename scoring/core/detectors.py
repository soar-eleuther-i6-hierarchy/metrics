"""Per-ordered-pair detectors over the recovered latents, computed from SAE outputs only.

Each detector is an [R, R] matrix, entry [p, c] scoring parent p and child c, with a NaN diagonal.
Undefined cells are NaN, not a filled-in value (`pmi` alone is Laplace-smoothed).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metrics.rules import nan_self_pairs as _nan_diag
from metrics.rules import score_pairs
from scoring.core import gates
from scoring.core.registry import DETECTOR_SIGN, DETECTORS, gate_constants

DT = torch.float64
_NAN = float("nan")

# --- scorability mask ---
# NaN unless both endpoints fire >= min_fire_count and co-fire >= support_min_joint: below
# that floor these co-firing values are arithmetic, not measurement.
MASKED_DETECTORS: tuple[str, ...] = ("coverage_R", "asymmetry_R", "pmi")

# Child side only (fire[c] >= min_fire_count): these live on the child's tokens, so a parent
# that contributes nothing there is negative evidence, not missing data.
CHILD_MASKED_DETECTORS: tuple[str, ...] = ("recon_2a", "recon_child_gain")

# Everything else is unmasked on purpose. Per-parent broadcasts do not depend on the column,
# and a NaN there would spread through `reads.wide_matrix` into `wide`. `s_res` reads no tokens.
# `token_freq_survival` applies `min_joint` itself, hence the separate `support_min_joint`.


def _mask_to_nan(m: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, m, torch.full_like(m, _NAN))


@dataclass(frozen=True)
class DetectorInputs:
    """SAE-side inputs for the R recovered latents over n held-out tokens.

    acts_rec [n, R]  activations of the matched latent per recovered feature
    W_unit   [R, D]  oriented, L2-normalized decoder rows (geometry channel)
    W_raw    [R, D]  raw decoder rows (reconstruction channel)
    h        [n, D]  the residual activations the SAE saw (recon channel)
    b_dec    [D]     decoder bias (recon channel)
    tokens   [n]     token id per row (frequency channel)
    vocab            token-id vocabulary size
    """

    acts_rec: torch.Tensor
    W_unit: torch.Tensor
    W_raw: torch.Tensor
    h: torch.Tensor | None = None
    b_dec: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    vocab: int = 0


def _broadcast_parent(vec: torch.Tensor, R: int) -> torch.Tensor:
    """A per-parent vector -> an [R, R] matrix constant across columns, NaN diagonal."""
    return _nan_diag(vec.reshape(R, 1).expand(R, R).clone())


def _broadcast_child(vec: torch.Tensor, R: int) -> torch.Tensor:
    """A per-child vector -> an [R, R] matrix with entry [p, c] = vec[c], NaN diagonal.

    The transpose of `_broadcast_parent`; swapping the two gives the same shape and no error.
    """
    return _nan_diag(vec.reshape(1, R).expand(R, R).clone())


def cofiring(Fm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(cofire[R,R], fire[R], N) from a boolean firing mask `Fm` [n, R]."""
    F = Fm.double()
    cofire = F.transpose(0, 1) @ F
    fire = F.sum(dim=0)
    return cofire, fire, int(Fm.shape[0])


def coverage_R(cofire: torch.Tensor, fire: torch.Tensor, eps: float) -> torch.Tensor:
    """Reverse coverage R(p|c) = cofire[p,c] / fire[c] = P(parent fires | child fires).

    A never-firing child gives NaN, not 0. `eps` is unused and kept for the signature.
    """
    fire_c = fire.reshape(1, -1)
    denom = torch.where(fire_c > 0, fire_c, torch.full_like(fire_c, _NAN))
    return _nan_diag(cofire / denom)


def _forward_F(cofire: torch.Tensor, fire: torch.Tensor, eps: float) -> torch.Tensor:
    """Forward coverage F(p|c) = cofire[p,c] / fire[p] = P(child fires | parent fires).

    A dead parent gives NaN. `eps` is unused.
    """
    fire_p = fire.reshape(-1, 1)
    denom = torch.where(fire_p > 0, fire_p, torch.full_like(fire_p, _NAN))
    return cofire / denom


def asymmetry_R(R_mat: torch.Tensor) -> torch.Tensor:
    """R(p|c) - R(c|p): >0 for a directed containment, ~0 for symmetric co-firing."""
    return _nan_diag(R_mat - R_mat.transpose(0, 1))


def pmi(cofire: torch.Tensor, fire: torch.Tensor, N: int, laplace: float) -> torch.Tensor:
    """Smoothed PMI = log((cofire+l)*N / ((fire_p+l)(fire_c+l))). Symmetric."""
    num = (cofire + laplace) * float(N)
    den = (fire.reshape(-1, 1) + laplace) * (fire.reshape(1, -1) + laplace)
    return _nan_diag(torch.log(num / den))


def s_res_cosine(W_unit: torch.Tensor) -> torch.Tensor:
    """Decoder-geometry overlap cos(d_p, d_c) between oriented unit decoders. Symmetric."""
    W = W_unit.double()
    return _nan_diag(W @ W.transpose(0, 1))


def fit_probe_directions(fit_h: torch.Tensor, fit_labels: torch.Tensor,
                         constants: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one linear probe per child on `fit_h` and return `(P [R, d], available [R])`.

    The positive-count check reads `fit_labels`, the labels the probe trains on, not the
    scoring labels. A child with too few positives or a failed probe is marked unavailable.
    """
    from metrics.sres import train_probe

    R = int(fit_labels.shape[1])
    d = int(fit_h.shape[1])
    P = torch.zeros((R, d), dtype=DT, device=fit_h.device)
    available = torch.zeros(R, dtype=torch.bool, device=fit_h.device)
    min_pos = int(constants["sres_min_probe_pos"])
    fire_thresh = constants["fire_thresh"]            # same firing convention as compute_all
    for c in range(R):
        pos = fit_labels[:, c] > fire_thresh
        if int(pos.sum()) < min_pos:
            continue                                  # untestable child -> column NaN
        probe = train_probe(
            fit_h, pos, seed=c,
            neg_ratio=int(constants["sres_neg_ratio"]),
            max_tokens=int(constants["sres_max_probe_tokens"]),
            steps=int(constants["sres_steps"]),
            lr=float(constants["sres_lr"]),
            min_neg=int(constants["sres_min_neg"]),
        )
        if probe is None:
            continue                                  # too few negatives -> column NaN
        P[c] = probe.to(P.device).double()
        available[c] = True
    return P, available


def s_res_from_directions(P: torch.Tensor, available: torch.Tensor,
                          W_unit: torch.Tensor) -> torch.Tensor:
    """Score frozen probe directions: out[p, c] = min(corr[p], corr[c]), corr = W_unit @ P[c].

    Unit decoders make `corr` a cosine. Asymmetric, since column c uses child c's probe.
    """
    Wu = W_unit.double()
    R = int(Wu.shape[0])
    out = torch.full((R, R), _NAN, dtype=DT, device=Wu.device)
    for c in range(R):
        if not bool(available[c]):
            continue
        corr = Wu @ P[c].to(Wu.device).double()       # [R] cosine of each decoder with the probe
        out[:, c] = torch.minimum(corr, corr[c])
    return _nan_diag(out)


def s_res_probe(acts_rec: torch.Tensor, h: torch.Tensor, W_unit: torch.Tensor,
                constants: dict, label_acts: torch.Tensor | None = None,
                fit_h: torch.Tensor | None = None,
                fit_labels: torch.Tensor | None = None) -> torch.Tensor:
    """Tree SAE's probe s_res: fit a probe per child, then `s_res_from_directions`.

    Labels default to `acts_rec`, the SAE's own firing. `fit_h`/`fit_labels` select a separate
    fitting draw (PRECOMMIT.md s6) and default to `h` and the scoring labels.
    """
    labels = acts_rec if label_acts is None else label_acts
    P, available = fit_probe_directions(
        h if fit_h is None else fit_h,
        labels if fit_labels is None else fit_labels,
        constants,
    )
    return s_res_from_directions(P, available, W_unit)


def s_res_variants(acts_rec: torch.Tensor, h: torch.Tensor, W_unit: torch.Tensor,
                   g_unit: torch.Tensor, A_rec: torch.Tensor,
                   constants: dict) -> dict[str, torch.Tensor]:
    """s_res variants for toy diagnostics. Reads ground truth (`g_unit`, `A_rec`), so not a scorer.

      cosine_g     : cosine over the true directions g
      probe_true_g : probe on true firing, against g
      probe_self_W : probe on the SAE's own firing, against learned decoders (the deployed form)
      probe_true_W : probe on true firing, against learned decoders
    """
    return {
        "cosine_g": s_res_cosine(g_unit),
        "probe_true_g": s_res_probe(A_rec, h, g_unit, constants, label_acts=A_rec),
        "probe_self_W": s_res_probe(acts_rec, h, W_unit, constants),
        "probe_true_W": s_res_probe(acts_rec, h, W_unit, constants, label_acts=A_rec),
    }


def edge_mask(R_mat: torch.Tensor, fire: torch.Tensor, tau: float, min_fire: int,
              cofire: torch.Tensor | None = None, min_joint: int = 0) -> torch.Tensor:
    """Inferred edge set, bool [R, R]: R(p|c) >= tau, both endpoints fire >= min_fire, and at
    least `min_joint` co-firing tokens when `cofire` is given."""
    keep = R_mat >= tau                                  # NaN >= tau -> False
    enough = fire >= min_fire
    keep = keep & enough.reshape(-1, 1) & enough.reshape(1, -1)
    if cofire is not None and min_joint > 0:
        keep = keep & (cofire >= min_joint)
    return keep


def outdegree(em: torch.Tensor, fire: torch.Tensor, N: int) -> torch.Tensor:
    """Per-parent kept-children count, broadcast across columns; NaN for a dead parent.
    Raw direction; the frozen sign flips it so a wide parent scores low."""
    R = em.shape[0]
    outdeg = em.double().sum(dim=1)
    outdeg = torch.where(fire > 0, outdeg, torch.full_like(outdeg, _NAN))
    return _broadcast_parent(outdeg, R)


def joint_child_J(F_mat: torch.Tensor, em: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent J(p) = min(1, sum of forward coverage over kept children), broadcast; NaN
    for a dead parent."""
    R = F_mat.shape[0]
    contrib = torch.where(em, F_mat, torch.zeros_like(F_mat))
    J = contrib.sum(dim=1).clamp(max=1.0)
    J = torch.where(fire > 0, J, torch.full_like(J, _NAN))
    return _broadcast_parent(J, R)


def joint_child_mass(acts_rec: torch.Tensor, Fm: torch.Tensor, em: torch.Tensor) -> torch.Tensor:
    """Per-parent share of activation energy on tokens where a kept child also fires,
    broadcast; NaN for a parent with zero energy."""
    R = acts_rec.shape[1]
    energy = (acts_rec.double() ** 2)                    # [n, R]
    energy_total = energy.sum(dim=0)                     # [R]
    r_mass = torch.full((R,), _NAN, dtype=DT, device=acts_rec.device)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        if float(energy_total[p]) <= 0.0:
            continue                                     # dead parent -> NaN
        if kids.numel() == 0:
            r_mass[p] = 0.0                              # live parent, no kept children
            continue
        any_child = Fm[:, kids].any(dim=1)               # [n]
        r_mass[p] = float(energy[any_child, p].sum() / energy_total[p])
    return _broadcast_parent(r_mass, R)


def sibling_redundancy(em: torch.Tensor, cofire: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent mean pairwise Jaccard over kept children, broadcast; NaN for fewer than two.
    Raw direction; the frozen sign flips it so a splitting parent scores low."""
    R = em.shape[0]
    red = torch.full((R,), _NAN, dtype=DT, device=em.device)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        k = int(kids.numel())
        if k < 2:
            continue
        cf = cofire[kids][:, kids]                       # [k, k] child co-firing
        fk = fire[kids]
        union = fk.reshape(-1, 1) + fk.reshape(1, -1) - cf
        jac = torch.where(union > 0, cf / union, torch.full_like(cf, _NAN))
        iu = torch.triu_indices(k, k, offset=1)          # upper triangle = each sibling pair once
        vals = jac[iu[0], iu[1]]
        vals = vals[~torch.isnan(vals)]
        if vals.numel():
            red[p] = float(vals.mean())
    return _broadcast_parent(red, R)


def _recon_gains(acts_rec: torch.Tensor, h: torch.Tensor, W_raw: torch.Tensor,
                 b_dec: torch.Tensor, Fm: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """`(gains [R, R], child_gain [R])` for `recon_2a` and `recon_child_gain`.

    gains[p, c] is the relative reconstruction damage from ablating p on the tokens where c
    fires. Its diagonal is `child_gain`, so it is kept here rather than NaN-ed.
    """
    a = acts_rec.double()
    W = W_raw.double()
    x_hat = a @ W + (b_dec.double() if b_dec is not None else 0.0)
    err = h.double() - x_hat                             # [n, D]
    dot = err @ W.transpose(0, 1)                        # [n, R] = <d_f, err_t>
    dnorm2 = (W ** 2).sum(dim=1)                         # [R]
    g = 2.0 * a * dot + (a ** 2) * dnorm2.reshape(1, -1)  # [n, R] per-token ablation gain
    Fc = Fm.double()                                     # [n, R] child-firing indicator
    numer = g.transpose(0, 1) @ Fc                       # [R(parent), R(child)]
    err_sq = (err ** 2).sum(dim=1)                       # [n]
    denom_c = err_sq @ Fc                                # [R(child)]
    out = numer / torch.where(denom_c > 0, denom_c, torch.full_like(denom_c, _NAN)).reshape(1, -1)
    return out, torch.diagonal(out).clone()


def recon_2a(acts_rec: torch.Tensor, h: torch.Tensor, W_raw: torch.Tensor,
             b_dec: torch.Tensor, Fm: torch.Tensor) -> torch.Tensor:
    """Parent gain: how much ablating p worsens reconstruction on tokens where child c fires.

    gain[p,c] = sum(g[t,p]) / sum(||err[t]||^2) over those tokens, with
    err = h - (acts @ W_raw + b_dec) and g[t,f] = 2 a_f<d_f,err_t> + a_f^2||d_f||^2.
    NaN for a never-firing child.
    """
    gains, _child = _recon_gains(acts_rec, h, W_raw, b_dec, Fm)
    return _nan_diag(gains)


def recon_child_gain(acts_rec: torch.Tensor, h: torch.Tensor, W_raw: torch.Tensor,
                     b_dec: torch.Tensor, Fm: torch.Tensor) -> torch.Tensor:
    """Per-child reconstruction gain from ablating the child on its own tokens, broadcast down
    each column (entry [p, c] = child_gain[c]); NaN column for a never-firing child.

    The `child_gain` half of `metrics.reconstruction.edge_reconstruction_condition`.
    """
    _gains, child = _recon_gains(acts_rec, h, W_raw, b_dec, Fm)
    return _broadcast_child(child, int(acts_rec.shape[1]))


def joint_child_supp(Fm: torch.Tensor, em: torch.Tensor,
                     fire: torch.Tensor) -> torch.Tensor:
    """Per-parent share of the parent's firing tokens where a kept child also fires, broadcast.

    `metrics.joint_child.r_supp`, the exact union that `joint_child_J` upper-bounds. A dead
    parent is NaN (metrics returns 0); a live parent with no kept children is 0.0.
    """
    R = int(em.shape[0])
    out = torch.full((R,), _NAN, dtype=DT, device=em.device)
    for p in range(R):
        if float(fire[p]) <= 0.0:
            continue                                     # dead parent -> NaN
        kids = em[p].nonzero(as_tuple=True)[0]
        if kids.numel() == 0:
            out[p] = 0.0
            continue
        union = Fm[:, p] & Fm[:, kids].any(dim=1)
        out[p] = float(union.sum()) / float(fire[p])
    return _broadcast_parent(out, R)


def sibling_redundancy_pc(Fm: torch.Tensor, em: torch.Tensor) -> torch.Tensor:
    """Per-parent mean pairwise sibling Jaccard within the parent's firing tokens, broadcast.

    `metrics.sibling_redundancy.parent_conditioned_redundancy`: the union is clamped at 1, so
    siblings that never fire inside the parent score 0. Unlike metrics, fewer than two kept
    children or a dead parent is NaN, not 0.
    """
    R = int(em.shape[0])
    out = torch.full((R,), _NAN, dtype=DT, device=em.device)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        k = int(kids.numel())
        if k < 2 or not bool(Fm[:, p].any()):
            continue
        sub = Fm[Fm[:, p]][:, kids].double()              # [m, k] within the parent's tokens
        cf = sub.transpose(0, 1) @ sub
        fi = sub.sum(dim=0)
        union = fi.reshape(1, -1) + fi.reshape(-1, 1) - cf
        jac = cf / union.clamp(min=1.0)
        offdiag = ~torch.eye(k, dtype=torch.bool, device=jac.device)
        out[p] = float(jac[offdiag].mean())
    return _broadcast_parent(out, R)


def token_freq_survival(Fm: torch.Tensor, tokens: torch.Tensor, vocab: int,
                        min_joint: int, high_mass: float, mid_mass: float,
                        min_fire_low: int) -> torch.Tensor:
    """Whether coverage survives removing common tokens: x / (1 + x), with x = R(p|c) on the
    rare-token buckets {1, 2} over R(p|c) on all tokens.

    NaN where unmeasurable. Not gated on the edge set, which would NaN the shared-token pairs
    this exists to score.
    """
    counts = torch.bincount(tokens, minlength=vocab).double()
    order = torch.argsort(counts, descending=True)
    cum = torch.cumsum(counts[order], dim=0) / counts.sum().clamp_min(1.0)
    bucket_sorted = torch.full((counts.numel(),), 2, dtype=torch.long)
    bucket_sorted[cum <= high_mass] = 0
    bucket_sorted[(cum > high_mass) & (cum <= high_mass + mid_mass)] = 1
    id_bucket = torch.empty_like(bucket_sorted)
    id_bucket[order] = bucket_sorted
    tok_bucket = id_bucket[tokens]                       # [n]

    Ffloat = Fm.double()
    rest = (tok_bucket > 0)                              # buckets 1 and 2 (rare)
    Fm_rest = Fm & rest.reshape(-1, 1)
    cofire_all = Ffloat.transpose(0, 1) @ Ffloat
    fire_all = Ffloat.sum(dim=0)
    Fr = Fm_rest.double()
    cofire_rest = Fr.transpose(0, 1) @ Fr
    fire_rest = Fr.sum(dim=0)

    R_all = cofire_all / fire_all.clamp_min(1.0).reshape(1, -1)
    R_rest = cofire_rest / fire_rest.clamp_min(1.0).reshape(1, -1)
    # x/(1+x) rather than a clamp, so the >1 tail stays ranked
    ratio = R_rest / R_all.clamp_min(1e-12)
    survival = ratio / (1.0 + ratio)

    # floor on total child firing, not rare-bucket firing: a zero rare-bucket rate is the signal
    undefined = ((fire_all.reshape(1, -1) < min_fire_low) | (R_all <= 0)
                 | (cofire_all < min_joint))
    survival = torch.where(undefined, torch.full_like(survival, _NAN), survival)
    return _nan_diag(survival)


# --- compute_all / compute_bundle ---
def _compute_raw(inputs: DetectorInputs, constants: dict,
                 s_res_mode: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Every detector in raw orientation, plus the intermediates the gates are built from, so
    gates and detectors share one set of firing counts, edge set and gains."""
    if s_res_mode not in ("cosine", "probe"):
        raise ValueError(f"s_res_mode must be 'cosine' or 'probe', got {s_res_mode!r}")
    if inputs.h is None or inputs.b_dec is None or inputs.tokens is None:
        missing = [n for n in ("h", "b_dec", "tokens") if getattr(inputs, n) is None]
        readers = "recon_2a / token_freq_survival"
        if s_res_mode == "probe":
            readers += " / s_res_probe (trains on h)"
        raise ValueError(
            f"compute_all needs {missing} ({readers} read them); "
            f"reduce_to_recovered must be given h/b_dec/tokens before scoring")
    eps = constants["coverage_eps"]
    R = int(inputs.acts_rec.shape[1])
    Fm = inputs.acts_rec > constants["fire_thresh"]
    cofire, fire, N = cofiring(Fm)
    R_mat = coverage_R(cofire, fire, eps)
    F_mat = _forward_F(cofire, fire, eps)
    em = edge_mask(R_mat, fire, constants["edge_tau"], constants["min_fire_count"],
                   cofire=cofire, min_joint=constants["min_joint"])
    gains, child_gain = _recon_gains(inputs.acts_rec, inputs.h, inputs.W_raw, inputs.b_dec, Fm)
    # after `em`, which is built from unmasked coverage: the edge set never sees the mask
    support, support_counts = gates.support_mask(cofire, fire, constants["min_fire_count"],
                                                 constants["support_min_joint"])
    child_ok = (fire >= float(constants["min_fire_count"])).reshape(1, -1).expand(R, R)

    raw = {
        "coverage_R": R_mat,
        "asymmetry_R": asymmetry_R(R_mat),
        "joint_child_J": joint_child_J(F_mat, em, fire),
        "pmi": pmi(cofire, fire, N, constants["pmi_laplace"]),
        "token_freq_survival": token_freq_survival(
            Fm, inputs.tokens, inputs.vocab, constants["min_joint"],
            constants["freq_high_mass"], constants["freq_mid_mass"],
            constants["freq_min_fire_low"]),
        "recon_2a": _nan_diag(gains),
        "s_res": (s_res_cosine(inputs.W_unit) if s_res_mode == "cosine"
                  else s_res_probe(inputs.acts_rec, inputs.h, inputs.W_unit, constants)),
        "sibling_redundancy": sibling_redundancy(em, cofire, fire),
        "joint_child_mass": joint_child_mass(inputs.acts_rec, Fm, em),
        "outdegree": outdegree(em, fire, N),
        "recon_child_gain": _broadcast_child(child_gain, R),
        "joint_child_supp": joint_child_supp(Fm, em, fire),
        "sibling_redundancy_pc": sibling_redundancy_pc(Fm, em),
    }
    # masked here rather than on the inputs, so the masked set is the explicit list
    for name in MASKED_DETECTORS:
        raw[name] = _mask_to_nan(raw[name], support)
    for name in CHILD_MASKED_DETECTORS:
        raw[name] = _mask_to_nan(raw[name], child_ok)

    # ctx["R_mat"] is the pre-mask coverage: `_mask_to_nan` returns a copy. The gates apply
    # support themselves, and an in-place mask would change what every gate is computed from.
    ctx = {"Fm": Fm, "cofire": cofire, "fire": fire, "N": N, "R": R,
           "R_mat": R_mat, "F_mat": F_mat, "em": em, "child_gain": child_gain,
           "support": support, "support_counts": support_counts}
    return raw, ctx


def _orient(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply each detector's frozen sign; any leftover +/-inf becomes NaN."""
    out: dict[str, torch.Tensor] = {}
    for name in DETECTORS:
        m = raw[name] * float(DETECTOR_SIGN[name])
        m = torch.where(torch.isinf(m), torch.full_like(m, _NAN), m)
        out[name] = _nan_diag(m)
    return out


def compute_all(inputs: DetectorInputs, constants: dict,
                s_res_mode: str = "cosine") -> dict[str, torch.Tensor]:
    """Every detector as an oriented [R, R] matrix (higher = more is-a-like), NaN diagonal.

    `s_res_mode` ("cosine" or "probe") picks the `s_res` variant; nothing else depends on it.
    The scorability mask applies here so every caller gets the same `coverage_R`, not a masked
    and an unmasked one under one name.
    """
    raw, _ctx = _compute_raw(inputs, constants, s_res_mode)
    return _orient(raw)


def compute_bundle(inputs: DetectorInputs, constants: dict, s_res_mode: str = "cosine",
                   probe_directions: torch.Tensor | None = None,
                   probe_available: torch.Tensor | None = None) -> dict:
    """`{"detectors", "gates", "support"}`: the gates share the detectors' intermediates, and
    `support` holds the support-mask counts.

    Without `probe_directions`/`probe_available` (from `fit_probe_directions`), `gate_sres_rank`
    is all NaN and the rules reading it are unscorable.
    """
    raw, ctx = _compute_raw(inputs, constants, s_res_mode)
    gc = gate_constants(constants)
    # raw gains, not oriented: `recon_rel_gain_min` is stated on the gain itself
    stats = gates.square_pair_stats(
        ctx["cofire"], ctx["fire"], ctx["R_mat"], ctx["em"], raw["recon_2a"], ctx["child_gain"],
        raw["token_freq_survival"], gc, probe_directions=probe_directions,
        probe_available=probe_available, W_unit=inputs.W_unit)
    gate_vals = score_pairs(stats, gc)["gates"]
    counts = ctx["support_counts"]

    missing = [g for g in gates.GATE_NAMES if g not in gate_vals]
    if missing:
        raise RuntimeError(f"compute_bundle did not produce {missing}; a registered gate "
                           f"silently absent would make its rules untestable for a harness "
                           f"reason, not a measurement one")
    return {"detectors": _orient(raw), "gates": gate_vals, "support": counts}
