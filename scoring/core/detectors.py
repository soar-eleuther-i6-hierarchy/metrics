"""
detectors — the per-ordered-pair scalars, computed from SAE outputs only.

Reads held-out activations and oriented unit decoders of the recovered latents; returns an
`[R, R]` matrix per detector, entry `[p, c]` = score for ordered pair parent=p, child=c.
Labels never enter here — truth lives only in `scoring.core.grid`.

Conventions (see `scoring.core.registry`):
  - firing = activation > 0 (BatchTopK: nonzero set is its top-k); diagonal is always NaN;
  - co-firing detectors (coverage_R, asymmetry_R, pmi) use fixed smoothing, so a zero-fire
    endpoint gives a finite value on purpose;
  - energy/reconstruction/frequency detectors and per-parent graph detectors use no
    smoothing, so a true 0/0 cell is NaN — never a filled-in value, never +/-inf;
  - `compute_all` applies each detector's frozen sign so higher == more is-a-like.

Implemented directly here (not in `metrics/`) so undefined cells aren't hidden by
zero-denominator clamping. `s_res`'s probe variant trains via `metrics.sres.train_probe`.

A SCORABILITY MASK runs inside `compute_all`, so it reaches every caller and not only the
benchmark -- see `MASKED_DETECTORS` for what it touches and `compute_all` for why that is the
intended scope and what it costs.
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

# --------------------------------------------------------------------------
# the scorability mask -- SELECTIVE, and the exclusions are the load-bearing part
# --------------------------------------------------------------------------
# Detectors that take the FULL support mask: both endpoints must fire at least
# `min_fire_count` times AND they must co-fire at least `support_min_joint` times. All three
# are pair-level co-firing quantities, and below that floor they are arithmetic rather than
# measurement -- a child firing 20 times inside a near-always-on parent reaches coverage 1.0
# on no evidence beyond base rate.
MASKED_DETECTORS: tuple[str, ...] = ("coverage_R", "asymmetry_R", "pmi")

# Detectors that take the CHILD side of it only (`fire[c] >= min_fire_count`). Both are
# defined over the tokens where the CHILD fires, so the joint count is not their support:
# a parent that contributes nothing on those tokens is a real measured answer, not a missing
# one, and NaN-ing it would delete the negative evidence.
CHILD_MASKED_DETECTORS: tuple[str, ...] = ("recon_2a", "recon_child_gain")

# EVERYTHING ELSE IS DELIBERATELY UNMASKED, and not out of caution.
#   * the per-parent broadcasts (`outdegree`, `joint_child_J`, `joint_child_supp`,
#     `joint_child_mass`, `sibling_redundancy`, `sibling_redundancy_pc`) have a value that does
#     not depend on the column at all. NaN-ing `outdegree[p, c]` because p and c rarely co-fire
#     deletes a number that was never about that pair, and it propagates through
#     `reads.wide_matrix` into `wide`, gutting the scorable population of every rule reading it.
#   * the geometry channel (`s_res` -> `G`, `S_res`) never reads tokens. A pair that never
#     co-fires still has a perfectly well-defined decoder cosine.
#   * `token_freq_survival` applies `min_joint` internally already (see its own guard), which
#     is why `support_min_joint` is a SEPARATELY NAMED constant: tuning the mask through
#     `min_joint` would silently retune the frequency detector as well.
#
# `em` is built from the UNMASKED coverage matrix, so the inferred edge set -- and therefore
# every per-parent broadcast computed from it -- is bit-identical to what it was before the
# mask existed. `edge_mask` carries its own joint-support guard.


def _mask_to_nan(m: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    return torch.where(keep, m, torch.full_like(m, _NAN))


@dataclass(frozen=True)
class DetectorInputs:
    """SAE-side inputs for the recovered latents (R of them), over n held-out tokens.

    acts_rec [n, R]  float activations of the matched latent per recovered feature
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
    """A per-CHILD vector -> an [R, R] matrix constant down COLUMNS, NaN diagonal.

    The transpose of `_broadcast_parent`, and the pair is why both exist by name: the two
    produce matrices of the same shape with the same values in them, so a swap is invisible
    except in the answer. `entry[p, c] = vec[c]` here; `vec[p]` there.
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

    A never-firing child gives 0/0 -> NaN, not a fake 0. `eps` is kept for signature
    compatibility only; it no longer affects a defined cell.
    """
    fire_c = fire.reshape(1, -1)
    denom = torch.where(fire_c > 0, fire_c, torch.full_like(fire_c, _NAN))
    return _nan_diag(cofire / denom)


def _forward_F(cofire: torch.Tensor, fire: torch.Tensor, eps: float) -> torch.Tensor:
    """Forward coverage F(p|c) = cofire[p,c] / fire[p] = P(child fires | parent fires).

    Same convention as `coverage_R`: a dead parent (fire[p]==0) gives NaN, not an
    epsilon-smoothed value.
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
    """Decoder-geometry overlap = cos(d_c, d_p) between oriented unit decoders.

    Equals cos(g_c, g_p) = alpha_c * sqrt(1 - alpha_p^2) for an is-a edge, 0 for the
    orthogonal null. Symmetric. Cheap analytic geometry oracle; the trained path uses
    `s_res_probe` instead.
    """
    W = W_unit.double()
    return _nan_diag(W @ W.transpose(0, 1))


def fit_probe_directions(fit_h: torch.Tensor, fit_labels: torch.Tensor,
                         constants: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one linear probe per child on `fit_h`, returning `(P, available)`.

    `P` is `[R, d]`, row `c` being child c's probe direction in the residual basis (or zeros
    where no probe exists); `available` is `[R]` bool. Separating this from the scoring step
    makes "freeze the fitted directions before computing scores" STRUCTURAL rather than a
    comment, and makes the directions a first-class object that can be persisted and hashed.

    The support gate reads `fit_labels`, because that is what the probe trains on. Reading the
    SCORING labels here would let a child with plenty of scoring-draw positives but almost none
    in the fitting draw produce a direction fitted on nothing.

    A child with too few positives, or one whose probe fails to train, gets `available[c] =
    False` and an all-NaN column downstream -- never a fake 0. Seed is the recovered position
    `c`, so the fit is deterministic.
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
    """Score FROZEN probe directions against unit decoders. A pure function of its arguments.

    `out[p, c] = min(corr[p], corr[c])` where `corr = W_unit @ P[c]`, asymmetric because column
    `c` uses child c's probe. Correlating against UNIT decoders is what makes `corr` a cosine
    (bounded by 1). The corpus is not reachable from here, which is the point of the split.
    """
    Wu = W_unit.double()
    R = int(Wu.shape[0])
    out = torch.full((R, R), _NAN, dtype=DT, device=Wu.device)   # inherit device (GPU-safe)
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
    """Probe-based s_res (Tree-SAE's probe metric). Trains a linear probe per child on the
    residual stream to predict its firing, then returns
    s_res(p,c) = min over {parent, child} of the probe's cosine with that latent's unit decoder.

    Positives for child c are `label_acts[:,c] > 0` (`label_acts` defaults to `acts_rec`, the
    deployed self-label; the diagnostic passes true firing instead). A child with too few
    positives (or an untestable probe) gives an all-NaN column, never a fake 0.

    Probe direction is correlated against UNIT decoders so `corr` is a cosine (<=1 bound);
    then `out[p,c] = min(corr[p], corr[c])`, asymmetric since column c uses child-c's probe.
    Seed is the recovered position c, so it's deterministic.

    `fit_h` / `fit_labels` select a SEPARATE FITTING DRAW (`PRECOMMIT.md` s6 step 2). They
    DEFAULT TO `h` / the scoring labels, i.e. to the unseparated behaviour, so all 13 existing
    call sites -- `compute_all` and `training/score_trained.py` among them -- stay byte-identical
    and the pilot reference and harness gate remain valid. Only the benchmark opts in.

    What the separation removes is the shared sampling noise between the fitted probe and both
    the co-firing clauses it is conjoined with and the `S_res` null quantile calibrated on the
    same draw. It does NOT remove the self-LABEL circularity (`probe_self_W` training on the
    SAE's own firing), which is untouched, and it does not make `S_res` held out the way `pmi`
    is: `S_res` is never evaluated on tokens at all.

    Note: this scores a normalized min-cosine signal, not gemma's raw-dot top-k rule — the
    two are not yet aligned.
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
    """s_res variants for toy calibration/diagnostics. Reads ground truth (`g_unit`, `A_rec`),
    so this is a diagnostic, NOT a firewalled detector — never call on the trained scoring path.

      cosine_g     : analytic cosine over the TRUE unit directions g.
      probe_true_g : probe on TRUE firing, correlated against g — calibrates the probe machinery.
      probe_self_W : probe on the SELF-label, learned decoders — the deployed detector.
      probe_true_W : probe on TRUE firing, learned decoders.

    Caller reports `self_label_bias = probe_self_W - probe_true_W` (AUROC-level) to isolate
    the self-label's circularity cost.
    """
    return {
        "cosine_g": s_res_cosine(g_unit),
        "probe_true_g": s_res_probe(A_rec, h, g_unit, constants, label_acts=A_rec),
        "probe_self_W": s_res_probe(acts_rec, h, W_unit, constants),
        "probe_true_W": s_res_probe(acts_rec, h, W_unit, constants, label_acts=A_rec),
    }


def edge_mask(R_mat: torch.Tensor, fire: torch.Tensor, tau: float, min_fire: int,
              cofire: torch.Tensor | None = None, min_joint: int = 0) -> torch.Tensor:
    """Inferred edge set: R(p|c) >= tau, both endpoints fire >= min_fire, and (when
    `cofire`/`min_joint` given) at least `min_joint` co-firing tokens. Bool [R,R].

    The joint-support gate kills chance edges from a rarely-firing child inside a
    near-always-on parent; NaN diagonal compares False so self-edges drop out automatically.
    """
    keep = R_mat >= tau                                  # NaN >= tau -> False
    enough = fire >= min_fire
    keep = keep & enough.reshape(-1, 1) & enough.reshape(1, -1)
    if cofire is not None and min_joint > 0:
        keep = keep & (cofire >= min_joint)
    return keep


def outdegree(em: torch.Tensor, fire: torch.Tensor, N: int) -> torch.Tensor:
    """Per-parent out-degree (kept-children count), broadcast across columns. Raw direction
    (higher = more children); `compute_all` flips sign so a wide superparent scores LOW. A
    dead parent has no evidence, so NaN rather than a confident 0."""
    R = em.shape[0]
    outdeg = em.double().sum(dim=1)
    outdeg = torch.where(fire > 0, outdeg, torch.full_like(outdeg, _NAN))
    return _broadcast_parent(outdeg, R)


def joint_child_J(F_mat: torch.Tensor, em: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent J(p) = sum over kept children of forward-coverage(p,c), capped at 1.
    A dead parent (never fires) is undefined, so NaN."""
    R = F_mat.shape[0]
    contrib = torch.where(em, F_mat, torch.zeros_like(F_mat))
    J = contrib.sum(dim=1).clamp(max=1.0)
    J = torch.where(fire > 0, J, torch.full_like(J, _NAN))
    return _broadcast_parent(J, R)


def joint_child_mass(acts_rec: torch.Tensor, Fm: torch.Tensor, em: torch.Tensor) -> torch.Tensor:
    """Per-parent r_mass(p) = share of the parent's activation energy landing on tokens where
    >=1 kept child also fires. Dead parent (zero energy) gives 0/0 -> NaN, never 0 or 0.5.
    """
    R = acts_rec.shape[1]
    energy = (acts_rec.double() ** 2)                    # [n, R]
    energy_total = energy.sum(dim=0)                     # [R]
    r_mass = torch.full((R,), _NAN, dtype=DT, device=acts_rec.device)  # match input device (CUDA-safe)
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
    """Per-parent mean pairwise Jaccard over its kept children (sibling overlap), broadcast
    across columns. Undefined (NaN) for <2 kept children. Raw direction (higher = more
    redundant); frozen sign flips it so a splitting parent scores LOW.
    """
    R = em.shape[0]
    red = torch.full((R,), _NAN, dtype=DT, device=em.device)  # match input device (CUDA-safe)
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
    """`(gains [R, R] with its diagonal INTACT, child_gain [R])`.

    Shared by `recon_2a` and `recon_child_gain`, which are the off-diagonal and the diagonal of
    one matrix. `gains[p, c]` is the relative reconstruction damage from ablating p over the
    tokens where c fires, so `gains[c, c]` is the damage from ablating the CHILD over its own
    tokens -- `metrics.reconstruction`'s `child_gain = g_child_sum / err_sum_c`, exactly.

    The diagonal is returned unmasked because `_nan_diag` would otherwise delete the only place
    `child_gain` is computed, which is why this quantity was missing from the package.
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
    """Reconstruction-ablation parent gain: how much removing the parent would worsen
    reconstruction, over tokens where the child fires.

    parent_gain[p,c] = sum(g[t,p]) / sum(||err[t]||^2) over tokens where child c fires, with
    err = h - (acts @ W_raw + b_dec) and g[t,f] = 2 a_f<d_f,err_t> + a_f^2||d_f||^2. A
    never-firing child gives 0 denominator -> NaN.
    """
    gains, _child = _recon_gains(acts_rec, h, W_raw, b_dec, Fm)
    return _nan_diag(gains)


def recon_child_gain(acts_rec: torch.Tensor, h: torch.Tensor, W_raw: torch.Tensor,
                     b_dec: torch.Tensor, Fm: torch.Tensor) -> torch.Tensor:
    """Per-CHILD reconstruction gain, broadcast down each column.

    `metrics.reconstruction.edge_reconstruction_condition`'s `child_gain`: how much worse
    reconstruction gets on the child's own tokens if the CHILD is ablated. It is the second
    half of her edge condition -- the parent must contribute, and the child must add something
    beyond the parent -- and the package carried only the first half.

    Broadcast down a COLUMN (`entry[p, c] = child_gain[c]`), matching her `.unsqueeze(0)`. A
    never-firing child gives 0/0 -> NaN, so its column is undefined rather than zero.
    """
    _gains, child = _recon_gains(acts_rec, h, W_raw, b_dec, Fm)
    return _broadcast_child(child, int(acts_rec.shape[1]))


def joint_child_supp(Fm: torch.Tensor, em: torch.Tensor,
                     fire: torch.Tensor) -> torch.Tensor:
    """Per-parent EXACT joint-child coverage: share of the parent's firing tokens on which at
    least one kept child also fires. Broadcast across columns.

    `metrics.joint_child.r_supp`, the union count over tokens. `joint_child_J` is the
    closed-form upper bound `min(1, sum of forward coverages)`, which double-counts every pair
    of co-firing siblings and therefore saturates near 1 for any parent with several children
    regardless of structure. Both are registered: the bound is what the frozen rules were
    calibrated against, the union is the quantity it was approximating.

    A dead parent is NaN (hers clamps the denominator to 1 and returns 0). A live parent with
    no kept children is 0.0 -- a real answer, not missing data.
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
    """Per-parent mean pairwise sibling Jaccard RESTRICTED to the parent's firing tokens.

    `metrics.sibling_redundancy.parent_conditioned_redundancy`, and her docstring says why it
    is the corrected form: the property under test is whether the children partition the
    PARENT's firing set, while the global Jaccard also scores co-firing where the parent is
    silent, which is irrelevant to this parent's partition. Registered beside the global
    `sibling_redundancy` rather than replacing it.

    Her arithmetic inside the defined domain, unchanged: the mean over the off-diagonal (each
    sibling pair counted twice, which a symmetric matrix makes identical to the upper-triangle
    mean), and the union clamped at 1 so two siblings that never fire inside the parent score
    0 redundancy rather than NaN -- there the answer really is "not redundant".

    Scoring's convention where hers differs: fewer than two kept children, or a parent that
    never fires, is NaN. Hers returns 0.0, which is indistinguishable from a measured
    partition and would let a childless parent look like a healthy one.
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
    """Frequency-controlled coverage survival: does an edge hold up once common tokens are
    removed?

    survival(p,c) = R(p|c) over rare-token buckets {1,2} divided by R(p|c) over all buckets;
    ~1 means a real relationship, ~0 means common-token coincidence. NaN where unmeasurable
    (no co-firing, too-rare firing, or too few joint co-fires). Deliberately NOT gated on
    being a kept edge, since that would NaN exactly the shared-token pairs this needs to score.
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
    # Squash ratio into [0,1) via x/(1+x): rank-preserving, avoids tying the >1 tail like a hard clamp would.
    ratio = R_rest / R_all.clamp_min(1e-12)
    survival = ratio / (1.0 + ratio)

    # Floor support on TOTAL child firing (not rare-bucket count), since zero rare-bucket rate is the signal, not missing data; min_joint kills chance pairs.
    undefined = ((fire_all.reshape(1, -1) < min_fire_low) | (R_all <= 0)
                 | (cofire_all < min_joint))
    survival = torch.where(undefined, torch.full_like(survival, _NAN), survival)
    return _nan_diag(survival)


# --------------------------------------------------------------------------
# the handoff
# --------------------------------------------------------------------------
def _compute_raw(inputs: DetectorInputs, constants: dict,
                 s_res_mode: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Every detector in its RAW orientation, plus the intermediates the gates need.

    Split out of `compute_all` so `compute_bundle` can build the gates from the same firing
    counts, edge set and reconstruction gains the detectors were computed from. Recomputing
    them alongside would let a gate and the metric it is supposed to agree with drift apart
    while both still look right.
    """
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
    # One pass for both halves of the reconstruction condition (`recon_2a` is the off-diagonal,
    # `recon_child_gain` the diagonal of the same matrix).
    gains, child_gain = _recon_gains(inputs.acts_rec, inputs.h, inputs.W_raw, inputs.b_dec, Fm)
    # Built AFTER `em`, so the edge set never sees the mask (see MASKED_DETECTORS).
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
    # Applied HERE, after every detector is built, rather than to the inputs. Masking `R_mat`
    # before `em` or before `asymmetry_R` would work too -- the support mask is symmetric in
    # the endpoints -- but it would make the set of masked detectors implicit in the dataflow
    # instead of a list anyone can read.
    for name in MASKED_DETECTORS:
        raw[name] = _mask_to_nan(raw[name], support)
    for name in CHILD_MASKED_DETECTORS:
        raw[name] = _mask_to_nan(raw[name], child_ok)

    # `ctx["R_mat"]` is the PRE-MASK coverage matrix: `_mask_to_nan` returns a new tensor, so
    # rebinding `raw["coverage_R"]` above left this one alone. `compute_bundle` wants it that
    # way -- `directed_coverage_gates` ANDs with `support` itself, so handing it the masked copy
    # would mask twice. Nothing numeric turns on it today (a masked cell is unsupported, and
    # `NaN >= tau` is False either way), but an in-place mask here would silently change what
    # every gate is computed from, which is why the copy is stated rather than assumed.
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
    """Return every detector as an oriented [R, R] matrix (higher == is-a-like), NaN diagonal.

    Applies each detector's frozen `DETECTOR_SIGN`; any leftover +/-inf becomes NaN.

    `s_res_mode` selects the s_res variant:
      "probe"  -- the real Tree-SAE probe metric; only mode allowed for a REPORTED s_res cell.
      "cosine" -- cheap analytic geometry oracle; diagnostic/test default, never for a reported grid.
    Only s_res depends on the mode; the other detectors are unaffected.

    THE SCORABILITY MASK APPLIES HERE, AND THEREFORE TO EVERY CALLER. `coverage_R`,
    `asymmetry_R` and `pmi` are NaN below the support floor and three detectors were added, so
    every consumer of this function sees both changes -- `scoring/oracle/score_dump.py`,
    `scoring/oracle/validate_metrics.py`, `scoring/trained/retrieval.py`,
    `training/score_trained.py`, `scoring/run_validate.py` and `scoring/run_scoring.py`, none
    of which asked for it. Their per-class tables gain three rows and recompute three existing
    ones over a smaller population.

    That is deliberate. The alternative is a masked `coverage_R` on one path and an unmasked one
    on another, which is two metrics under one name in a single repo -- the exact duplication
    that `metrics/` versus `scoring/` already cost this project once. A pair whose endpoints
    barely co-fire has a coverage of 1.0 by arithmetic on either path, and that is not a
    measurement on either path.

    What it does NOT come with is a schema bump for those packages: `registry.REPORT_SCHEMA`
    covers the benchmark artifacts only. A table produced by one of the six above before this
    change and one produced after are not comparable, and nothing in their own provenance says
    so. Measured on `only_isa` at 3k tokens: 44.1% of ordered pairs fall below the floor.
    """
    raw, _ctx = _compute_raw(inputs, constants, s_res_mode)
    return _orient(raw)


def compute_bundle(inputs: DetectorInputs, constants: dict, s_res_mode: str = "cosine",
                   probe_directions: torch.Tensor | None = None,
                   probe_available: torch.Tensor | None = None) -> dict:
    """`{"detectors", "gates", "support"}`, the gates built from the same firing counts, edge
    set and gains as the detectors. `support` holds the support-mask counts.

    `probe_directions`/`probe_available` come from `fit_probe_directions` on the separate fitting
    draw; without them `gate_sres_rank` is all NaN and the rules reading it are unscorable.
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
