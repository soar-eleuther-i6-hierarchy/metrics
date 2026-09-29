"""Per-ordered-pair detectors over the recovered latents, computed from SAE outputs only.

Each detector is an [R, R] matrix, entry [p, c] scoring parent p and child c, with a NaN diagonal.
Undefined cells are NaN, not a filled-in value.

The formulas are `metrics/`'s, called with `undefined=NaN`. This module holds the square frame
(NaN diagonal, masks, orientation, per-parent broadcasts) and the sufficient statistics computed
from token activations. `G`, the decoder cosine, is the one formula defined here: a trial metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metrics.coverage import (
    coverage_asymmetry,
    coverage_legs,
    joint_child_coverage_upper,
    keep_edges,
)
from metrics.independence_null import independence_scores
from metrics.joint_child import r_mass, r_supp
from metrics.outdegree import kept_outdegree
from metrics.reconstruction import (
    edge_reconstruction_condition,
    per_token_ablation_gain,
)
from metrics.rules import (
    GATE_NAMES,
    METRIC_SETTINGS,
    SYNTHETIC_TOYS,
    GateConstants,
    MetricSettings,
    score_pairs,
    squash,
)
from metrics.rules import nan_self_pairs as _nan_diag
from metrics.sibling_redundancy import parent_conditioned_redundancy
from metrics.sibling_redundancy import sibling_redundancy as global_sibling_redundancy
from metrics.sres import sres_scores
from metrics.token_control import frequency_buckets, frequency_controlled_coverage
from scoring.core.gates import probe_correlations, square_pair_stats, support_mask
from scoring.core.registry import DETECTOR_SIGN, DETECTORS

DT = torch.float64
_NAN = float("nan")

# --- scorability mask ---
# NaN unless both endpoints fire >= min_fire_count and co-fire >= min_joint: below that floor
# these co-firing values are arithmetic, not measurement.
MASKED_DETECTORS: tuple[str, ...] = ("coverage_R", "asymmetry_R", "pmi")

# Child side only (fire[c] >= min_fire_count): these live on the child's tokens, so a parent
# that contributes nothing there is negative evidence, not missing data.
CHILD_MASKED_DETECTORS: tuple[str, ...] = ("recon_2a", "recon_child_gain")

# Everything else is unmasked on purpose. Per-parent broadcasts do not depend on the column,
# and a NaN there would spread through `reads.wide_matrix` into `wide`.
# `token_freq_survival` applies `min_joint` itself.


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


def coverage_R(cofire: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Reverse coverage R(p|c) = P(parent fires | child fires), from `metrics.coverage.coverage_legs`.

    A never-firing child gives NaN, not 0.
    """
    R, _ = coverage_legs(cofire, fire, fire, undefined=_NAN)
    return _nan_diag(R)


def _forward_F(cofire: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Forward coverage F(p|c) = P(child fires | parent fires); a dead parent gives NaN.

    The diagonal is kept: `joint_child_coverage_upper` multiplies by the edge mask, and a NaN
    times False is NaN.
    """
    _, F = coverage_legs(cofire, fire, fire, undefined=_NAN)
    return F


def asymmetry_R(R_mat: torch.Tensor, F_mat: torch.Tensor | None = None) -> torch.Tensor:
    """R(p|c) - R(c|p), from `metrics.coverage.coverage_asymmetry`: >0 for a directed containment,
    ~0 for symmetric co-firing. `F_mat` defaults to `R_mat.T`, the same off the diagonal."""
    F_mat = R_mat.transpose(0, 1) if F_mat is None else F_mat
    return _nan_diag(coverage_asymmetry(R_mat, F_mat))


def pmi(cofire: torch.Tensor, fire: torch.Tensor, N: int, min_joint: int) -> torch.Tensor:
    """PMI = log(P(p, c) / (P(p) P(c))), from `metrics.independence_null.independence_scores`;
    NaN under `min_joint` co-firing tokens. Symmetric."""
    return _nan_diag(independence_scores(cofire, fire, fire, N, min_joint=min_joint)["pmi"])


def decoder_cosine(W_unit: torch.Tensor) -> torch.Tensor:
    """`G`, the decoder cosine cos(d_p, d_c) between oriented unit decoders. Symmetric."""
    W = W_unit.double()
    return _nan_diag(W @ W.transpose(0, 1))


def fit_probe_directions(fit_h: torch.Tensor, fit_labels: torch.Tensor, *,
                         gates: GateConstants = SYNTHETIC_TOYS,
                         settings: MetricSettings = METRIC_SETTINGS
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one linear probe per child on `fit_h` and return `(P [R, d], available [R])`.

    The positive-count check reads `fit_labels`, the labels the probe trains on, not the
    scoring labels. A child with too few positives or a failed probe is marked unavailable.
    """
    from metrics.sres import train_probe

    R = int(fit_labels.shape[1])
    d = int(fit_h.shape[1])
    P = torch.zeros((R, d), dtype=DT, device=fit_h.device)
    available = torch.zeros(R, dtype=torch.bool, device=fit_h.device)
    for c in range(R):
        pos = fit_labels[:, c] > gates.fire_threshold    # same firing as compute_bundle
        if int(pos.sum()) < settings.probe_min_pos:
            continue                                  # untestable child -> column NaN
        probe = train_probe(
            fit_h, pos, seed=c,
            neg_ratio=settings.probe_neg_ratio,
            max_tokens=settings.probe_max_tokens,
            steps=settings.probe_steps,
            lr=settings.probe_lr,
            min_neg=settings.probe_min_neg,
        )
        if probe is None:
            continue                                  # too few negatives -> column NaN
        P[c] = probe.to(P.device).double()
        available[c] = True
    return P, available


def s_res_from_directions(P: torch.Tensor, available: torch.Tensor,
                          W_unit: torch.Tensor) -> torch.Tensor:
    """Tree SAE's S_res from frozen probe directions, via `metrics.sres.sres_scores`; NaN diagonal.

    Unit decoders make each correlation a cosine. Asymmetric, since column c uses child c's probe.
    """
    corr = probe_correlations(P, available, W_unit)
    ids = torch.arange(int(corr.shape[0]), device=corr.device)
    return _nan_diag(sres_scores(corr, ids, ids, available))


def edge_mask(R_mat: torch.Tensor, fire: torch.Tensor, tau: float, min_fire: int,
              cofire: torch.Tensor | None = None, min_joint: int = 0) -> torch.Tensor:
    """Inferred edge set, bool [R, R], from `metrics.coverage.keep_edges`: R(p|c) >= tau, both
    endpoints fire >= min_fire, and at least `min_joint` co-firing tokens when `cofire` is given.
    The NaN diagonal of `R_mat` keeps self-pairs out, since NaN >= tau is False."""
    return keep_edges(R_mat, fire, fire, tau, min_fire, cofire=cofire, min_joint=min_joint)


def outdegree(em: torch.Tensor, fire: torch.Tensor, N: int) -> torch.Tensor:
    """Per-parent kept-children count, from `metrics.outdegree.kept_outdegree`, broadcast across
    columns; NaN for a dead parent. Raw direction; the frozen sign flips it so a wide parent
    scores low."""
    return _broadcast_parent(kept_outdegree(em, fire, undefined=_NAN), int(em.shape[0]))


def joint_child_J(F_mat: torch.Tensor, em: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent J(p) = min(1, sum of forward coverage over kept children), from
    `metrics.coverage.joint_child_coverage_upper`, broadcast; NaN for a dead parent."""
    J = joint_child_coverage_upper(F_mat, em)
    J = torch.where(fire > 0, J, torch.full_like(J, _NAN))
    return _broadcast_parent(J, int(F_mat.shape[0]))


def joint_child_mass(acts_rec: torch.Tensor, Fm: torch.Tensor, em: torch.Tensor) -> torch.Tensor:
    """Per-parent share of activation energy on tokens where a kept child also fires, from
    `metrics.joint_child.r_mass`, broadcast; NaN for a parent with zero energy."""
    R = acts_rec.shape[1]
    energy = (acts_rec.double() ** 2)                    # [n, R]
    union_energy = torch.zeros(R, dtype=DT, device=acts_rec.device)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        if kids.numel():
            union_energy[p] = energy[Fm[:, kids].any(dim=1), p].sum()
    return _broadcast_parent(r_mass(union_energy, energy.sum(dim=0), undefined=_NAN), R)


def sibling_redundancy(em: torch.Tensor, cofire: torch.Tensor, fire: torch.Tensor) -> torch.Tensor:
    """Per-parent mean pairwise Jaccard over kept children, from
    `metrics.sibling_redundancy.sibling_redundancy`, broadcast; NaN for fewer than two. Raw
    direction; the frozen sign flips it so a splitting parent scores low."""
    R = int(em.shape[0])
    red = torch.full((R,), _NAN, dtype=DT, device=em.device)
    # max_children=R: every kept child counts, none subsampled
    for p, row in global_sibling_redundancy(em, cofire, fire, max_children=R).items():
        red[p] = row["redundancy"]
    return _broadcast_parent(red, R)


def _recon_gains(acts_rec: torch.Tensor, h: torch.Tensor, W_raw: torch.Tensor,
                 b_dec: torch.Tensor, Fm: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """`(gains [R, R], child_gain [R])` for `recon_2a` and `recon_child_gain`, from
    `metrics.reconstruction`.

    gains[p, c] is the relative reconstruction damage from ablating p on the tokens where c
    fires. Its diagonal is `child_gain`, so it is kept here rather than NaN-ed.
    """
    a = acts_rec.double()
    W = W_raw.double()
    x_hat = a @ W + (b_dec.double() if b_dec is not None else 0.0)
    err = h.double() - x_hat                             # [n, D]
    g = per_token_ablation_gain(a, err, W)               # [n, R]
    Fc = Fm.double()                                     # [n, R] child-firing indicator
    g_sum = g.transpose(0, 1) @ Fc                       # [R(parent), R(child)]
    err_sum = (err ** 2).sum(dim=1) @ Fc                 # [R(child)]
    out = edge_reconstruction_condition(err_sum, g_sum, torch.diagonal(g_sum), undefined=_NAN)
    return out["parent_gain"], out["child_gain"]


def joint_child_supp(Fm: torch.Tensor, em: torch.Tensor,
                     fire: torch.Tensor) -> torch.Tensor:
    """Per-parent share of the parent's firing tokens where a kept child also fires, broadcast.

    `metrics.joint_child.r_supp`, the exact union that `joint_child_J` upper-bounds. A dead
    parent is NaN; a live parent with no kept children is 0.0.
    """
    R = int(em.shape[0])
    union = torch.zeros(R, dtype=DT, device=em.device)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        if kids.numel():
            union[p] = float((Fm[:, p] & Fm[:, kids].any(dim=1)).sum())
    return _broadcast_parent(r_supp(union, fire, undefined=_NAN), R)


def sibling_redundancy_pc(Fm: torch.Tensor, em: torch.Tensor) -> torch.Tensor:
    """Per-parent mean pairwise sibling Jaccard within the parent's firing tokens, from
    `metrics.sibling_redundancy.parent_conditioned_redundancy`, broadcast.

    Siblings that never fire inside the parent score 0; fewer than two kept children or a dead
    parent is NaN.
    """
    R = int(em.shape[0])
    out = torch.full((R,), _NAN, dtype=DT, device=em.device)
    for p in range(R):
        kids = em[p].nonzero(as_tuple=True)[0]
        out[p] = parent_conditioned_redundancy(Fm[:, p], Fm[:, kids], undefined=_NAN)
    return _broadcast_parent(out, R)


def token_freq_survival(Fm: torch.Tensor, tokens: torch.Tensor, vocab: int, min_joint: int,
                        settings: MetricSettings) -> torch.Tensor:
    """Whether coverage survives removing common tokens, from `metrics.token_control`: x / (1 + x),
    with x = R(p|c) on the rare-token buckets {1, 2} over R(p|c) on all tokens.

    NaN where unmeasurable. Not gated on the edge set, which would NaN the shared-token pairs
    this exists to score.
    """
    counts = torch.bincount(tokens, minlength=vocab)
    id_bucket = frequency_buckets(counts, high_mass=settings.freq_high_mass,
                                  mid_mass=settings.freq_mid_mass)
    if int(id_bucket.max()) >= settings.n_freq_buckets:
        raise ValueError(f"n_freq_buckets={settings.n_freq_buckets} leaves out bucket "
                         f"{int(id_bucket.max())}; its tokens would drop from both coverages")
    tok_bucket = id_bucket[tokens]                                    # [n]
    Fb = [(Fm & (tok_bucket == k).reshape(-1, 1)).double()
          for k in range(settings.n_freq_buckets)]
    cofire_b = torch.stack([f.transpose(0, 1) @ f for f in Fb])       # [buckets, R, R]
    fire_b = torch.stack([f.sum(dim=0) for f in Fb])                  # [buckets, R]
    cofire_all = cofire_b.sum(dim=0)
    measurable = (cofire_all >= min_joint) & (cofire_all > 0)
    # unclamped, and x/(1+x) below rather than a clamp, so the >1 tail stays ranked; a child
    # that never fires on rare tokens scores 0: a zero rare-bucket rate is the signal
    ratio = frequency_controlled_coverage(cofire_b, fire_b, measurable,
                                          min_fire_low=settings.freq_min_fire_low,
                                          clamp_max=None,
                                          no_rare_firing_scores_zero=True)["survival"]
    return _nan_diag(squash(ratio))


# --- compute_bundle ---
def _compute_raw(inputs: DetectorInputs, gates: GateConstants, settings: MetricSettings
                 ) -> tuple[dict[str, torch.Tensor], dict]:
    """Every detector in raw orientation, plus the intermediates the gates are built from, so
    gates and detectors share one set of firing counts, edge set and gains."""
    if inputs.h is None or inputs.b_dec is None or inputs.tokens is None:
        missing = [n for n in ("h", "b_dec", "tokens") if getattr(inputs, n) is None]
        raise ValueError(
            f"compute_bundle needs {missing} (recon_2a / token_freq_survival read them); "
            f"the DetectorInputs must carry h/b_dec/tokens before scoring")
    R = int(inputs.acts_rec.shape[1])
    Fm = inputs.acts_rec > gates.fire_threshold
    cofire, fire, N = cofiring(Fm)
    R_mat = coverage_R(cofire, fire)
    F_mat = _forward_F(cofire, fire)
    em = edge_mask(R_mat, fire, gates.edge_tau, gates.min_fire_count,
                   cofire=cofire, min_joint=gates.min_joint)
    gains, child_gain = _recon_gains(inputs.acts_rec, inputs.h, inputs.W_raw, inputs.b_dec, Fm)
    # after `em`, which is built from unmasked coverage: the edge set never sees the mask
    support, support_counts = support_mask(cofire, fire, gates.min_fire_count, gates.min_joint)
    child_ok = (fire >= float(gates.min_fire_count)).reshape(1, -1).expand(R, R)

    raw = {
        "coverage_R": R_mat,
        "asymmetry_R": asymmetry_R(R_mat),
        "joint_child_J": joint_child_J(F_mat, em, fire),
        "pmi": pmi(cofire, fire, N, gates.min_joint),
        "token_freq_survival": token_freq_survival(Fm, inputs.tokens, inputs.vocab,
                                                   gates.min_joint, settings),
        "recon_2a": _nan_diag(gains),
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


def compute_bundle(inputs: DetectorInputs, *, gates: GateConstants = SYNTHETIC_TOYS,
                   settings: MetricSettings = METRIC_SETTINGS,
                   probe_directions: torch.Tensor | None = None,
                   probe_available: torch.Tensor | None = None) -> dict:
    """`{"detectors", "gates", "support"}`: the gates share the detectors' intermediates, and
    `support` holds the support-mask counts.

    Without `probe_directions`/`probe_available` (from `fit_probe_directions`), `gate_sres_rank`
    is all NaN and the rules reading it are unscorable.
    """
    raw, ctx = _compute_raw(inputs, gates, settings)
    # raw gains, not oriented: `recon_rel_gain_min` is stated on the gain itself
    stats = square_pair_stats(
        ctx["cofire"], ctx["fire"], ctx["R_mat"], ctx["em"], raw["recon_2a"], ctx["child_gain"],
        raw["token_freq_survival"], gates, probe_directions=probe_directions,
        probe_available=probe_available, W_unit=inputs.W_unit, n_tokens=int(ctx["N"]))
    gate_vals = score_pairs(stats, gates)["gates"]
    counts = ctx["support_counts"]

    missing = [g for g in GATE_NAMES if g not in gate_vals]
    if missing:
        raise RuntimeError(f"compute_bundle did not produce {missing}; a registered gate "
                           f"silently absent would make its rules untestable for a harness "
                           f"reason, not a measurement one")
    return {"detectors": _orient(raw), "gates": gate_vals, "support": counts}
