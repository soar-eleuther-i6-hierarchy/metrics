"""
gates — the FIXED-threshold pass/fail rules, in the square `[R, R]` frame.

Every rule here is a transcription of one that already exists in `metrics/` and `config.py`,
moved into scoring's frame and its NaN conventions. Nothing is fitted: a gate compares a
measured quantity against a constant chosen once, so the same rule means the same thing on a
24-feature toy and on a 16k-latent SAE. That is the property the null-quantile rule it replaces
did not have -- a q99 cut accepts 1% of whatever population it is pointed at, so it is
computable at Gemma scale and useless there.

TRISTATE, NEVER BOOLEAN. Each gate is a float matrix over {1.0, 0.0, NaN}:

    1.0   the rule holds for this ordered pair
    0.0   the rule does not hold
    NaN   the pair was never measurable, so neither answer was earned

The NaN state is not decoration. A boolean gate has no way to say "no evidence", so an
unmeasurable pair silently becomes a rejection, and a conjunction of rejections reads as a
confident negative. `predicates.passes` / `predicates.fails` are both written literally against
the finite values, so NaN satisfies NEITHER.

Booleans must also not be stored as booleans anywhere downstream, for three separately silent
reasons, all verified: `Tensor.fill_diagonal_(nan)` on a bool tensor sets the diagonal TRUE
rather than raising; `torch.isfinite` on a bool tensor is all-True, so a finiteness gate would
call an undefined cell scorable; and multiplying a flag by a `DETECTOR_SIGN` is meaningless.
Hence float64 tristates, and hence gates live here rather than in `DETECTORS`.

DIAGONAL. Every gate NaNs its own diagonal, for the same reason the detectors do: a feature is
not its own parent, and `metrics.in_block.duplicate` would otherwise read True there (a feature
covers itself perfectly in both directions).

WHERE THE CONSTANTS COME FROM. `config.py`, unchanged in value:
`EDGE_TAU` 0.5, `MIN_FIRE_COUNT` 20, `MIN_JOINT` 30, `RECON_REL_GAIN_MIN` 0.01,
`SRES_RANK_TOP_K` 5, `SUPERPARENT_OUTDEG_FRAC` 0.30, `FREQ_SURVIVAL_MIN` 0.5.
They are eyeballed values, not derived ones (Chanin's absorption paper sets its cutoffs the
same way and says so). Adopting them relocates the arbitrariness somewhere visible; it does not
remove it. The defence is that one fixed rule applies identically to every arm of a comparison,
so the choice cancels -- which is what a dose-response with a frozen rule already does.
"""

from __future__ import annotations

import torch

DT = torch.float64
_NAN = float("nan")

# The registered gate names, in a fixed order. `gate_`-prefixed so a collision with a metric
# name is structurally impossible rather than merely unlikely.
GATE_NAMES: tuple[str, ...] = (
    "gate_support", "gate_parent_of", "gate_duplicate", "gate_recon",
    "gate_sres_rank", "gate_superparent", "gate_freq_survives",
)

# The constants the gates compare against. ONE list: the freeze record and every artifact's
# `__meta__` both stamp it, and two copies would let an eighth constant land in one and not the
# other -- visible in neither, because each side would look complete on its own.
GATE_CONSTANT_KEYS: tuple[str, ...] = (
    "edge_tau", "min_fire_count", "support_min_joint", "recon_rel_gain_min",
    "sres_rank_top_k", "superparent_outdeg_frac", "freq_survival_min_raw",
)


# Which REGISTERED METRICS each gate decides on. Provenance, and load-bearing: the expressions
# name gates, so without this nothing connects a rule to the metric distributions reported
# beside it, and `evaluate.constant_flags`' `constant_target` -- the flag that caught the
# degenerate superparent separation -- has no target to attach to any metric.
#
# `gate_support` lists nothing on purpose: it is built from raw firing and co-firing counts,
# which are not registered metrics, and inventing an association would attach a target to a
# metric no rule decides on.
GATE_SOURCES: dict[str, tuple[str, ...]] = {
    "gate_support": (),
    "gate_parent_of": ("coverage_R", "asymmetry_R"),
    "gate_duplicate": ("coverage_R", "asymmetry_R"),
    "gate_recon": ("recon_2a", "recon_child_gain"),
    "gate_sres_rank": ("S_res",),
    "gate_superparent": ("outdegree", "wide"),
    "gate_freq_survives": ("token_freq_survival",),
}
assert set(GATE_SOURCES) == set(GATE_NAMES), "a gate has no declared source metrics"


def nan_diag(m: torch.Tensor) -> torch.Tensor:
    """Self-pairs are not pairs. Float only -- on a bool tensor this sets the diagonal True."""
    if not m.is_floating_point():
        raise TypeError("nan_diag needs a float tensor; on a bool tensor fill_diagonal_(nan) "
                        "silently writes True")
    m = m.clone()
    m.fill_diagonal_(_NAN)
    return m


def tristate(flag: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    """`{1.0, 0.0, NaN}` from a pass flag and the domain on which it was measurable.

    `defined` is required, not optional. Every gate has a domain outside which its inputs do
    not exist -- a pair that never co-fires, a child with no probe, a parent that never fires --
    and defaulting it to all-true converts every one of those into a confident rejection.
    """
    if flag.shape != defined.shape:
        raise ValueError(f"flag {tuple(flag.shape)} and defined {tuple(defined.shape)} "
                         f"describe different frames")
    out = torch.where(defined, flag.to(DT), torch.full(flag.shape, _NAN, dtype=DT,
                                                       device=flag.device))
    return nan_diag(out) if out.ndim == 2 and out.shape[0] == out.shape[1] else out


def squash(ratio: float) -> float:
    """`x / (1 + x)`, the transform `detectors.token_freq_survival` reports its ratio under.

    A threshold stated on the RAW ratio must be squashed before it is compared against that
    detector. `config.FREQ_SURVIVAL_MIN = 0.5` means "the rare-token coverage is at least half
    the all-token coverage"; squashed that is 0.333. Comparing the reported value against 0.5
    unconverted would demand a raw ratio of 1.0 -- twice as strict as intended, and with no
    error anywhere to say so.
    """
    x = float(ratio)
    return x / (1.0 + x)


# --------------------------------------------------------------------------
# support
# --------------------------------------------------------------------------
def support_mask(cofire: torch.Tensor, fire: torch.Tensor, min_fire: int,
                 min_joint: int) -> tuple[torch.Tensor, dict]:
    """`(bool [R, R], counts)`: which ordered pairs carry enough evidence to be scored at all.

    A pair is supported when both endpoints fire at least `min_fire` times and they co-fire at
    least `min_joint` times. This is `metrics.coverage.keep_edges`' guard promoted from an
    edge-set filter to a SCORABILITY test: without it, a child firing 20 times inside a
    near-always-on parent reaches coverage 1.0 on no evidence beyond base rate.

    The counts are returned rather than logged because the excluded fraction is a first-class
    number -- it says how much of the frame the guard removed, which is the difference between
    "the rule rejected these pairs" and "the rule could not see them". They EXCLUDE the
    diagonal, which is not a pair.

    The diagonal of the mask itself is left as the arithmetic produces it (a feature co-fires
    with itself on every token it fires) because every masked detector already NaNs its own
    diagonal, so the value there cannot reach a reported number.
    """
    enough = fire >= float(min_fire)
    keep = enough.reshape(-1, 1) & enough.reshape(1, -1) & (cofire >= float(min_joint))
    R = int(keep.shape[0])
    off = ~torch.eye(R, dtype=torch.bool, device=keep.device)
    n_pairs = int(off.sum())
    n_supported = int((keep & off).sum())
    counts = {
        "n_pairs": n_pairs,
        "n_supported": n_supported,
        "n_excluded": n_pairs - n_supported,
        "frac_excluded": (n_pairs - n_supported) / n_pairs if n_pairs else float("nan"),
        # By cause, because the two have different remedies: a low-firing endpoint needs more
        # tokens, a low joint count may mean the pair is simply unrelated. NOT A PARTITION --
        # a pair can fail both, so these overlap and do not sum to `n_excluded`.
        "n_excluded_low_fire": int(((~(enough.reshape(-1, 1) & enough.reshape(1, -1))) & off).sum()),
        "n_excluded_low_joint": int(((cofire < float(min_joint)) & off).sum()),
        "n_excluded_both_causes": int(
            ((~(enough.reshape(-1, 1) & enough.reshape(1, -1)))
             & (cofire < float(min_joint)) & off).sum()),
        "min_fire": int(min_fire), "min_joint": int(min_joint),
    }
    return keep, counts


def support_gate(support: torch.Tensor) -> torch.Tensor:
    """`gate_support` as a reported tristate.

    Its `defined` domain really is the whole frame, and that is not a default: firing and
    co-firing counts exist for every pair in a scored universe, so "is this pair supported"
    always has an answer. It is registered so a rule can require support explicitly and so the
    excluded population is visible per pair, not only as an aggregate count.
    """
    return tristate(support, torch.ones_like(support, dtype=torch.bool))


# --------------------------------------------------------------------------
# directed coverage: the parent / duplicate split  (metrics/in_block.py)
# --------------------------------------------------------------------------
def directed_coverage_gates(R_mat: torch.Tensor, support: torch.Tensor,
                            tau: float) -> dict[str, torch.Tensor]:
    """`{gate_parent_of, gate_duplicate}` from coverage asymmetry at a FIXED tau.

    Transcribed from `metrics.in_block.directed_coverage`, whose header is "Direction and
    duplicates from coverage asymmetry":

        parent_of[p, c] = R[p, c] >= tau AND R[c, p] <  tau
        duplicate[p, c] = R[p, c] >= tau AND R[c, p] >= tau

    both restricted to supported pairs. Together they are the fixed-threshold form of the two
    coverage clauses the quantile rule spelled `HIGH(coverage_R) AND HIGH(asymmetry_R)`:
    containment in one direction, and NOT in the other, which is what separates a real
    parent from a co-extensive rename.

    `parent_of` is antisymmetric by construction, so the induced graph is acyclic; duplicates
    are reported separately and are never an edge, which is what would create 2-cycles.

    ONE FRAME DIFFERENCE from `metrics/`, and it matters. There `R = cofire / fire.clamp(1.0)`,
    so a never-firing endpoint gets coverage 0 rather than NaN, and the diagonal is removed by
    an explicit `~eye`. Here `R_mat` is `detectors.coverage_R`, which is NaN on the diagonal and
    NaN for a dead child. `NaN >= tau` is False, so an undefined forward direction cannot pass;
    and because `support` already requires both endpoints to fire, the reverse comparison
    `~(R[c,p] >= tau)` is never reading a NaN as "not covered" inside the domain that survives.
    """
    ge = R_mat >= float(tau)                       # NaN >= tau -> False
    ge_rev = ge.transpose(0, 1)
    return {
        "gate_parent_of": tristate(ge & ~ge_rev, support),
        "gate_duplicate": tristate(ge & ge_rev, support),
    }


# --------------------------------------------------------------------------
# reconstruction contribution  (metrics/reconstruction.py)
# --------------------------------------------------------------------------
def recon_contributes(parent_gain: torch.Tensor, child_gain: torch.Tensor,
                      rel_gain_min: float) -> torch.Tensor:
    """`gate_recon`: BOTH endpoints carry reconstruction mass on the child's tokens.

    `metrics.reconstruction.edge_reconstruction_condition`'s pass rule, unchanged:

        parent_gain[p, c] >= rel_gain_min  AND  child_gain[c] >= rel_gain_min

    `parent_gain` is `detectors.recon_2a` and `child_gain` is the per-child vector
    `detectors.recon_child_gain` reads, which is the SAME quantity for the pair (c, c) -- the
    diagonal `_nan_diag` deletes. It is broadcast down a COLUMN (entry [p, c] takes
    `child_gain[c]`, constant in p), which is her `.unsqueeze(0)`. Taking it from the row
    instead would test whether the PARENT reconstructs its own tokens, a different claim that
    happens to produce plausible numbers.

    Defined where both gains are finite. Hers clamps the denominator at 1e-12 instead, which
    turns a child that never fires into a gain of ~0 and therefore a confident failure; the
    two agree on every child that fires.
    """
    if child_gain.ndim == 1:
        child_gain = child_gain.reshape(1, -1).expand_as(parent_gain)
    flag = (parent_gain >= float(rel_gain_min)) & (child_gain >= float(rel_gain_min))
    defined = torch.isfinite(parent_gain) & torch.isfinite(child_gain)
    return tristate(flag, defined)


# --------------------------------------------------------------------------
# S_res, Tree SAE's rank rule  (metrics/sres.py)
# --------------------------------------------------------------------------
def sres_rank_gate(P: torch.Tensor, available: torch.Tensor, W_unit: torch.Tensor,
                   top_k: int) -> torch.Tensor:
    """`gate_sres_rank`: both decoders are in the child probe's top-k correlations.

    `metrics.sres.sres_rank_check`, which is Tree SAE's operational rule and carries NO numeric
    threshold at all: for child c's probe direction, correlate it against every dictionary
    decoder and require BOTH the parent's and the child's own decoder to rank inside the top k.

    This is the one gate the scoring package was missing outright. `detectors.s_res` computes
    the same `min(corr[p], corr[c])` value Tree SAE defines, but a THRESHOLD on it cannot be
    chosen: healthy parent and child are close to orthogonal, which caps the min at about
    1/sqrt(2), so any tau above 0.707 rejects every healthy pair by construction. A rank rule
    sidesteps that, and it scales correctly too -- accepting k per child is O(L) pairs rather
    than the O(L^2) a fixed cut admits.

    Defined only on columns whose probe trained (`available[c]`). A child with too few positives
    gets an all-NaN column, never a column of confident failures.

    THE COMPETITOR POOL IS THE SCORED FRAME, NOT THE WHOLE DICTIONARY, and that is a real
    caveat. `metrics/sres.py` states the rule over "ALL dictionary features"; `W_unit` here is
    already `[R, d]` after the recovered reduction, so on a TRAINED read with R < L the bar is
    top-k of R, which is weaker than top-k of L and moves with recovery as well as with
    structure. On the oracle and synthetic reads R == F and on a latent-frame Gemma read
    R == L, so the two coincide there. Widening the pool would need the full decoder, which the
    scored frame does not carry; the number to watch is `n_recovered_features`.

    Measured on `only_isa`: the null pass rate falls from 13.6% at R=24 to 1.26% at R=240,
    roughly as 1/R. That is the scaling a rank rule buys and a fixed quantile does not -- a q99
    cut accepts 1% of any population at any size.
    """
    Wu = W_unit.to(DT)
    R = int(Wu.shape[0])
    flag = torch.zeros((R, R), dtype=torch.bool, device=Wu.device)
    defined = torch.zeros((R, R), dtype=torch.bool, device=Wu.device)
    k = int(top_k)
    for c in range(R):
        if not bool(available[c]):
            continue
        corr = Wu @ P[c].to(Wu.device).to(DT)          # [R] cosine of each decoder with probe c
        order = torch.argsort(corr, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.numel(), device=order.device)
        in_top = ranks < k
        flag[:, c] = in_top & in_top[c]                # parent in top-k AND child in its own
        defined[:, c] = True
    return tristate(flag, defined)


# --------------------------------------------------------------------------
# superparent  (metrics/outdegree.py)
# --------------------------------------------------------------------------
def superparent_flag(em: torch.Tensor, fire: torch.Tensor, outdeg_frac: float,
                     min_fire: int) -> torch.Tensor:
    """`gate_superparent`: EITHER endpoint holds a large share of the candidate children.

    `metrics.outdegree.find_superparents`, whose docstring is explicit that the flag is on
    out-degree ALONE: `outdeg >= outdeg_frac * n_children`. The old out-degree-AND-firing-rate
    gate survives there only as a reported `strict` attribute, because the AND let a parent with
    41.9% firing and 21.9% fan-out through. It is deliberately not adopted here.

    EITHER endpoint, not just the candidate parent, which is the fixed-threshold form of the
    `wide` transformation the quantile rule used: `wide = min(oriented outdegree both ways)` is
    `-max(raw outdegree both ways)`, so `LOW(wide)` accepted a pair when the LARGER of the two
    out-degrees was extreme. This gate keeps that, so it stays a function of the two endpoints
    and belongs in `evaluate.ENDPOINT_BROADCAST`.

    `n_children` is `R - 1`, not `R`. Hers is the size of a DISJOINT child block, where every
    member is a candidate; in a square frame a feature is not a candidate child of itself, so
    `R - 1` is the count of candidates that the out-degree was actually measured over.

    Defined where the endpoint fires at least `min_fire` times, NOT merely where it fires at
    all. `edge_mask` requires BOTH endpoints to clear `min_fire_count` before an edge exists, so
    a feature firing 1-19 times has an out-degree of exactly 0 by construction -- and reporting
    that as "not a superparent" is arithmetic dressed as a measurement. It is the same
    distinction the tristate exists for, one level up: absence of evidence, not evidence of
    narrowness. `superparent_v5` is this gate alone, so its denominators are the ones that
    would fill with those pairs.
    """
    R = int(em.shape[0])
    n_children = max(R - 1, 1)
    outdeg = em.to(DT).sum(dim=1)
    testable = fire >= float(min_fire)
    flag_f = (outdeg >= float(outdeg_frac) * n_children) & testable
    flag = flag_f.reshape(-1, 1) | flag_f.reshape(1, -1)
    defined = testable.reshape(-1, 1) & testable.reshape(1, -1)
    return tristate(flag, defined)


# --------------------------------------------------------------------------
# frequency survival  (metrics/token_control.py + config.FREQ_SURVIVAL_MIN)
# --------------------------------------------------------------------------
def freq_survives_gate(survival: torch.Tensor, tau_squashed: float) -> torch.Tensor:
    """`gate_freq_survives`: the edge holds up once high-frequency tokens are conditioned away.

    `survival >= tau_squashed`, with equality assigned to SURVIVING -- the same side the
    deleted `SURVIVES` predicate assigned it to, kept so the boundary case does not change
    hands in a rewrite.

    `tau_squashed` must already be through `squash`: `detectors.token_freq_survival` reports
    `ratio / (1 + ratio)`, while `config.FREQ_SURVIVAL_MIN` is stated on the raw ratio. Pass
    `squash(config.FREQ_SURVIVAL_MIN)`, never the raw constant.

    Defined where the detector is: it already NaNs a pair with no co-firing, too-rare firing or
    too few joint co-fires, which is precisely where the ratio has no denominator.
    """
    return tristate(survival >= float(tau_squashed), torch.isfinite(survival))
