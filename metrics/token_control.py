"""
Metric 5 - Token-frequency-controlled coverage (idea from Tokenized SAEs).

Hypothesis under test: poly-parenting / superparents are driven by
high-frequency tokens (spaces, punctuation, "the"). If an edge's coverage is
carried by those tokens, conditioning them away should destroy it; a genuine
semantic edge should survive.

We bucket token ids by their share of corpus MASS (not rank): bucket 0 =
most frequent ids covering the top ~50% of all tokens, bucket 1 = next ~40%,
bucket 2 = the long tail. The streaming pass accumulates fire/cofire counts
per bucket; here we recompute reverse coverage per bucket and report a
survival ratio:

    survival(p, c) = R_low+mid(p, c) / R_all(p, c)

    ~1  -> edge holds on rare tokens too (genuine)
    ~0  -> edge exists only on frequent tokens (frequency capture)
"""

from __future__ import annotations

import torch


def frequency_buckets(
    token_counts: torch.Tensor,   # [vocab] corpus counts per token id
    high_mass: float = 0.50,
    mid_mass: float = 0.40,
) -> torch.Tensor:
    """Assign every token id a bucket 0 (high) / 1 (mid) / 2 (low) by cumulative mass."""
    counts = token_counts.double()
    order = torch.argsort(counts, descending=True)
    csum = torch.cumsum(counts[order], dim=0) / counts.sum().clamp(min=1.0)
    bucket_sorted = torch.full_like(order, 2)
    bucket_sorted[csum <= high_mass + mid_mass] = 1
    bucket_sorted[csum <= high_mass] = 0
    buckets = torch.empty_like(order)
    buckets[order] = bucket_sorted
    return buckets


def local_frequency_buckets(
    tokens: torch.Tensor,         # [b, seq] token ids
    keep: torch.Tensor,           # [b, seq] bool — positions entering the statistics
    vocab: int,
    high_mass: float = 0.50,
    mid_mass: float = 0.40,
) -> torch.Tensor:
    """Bucket each kept POSITION by its token's frequency *within its own window*.

    `frequency_buckets` above is a function of the token id: a bucket is a property
    the id carries everywhere in the corpus. This is a function of (id, window): the
    same id can be bucket 0 in one context and bucket 2 in another.

    The distinction is the whole question on a PCFG corpus. `zipf_exponent` weights
    terminal *ranks* and the generator re-permutes which id holds which rank per
    document, so concentration is real inside a document and averages to flat across
    the corpus — top 10 ids hold 59.2% within a document and 1.3% corpus-wide at
    zipf=1.5. A global bucketing sees nothing there; a local one sees the structure
    the knob actually creates.

    Returns [b, seq] with bucket 2 (tail) at dropped positions, which never enter the
    statistics anyway.
    """
    out = torch.full_like(tokens, 2)
    for i in range(tokens.shape[0]):
        row = tokens[i][keep[i]]
        if row.numel() == 0:
            continue
        counts = torch.bincount(row, minlength=vocab).double()
        out[i][keep[i]] = frequency_buckets(counts, high_mass, mid_mass).to(out.dtype)[row]
    return out


def frequency_controlled_coverage(
    cofire_by_bucket: torch.Tensor,   # [K, P, C] co-firing counts per bucket
    fire_c_by_bucket: torch.Tensor,   # [K, C]    child firing counts per bucket
    edge_mask: torch.Tensor,          # [P, C]    kept edges (from all-token coverage)
    min_fire_low: int = 5,
    *,
    clamp_max: float | None = 1.5,
    floor: str = "rare",
) -> dict[str, torch.Tensor]:
    """Per-edge reverse coverage per bucket + survival on non-frequent tokens.

    survival[p, c] = R restricted to buckets 1+2 (mid+low), divided by R_all.
    Edges whose child barely fires outside bucket 0 (fewer than min_fire_low
    tokens) are marked untestable (survival = nan) rather than failed.

    `clamp_max=None` leaves the ratio unclamped. `floor="total"` applies min_fire_low to all of
    the child's firing rather than its rare-token firing. `edge_mask` can be any [P, C] mask of
    the pairs to report, e.g. a support mask.
    """
    # Review note: under floor="rare" a child that never fires on rare tokens is untestable,
    # yet R_rest = 0 on such a child is the plainest sign of a frequency-driven edge. The
    # synthetic benchmark passes floor="total", which scores it 0; worth checking which one
    # this pipeline wants.
    if floor not in ("rare", "total"):
        raise ValueError(f"floor must be 'rare' or 'total', got {floor!r}")
    cf = cofire_by_bucket.double()
    fc = fire_c_by_bucket.double()

    R_all = cf.sum(0) / fc.sum(0).clamp(min=1.0).unsqueeze(0)          # [P, C]
    cf_rest = cf[1:].sum(0)                                            # [P, C]
    fc_rest = fc[1:].sum(0)                                            # [C]
    R_rest = cf_rest / fc_rest.clamp(min=1.0).unsqueeze(0)             # [P, C]

    survival = R_rest / R_all.clamp(min=1e-12)
    if clamp_max is not None:
        survival = survival.clamp(max=clamp_max)  # noise guard: tiny denominators
    fc_floor = fc_rest if floor == "rare" else fc.sum(0)
    untestable = (fc_floor < min_fire_low).unsqueeze(0).expand_as(survival)
    survival = torch.where(untestable, torch.nan, survival)
    survival = torch.where(edge_mask, survival, torch.nan)

    return {"R_all": R_all, "R_rest": R_rest, "survival": survival}
