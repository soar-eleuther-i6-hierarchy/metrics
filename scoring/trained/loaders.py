"""Load a trained sae-training checkpoint into the `LoadedSAE` the scorers read.

The only module that imports `sae_training`, and lazily, so the rest of the package imports without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class LoadedSAE:
    encode: Callable[[torch.Tensor], torch.Tensor]
    W_dec: torch.Tensor          # [S, D] decoder rows (one per latent)
    b_dec: torch.Tensor          # [D]
    meta: dict[str, object]
    arch: str | None = None      # activation_function, e.g. "batch_topk"; provenance only


def load_sae(ckpt_dir: str | Path) -> LoadedSAE:
    """Load a checkpoint and its `toy_meta.json`; `variant` picks Matryoshka or vanilla BatchTopK.

    `encode` returns float64 activations."""
    ckpt = Path(ckpt_dir)
    meta = json.loads((ckpt / "toy_meta.json").read_text(encoding="utf-8"))
    variant = str(meta.get("variant", "matryoshka"))
    if variant == "vanilla":
        from sae_training.architectures.base import VanillaSAE as SAECls  # lazy
    else:
        from sae_training.architectures.matryoshka import MatryoshkaSAE as SAECls  # lazy

    sae = SAECls.from_pretrained(str(ckpt), device="cpu")
    sae.eval()

    def encode(h: torch.Tensor) -> torch.Tensor:
        # sae.encode returns (hidden, pre_acts); hidden is the thresholded output
        with torch.no_grad():
            hidden = sae.encode(h.to(torch.float32), use_threshold=True)[0]
        return hidden.detach().to(torch.float64)

    W_dec = sae.W_dec.detach().to(torch.float64)        # stored as [S, D], no transpose
    if tuple(W_dec.shape) != (int(sae.config.d_sae), int(sae.config.d_in)):
        raise ValueError(f"unexpected W_dec orientation {tuple(W_dec.shape)}; "
                         f"expected (d_sae, d_in) = ({sae.config.d_sae}, {sae.config.d_in})")
    return LoadedSAE(
        encode=encode,
        W_dec=W_dec,
        b_dec=sae.b_dec.detach().to(torch.float64),
        meta=meta,
        arch=str(getattr(sae.config, "activation_function", None) or type(sae.config).__name__),
    )
