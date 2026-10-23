"""Ground-truth calibration on Bussmann's toy, with a *trained* SAE.

`synthetic_toy_world.py` hands the five metrics hand-built statistics: it proves the maths
but not that the metrics survive a real training run. This closes that gap:

  1. rebuild the toy hierarchy from the team's repo (sae-training/configs/tree.json),
  2. load the Matryoshka SAE we trained on it (outputs/toy_trained/, GPU 3),
  3. match each learned latent to the true feature it recovered (decoder cosine),
  4. cache the statistics the metrics read, computed from the LEARNED latents,
  5. run the five production metrics and compare the edges they keep against the
     known parent->child tree.

So this is calibration on ground truth AND on trained features. If the metrics
recover the tree here, they are trustworthy on the real gemma-2-2b SAE.

Run:
    PYTHONPATH=src python3 validation/calibrate_on_trained_toy.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Which checkpoint to grade. The default is the reference-model Matryoshka trained by
# validation/notebooks/train_and_calibrate_on_toy.ipynb (40k steps, seed 0),
# which lived in sae-training/scripts/ when this run was made: it is the only run
# that learned all 20 features, so it grades the battery without the confound of edges
# whose endpoints were never available to find.
#
# outputs/toy_trained/ holds one directory per checkpoint, each with its own cfg.json
# and sae_weights.safetensors:
#
#   matryoshka_toy/    relu + adaptive L1, upstream reference implementation (default)
#   batch_topk_toy/    batch_topk k=2, this repo's trainer -- an independent second run
#                      on a different architecture, kept as a stability check
#
# Grade the second with:
#   EXP0_TOY_CKPT=outputs/toy_trained/batch_topk_toy \
#   EXP0_TOY_CALIB_OUT=outputs/batch_topk_toy_calibration.json \
#   python3 validation/calibrate_on_trained_toy.py
#
# EXP0_TOY_CKPT points this at another checkpoint without editing the file, so a
# comparison across checkpoints cannot silently become a comparison across script edits.
CKPT = Path(os.environ.get(
    "EXP0_TOY_CKPT", ROOT / "outputs" / "toy_trained" / "matryoshka_toy"))

# Where the result lands. Defaults beside the other reports; overridden when grading a
# checkpoint that is not the canonical one, so a side experiment cannot overwrite the
# number the paper's tables and figures read.
OUT_JSON = Path(os.environ.get(
    "EXP0_TOY_CALIB_OUT", ROOT / "outputs" / "trained_toy_calibration.json"))


def _find_tree() -> Path:
    """Locate the team repo's tree.json.

    `sae-training` is a separate repo, so where it sits is the user's choice.
    Default is beside experiment_0 (the layout in the README); an older layout
    nested it inside. EXP0_SAE_TRAINING overrides both.
    """
    env = os.environ.get("EXP0_SAE_TRAINING")
    roots = [Path(env)] if env else [ROOT.parent / "sae-training", ROOT / "sae-training"]
    for r in roots:
        if (r / "configs" / "tree.json").is_file():
            return r / "configs" / "tree.json"
    raise SystemExit(
        "cannot find sae-training/configs/tree.json (looked in: "
        + ", ".join(str(r) for r in roots)
        + ").\nClone https://github.com/soar-eleuther-i6-hierarchy/sae-training "
        "beside experiment_0, or set EXP0_SAE_TRAINING to its path."
    )


from metrics import (                                             # noqa: E402
    coverage_legs, keep_edges, edge_reconstruction_condition,
    frequency_controlled_coverage, frequency_buckets,
)
from metrics.reconstruction import per_token_ablation_gain        # noqa: E402
from metrics.sres import sres_rank_check, train_probe             # noqa: E402
from metrics.sibling_redundancy import parent_conditioned_redundancy  # noqa: E402


def score_per_token(x, fired, gt, W_dec, truth, m, parent_lat, child_lat, recovered):
    """The probe functions, on LEARNED latents against the known tree.

    Tier 1 grades these on hand-built statistics: the parent direction is one we
    chose, so "the rank rule finds it" is a statement about the arithmetic. Here
    the parent is a direction the SAE had to learn, which is the only place the
    question can be asked -- gemma has no known answer, and the synthetic toy has
    no training run.

    `x` is the toy's activation vector, which IS its residual stream: the object
    the SAE decomposes, and what stage 03 trains its probes on. No token cache is
    needed because nothing here is streamed.

    One limit is structural and stated in the result. The dictionary is 20
    latents, so a top-5 rank rule passes an unrelated parent at k/D = 25% by
    chance. This tier can therefore confirm that a LEARNED true parent is
    accepted; it cannot show an unrelated one is rejected. gemma's 32768 puts
    that null at 0.015%.
    """
    lat_of = {t: i for i, t in enumerate(m) if t >= 0}
    rows, red = [], {}
    for (tp_, tc_) in sorted(truth):
        if tp_ not in lat_of or tc_ not in lat_of:
            rows.append({"edge": f"{tp_} -> {tc_}", "testable": False,
                         "why": "endpoint never learned"})
            continue
        gp, gc = lat_of[tp_], lat_of[tc_]
        probe = train_probe(x.float(), fired[:, gc].bool(), seed=gc)
        if probe is None:
            rows.append({"edge": f"{tp_} -> {tc_}", "testable": False,
                         "why": "too few negatives for a probe"})
            continue
        corr = probe.double() @ W_dec.double().T
        ok, det = sres_rank_check(corr, gp, gc, 5)
        rows.append({"edge": f"{tp_} -> {tc_}", "testable": True, "pass": bool(ok), **det})

    # sibling redundancy inside each true parent's own firing set, and the fact
    # under it: how often the children co-fire in the GRAMMAR against how often
    # the latents that recovered them co-fire. The tree makes them mutually
    # exclusive, so the first column is zero by construction and any gap is the
    # SAE's. Carried as numbers so the page can plot it instead of quoting it.
    cofire = {}
    for tp_ in sorted({p for p, _ in truth}):
        true_kids = [c for (p, c) in truth if p == tp_]
        kids = [lat_of[c] for c in true_kids if c in lat_of]
        if tp_ in lat_of and len(kids) >= 2:
            red[str(tp_)] = round(parent_conditioned_redundancy(
                fired[:, lat_of[tp_]].bool(), fired[:, kids].bool()), 4)
            learned = [c for c in true_kids if c in lat_of]
            g = t = 0
            for a in range(len(learned)):
                for b in range(a + 1, len(learned)):
                    g += int(((gt[:, learned[a]] > 0) & (gt[:, learned[b]] > 0)).sum())
                    t += int((fired[:, lat_of[learned[a]]].bool()
                              & fired[:, lat_of[learned[b]]].bool()).sum())
            cofire[str(tp_)] = {"ground_truth": g, "learned": t, "children": learned}

    testable = [r for r in rows if r["testable"]]
    n_pass = sum(r["pass"] for r in testable)
    return {
        "n_testable": len(testable), "n_pass": n_pass,
        "chance_pass_rate": round(5 / W_dec.shape[0], 4),
        "edges": rows,
        "parent_conditioned_redundancy": red,
        "child_cofire": cofire,
        "redundancy_threshold": 0.5,
    }


# --------------------------------------------------------------------------
# toy hierarchy: read the team's Tree config directly. We reimplement the tiny
# sampler here so drawing samples needs none of the repo's heavy deps.
# --------------------------------------------------------------------------
class Node:
    def __init__(self, d, nxt):
        self.p = d["active_prob"]
        self.readout = d.get("is_read_out", True)
        self.excl = d.get("mutually_exclusive_children", False)
        self.idx = nxt[0] if self.readout else None
        if self.readout:
            nxt[0] += 1
        self.children = [Node(c, nxt) for c in d.get("children", [])]


def build_tree():
    return Node(json.loads(_find_tree().read_text()), [0])


def true_edges(tree) -> set[tuple[int, int]]:
    """Parent->child edges over READ-OUT features (hidden children have no index)."""
    edges = set()

    def walk(n):
        if n.idx is not None:
            for c in n.children:
                if c.idx is not None:
                    edges.add((n.idx, c.idx))
        for c in n.children:
            walk(c)

    walk(tree)
    return edges


def n_features(tree) -> int:
    m = [0]

    def walk(n):
        if n.readout:
            m[0] += 1
        for c in n.children:
            walk(c)

    walk(tree)
    return m[0]


def sample(tree, n, F, gen) -> torch.Tensor:
    """[n, F] binary ground-truth activations, honouring the two structural rules:
    a child can fire only if its parent fires, and exclusive siblings never co-fire.

    Inside an exclusive group ``active_prob`` is the multinomial PICK WEIGHT, and the
    picked sibling then fires with certainty -- ``Tree.sample`` in the training repo
    passes ``force_active=True`` there. Drawing a second Bernoulli on the picked child
    applies its probability twice, which dropped the child firing rate by ~5x (0.006
    instead of 0.030 on this tree) and graded the SAE on a distribution it was never
    trained on. ``forced`` is what keeps this sampler on the training distribution.
    """
    out = torch.zeros(n, F)

    def rec(node, mask, forced=False):
        fire = mask if forced else mask & (torch.rand(n, generator=gen) < node.p)
        if node.idx is not None:
            out[fire, node.idx] = 1.0
        if node.excl and node.children:
            probs = torch.tensor([c.p for c in node.children])
            pick = torch.multinomial(probs, n, replacement=True, generator=gen)
            for i, c in enumerate(node.children):
                rec(c, fire & (pick == i), forced=True)
        else:
            for c in node.children:
                rec(c, fire)

    rec(tree, torch.ones(n, dtype=torch.bool))
    return out


# --------------------------------------------------------------------------
# the trained SAE
# --------------------------------------------------------------------------
def load_sae():
    return load_file(str(CKPT / "sae_weights.safetensors")), json.loads((CKPT / "cfg.json").read_text())


def is_reference_sae(w) -> bool:
    """Does this checkpoint come from the upstream reference model?

    ``normalizer.running_avg`` is a parameter only the upstream MatryoshkaSAE
    (noanabeshima/matryoshka-saes) carries; this repo's ``architectures/matryoshka.py``
    has no normaliser. Keying on a tensor that must exist beats keying on a cfg field
    a hand-written checkpoint might set either way.
    """
    return "normalizer.running_avg" in w


def encode(w, x, cfg):
    """Replicate the trained SAE's activation. The toy SAE uses batch_topk with no
    saved inference threshold, so we apply the same batch-wide top-(k*n) selection
    the architecture uses at train time; relu alone would leave the codes near-zero.
    """
    if is_reference_sae(w):
        return encode_decode(w, x, cfg)[0]
    pre = (x - w["b_dec"]) @ w["W_enc"] + w["b_enc"]
    act = cfg.get("activation_function", "relu")
    if act == "relu":
        return torch.relu(pre)
    post = torch.relu(pre)
    k = cfg["k"]
    if act == "topk":
        top = post.topk(k, dim=-1)
        out = torch.zeros_like(post)
        return out.scatter_(-1, top.indices, top.values)
    # batch_topk: keep the k*n largest post-relu activations across the whole batch
    flat = post.flatten()
    keep = min(k * post.shape[0], flat.numel())
    top = flat.topk(keep)
    out = torch.zeros_like(flat)
    out.scatter_(-1, top.indices, top.values)
    return out.reshape_as(post)


def encode_decode(w, x, cfg):
    """``(activations, residual)`` for either family of toy SAE.

    Two forward passes reach this script and they differ in ways that silently move
    every statistic downstream -- co-firing counts, ablation gains, the residual norm:

    * this repo's ``architectures/matryoshka.py`` centres the input by ``b_dec`` and
      returns the post-activation codes directly;
    * the upstream reference model centres nothing, rescales its input by a
      running-average normaliser, and reports activations multiplied by ``||W_dec||``
      before undoing that rescaling (``get_acts`` in its ``sae.py``).

    Reusing the first path on a checkpoint of the second kind loads without error and
    reports numbers that look plausible, which is exactly why the split is explicit.
    """
    if not is_reference_sae(w):
        acts = encode(w, x, cfg)
        return acts, x - (acts @ w["W_dec"] + w["b_dec"])

    # normalize(x) = x * sqrt(d) / running_avg; unnormalize divides by the same factor.
    scale = (x.shape[-1] ** 0.5) / w["normalizer.running_avg"]
    codes = torch.relu((x * scale) @ w["W_enc"] + w["b_enc"])
    x_hat = (codes @ w["W_dec"] + w["b_dec"]) / scale
    acts = (codes * w["W_dec"].norm(dim=1)) / scale
    return acts, x - x_hat


def match_latents(w, true_dirs):
    """Each latent -> the true feature its decoder points at (-1 if none)."""
    W = w["W_dec"] / w["W_dec"].norm(dim=1, keepdim=True).clamp(min=1e-8)
    T = true_dirs / true_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = W @ T.T
    best = cos.argmax(dim=1)
    best[cos.max(dim=1).values < 0.4] = -1
    return best


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    tree = build_tree()
    truth = true_edges(tree)
    F = n_features(tree)
    print(f"toy: {F} read-out features, {len(truth)} true parent->child edges")
    print(f"  true edges: {sorted(truth)}")

    w, cfg = load_sae()
    # Identity embedding, per the paper -- unless the checkpoint shipped the directions
    # it was actually trained on. The notebook jitters feature norms by ~5%, so grading
    # it against a clean identity would feed the SAE activations it never saw. Matching
    # is unaffected (match_latents normalises), the co-firing statistics are not.
    true_dirs = w["true_feats"] if "true_feats" in w else torch.eye(F)
    if "true_feats" in w:
        print(f"using the checkpoint's own feature directions "
              f"(norms {true_dirs.norm(dim=1).min():.3f}-{true_dirs.norm(dim=1).max():.3f})")
    match = match_latents(w, true_dirs)
    recovered = {int(t) for t in match if t >= 0}
    print(f"\ntrained Matryoshka SAE: recovered {len(recovered)}/{F} true features")

    n = 200_000
    gt = sample(tree, n, F, gen)                       # [n, F] ground-truth firings
    x = gt @ true_dirs                                 # model input
    acts, resid = encode_decode(w, x, cfg)             # LEARNED latent activations

    fired = (acts > 1e-3).double()
    g = per_token_ablation_gain(acts.double(), resid.double(), w["W_dec"].double())
    err = (resid.double() ** 2).sum(dim=1)

    true_parents = sorted({p for p, _ in truth})
    true_children = sorted({c for _, c in truth})
    m = match.tolist()
    parent_lat = [i for i, t in enumerate(m) if t in true_parents]
    child_lat = [i for i, t in enumerate(m) if t in true_children]
    print(f"latents matched to parents: {len(parent_lat)}, to children: {len(child_lat)}")
    if not parent_lat or not child_lat:
        print("SAE did not recover both parents and children; stop.")
        return

    fp = fired[:, parent_lat]
    fc = fired[:, child_lat]
    cofire = fp.T @ fc
    fire_p, fire_c = fp.sum(0), fc.sum(0)

    R, _ = coverage_legs(cofire, fire_p, fire_c)
    edge_mask = keep_edges(R, fire_p, fire_c, 0.5, 20)
    recon = edge_reconstruction_condition(
        fc.T @ err, g[:, parent_lat].T @ fc, (fc * g[:, child_lat]).sum(0), 0.01)

    token_ids = fired.argmax(dim=1).long()             # token-like id per row
    vocab = int(token_ids.max()) + 1
    counts = torch.zeros(vocab, dtype=torch.float64)
    counts.scatter_add_(0, token_ids, torch.ones(n, dtype=torch.float64))
    buckets = frequency_buckets(counts, 0.5, 0.4)[token_ids]
    cbb = torch.zeros(3, fp.shape[1], fc.shape[1], dtype=torch.float64)
    fcb = torch.zeros(3, fc.shape[1], dtype=torch.float64)
    for k in range(3):
        sel = (buckets == k).double().unsqueeze(1)
        cbb[k] = fp.T @ (fc * sel)
        fcb[k] = (fc * sel).sum(0)
    fcov = frequency_controlled_coverage(cbb, fcb, edge_mask)

    survivors = edge_mask & recon["passes"] & (fcov["survival"] >= 0.5)

    found = set()
    for pi in range(survivors.shape[0]):
        for ci in range(survivors.shape[1]):
            if survivors[pi, ci]:
                found.add((m[parent_lat[pi]], m[child_lat[ci]]))

    tp, fp_, fn = found & truth, found - truth, truth - found
    prec = len(tp) / max(len(found), 1)
    rec = len(tp) / max(len(truth), 1)
    print("\n=== calibration on the TRAINED toy ===")
    print(f"recovered edges: {sorted(found)}")
    print(f"true positives : {len(tp)} / {len(truth)}   false pos: {len(fp_)}   false neg: {len(fn)}")
    if fn:
        print(f"  missed edges : {sorted(fn)}")
    if fp_:
        print(f"  spurious     : {sorted(fp_)}")
    print(f"precision {prec:.2f}   recall {rec:.2f}")
    print(f"VERDICT: {'PASS' if prec >= 0.8 and rec >= 0.8 else 'NEEDS WORK'}")

    pt = score_per_token(x, fired, gt, w["W_dec"], truth, m, parent_lat, child_lat, recovered)
    print(f"\nprobe S_res on LEARNED latents: {pt['n_pass']}/{pt['n_testable']} true edges "
          f"accepted (chance {pt['chance_pass_rate']:.0%} at k/D)")
    for r in pt["edges"]:
        if r["testable"]:
            print(f"  {r['edge']:<10} parent rank {r['parent_rank']:>2}  "
                  f"child rank {r['child_rank']:>2}  {'pass' if r['pass'] else 'FAIL'}")
        else:
            print(f"  {r['edge']:<10} untestable — {r['why']}")
    print(f"parent-conditioned sibling redundancy: {pt['parent_conditioned_redundancy']}")

    # per-edge verdict rows for the report, ordered parent then child
    def edge_row(e):
        p, c = e
        cat = "recovered" if e in found else ("missed: child not learned"
              if c not in recovered else "missed")
        return {"edge": f"{p} -> {c}", "parent": p, "child": c,
                "found": e in found, "category": cat}

    result = {
        "n_features": F,
        # The tree itself, not just its edges. Reporting has to be able to SHOW what an
        # edge is -- reviewer feedback was that a recovery figure means nothing when the
        # tree behind it is never displayed -- and reading it back out of sae-training
        # would make a figure depend on a second repo being checked out beside this one.
        "tree": json.loads(_find_tree().read_text()),
        "n_recovered_features": len(recovered),
        "recovered_features": sorted(recovered),
        "true_edges": sorted(truth),
        "found_edges": sorted(found),
        "true_positives": len(tp), "false_positives": len(fp_), "false_negatives": len(fn),
        "precision": prec, "recall": rec,
        "cfg": cfg,
        "edge_rows": [edge_row(e) for e in sorted(truth)] +
                     [{"edge": f"{p} -> {c}", "parent": p, "child": c, "found": True,
                       "category": "spurious"} for (p, c) in sorted(fp_)],
        "missed_children": sorted({c for _, c in fn if c not in recovered}),
        "per_token": pt,
    }
    out = OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
