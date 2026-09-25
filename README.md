<style>
.x0nav{position:sticky;top:0;z-index:999;background:#fff;border-bottom:1px solid #E3DAFB;
font:500 13px/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0 0 14px;}
.x0nav .row{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding:9px 18px;}
.x0nav .row+.row{border-top:1px solid #F1ECFD;}
.x0nav a{text-decoration:none;color:#5A6B7B;}
.x0nav a:hover{color:#7C22CE;}
.x0nav .brand{font-weight:700;color:#7C22CE;letter-spacing:.2px;}
.x0nav .on{color:#7C22CE;font-weight:700;}
.x0nav .lbl{color:#9AA7B3;font-size:11px;text-transform:uppercase;letter-spacing:.7px;}
.x0nav .pill{border:1px solid #E3DAFB;border-radius:7px;padding:5px 10px;background:#F6F3FE;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{width:1px;height:17px;background:#E3DAFB;}
.x0nav .gh{display:inline-flex;align-items:center;gap:5px;margin-left:auto;}
.x0nav .gh svg{width:15px;height:15px;fill:currentColor;display:block;}
.x0nav details.dens{flex:1 1 100%;}
.x0nav details.dens summary{cursor:pointer;list-style:none;user-select:none;display:inline-block;}
.x0nav details.dens summary::-webkit-details-marker{display:none;}
.x0nav details.dens summary::after{content:"▸";margin-left:4px;}
.x0nav details.dens[open] summary::after{content:"▾";}
.x0nav details.dens summary:hover{color:#7C22CE;}
.x0nav .drow{display:flex;flex-wrap:wrap;align-items:center;gap:13px;padding-top:9px;}
@media (prefers-color-scheme:dark){
.x0nav{background:#141414;border-bottom-color:#2E2E2E;}
.x0nav .row+.row{border-top-color:#242424;}
.x0nav a{color:#A9B4BF;}
.x0nav .brand,.x0nav a:hover,.x0nav .on{color:#C79BF2;}
.x0nav .pill{background:#1E1830;border-color:#3A2B57;}
.x0nav .pill.on{background:#7C22CE;color:#fff;border-color:#7C22CE;}
.x0nav .sep{background:#2E2E2E;}
.x0nav details.dens summary:hover{color:#C79BF2;}}
</style><nav class="x0nav"><div class="row"><a class="brand" href="./">SOAR I-6 · metrics</a><a class="" href="./outputs/">Results</a><a class="" href="./outputs/synthetic_toy_calibration.html">Synthetic Toy Calibration</a><a class="" href="./outputs/trained_toy_calibration.html">Trained Toy Calibration</a><a class="" href="./outputs/pcfg-matryoshka/">pcfg-matryoshka</a><a class="" href="./outputs/gemma-2-2b/">gemma-2-2b</a><a class="gh" href="https://github.com/soar-eleuther-i6-hierarchy/metrics" title="Browse the code on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>Code</a></div></nav>

# Implement Metrics (SOAR I-6)

Grades candidate parent→child edges between the nested blocks of a **Matryoshka SAE** on
`google/gemma-2-2b` (residual stream, layers 1–24). A family of competing metrics decides which
"edges" are real hierarchy and which are frequency, splitting or co-occurrence artifacts.

The SAE has `D_SAE = 32768` features in 5 nested blocks with prefix lengths
`[128, 512, 2048, 8192, 32768]` → `B0=[0,128) B1=[128,512) B2=[512,2048) B3=[2048,8192) B4=[8192,32768)`.
Cross-block edges are computed between **adjacent** blocks only.

**Live site:** [soar-eleuther-i6-hierarchy.github.io/metrics](https://soar-eleuther-i6-hierarchy.github.io/metrics/)


| Where to go                | What is there                                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [metrics/](metrics/)       | every metric: formula, threshold, what it catches, what it is blind to                                                                 |
| [outputs/](outputs/)       | all results — dashboards, reports, per-layer pages, validation tiers                                                                  |
| [validation/](validation/) | the two calibration tiers scored against a known tree (synthetic toy, trained toy) plus the qualitative Tier 3 and the lateral control |

## Install and run

Run everything from the `metrics/` directory.

```bash
pip install torch sae_lens datasets plotly numpy matplotlib scipy   # scipy: NNLS for the synthetic dictionaries (synthdict/activations.py)

python3 run_pipeline.py         # every stage in order, refusing any whose inputs are missing
python3 run_pipeline.py --list  # show the order and what is already satisfied
python3 run_pipeline.py --from 02   # resume after the slow one
```

## What each metric catches

The validation table: **metrics (rows) × properties (columns)**. A candidate pair can look like an edge for seven different reasons. Only one of them is real
hierarchy; the rest are the pathologies the metrics have to separate out. Each metric is a partial
detector, so the question for every metric is not "is it correct" but **which column does it add**.


| Metric                                                | Parent→child | Absorption | Splitting | Superparent | Siblings | Frequency coincidence | Concept co-occurrence |
| ------------------------------------------------------- | :-------------: | :----------: | :---------: | :-----------: | :--------: | :---------------------: | :---------------------: |
| **1a** Coverage — reverse `R`                        |      ◐      |     ✗     |    ✗    |     ✗     |    ✗    |          ✗          |          ✗          |
| **1b** Coverage — forward `F`                        |      ◐      |     ✗     |    ✗    |     ◐     |    ✗    |          ✗          |          ✗          |
| **1c** Joint-child `R_supp` / `R_mass` / energy share |      ◐      |     ✗     |    ✅    |     ✅     |    ✗    |          ✗          |          ✗          |
| **2a** Reconstruction ablation (contribution filter)  |      ◐      |     ✗     |    ✗    |     ✗     |    ✗    |          ◐          |          ✗          |
| **2b** `S_res` probe, rank-scored                     |      ✅      |     ◐     |    ✗    |     ◐     |    ◐    |          ✅          |          ✗          |
| **3** Sibling redundancy                              |      ◐      |     ✗     |    ✅    |     ✗     |    ✅    |          ✗          |          ✗          |
| **4** Out-degree / superparent                        |      ✗      |     ✗     |    ✗    |     ✅     |    ✗    |          ◐          |          ✗          |
| **5** Token-frequency control                         |      ◐      |     ✗     |    ✗    |     ◐     |    ✗    |          ✅          |          ✗          |
| **6** Independence null (PMI / Dev)                   |      ◐      |     ✗     |    ✗    |     ✅     |    ✗    |          ✅          |          ✗          |
| **7** In-block directed coverage                      |      ✅      |     ✗     |    ✅    |     ✗     |    ✅    |          ✗          |          ✗          |

✅ detects this property · ◐ partial — necessary but not sufficient, or only in some regimes · ✗ blind

**The properties.** *Parent→child*: the child is a genuine refinement of the parent. *Absorption*:
the child has absorbed a case from the parent, so the parent goes **silent** exactly where the child
fires. *Splitting*: the "children" are near-copies of one another. *Superparent*: one parent fans out
over most of the next block and fires on a huge share of tokens. *Siblings*: the two features are
co-level (co-hyponyms or co-extensive duplicates), not parent and child. *Frequency coincidence*: the
co-firing is base rate, carried by high-frequency tokens. *Concept co-occurrence*: both features are
specific and genuinely unrelated (`enzyme`, `CT scan`) but share a latent topic (`biology`), so they
co-fire in the same documents.

**Two columns are still open — and are now demonstrated rather than argued.**
[`validation/synthetic_toy_world.py`](validation/synthetic_toy_world.py) carries a scored negative control for each:
an absorbed child (the true edge has `R = 0.00` and never enters the candidate set) and a
shared-topic pair (a non-edge that clears coverage, reconstruction, the frequency control *and*
PMI). Both rows pass when the battery does **nothing**, so a regression turns a stated limitation
into a visible failure.

- **Concept co-occurrence is caught by nothing.** `enzyme` → `CT scan` passes coverage (they co-fire),
  passes reconstruction (both carry mass on biology tokens), passes token-frequency control (biology
  tokens are not frequent), and passes PMI (they are *not* independent — the shared topic makes
  PMI > 0). Closing it needs a **model-based null**: a topic model / LDA-style `M` that removes the
  shared concept and asks whether the residuals `a = enzyme − biology`, `b = CT − biology` are still
  dependent. Not implemented in this tranche.

  **It is not hypothetical, and it is the commonest way a survivor is wrong.** Reading all 48
  surviving edges at B0→B1 against Neuronpedia labels (8 per layer, six layers), eight are a
  semantic parent with a function-word or formatting child — *"formal legal terminology"* → the word
  **"the"**, *"code, formulas and citations"* → **blank lines**. Every one clears the whole battery.
  So this column costs roughly **one sixth of what survives**.
- **Absorption** is only reachable through decoder geometry (`S_res`), and even there the edge never
  arrives: coverage gates the candidate set, and an absorbed child has low `R` by construction, so
  the pair is dropped before any later metric sees it.

Cells are read off each metric's construction (see the module docstrings) together with the
calibrations below — they state what a metric is *able* to separate, not a measured accuracy on the
real SAE. Every cell is now exercised by a Tier-1 row: the calibration covers **21/21 metric
functions**, which it did not until 7 August.

One cell needs its own caveat. **2b's ability to disfavour a superparent is bounded by dictionary
size.** The rank rule is a geometry test, so an unrelated parent passes whenever chance puts it in
the top *k* of *D* — `k/D`, which is 0.015% on gemma's 32768 latents but 0.28% on a 1792-latent PCFG
dictionary and 11.9% on the toy. It ranks a superparent far down (median rank 24 of 42 against the
true parent's 1), and the top-*k* cutoff is what lets chance through. An `S_res` pass rate is only
comparable between dictionaries of similar size.

### Why trust the table: three tiers

Each tier gives up one guarantee and gains one dose of reality; a metric we trust has to hold across
all three. ("Tier", not "layer", to avoid confusion with the model's residual-stream layers.)


| Tier                 | What it is                                                                                                             | Ground truth?          | What it proves                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **1. Synthetic toy** | a known 5-parent tree plus six injected structures, reduced to cached stats*and* to the per-token view the probes need | yes, by construction   | the maths is right —**14/14 rows**, covering all 21 metric functions, each pathology caught by its intended metric |
| **2. Trained toy**   | a Matryoshka SAE actually trained on the known tree; metrics run on the*learned* features                              | yes, the tree is known | the metrics survive real training — precision 1.00, recall 1.00, 0 false positives, all 20 features learned        |
| **3. Released SAE**  | the published`gemma-2-2b` Matryoshka SAE, read against Neuronpedia labels                                              | none — human reading  | 48 survivors read against autointerp labels                                                                         |

Tier 1 is certain but artificial; Tier 3 is realistic but has no ground truth and is a checkpoint we
did not train, which is what *released* names — the rungs below it are real SAEs too. Tier 2 is the
only rung with both a trained SAE and a known answer.

**Tier 2 does not isolate the variable it is named for.** It runs on a clean 20-feature tree while
Tier 1 grades a larger pathology-injected world, so it changes the toy as well as the statistics.
What it does isolate is **blame** — a missed edge counts against a metric only if the SAE learned
both endpoints — and that is what it is for.

**The PCFG SAE is a control, not a rung.** A tier earns its place by scoring the battery against a
known answer, and the PCFG run has none: its grammar is known, but nothing maps a latent to a grammar
symbol, so it reports the same battery outputs as Tier 3 rather than a recovery score. The helper a
mapping would be built from exists — `pcfg_bridge.grammar.vocab.role_of(token_id)` in the PCFG repo —
and until it is consumed, the run answers a different question: what corpus complexity does to the
same battery, between a hand-built world and natural language. Its numbers are reported beside gemma
in [outputs/README.md](outputs/README.md#other-sources), not on the ladder.

**Scope.** Tier 1 scores **14/14 rows**, covering **21/21 metric functions** —
including `S_res` and in-block directed coverage, which until 7 August were graded by nothing. The
page used to say the probe functions were "calibrated in Tier 2"; Tier 2 imports five functions and
none of them is one of those. Two rows are negative controls that pass when the battery does *not*
act, so absorption and shared-topic co-occurrence are demonstrated limitations rather than asserted
ones. Full detail, per-metric scorecards and how to run each tier:
**[outputs/README.md](outputs/README.md#how-the-metrics-are-validated-three-tiers)**.
