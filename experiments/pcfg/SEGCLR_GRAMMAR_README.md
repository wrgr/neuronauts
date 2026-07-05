# SegCLR + compartment grammar for proofreading — overview

Can Google's public **SegCLR** embeddings (local morphology fingerprints) plus a
**compartment/topology grammar** detect and fix segmentation errors (false merges,
false splits) on the proofread MICrONS minnie65 column?

**One-line answer:** *Structure proposes, SegCLR compares — and SegCLR encodes
cell **type/compartment**, not individual **identity**, which is exactly why it is
strong for splits and weak for same-type merges.*

Detailed log with every number: [`FINDINGS_compartment_grammar.md`](FINDINGS_compartment_grammar.md).

---

## The headline finding (read this first)

**SegCLR embeddings encode cell type / compartment, not individual cell identity.**
Proven: cluster the SegCLR nodes of one neuron → you get axon-vs-dendrite
(ARI 0.69–0.86); pool two different cells and cluster → it splits by compartment
(ARI ≈ 0.7–0.84) and by *cell identity ARI ≈ 0* (10/10 pairs). It cannot tell two
cells apart. Cause: SegCLR is trained with a ~4 µm receptive field + local
positives, so it learns *local morphology*, which is compartment/type.

This explains everything:

| task | who solves it | why |
|---|---|---|
| **multi-soma merge** | grammar (topology) | two somas = 2 big-radius clusters — nothing to do with SegCLR |
| **false split (stitch fragments)** | geometry + SegCLR | local: the true continuation is the adjacent, same-type node → most similar |
| **same-type / same-compartment merge** | **unsolved** | needs identity; SegCLR has only type |

---

## Results (leakage-safe on the proofread column, tangential train/eval split)

**Splits** (stitch m343 fragments back into cells):
- Most are trivial — proximity alone gets ~1.00 (the true continuation is the only
  thing nearby).
- On *contested* endpoints (a wrong-cell candidate also within reach, n=12):
  distance 0.50 → geometry 0.75 → **SegCLR 1.00**. ⚠ But see caveats: n=12, a
  lenient "same-neuron" metric, and SegCLR only wins when the distractor is a
  *different type*. **Honest read: geometry does the work; SegCLR disambiguates
  type on a small hard slice.**

**Merges:**
- Multi-soma: **AUC 1.0** (grammar).
- A↔D crossing (cross-compartment): weak/partial — real 1-soma merges fired 1/3;
  false positives on clean cells. *Not every merge crosses a compartment.*
- Same-compartment (the residual): every SegCLR attempt fails —
  absolute step (within-cell drift), comparative walk (4.8× localization but
  useless as a *detector*, AUC 0.66), branch-point proposer (seam ranks 149/149),
  global spectral (ARI 0.17, splits by compartment not cell). **All fail for the
  one reason above: no identity signal.**

---

## Code map

Core (`neuronauts/`):
- `segclr.py` — dependency-free loader for the public SegCLR CSV-zip shards over
  HTTPS byte-range (no `connectomics` pkg, no auth); `md5_shard` (bytewidth=64),
  64-dim nm-coord embeddings, per-segment cache, spatial assignment.
- `soma_clusters.py` — verified multi-soma routine (large-radius vertex clusters).

Experiment thread (`experiments/pcfg/`):
- `compartments.py` — `label_compartments`: per-vertex {soma, axon, dend, unknown}
  from synapse polarity (pre=axon, post=dend, diffused) + soma caliber.
- `compartment_grammar.py` — `object_signals` (multi-soma + A↔D crossing +
  candidate seam), `build_merged_object` / `extract_axon_piece` (synthetic merges).
- `walk_detector.py` — the SegCLR walk toolkit:
  - splits: `fragment_endpoints`, `endpoint_join_score` (geometry + SegCLR),
    `stitch_fragments`;
  - merges: `comparative_split_score`, `rank_cut_candidates` (branch proposer),
    `neuron_fragment_ids` / `assign_segclr_to_skeleton` (m343→skeleton via seg vol).
- `column_split.py` — proofread-column loader + tangential PCA train/eval split.
- `run_compartment_grammar.py` — CLI: `--exp0/1/2` (SegCLR value probes), `--m1`
  (compartment sanity).

## Key data facts (verified)
- "m343" = a 2022 segmentation snapshot, **not** a queryable materialization; SegCLR
  ids are m343 segment ids. Map current→m343 via the `seg_m343` cloudvolume.
- SegCLR nodes are **~1 µm apart**; a current neuron shatters into **~40–90 m343
  fragments**, ~10–22k nodes total, ~1 node/µm cable.
- The CAVE skeleton service needs `cloud-volume` (else fetches silently
  negative-cache to empty — a real footgun).

## How to run
```bash
uv sync --extra dev --extra cave       # + `uv pip install cloud-volume`
export token=<CAVE token>              # 32-hex middle-auth token
python -m experiments.pcfg.run_compartment_grammar --exp2   # SegCLR top-1 retrieval
python -m experiments.pcfg.run_compartment_grammar --m1     # compartment labeling sanity
```

## Honest limitations
- Column evals use modest n (7 real merges; 12 contested split endpoints); numbers
  are directional. Split metric is lenient (same-neuron, not true continuation).
- Synthetic merges may under-power the A↔D signal; the real-merge check corrects this.
- The same-type merge residual is genuinely open; the promising lever is **not**
  more SegCLR modeling but EM-native seam features, or a supervised
  identity-metric head (see the identity-decodability test).
