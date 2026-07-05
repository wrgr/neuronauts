# Two-cue abstaining auto-proofreader — findings

Error **detection + correction** framed like a trained proofreader: every candidate
edit must satisfy a **global shape grammar** (Pillar 1) *and* a **local EM
ultrastructure** cue (Pillar 2), fused by a **calibrated abstaining combiner**
(Pillar 3), with the residual deferred to a ranked queue. This documents what is
built, what is validated, and — honestly (CLAUDE.md) — what does **not** yet work.

## What is built and tested (offline, no network)

| Pillar | Module | Tests |
|---|---|---|
| 1 global grammar energy | `grammar_energy.py` | `tests/test_grammar_energy.py` (4) |
| 2 local EM ultrastructure | `local_evidence.py` | `tests/test_local_evidence.py` (5) |
| 3 combiner + candidates + queue | `complementarity.py`, `queue.py`, `pipeline.py` | `tests/test_proofread_queue.py` (3) |

## Pillar 1 — grammar energy (validated on real merges)

`grammar_energy` scores ungrammaticality (multi-soma, A↔D-not-via-soma, caliber
jump, disconnection); `cut_delta_energy` / `join_delta_energy` give per-edit ΔE.
On real m343 false-merges that split into two current roots, joining the two pieces
scores **ΔE < 0 (rejected)** when it fuses two somata; clean neurons score E≈0.6–1.6.
**Crucially, same-compartment merges score ΔE ≈ 0 — grammar is blind to them.** That
blind spot is the whole reason for Pillar 2.

## Pillar 2 — local EM cue (validated on real EM at proper cut-face sites)

`local_evidence(pos_a, pos_b, embed_fn)` returns `cutface_sim` (committed contrastive
cut-face encoder over the two cross-sections) and `barrier` (dark-membrane profile
along the connecting axis) from one bulk mip-1 EM+seg fetch.

Re-ID sanity on a real MICrONS box (`experiments/proofread/val_local_evidence.py`): two
z-separated faces of the **same** neurite score mean `cutface_sim` **+0.58** vs
**+0.30** for **different** neurites (separable); `barrier ≈ 0` on continuous
cytoplasm. **The cue carries same/different-process signal at genuine cross-section
sites.**

## Ground truth (Task 23)

`experiments.pcfg.run_synapse_correction.fetch_side_table` over a single 24 µm box at
the proofread column center (v117 → v1718), cached at
`cache/sidetable/col_n1_v1718.npz` (gitignored; ~15 min to refetch). `summarize_edits`:

```
sides=17874  v117_roots=5362  later_roots=5376
split_roots=12   (false MERGES to cut)
merge_targets=2  (false SPLITS to join)
```

## Complementarity result — HONEST, and it does **not** yet clear the bar

Unified "do these two synapse-sides belong to the same cell?" over both strata
(within-root y=0 = false merge/CUT; cross-root y=1 = false split/JOIN), local-site
filter ≤6 µm, leakage-safe GroupKFold OOF logistic. Command:

```
PYTHONPATH=. python -m experiments.proofread.run_complementarity \
  --sidetable cache/sidetable/col_n1_v1718.npz --max-candidates 80 --max-pair-nm 6000 \
  --out out/complementarity_n1.json
```

**80 pairs (24 local cut-errors, 0 join candidates), 57 cell groups:**

| cue | AUC (OOF) |
|---|---|
| shape / grammar (point-cloud, AutoProof-style baseline) | **0.487** (≈ chance) |
| local EM ultrastructure (Pillar 2) | **0.601** |
| joint combiner | **0.536** |

Cue direction is correct among the 54/80 sites with a valid cross-section:
same-cell `cutface_sim` 0.502 vs different 0.405; `barrier` 0.041 vs 0.115. The
ranked queue proposes 8 confident CUTs at **precision 0.125 — below the 0.30 base
rate.** Abstention correctly defers 72/80, but the confident auto-edits are **not
trustworthy on this substrate.**

**Read this straight:** the local-EM cue is the single most informative stream
(0.601 > shape 0.487), which is directionally consistent with the two-cue thesis,
but the joint combiner does **not** beat local-alone and end-to-end auto-correction
is **not** deployable here. This is a negative/underpowered result, not a win.

### Why it fails here (diagnosed, not hand-waved)

1. **Wrong sample point.** Candidates are sampled at **synapse-cleft positions**, not
   on the neurite. 26/80 sites have no cross-section footprint at all (`ok=0`), and a
   cleft cross-section is not the neurite cross-section the encoder was trained on.
2. **Wrong site for a cut.** A false-merge's error is the **seam** between two lobes;
   our within-root pairs are two arbitrary synapse sides (median 11 µm apart), so the
   local cue is not read at the place the human would cut.
3. **Underpowered.** One box gives 24 local cut-errors and **0** usable join
   candidates — too few, and the join direction (where Pillar 2 should shine) is
   untested. Contrast the clean re-ID result above, which *does* separate at proper
   cut-face sites.

The failure traces to **site placement and sample size, not the cue** (which
separates at proper sites) — so the fix is mechanical, not a dead end.

## Honest next steps

1. **Localize the edit site geometrically** (skeleton / `walk_detector` seam +
   endpoints), sample both cues **on the neurite at the seam/tip**, not at the
   synapse cleft. This is the single change most likely to move the numbers.
2. **More boxes** to accumulate join candidates (false splits), the direction the
   local cue is built for; re-run complementarity with both directions populated.
3. Only then wire confident edits through **matching** (not agglomeration) and report
   synapse-pair line-graph F1 before/after against the oracle (0.928) / greedy (0.14)
   anchors.

## Positioning

Baseline = shape/synapse (AutoProof-style); our added cue = local EM ultrastructure.
The decomposition and abstaining queue are built and tested; the *complementarity
claim is not yet supported by data* and is gated on step 1 above.
