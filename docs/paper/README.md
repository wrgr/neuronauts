# ⚠️ Superseded drafts — numbers pending re-derivation

**Status: 2026-09-01. Do not submit, circulate, or quote figures from these
drafts until the numbers are re-derived on real data.**

The manuscripts in this directory —

| File | Headline claim |
|---|---|
| `TreeGrammar_Connectomics_2026.{tex,pdf}` | Tree-grammar connectomics, EXP-035 "restored SOTA" |
| `MICCAI_2026_Neuronauts.{tex,pdf}` | Active micro-volumetric inference + contrastive morphological DNA |
| `MICCAI_2026_Neuronauts_Global_Assembly.md` | Same, markdown source |
| `walkthrough.md` | EXP-035 metric recovery and confusion matrices |

— take their quantitative results from the `EXP-020`–`EXP-050` benchmark
series, which is **not real-data evidence**. Verified by direct inspection of
the scripts (see `EXPERIMENT_LOG.md`, `docs/consolidation_plan.md` §1.4):

1. **The test worlds are synthetic.** Every benchmark in that range imports
   `treestitch.worldbuild.frankenmerge_adjacent` and applies it at **45–50%**
   to real proofread skeletons that were themselves **synthetically cut**.
   `walkthrough.md`'s EXP-035 is
   `frankenmerge_adjacent(pieces_rec, 0.45, rng, radius_nm=9500.0)`.
2. **The engines are untrained.** 15 of the 26 modules in
   `neuronauts/morpho_grammar/` draw random numbers at runtime and **none**
   loads a checkpoint. The reported accuracies are those of randomly
   initialised models.
3. **This was found independently once before.** `results/exp051_evaluation.md`
   audited EXP-049 and reached the same conclusion, including that the
   SANTIAGO infiller "initializes random matrices at runtime rather than
   loading a trained real-data grammar checkpoint."

Consequently the SOTA comparison tables — which place our numbers beside FFN
(Januszewski 2018), Janelia multicut (Beier 2017), DeepMulticut (Li 2024) and
FlyWire (Dorkenwald 2024), all measured on real data — compare untrained
engines on synthetic damage against published methods on real volumes. That
comparison is not sound and must not appear in a submission.

The code these drafts describe now lives in
[`attic/morpho_grammar/`](../../attic/morpho_grammar/) and
[`attic/benchmarks_semi_synthetic/`](../../attic/benchmarks_semi_synthetic/).

## What a re-derivation requires

`docs/consolidation_plan.md` §6 defines the path back. The relevant steps:

- **EXP-057/058** establish the real substrate, ground-truth overlay, and the
  baseline ladder (do-nothing, random, proximity, current checkpoints, oracle),
  so any claim has a floor and a ceiling beside it.
- **EXP-064** compares scorers on one fixed, label-blind candidate panel.
- **EXP-069** re-runs the best engine from this line under that protocol with a
  *trained* grammar. If it wins there, these numbers can be restated; if it
  does not, the drafts need rewriting around the results that did hold up
  (`docs/consolidation_plan.md` §1.5).

Literature numbers belong in a separate table labelled "different data,
different protocol" and must never share a row with ours.
