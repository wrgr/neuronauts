# neuronauts-bench v1

The canonical train/val/test set. Real MICrONS data only, built from real
proofreading lineage. Nothing here is generated.

Rebuild (synapse and lineage fetches are cached, so a rebuild is fast):

```bash
python scripts/build_bench_v1.py \
  --train-regions OOC3 P1a A E --val-regions P1b --test-regions P1c \
  --seam-buffer-nm 25000 --limit 20000 --min-syn-per-fragment 1
python scripts/verify_split.py
```

## What a label means

`base v117 → labels v1718`. Every synapse's supervoxel is resolved to a root at
both versions:

- **true merge pair** — two v117 roots resolving to the same v1718 root. The
  segmentation split one neuron; a proofreader merged it. These are the
  positives a merge model must recover.
- **frankenmerge** — one v117 root spanning two v1718 roots. The segmentation
  merged two neurons; a proofreader split them. EXP-056 showed geometry alone
  cannot cleave these, so they are the label-noise floor, not free supervision.

## The splits

| Split | Regions | Observations | v117 roots | True merge pairs | Frankenmerges |
|---|---|---:|---:|---:|---:|
| train | OOC3, P1a, A, E | 76,429 | 52,261 | 351 | 756 |
| val | P1b | 16,244 | 11,375 | 59 | 181 |
| test | P1c | 20,000 | 12,287 | **153** | 203 |

**The full candidate population is included.** Median synapses per v117 root is
1, and 8,823 of the test region's 12,287 roots have exactly one synapse. That
sliver/singleton tail is not noise — it is the confuser set that defeated
EXP-051/052/053A (EXP-052: 1,023 usable roots against 10,218 singleton
confusers). An earlier build of this dataset filtered it with
`min_syn_per_fragment=3` and thereby discarded 87% of candidate roots and 68% of
true merge pairs, making the benchmark easier than the task. The default is now
1; any higher value prints what it costs at build time, and every manifest
records `population_unfiltered`.

P1a/P1b/P1c are z-thirds of P1, the densely proofread region where ~100% of
somata have v117 ≠ v1718. OOC3 is ~140 µm away from the test region and carries
substantial real signal (see the finding below), which makes train→test a
genuine cross-region generalization test rather than a within-neighbourhood one.

## Why you can trust the split

Verified by `scripts/verify_split.py`, which re-derives the checks from the
written manifests rather than trusting the builder's own logic:

- **Root-disjoint.** No v117 or v1718 root appears in two splits. Dedup removed
  2,165 roots from train and 2,194 from val that also occur in a higher-priority
  split (test > val > train). This is the exact guarantee.
- **Spatially separated** by ≥ 25,000 nm at every cross-split seam. The buffer
  is applied only to faces that actually face another split, because the
  training regions are just 70,000 nm deep in y and a uniform inset would
  collapse them.
- **Gated.** The build refuses to write a dataset failing its acceptance gates.
  It did exactly that on an early attempt: an assignment that put OOC3 in test
  left train with 11 merge pairs against a required 20, and the build aborted
  rather than emit it.

Phase 2.11 measured what skipping this costs: out-of-sample ARI fell 0.901 →
0.752 and fk_split 0.350 → 0.000 once boundary leakage was removed. The leaked
numbers were the optimistic ones.

## Honest limitations

- **This is a 20,000-synapse sample per region, not full coverage, and the
  sampled fraction is small.** Using EXP-052's documented density (~0.9 pre-side
  synapses/µm³), P1c's 541,670 µm³ would hold on the order of 10⁵ synapses, so
  20,000 covers only a few percent. (A direct density probe timed out, so that
  is an inference from a documented figure, not a measurement.) The real
  confuser population is therefore larger still and every precision figure is
  optimistic. The clean fix for `bench_v2` is to size regions to the fetch
  budget — a ~30 µm box where one ~20k fetch *is* the whole population, as
  EXP-052 used — rather than sampling a large box. Counts here are lower bounds.
- **Why the cap is 20,000.** Measured on this endpoint: `limit=20,000` returns
  in ~51s, `limit=50,000` in ~261s, and `limit=200,000` exceeds `lineage.py`'s
  own 300s request timeout and returns nothing. Response time tracks `limit`,
  not bbox size — a *narrower* 10 µm tile at limit 50,000 timed out while a
  20 µm tile succeeded. Deeper coverage needs many narrow tiles at ~20k each, or
  a raised timeout.
- **Merge-pair counts are sampling-density dependent.** Each P1 z-third sampled
  at 20,000 synapses yields as many or more pairs (16/32/22) than all of P1
  sampled at 20,000 (16), because a fixed sample of a smaller volume is denser.
  A pair is only observable when *both* fragments land in the sample.
  Consequently a region reported as having "no edit signal" at a capped fetch
  may simply be under-sampled — do not read the zeros in
  `docs/region_inventory.md` as proof of absence. That inventory was also built
  with `min_syn>=3`, so its root and pair counts are likewise low; the survey
  default is now 1.
- **Frankenmerges outnumber merge pairs ~5-10x** in every region measured. The
  dominant real error in v117→v1718 is a merge needing a split, not a split
  needing a merge. Much prior work emphasised the latter.
- **`v117 fragment = atom` is an approximation.** These splits contain 1,140
  mixed-lineage roots that cannot be cleaved by geometry (EXP-056). They are
  recorded per region so evaluations can report with and without them.
- Only the `pre` synapse side is included.

## Files

- `manifests/{train,val,test}.json` — committed. Bboxes, per-region stats, and
  the full v117/v1718 root id lists that define the split.
- `manifests/dataset.json` — provenance, per-split manifest hashes, gate status.
- `regions/*.npz` — bulk per-region arrays; gitignored and rebuildable.

**Dataset manifest sha256:**
`58f59da287b4331d2f01d10357b4f2b66e33aaae6421d13b6f96fed9d4dbc24f`
(supersedes `f4185886…`, which was built with the sliver filter described above)

Stamp it into every `ResultsRecord` produced from this dataset
(`neuronauts.results_schema`). The test manifest is locked: changing it means
building `bench_v2`, not editing in place, and CI refuses silent edits.
