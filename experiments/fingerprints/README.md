# Thread: fingerprints

**Goal.** Give each segmentation fragment (a v117 "atom") a **connectivity
fingerprint** — an identity signature derived from *who it synapses with* (and
where) — so that fragments of the same neuron can be retrieved as siblings.
Where [tree_dna](../tree_dna/README.md) encodes a fragment's own *morphology*,
fingerprints encode its *connectivity context*: synaptic partners, partner
co-occurrence (TF-IDF over partner roots), and spatial proximity.

> **fingerprints ≠ tree-DNA.** They are two different identity cues for the same
> stitching problem and live on different branches. Earlier docs conflated them;
> this thread is the connectivity one.

**Status:** external — the work lives on an unmerged branch, not in this tree.
See [Findings](#findings-from-small-e2e-test-b9k2g) below for the quantitative
summary from the earlier v117-atom probe.

## Where the code is

- **`claude/neuron-fingerprints-connectivity-jg95xp`** — the primary branch
  (44 commits ahead of `main`, 0 behind as of 2026-06-29: strictly ahead and
  PR-ready). Not merged here yet.
- **`claude/small-e2e-test-B9k2g`** — related earlier probe: v117-atom
  split-recovery via sibling retrieval (TF-IDF / cosine / proximity AUROC on real
  v117 atoms). Its finding — real v117 atoms have too few synapses (median ~10)
  to carry strong fingerprint content at the atomic scale — is the constraint
  this thread is up against.

To read the branch: `git fetch origin claude/neuron-fingerprints-connectivity-jg95xp`
then `git show FETCH_HEAD:<path>`.

## Relationship to other threads

- Complements [tree_dna](../tree_dna/README.md): morphology (shape) + connectivity
  (partners) are independent evidence for "same neuron".
- Feeds the same global-assembly goal as
  [cell_assignment](../cell_assignment/README.md) and the co-assignment /
  stitch branches (`synapse-coassign`, `abstract-tree-stitch`).

## Findings from `small-e2e-test-B9k2g`

Source: `FRAGMENT_ASSEMBLY_RESULTS.md` on branch `claude/small-e2e-test-B9k2g`
(250 v1412 cells, 927k synapses; branch closed after findings folded here).

### Synthetic fragment assembly (balanced PCA bisection)

Each cell PCA-bisected into K disjoint fragments; top-1 = sibling recovered
as the single nearest neighbor out of N−1 candidates.

| K | conn-tfidf top-1 | top-10 | AUROC | proximity AUROC |
|--:|--:|--:|--:|--:|
| 2 (500 fragments) | **23.0%** | 65.6% | **0.952** | 0.789 |
| 4 (1000)          | **30.0%** | 75.4% | **0.912** | 0.819 |
| 8 (2000)          | **33.8%** | 72.4% | **0.851** | 0.825 |

Key: **the fingerprint is a near-perfect narrower** — the median sibling ranks
within ~1% of the haystack (narrowing factor 0.99+). Top-1 undersells it;
top-50 hits 92% at K=2, 94% at K=4. Plain cosine and Jaccard trail TF-IDF;
TF-IDF's downweighting of high-degree hub cells is doing real work.

Proximity fails at top-1 (0%) when the PCA bisection is adversarial (sibling
halves placed far apart by construction), but connectivity stays strong.
The trained path encoder (30 µm windows) scores below chance (AUROC 0.43–0.53)
— the encoder's spatial scale doesn't compose into cell-level identity.

### Real v117 atoms (the actual working scale)

Each v1412 cell decomposes into v117 root_ids (median **74 atoms/cell**,
median **10 synapses/atom**). Ground truth: same v1412 parent = sibling.

| min-syn | atoms | cells | conn-tfidf AUROC | proximity AUROC |
|--:|--:|--:|--:|--:|
| 5   | 6944 | 237 | 0.517 | **0.680** |
| 20  | 1729 | 213 | 0.577 | **0.737** |
| 50  |  770 | 159 | 0.679 | **0.800** |
| 100 |  343 | 105 | 0.763 | **0.844** |
| 200 |  142 |  56 | 0.871 | 0.885 |

**The synthetic experiment over-promised.** At the realistic atomic scale
(min-syn 5), TF-IDF AUROC is 0.52 vs 0.68 for proximity. Real atoms have a
median of ~10 synapses — not enough fingerprint content. Filtering to
min-syn ≥ 100 recovers AUROC to 0.76, but at the cost of 74% of cells (237→63)
and 95% of atoms. Proximity beats connectivity at every threshold.

### Split recovery vs. merge detection: different features

Partner-set coherence (TF-IDF cosine between two halves of a fragment):
AUROC **0.69** for detecting wrongful merges, vs. **0.91** for spatial
coherence. The features are complementary: use connectivity for *split
recovery* (where proximity is adversarial), spatial coherence for *merge
detection* (where the two fused cells are physically far apart).

### Design implications for the fingerprints thread

1. **Need ≥ ~100 synapses per atom** for the fingerprint to carry signal.
   Possible fix: aggregate atoms into "scaffolds" before fingerprinting, or
   route small atoms to a different mechanism.
2. **TF-IDF as a candidate filter, not a ranker.** Even at AUROC 0.52, the
   narrowing factor stays at 0.985+ — fingerprint narrows 6944 candidates to
   ~50 for downstream disambiguation.
3. **The path encoder's 30 µm scale is the bottleneck.** AUROC 0.43 on this
   task. Extending it to whole-cell or multi-scale representations is the lever.
4. **Combine proximity + connectivity** at the atomic scale. Neither alone
   exceeds AUROC 0.70 at realistic min-syn; together they plausibly do.

## Graduation

When it lands: open a PR from `claude/neuron-fingerprints-connectivity-jg95xp`,
and if the fingerprint retrieval beats the baselines above on real v117 atoms,
fold the encoder into the `represent/` stage alongside tree-DNA. The threshold
to beat is proximity AUROC 0.68 at min-syn 5 (all atoms) or 0.84 at min-syn
100 (large atoms only).
