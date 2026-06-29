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

## Graduation

When it lands: open a PR from the branch, and if the fingerprint retrieval beats
proximity/connectivity baselines on real v117 atoms, fold the encoder into the
`represent/` stage alongside tree-DNA. Until then it stays external.
