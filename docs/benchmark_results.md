# Partition Benchmark Results

**Date:** 2026-06-13  
**Epochs:** 80  **Seed:** 0  **Device:** cpu

## SOTA Context

| Method | Input | Training | ARI | merge_P | fk_detect |
|---|---|---|---|---|---|
| FFN/Pathfinder ¹ | Raw EM voxels | Voxel labels | ~0.95 * | N/A | No |
| NEURD ² | 3-D mesh | Rule-based | ~0.80 * | N/A | No |
| AutoProof ³ | Seg+mesh | Expert edits | N/A | ~0.97 * | No |
| Union-find (ours) | Synapse coords | Version history | see below | see below | No |
| edge\_cc (ours) | Synapse coords | Version history | see below | see below | **Yes** |

> \* Published on different benchmarks; not directly comparable to ARI.
> ¹ Januszewski et al. 2018  ² Bae et al. 2021  ³ Dorkenwald et al. 2023

## Synthetic Benchmark Results

| n\_obj | method | ARI | clusters | merge\_P | merge\_R | over | Bar1 | Bar2 |
|---|---|---|---|---|---|---|---|---|
| 4 | union-find | 1.0000 | 4/4 | 1.000 | 1.000 | 0.000 | — | — |
| 4 | edge\_cc | 1.0000 | 4/4 | 1.000 | 1.000 | 0.000 | PASS | PASS |
| 8 | union-find | 1.0000 | 8/8 | 1.000 | 1.000 | 0.000 | — | — |
| 8 | edge\_cc | 1.0000 | 8/8 | 1.000 | 1.000 | 0.000 | PASS | PASS |
| 16 | union-find | 1.0000 | 16/16 | 1.000 | 1.000 | 0.000 | — | — |
| 16 | edge\_cc | 1.0000 | 16/16 | 1.000 | 1.000 | 0.000 | PASS | PASS |

## Notes

- Bar 1: edge\_cc ARI ≥ union-find ARI **and** merge\_P ≥ union-find merge\_P
- Bar 2: merge\_P > 0.90 and merge\_R > 0.70 (operational threshold)
- Bar 3: frankenmerge\_split\_recall > 0.5 — requires real CAVE data (not shown)
- Synthetic graphs have one v117 fragment per neuron (no real frankenmerges)
