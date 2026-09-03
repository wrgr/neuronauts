# Minnie65 Non-Zero Box Count Analysis

> How many distinct (non-overlapping) boxes contain synapses at different box sizes?  
> Smaller boxes → more boxes → more global coverage from training.

## Geometric Estimates

Using two assumed extents:

- **Full extent** — dataset_builder bounds: 3.5 mm × 2.4 mm × 0.73 mm
- **Core ~1 mm³** — proofread neuropil (MICrONS docs)

| Box (µm) | Full extent | Core ~1 mm³ |
|----------|-------------|-------------|
| 4        | 95.6M       | 15.6M       |
| 6        | 28.2M       | 4.57M       |
| 10       | 6.13M       | 1M          |
| 20       | 756k        | 125k        |
| 30       | 223k        | 36k         |
| 40       | 94k         | 15.6k       |
| 60       | 28k         | 4.1k        |
| 80       | 11.6k       | 1.7k        |
| 100      | 5.9k        | 1k          |
| 150      | 1.5k        | 216         |
| 200      | 612         | 125         |

## Implications

**Two different axes:**

- **Bigger boxes** → more global structure *per box* (each example sees more neurites, more synapses, more topology within one forward pass)
- **Smaller boxes** → more boxes total → more spatial coverage and training diversity (127× more 6 µm than 30 µm boxes in the core)

- **6 µm vs 30 µm:** ~127× more boxes (core), but each 6 µm box is more local
- **6 µm vs 15 µm:** ~16× more boxes
- **10 µm vs 30 µm:** ~27× more boxes

**Trade-offs:**

- Smaller boxes have fewer synapses per box — lower `min_positive_pairs` (e.g. 2 for 6 µm, 5 for 30 µm)
- Grammar training is CAVE-only (no EM fetch) — scales with number of boxes
- GAT/agent simulation requires EM — cost per box scales with volume (S³)
- Bigger boxes give the GAT more context per pass; smaller boxes give more independent training examples

**Recommendation:** For more global structure *within* each inference, use larger boxes (20–40 µm). For more training examples and spatial diversity (grammar-only, CAVE-only), smaller boxes (6–10 µm) scale better.

**Pushing to 200 µm:**

- **Box count:** 125 (core) or 612 (full) — the whole 1 mm³ fits in ~125 non-overlapping 200 µm cubes.
- **Global structure:** Each 200 µm box sees a large chunk of the connectome — dozens of neurons, thousands of synapses, substantial arbor topology.
- **Bottleneck: EM volume.** At MIP-2 (32×32×40 nm), a 200 µm box ≈ 6250×6250×5000 voxels ≈ **195 GB** raw. Current pipeline fetches full EM per box — not feasible on typical hardware.
- **Workaround:** Grammar-only (CAVE synapses, no EM) could use 200 µm boxes; GAT/agents need EM. For big boxes with agents, either tile internally (load in chunks) or use MIP 3 (64×64×40 nm → 4× less voxels, ~49 GB per 200 µm box — still large).
- **Sweet spot for EM:** 40–80 µm keeps volume tractable (40 µm ≈ 1.5 GB, 80 µm ≈ 12 GB at MIP-2) while giving more global context than 6–30 µm.

## Running the Analysis

```bash
# Geometric estimates only (no network)
python attic/one_off_analyses/analyze_minnie65_boxes.py --estimate-only

# Actual counts from CAVE sample (requires caveclient, network)
python attic/one_off_analyses/analyze_minnie65_boxes.py --sample 50000 --source cave

# From static synapse CSV (if downloaded)
python attic/one_off_analyses/analyze_minnie65_boxes.py --static-dir data/microns_static --version 1078 --sample 200000 --source static
```

---

*See `attic/one_off_analyses/analyze_minnie65_boxes.py` for the analysis script.*
