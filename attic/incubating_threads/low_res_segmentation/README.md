# Thread: low_res_segmentation

**Goal.** Reconstruct neurons at coarse resolution (128×128×120 nm) from
EM volumes, using membrane guidance and synapse positions to grow per-cell
segments without requiring full-resolution tracing.

**Status:** incubating. Runs on synthetic connectomes; real-CAVE scaffold
tested against Minnie65 soma boxes.

## Modules

| File | Role |
|------|------|
| `low_res_segmentation.py` | `LowResNeuronSegmentationPipeline` — downsampling, coordinate transforms, synapse-guided component labeling |
| `high_precision_segmentation.py` | `HighPrecisionSegmentation3D` — watershed + level-set refinement |
| `conservative_scaffolding.py` | `ConservativeScaffoldingPipeline` — conservative grower with per-cell stop |
| `evaluate_neuron_purity.py` | Purity metrics + synthetic connectome builder |

## Scripts

```bash
# Synthetic smoke test (no CAVE needed)
python attic/incubating_threads/low_res_segmentation/test_conservative_scaffolding.py

# Real Minnie65 CAVE box
CAVE_TOKEN=<token> python attic/incubating_threads/low_res_segmentation/test_cave_scaffolding.py

# Find soma box + full scaffold
CAVE_TOKEN=<token> python attic/incubating_threads/low_res_segmentation/find_soma_box.py
```

## Tests

```bash
pytest tests/test_low_res_segmentation.py tests/test_high_precision_segmentation.py -v
```

## Relation to main pipeline

Operates on EM volumes rather than synapse-table-only — complementary to the
default CellGNN path. A natural fusion point once the global-assembly roadmap
incorporates EM signal at coarse scale.

## Graduation

Promote to core once purity metrics on real Minnie65 boxes match or exceed
the CellGNN baseline F1 of **0.272** on the same boxes.
