# neuronauts Research Program

## Goal
Maximize **line graph F1** on the synthetic test volume, then transfer to MICrONS.

Line graph F1 measures how accurately the agent swarm recovers neuron connectivity:
- TP = two synapses correctly identified as co-incident on the same neuron
- FP = false merge (synapses incorrectly linked)
- FN = false split (synapses that should be linked but aren't)

The metric is computed in `neuronauts/line_graph.py` and returned by `neuronauts/run.py`.
The scalar to optimize is `metrics.f1` (higher = better, max 1.0).

## What you may edit
Primary experiment entrypoint: `neuronauts/run.py`.

For autoresearch-style sweeps, edit the CONFIG block between the
`# EXPERIMENT CONFIG` and `# END CONFIG` markers.

Tunable parameters:
- `AgentConfig` fields (sensor weights, speed, thresholds)
- `N_AGENTS` and `SYNAPSE_SPAWN_FRACTION`
- `MERGE_RADIUS` and `MIN_PATH_LENGTH`
- `POLARITY_CAPTURE_RADIUS`
- Field parameters (`MEMBRANE_SIGMA`, `SYNAPSE_ATTRACTION_RADIUS`, etc.)

Keep package internals intact unless you are intentionally changing core behavior.

## How to run one experiment
```
python -m neuronauts.run
```
This runs on the synthetic 96x96x96 volume and prints `val_f1 = X.XXXX`.

## Evaluation budget
Each run takes ~30-90 seconds on CPU. You have a fixed time budget.
Run as many experiments as possible, keep changes that improve val_f1.

## What good looks like
- val_f1 > 0.5 on synthetic data is a reasonable first target
- val_f1 > 0.7 would be competitive with the 2020 APL BRAIN paper on FIB-25

## Known failure modes to watch for
- Agents all dying early (raise `max_steps` or lower `membrane_threshold`)
- Over-merging (val_f1 drops, FP rises — lower `merge_radius`)  
- Under-merging (FN rises — raise `merge_radius` or `synapse_capture_radius`)
- Agents not reaching synapses (raise `w_synapse_attraction`)

## Biological context
- Agents trace neuron processes through 3D EM volumes
- Membranes = dark boundaries in the image (Sobel gradient detects them)
- Synapses = target waypoints (agents spawn at them and seek them)
- Two agents merge into one neuron if their paths cross
- The line graph edge between two synapses = they share a neuron

## Transfer to real data (future)
After optimizing on synthetic, run on MICrONS mip-2 data:
```python
from neuronauts.fetch import fetch_volume, fetch_synapses
# See fetch.py for bbox_nm format
```
Ground truth is available from CAVE (minnie65_public, no token required).
