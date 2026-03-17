# Neuronauts Research Program

## The problem

You are recovering a **connectome** — a directed graph of which neurons connect
to which — from a 3D electron microscopy volume.  You do this by deploying a
swarm of virtual agents that trace neuron processes through the volume.

The goal is not to label every voxel.  You only need to determine which synapses
share a neuron.  Two synapses on the same neuron = one edge in the connectome.

---

## Fixed inputs  (do not modify how these are loaded)

```
volume            : (X, Y, Z) uint8    — EM intensity volume
cached_membrane   : (X, Y, Z) float32  — optional learned membrane probability
                 In real-data mode, `run.py` may load a cached membrane volume
                 for the same box from `cache/membranes`. If none is present,
                 it falls back to Sobel on the EM volume.

synapse_pre_pts  : (N, 3) float32  — pre-synaptic  site locations in voxels
synapse_post_pts : (N, 3) float32  — post-synaptic site locations in voxels
```

Both synapse arrays come from CAVE ground truth. They are your spawn points,
navigation targets, and the objects whose co-incidence you are recovering.

The membrane cache is a preprocessing artifact, not a new inference target for
the optimizer. Treat it as a fixed input once it exists for a box.

---

## Fixed output  (do not modify `line_graph.py`)

```
val_f1 : float in [0, 1]
```

**What val_f1 measures:**  Build a graph where nodes = synapses and edges =
"these two synapses share a neuron."  Compare to the ground-truth graph from
CAVE root IDs.

- **TP** = correctly identified co-incident synapse pairs
- **FP** = false merges  (linked synapses that belong to different neurons)
- **FN** = false splits  (missed links between synapses on the same neuron)
- **F1** = 2·P·R / (P+R)

Precision and recall are printed separately.  They tell you which direction you
are failing:

| P low, R high | over-merging  — agents crossing membranes          |
| P high, R low | under-merging — agents not linking same-neuron paths|

---

## Key architectural decisions  (already in the code — preserve these)

### Pre/post role separation
Agents are split into two independent populations at merge time:

- **Pre-role agents** — those that visited pre-synaptic sites — are merged
  only among themselves into "pre neurons" (axons).
- **Post-role agents** — those that visited post-synaptic sites — are merged
  only among themselves into "post neurons" (dendrites).

An edge is added to the graph only when a pre-neuron and a post-neuron each
claim opposite sides of the same synapse.  This means a false merge between
two pre-side agents can never create a spurious edge with a post-side agent,
and vice versa.  The separation is enforced in `_merge_role_groups()`.

**Do not break this invariant.**  It is what makes the pre→pre and post→post
constraint meaningful at small volume scales.

### Merge gating
Proximity alone does not trigger a merge.  Two agents also need a minimum
number of shared synapse hits (`ROLE_MERGE_MIN_SHARED_HITS`) and a minimum
path-overlap fraction (`MERGE_OVERLAP_THRESHOLD`).  This prevents merging
agents that happen to pass near each other but are tracing different processes.

### Synapse capture radius
The capture radius (`synapse_capture_radius`) is intentionally small (≈1–3
voxels) to avoid spurious claims at process boundaries.  If it is too large,
agents on adjacent processes both claim the same synapse, corrupting the graph.

---

## What you may modify  (everything in `run.py` between the CONFIG markers)

You are not limited to tuning scalar weights.  You may:

- Replace the sensor weight vector with any policy expressible in numpy
- Change how membrane/exploration signals translate into agent velocity updates
- Replace the distance-based merge criterion with any data-driven criterion
- Change spawn strategy: density, pre/post fraction, jitter scale
- Add a small numpy MLP if it meaningfully improves F1
- Tune `MERGE_RADIUS`, `MERGE_OVERLAP_THRESHOLD`, `ROLE_MERGE_MIN_SHARED_HITS`
- Tune `POLARITY_CAPTURE_R` and `MAX_SYNAPSES_PER_NEURON`

You may NOT:

- Use ground-truth segmentation (neuron IDs) during inference
- Modify `line_graph.py`, `merge.py`, `fetch.py`, or `vectorized.py`
- Add unsupported new inputs beyond the EM volume, optional cached membrane
  field, and synapse locations
- Hardcode values that only work for one specific subvolume

---

## How to run one experiment

```bash
python -m neuronauts.run --data-mode real --membrane-source auto
```

Ends with:
```
val_f1 = X.XXXX
```

Each run should complete in under 60 seconds.  Reduce `N_AGENTS` or
`max_steps` if it takes longer.

For a quick synthetic smoke run:
```bash
python -m neuronauts.run --cases 1 --benchmark-mode fixed_validation
```

For a quick real-data validation run:
```bash
python -m neuronauts.run --data-mode real --real-boxes-per-eval 1 --membrane-source auto
```

---

## Optimization target

Maximize `val_f1`.  Keep any change that improves it.  Discard changes that
do not.  Record every experiment.

Milestones:
- val_f1 > 0.50 : competitive with heuristic baselines
- val_f1 > 0.60 : current best on synthetic benchmark
- val_f1 > 0.70 : matches Drenkow et al. 2020 on FIB-25
- val_f1 > 0.85 : exceeds prior published work

---

## Failure mode reference

| Symptom                      | Diagnosis                | Things to try                                      |
|------------------------------|--------------------------|----------------------------------------------------|
| P low, R high                | Over-merging             | Lower MERGE_RADIUS; raise MERGE_OVERLAP_THRESHOLD  |
| P high, R low                | Under-merging            | Raise MERGE_RADIUS; more agents; more steps        |
| val_f1 flat across runs      | Policy insensitive       | Change architecture, not just weight values        |
| Agents not reaching synapses | Navigation failure       | Raise w_synapse_attraction; inspect membrane field |
| Cached and Sobel disagree    | Preprocess mismatch      | Rebuild membrane cache; inspect one box manually   |
| All synapses claimed step 0  | Capture radius too large | Lower synapse_capture_radius                       |
| Neurons >> true neuron count | Under-merging            | Raise MERGE_RADIUS or lower ROLE_MERGE_MIN_SHARED_HITS |
| Neurons << true neuron count | Over-merging             | Lower MERGE_RADIUS; raise MERGE_OVERLAP_THRESHOLD  |

---

## Experiment log

Format: `[brief config note] → val_f1=X.XXX (P=X.XX R=X.XX)`

Baseline (Sobel, heuristic sensors, fixed_validation case 0):
  val_f1 = 0.4909 (P=0.587 R=0.422)
