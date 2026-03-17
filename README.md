# neuronauts

`neuronauts` is a Python package for agent-based connectome recovery experiments. It is structured to support a small `autoresearch`-style workflow with a fixed benchmark harness, one main experiment file, and a human-written `program.md`.

The current benchmark is synthetic line-graph recovery. Agents move through a 3D EM-like volume, touch pre/post synaptic sites, merge into neuron hypotheses, and are scored by line-graph F1.

## What is here

- Installable package: `neuronauts/`
- Main experiment entrypoint: `python -m neuronauts.run`
- Core metric: `neuronauts.line_graph`
- Synthetic data generator: `neuronauts.fetch.make_test_volume`
- Oracle regression test for the pre/post merge bug: `tests/test_run.py`
- Human instruction file: `program.md`
- Whitepaper: `docs/whitepaper.md`
- 5-minute loop runner: `scripts/iterative_loop.py`
- Codex outer optimizer: `scripts/codex_optimize.py`
- Membrane U-Net trainer: `scripts/train_membrane_unet.py`
- Membrane cache builder: `scripts/cache_membrane_volume.py`

## Quick start

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run one benchmark batch:

```bash
python -m neuronauts.run
```

Run the regression test:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Install optional membrane-model dependencies when you want learned membrane preprocessing:

```bash
python -m pip install -e .[membrane]
```

To start the autonomous loop, point Codex at [program.md](/Users/wgray13/projects/neuronauts/program.md).

## Project layout

```text
neuronauts/
  agent.py
  fetch.py
  fields.py
  line_graph.py
  merge.py
  run.py
  vectorized.py
docs/
  whitepaper.md
scripts/
  cache_membrane_volume.py
  codex_optimize.py
  iterative_loop.py
  train_membrane_unet.py
tests/
  test_run.py
```

## Codex Optimization Loop

The actual `autoresearch`-style optimizer is:

```bash
python scripts/codex_optimize.py --repeat-until-interrupt
```

That script:

- asks Codex to make one focused edit to `neuronauts/run.py`
- runs the regression test
- runs fixed real-data validation on the same boxes every iteration
- keeps or reverts the edit based on fixed-validation F1
- logs each proposal under `run_logs/codex_optimize/`

Useful options:

```bash
python scripts/codex_optimize.py --iterations 1 --skip-live-loop
python scripts/codex_optimize.py --minutes 5 --log-dir run_logs/codex_session_a
python scripts/codex_optimize.py --improvement-threshold 0.0005
```

## Iterative 5-minute loops

The helper script below runs repeated fixed-time benchmark iterations and logs per-run plus per-iteration metrics:

```bash
python scripts/iterative_loop.py --minutes 5 --repeat-until-interrupt
```

Useful options:

```bash
python scripts/iterative_loop.py --minutes 5 --python .venv/bin/python
python scripts/iterative_loop.py --minutes 5 --log-dir run_logs/loop_a
python scripts/iterative_loop.py --minutes 5 --benchmark-mode fixed_validation --cases 5
python scripts/iterative_loop.py --minutes 5 --iterations 3
```

Each 5-minute iteration writes:

- `iteration_XXX/summary.tsv`: per-run metrics within that iteration
- `iteration_XXX/iteration_stats.json`: aggregate metrics for that iteration
- `iteration_summary.tsv`: one row per 5-minute iteration with mean/best/min F1, mean precision, and mean recall

That makes it straightforward to monitor progress over time and plot iteration-level curves.

Plotting example:

```bash
python scripts/plot_iterations.py run_logs/loop_a/iteration_summary.tsv --output run_logs/loop_a/iteration_metrics.png
```

## Membrane U-Net

For a first-pass learned membrane signal, train the included small 2D U-Net on the external tif dataset repo:

```bash
python scripts/train_membrane_unet.py \
  --dataset-dir /path/to/unet_image_segmentation/data \
  --output models/membrane_unet.pt
```

Then fetch one MICRONS box, predict membranes slice-wise, and cache the result:

```bash
python scripts/cache_membrane_volume.py \
  --checkpoint models/membrane_unet.pt \
  --center-nm 1153592,793592,655640 \
  --side-um 6.0 \
  --mip 2 \
  --cache-dir cache/membranes
```

Real-data runs can then use the cache automatically:

```bash
python -m neuronauts.run \
  --data-mode real \
  --membrane-source auto \
  --membrane-cache-dir cache/membranes
```

## Running on real data later

The package includes MICrONS-oriented fetch helpers:

```python
from neuronauts.fetch import fetch_volume, fetch_synapses, make_cube_bbox_nm

bbox_nm = make_cube_bbox_nm(center_nm=(200_000, 200_000, 20_000), side_um=6.0)
```

Those functions depend on optional external packages:

- `cloud-volume`
- `caveclient`

Install them only when moving from synthetic to MICrONS data.

The current recommended real-data chunk size is approximately `6 x 6 x 6 um`,
which corresponds to a `bbox_nm` side length of `6000` nm.

## References

- Karpathy, A. `autoresearch`. GitHub. <https://github.com/karpathy/autoresearch>
- Silversmith, W. `cloud-volume`. GitHub. <https://github.com/seung-lab/cloud-volume>
- CAVEconnectome. `CAVEclient`. GitHub. <https://github.com/CAVEconnectome/CAVEclient>
- MICrONS Consortium et al. “Functional connectomics spanning multiple areas of mouse visual cortex.” *Nature* (2021). <https://www.nature.com/articles/s41586-021-03778-x>
- Bae, J. A. et al. “Digital museum of retinal ganglion cells with dense anatomy and physiology.” *Cell* (2024). CAVE-style data infrastructure context. <https://www.cell.com/cell/fulltext/S0092-8674(24)00308-4>
