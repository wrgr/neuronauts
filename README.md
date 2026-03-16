# neuronauts

`neuronauts` is a Python package for agent-based connectome recovery experiments. It is structured to support the `karpathy/autoresearch` workflow: a small, explicit experiment surface, short repeated improvement loops against a scalar metric, and benchmark generation policy defined in config instead of hidden in ad hoc scripts.

The current benchmark is synthetic line-graph recovery. Agents move through a 3D EM-like volume, touch pre/post synaptic sites, merge into neuron hypotheses, and are scored by line-graph F1.

## What is here

- Installable package: `neuronauts/`
- Main experiment entrypoint: `python -m neuronauts.run`
- Core metric: `neuronauts.line_graph`
- Synthetic data generator: `neuronauts.fetch.make_test_volume`
- Oracle regression test for the pre/post merge bug: `tests/test_run.py`
- Whitepaper: `docs/whitepaper.md`
- 5-minute loop runner: `scripts/iterative_loop.py`

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
  iterative_loop.py
tests/
  test_run.py
```

## Autoresearch workflow

The intended pattern is:

1. Treat `neuronauts/run.py` as the main experiment file.
2. Keep edits focused on the config block and clearly motivated algorithm changes.
3. Run short, repeated local loops against `val_f1`.
4. Preserve any regression tests that reveal specific failure modes.

This repo already includes the first oracle-discovered structural fix: pre-role and post-role waypoint grouping are merged separately before synapse edges are assigned. That avoids false positive line-graph edges caused by mixing pre and post evidence into one merged neuron.

## Iterative 5-minute loops

The helper script below repeatedly runs the benchmark for a fixed wall-clock budget and logs every run:

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

Recommended loop discipline:

1. Edit one thing.
2. Run the regression test.
3. Run a 5-minute benchmark loop.
4. Keep only changes that improve the metric or close a known failure mode.
5. Write down the best observed mean `val_f1` and the exact config snapshot that produced it.

By default, each benchmark run evaluates multiple fresh synthetic cases. Use `--benchmark-mode fixed_validation` only when you want strict apples-to-apples regression tracking.

Each 5-minute iteration writes:

- `iteration_XXX/summary.tsv`: per-run metrics within that iteration
- `iteration_XXX/iteration_stats.json`: aggregate metrics for that iteration
- `iteration_summary.tsv`: one row per 5-minute iteration with mean/best/min F1, mean precision, and mean recall

That makes it straightforward to monitor progress over time and plot iteration-level curves.

Plotting example:

```bash
python scripts/plot_iterations.py run_logs/loop_a/iteration_summary.tsv --output run_logs/loop_a/iteration_metrics.png
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
