# outer_loop_and_viz — training loops and visualizers around a retired track

**Era.** Spring 2026, around the box-local CellGNN pipeline and the
semi-synthetic benchmark generation. Moved here 2026-09-01 and 2026-09-02; the
per-file audit that justified each move is in
[`../README.md`](../README.md#the-outer-loop--dashboard--viz-move-2026-09-01)
and is not repeated here.

## What is in here

| Group | Files | What it was |
|---|---|---|
| Outer training loops | `watch_and_eval.sh`, `eval_at_t099.sh`, `eval_path_models.sh`, `run_feature_ablation.sh`, `run_k_ablation.sh`, `run_timing_pipeline.sh` | Shell `for`-loops that drove `scripts/train.py train-cell-gnn` / `evaluate` to produce the K-hop and per-feature ablation checkpoints for the box-local CellGNN, plus a checkpoint watcher and an end-to-end timing harness. |
| Document rendering | `render_whitepaper_pdf.sh` | A `pandoc`/`xelatex` wrapper over `docs/whitepaper.md`. |
| Dashboards | `dashboard/` (`app.py`, `streamlit_app.py`, `results_explorer.py`, `neuroglancer_export.py`, `templates/index.html`) | A Flask "v2 performance dashboard" keyed to the v1 `run_research_cycle` pipeline, plus two Streamlit result-bundle viewers. |
| Visualizers | `generate_dashboard.py`, `export_viz_data.py`, `viz_pipeline.py` | A Three.js HTML dashboard with its headline numbers hardcoded into the HTML string rather than computed; a data exporter that cuts real skeletons into synthetic pieces, fabricates synapses, and injects frankenmerges at 35%; and a 588-line Plotly viewer over `BoxCache` / `shared_grammar_model` / `cell_graph`. |
| Generated artifacts | `viz_synthetic_artifacts/connectome_visualizer.html`, `viz_synthetic_artifacts/sample_connectome_viz.json` | What `export_viz_data.py` produced. `results/exp051_evaluation.md` flagged the JSON independently: "contains synthetic IDs and zero links." |

## Why it is here

Two reasons, and they apply to different files. The loops train and evaluate a
track whose ceiling is architectural, not a tuning gap — the box-local CellGNN
tops out at held-out F1 0.272 because a 30 µm box cannot hold a larger neuron.
The visualizers render the synthetic-frankenmerge world described in
`docs/consolidation_plan.md` §1.4, so what they show is not real reconstruction
error.

## What replaced it

- **Views:** [`neuronauts/meshing/`](../../neuronauts/meshing/) serves precomputed
  meshes and skeletons to Neuroglancer against real data, and
  [`neuronauts/report/`](../../neuronauts/report/) renders `results/reports/` —
  Markdown, figures and Neuroglancer state — for every registered result. Between
  them they cover what `dashboard/` was for.
- **Outer loops:** the runner. `neuronauts/experiments/_runner.py` reads a
  predeclared criterion, refuses to report when a prerequisite fails, and writes
  `results/EXP-xxx/`. A sweep is now an experiment with a bar, not a shell loop.

Two things here are *not* retired and are easy to confuse with things that are:
the top-level `viz/` directory stays (it is `neuronauts/meshing/`'s default
output root), and `treestitch/ngl_export.py` stays (three live modules import it).

## Route back

None registered. A dashboard or viewer feature returns as a `report/` or
`meshing/` module built against real data — not by reviving a file from here.
