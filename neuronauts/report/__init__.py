"""Experiment reports: provenance, normalised records, figures, and viewers.

Every experiment in this repo ends in a JSON file under ``results/`` plus, for
the benchmarks, a hand-written ``results/expNNN_evaluation.md``. The JSONs are
honest but each has its own shape, so reading across experiments means
re-learning each schema. This package puts one layer over them:

  provenance.py  capture *and* audit run provenance (git state, input hashes,
                 environment, parameters); ``stamp``/``write_result`` for new
                 scripts, ``completeness`` to grade what old ones recorded
  registry.py    discover result files and normalise them into
                 ``ExperimentRecord`` -- status, provenance, population,
                 headline scalars, and every sweep-like table
  figures.py     matplotlib figures derived from a record (sweeps, operating
                 points, grids, percentiles, pair counts)
  render.py      one Markdown report per experiment plus an index with a
                 dependency graph between experiments
  ngl.py         base Neuroglancer state builder in nanometre coordinates,
                 plus views for a region, an experiment's box, and one atom

Command line:

    uv run python scripts/build_reports.py          # every result -> results/reports/
    uv run python scripts/ngl_view.py atom <id>     # Neuroglancer link for one atom
"""

from neuronauts.report.provenance import (
    capture_provenance, completeness, stamp, write_result,
)
from neuronauts.report.registry import ExperimentRecord, discover, load_record

__all__ = [
    "ExperimentRecord", "capture_provenance", "completeness", "discover",
    "load_record", "stamp", "write_result",
]
