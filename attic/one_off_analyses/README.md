# one_off_analyses — single-use analyses, exports and probes

**Era.** Mixed, March–August 2026. These are not a research track; each one
answered a single question once and was never wired into a pipeline.

| Script | What it did | Referenced by |
|---|---|---|
| `analyze_minnie65_boxes.py` | Surveyed candidate Minnie65 boxes for soma density. | `docs/minnie65_box_analysis.md` (its written output). |
| `benchmark_boundary_search.py` | Boundary-search sweep for the box-local CellGNN track. | `models/README.md`, as the provenance of one checkpoint. |
| `benchmark_synapse_membership_box.py` | Measured synapse membership inside a fixed box. | Nothing. |
| `characterize_v117_to_v1718.py` | Characterized the v117 → v1718 segmentation delta. | Nothing. Superseded by `docs/seg_117_to_1412.md` and `scripts/probe_seg_mapping.py`. |
| `export_merge_dataset.py` | Exported a merge-pair dataset for the CellGNN track. | Nothing. |
| `fetch_edit_locations.py` | Fetched edit locations from CAVE. | Nothing. Superseded by `scripts/fetch_edit_history.py`, which the soma-seeded thread uses. |
| `verify_cache_lineage.py` | One-time lineage check on a training cache. | Nothing. |

**What replaced them.** For anything that touches the current substrate:
`scripts/` now holds only builders and probes that write `data/substrate/` or
`data/external/`, and `docs/MAP.md` §1.3 lists them. For the segmentation-delta
questions specifically, `scripts/probe_seg_mapping.py` is the live version.

**Route back.** None of these is gated by an experiment. If one turns out to
answer a live question, copy the part you need into a probe under `scripts/`
rather than reviving the whole file — that is the repo's stated habit
(`CLAUDE.md`: "prefer the smallest experiment that answers the question").
