# Meshing results for Neuroglancer and other 3D tools

`neuronauts.meshing` turns anything in this project that has a shape — a
v117 atom's real L2 adjacency, a kimimaro/CAVE skeleton, a `Fragment`
collection, an assembled neuron — into a triangle mesh and writes it as a
bundle that Neuroglancer, Blender, MeshLab, or three.js can open. Only numpy
is required; `cloud-volume` is used by the tests to cross-check the
precomputed bytes, and the `neuroglancer` package is an optional local viewer.

## Quick start

```bash
# No data on disk required — sanity-check the pipeline:
python scripts/mesh_results.py demo --out viz/mesh/demo
python scripts/mesh_results.py serve --dir viz/mesh
# open the URL in viz/mesh/demo/url.txt (or viz/mesh/demo/index.html)

# The 8 most-connected v117 atoms in the harness's 100 µm region:
python scripts/mesh_results.py harness \
  --geom-dir data/substrate/geom \
  --population data/substrate/c100um/population.npz \
  --top-n 8 --min-synapses 10 \
  --out viz/mesh/harness_top8

# A kimimaro box archive (cell_graph.precompute_self_skeletons_for_cache output):
python scripts/mesh_results.py kimimaro --archive <box_hash>.npz --out viz/mesh/box

# A Fragment collection (neuronauts.schemas.save_fragments), grouped by an
# assembly result's fragment_to_neuron mapping:
python scripts/mesh_results.py fragments \
  --fragments data/some_fragments.npz \
  --fragment-to-neuron fragment_to_neuron.json \
  --out viz/mesh/assembly_run
```

`serve` hosts the bundle's parent directory over CORS HTTP (Neuroglancer
fetches precomputed sources itself, from the browser, so the files have to be
reachable over `http://`, with CORS headers, not just readable from disk).

## Pipeline

```
SkeletonGeometry            (skeleton.py)   vertices_nm [V,3], edges [E,2], radii_nm [V]
      │  adapters (sources.py): harness atoms · kimimaro archives ·
      │  CAVE/lineage skeleton dicts · Fragment / SegmentFragment
      ▼
tube_mesh()                  (tube.py)      swept-tube triangle mesh, joints capped with spheres
      │
      ▼
export_bundle()               (bundle.py)   mesh one segment id per result, write:
      │                                       mesh/, skeleton/  — Neuroglancer precomputed
      │                                       groups/           — merged mesh per assembly group
      │                                       export/all.obj|.ply — other tools
      │                                       segments.json     — manifest (stats, colors, groups)
      │                                       state.json, url.txt, index.html
      ▼
serve_forever() / serve_in_thread()   (serve.py)   CORS HTTP server for the bundle
```

### Why a tube sweep, not marching cubes

The harness deliberately does not keep voxels (see
`docs/grammar_harness_handoff.md`) — it keeps the real L2 chunk-graph adjacency
plus each chunk's distance-transform caliber (`mean_dt_nm` / `max_dt_nm`).
That graph-plus-radius *is* the geometry on hand, so meshing sweeps a tube
along it: one ring of vertices per skeleton vertex, oriented by a
parallel-transported frame (no twist on a bending neurite), joined by quads
along every unbranched run, with a subdivided-octahedron sphere at every tip
and branch point so runs meet without cracks.

### Grouping = assembly, without re-meshing

Each result keeps its own mesh under its own segment id. An assembly's
grouping (which ids are "the same neuron") is carried separately as
Neuroglancer `equivalences` in `state.json` — selecting one member selects
and colors the whole group — so trying a different assembly means writing a
new grouping, not re-meshing. `groups/` additionally writes one physically
merged mesh + skeleton per group, for tools that have no notion of
equivalences.

### Formats

| Target | Format | Notes |
|---|---|---|
| Neuroglancer | `neuroglancer_legacy_mesh` + `neuroglancer_skeletons` precomputed | nanometres, identity transform; mesh bytes are byte-identical to cloud-volume's own encoder (`tests/test_meshing.py`) |
| Neuroglancer segment list | `neuroglancer_segment_properties` | searchable label, sortable numeric columns (cable length, face count, synapse count, …), filter tags |
| Blender / MeshLab | `.obj` | one `o` group per segment; micrometres by default (`--obj-scale`) |
| three.js / ParaView / generic | `.ply` | binary little-endian, optional per-vertex RGB by group |

## Adapters (`neuronauts/meshing/sources.py`)

| Source | Function |
|---|---|
| v117 atom, real L2 adjacency + pooled caliber | `HarnessAtomGeometry` / `load_harness_atoms` |
| kimimaro box skeleton archive | `kimimaro_archive_skeletons` |
| CAVE skeleton dict (`fetch.load_skeleton`), any `{vertices/edges/radii}` dict | `SkeletonGeometry.from_dict` |
| `neuronauts.schemas.Fragment`, `global_merge.schemas.SegmentFragment` | `SkeletonGeometry.from_fragment` |
| `GlobalAssemblyResult.fragment_to_neuron` | `groups_from_fragment_to_neuron` |
| `NeuronHypothesis` list | `groups_from_neuron_hypotheses` |

The harness adapter reads directly from `neuronauts.harness.geometry.
AtomGeometryStore` (per-atom L2 node ids + real `lvl2_graph` adjacency) and
`neuronauts.harness.topology.L2Attributes` (pooled `pos_nm` / caliber). An
atom whose attribute coverage is incomplete — the attribute fetch can trail
the topology fetch — is dropped by default (`--min-coverage 0.99`) rather than
drawn with missing nodes at the origin; `mesh_results.py harness` reports what
was dropped and why.

## Triangle budget

Meshing is per-vertex (roughly `2 * sides` faces per skeleton edge, plus one
sphere per tip/branch), so it scales with the input, not with a fixed grid.
The 8 most-connected v117 atoms in the 100 µm harness region (up to ~41k L2
nodes each) produce about 2.5M triangles at the default `sides=6`; that is
fine for a handful of atoms but not for meshing the full ~280k-atom
population in one bundle — use `--top-n` / `--min-synapses` / `--atom-ids` to
pick a subset, or raise `--min-radius-nm` and lower `--sides` for a coarser,
faster preview.

## Testing

`tests/test_meshing.py` checks: skeleton invariants (edge canonicalisation,
NaN dropping), chain decomposition (every edge partitioned exactly once,
including forks and pure cycles), mesh correctness (in-range faces, outward
winding via positive signed volume, radius floor/cap), format round-trips
through this project's own decoder, and — the strongest check available
without a GPU — that cloud-volume's own encoder produces byte-identical mesh
bytes and that cloud-volume's decoder reads our skeleton bytes back correctly
(skipped automatically if `cloud-volume` is not installed).
