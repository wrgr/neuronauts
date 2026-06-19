# NeuronautS data cache (git-lfs)

Reproducible, **provenance-tagged** caches of expensive CAVE fetches. Tracked
with git-lfs (`cache/**/*.npz`) so the fetch cost is paid **once** and shared
across runs and machines.

## Layout

| Dir | Contents | Produced by |
|---|---|---|
| `l2_skeleton/` | One `<sha1>.npz` per v117 fragment skeleton | `lineage.l2_skeleton()` |
| `synapse/` | One `<sha1>.npz` per region synapse fetch | `lineage.fetch_region_synapses()` |

Each dir has a `PROVENANCE.json` manifest describing the global constants
(datastack, table, algorithm). Each `.npz` additionally embeds a per-file
provenance blob under the `__provenance__` key — stripped automatically on load,
so callers only ever see the data arrays.

## Provenance fields

**L2 skeletons** — datastack, `l2_table`, `root_id`, `max_l2_nodes`, `seed`,
algorithm (`rep_coord_nm → kNN(k=6) → Kruskal MST`), `code_version` (git commit),
`fetched_at`.

**Synapse fetches** — datastack, `synapse_table`, `materialization_version`,
`side`, `limit`, `bbox_nm`, `code_version`, `fetched_at`.

## Auditing a cache file

```python
from neuronauts.data.lineage import read_cache_provenance
print(read_cache_provenance("cache/l2_skeleton/<sha1>.npz"))
# {'cache_kind': 'l2_skeleton', 'datastack': 'minnie65_public',
#  'root_id': 864691..., 'max_l2_nodes': 2000, 'seed': 0,
#  'code_version': 'b69a077', 'fetched_at': '2026-06-19T...'}
```

## Using the cache

Point the library at this directory (defaults are `/tmp/...`, which is
ephemeral):

```bash
export NEURONAUTS_L2_CACHE_DIR=$PWD/cache/l2_skeleton
export NEURONAUTS_SYNAPSE_CACHE_DIR=$PWD/cache/synapse
```

A cache miss fetches from CAVE and writes a new entry (with provenance); a hit
returns instantly with no network call. Set the var to `""`/`"0"`/`"off"` to
disable caching entirely.

## Validity

Cache keys encode every field that changes the result (datastack, table,
version, bbox, root_id, algorithm params). A different CAVE version or a
different bbox produces a different key, so entries never collide. If the
skeletonization algorithm changes, bump the params or clear the dir — the
`code_version` in each blob lets you detect stale entries.
