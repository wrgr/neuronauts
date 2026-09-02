# Real merge topology vs. the MST proxy — a fixed bug, a real finding, and an open question

> Prompted directly: an MST over atom centroids has no notion of which
> *specific* endpoint continues into which other endpoint, and no notion of
> spine-vs-shaft attachment geometry — it just asks "can these be connected by
> *some* short chain," which is not the same graph as the neuron's real
> branching topology. The real answer isn't a geometric proxy at all: v117
> fragments were literally merged into the proofread cell by specific, logged
> CAVE operations. This thread checks whether we can use that log directly.

## A dead module, fixed

`neuronauts/edit_history.py::fetch_edit_log` has returned an empty list for
every call, silently, for an unknown period. Verified directly: it calls
`client.chunkedgraph.get_tabular_changelog(root_id)`, and current caveclient
has no such method — the real one is `get_tabular_change_log` (second
underscore) — wrapped in a bare `except Exception: pass` that swallowed the
resulting `AttributeError` and returned `[]`, which reads exactly like "this
cell has no edit history," not "this code is broken." Confirmed by bypassing
the module and calling caveclient directly: **1,039 real operations** for one
gold-proofread cell (owner root `864691134920077322`), each with an operation
id, a millisecond timestamp, before/after root ids, an `is_merge` flag, and
the proofreader's name (Forrest Collman, Casey Schneider-Mizell, and others —
real MICrONS proofreaders). Fixed in `neuronauts/data/lineage.py` — sorry,
`neuronauts/edit_history.py` — and reverified: `fetch_edit_log` now returns
the same 1,039 operations (378 merge, 661 split), matching the raw call
exactly.

This has value independent of where the rest of this thread goes: it is the
real, recorded topology this project has needed since at least the
`error_correction` thread was scoped, and it was never working.

## The bridge-object question, tested once, inconclusively

For that same cell: 461 "leaf" root ids appear in its merge history (a
`before_root_ids` entry that never itself appears as an `after_root_ids` —
i.e., the earliest state on that lineage branch, within this operation log).
Resolved to v117 (`root_at_version`, 460/461 succeeded) and checked against
`data/substrate/c100um/population.npz` (every v117 object with ≥1 synapse in
our region): **only 64 of 460 (14%) match one of our known atoms.** The
remaining 396 are not in our synapse-anchored population at all.

That is suggestive of exactly what was asked — real connective tissue that our
population construction cannot see because it carries no local synapses — but
**it is not confirmed**, because "absent from population.npz" has two very
different causes that were not distinguished:

1. **Genuinely inside our 100 µm box with zero local synapses** — a true
   invisible bridge, invisible specifically because our region/population
   construction is entirely synapse-table driven (`neuronauts/harness/population.py`
   enumerates objects by which supervoxels a synapse touches; an object with no
   synapse anywhere in the box is never fetched, at any tier, regardless of how
   much real cable it contributes).
2. **Physically elsewhere in this cell's much larger extent.** A proofread
   MICrONS neuron can span the ~1.3 mm column; this cell's edit history covers
   its *entire* reconstruction, not just the fragment inside our harness cube.
   Most of the 396 could simply be real objects from a different part of the
   same cell, nowhere near our region, which would say nothing about local
   bridges.

Telling these apart needs each absent leaf's actual coordinate, checked
against our box. Two attempts at that check both timed out (10 min, then
5 min) with no output, and the reason is now understood, not mysterious: some
merge-tree leaves resolve to v117-era roots that had already absorbed an
enormous amount of tissue before any proofreading — one sampled root had
**689,734 supervoxels** — and `root_leaves(root_id, stop_layer=1)` had no
bounds parameter, so it always fetches every leaf of the object, however
large. `root_at_version` compounds this: it calls `root_leaves` unbounded
just to take the first element, so building a list via `root_at_version` over
461 arbitrary leaves can stall on a single huge one with no visible progress.

Fixed the specific gap, additively: `root_leaves` now takes an optional
`bounds` parameter (`neuronauts/harness/substrate.py::region_bounds` produces
the string), so a caller who only needs "does this object touch my region"
gets a cheap, spatially-restricted answer instead of enumerating the whole
object. Existing callers (`root_at_version`'s two call sites, both on small
nucleus-seed roots) are unaffected — the parameter defaults to `None`.

## What is not yet done

The actual inside/outside spatial check, using the now-fixed bounded fetch, on
a properly time-boxed run (a background agent with an explicit per-item
timeout and progress logging, not a bare loop in a foreground shell — the two
hangs above cost real time for zero result and should not be repeated the same
way). Concretely, for a sample of the 396 absent leaves:

```python
r117 = root_at_version(leaf, 117)
svs_in_box = root_leaves(r117, stop_layer=1, bounds=bstr)  # bstr from region_bounds
inside = svs_in_box is not None and len(svs_in_box) > 0
```

If a meaningful share come back `inside`, that confirms the bridge-object
hypothesis directly and reframes candidate generation: the missing partners
found in EXP-060B are not simply far away, some fraction are literally
invisible to any synapse-anchored substrate, and finding them needs a
different, non-synapse-gated fetch (e.g., a spatial sweep of the segmentation
directly, independent of the synapse table) rather than a better search over
the population we already have.

## What this changes about EXP-058/060/060B/061's ground truth

Independent of the bridge question: the MST-over-centroids denominator used
throughout today's candidate-generation experiments is a proxy, and the
critique that motivated this thread stands regardless of how the bridge
question resolves. The real ground truth — which specific v117 objects were
merged with which, in what order, confirmed by a human proofreader — is now
fetchable via the fixed `fetch_edit_log`. Replacing the MST denominator with
edit-history-derived pairwise ground truth is the natural next major revision
to the EXP-058/060/060B/061 series, not a footnote to it. Not attempted here;
flagged as the priority follow-on.
