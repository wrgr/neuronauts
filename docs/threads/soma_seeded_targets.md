# Soma-seeded targets: the task the grammar actually performs

*Opened 2026-09-02. Records a framing decision made with the user, the seed
census that follows from it, and the first re-cut of the truth set. Numbers
marked (partial) are from the 8 seeded cells whose level-2 graphs were cached at
the time of writing; the full 103-seed re-cut is a pending update.*

## The decision

Two tasks hide in "assembly", and every proposer in this repo through EXP-073
measured the first:

1. **Find all the joins in the box** — pairwise. Any fragment–fragment contact a
   true cell makes inside the volume is a target. Scored with distance, caliber,
   elongation, attachment angle. Collapsed at every radius, dust floor and read
   resolution tried (precision ≈ 0.09%, EXP-072 and its probes).
2. **Recover a particular root process** — seeded. Start at a neuronal cell
   body and grow a tree outward; the target is that cell's in-box connected
   component. Scored as how much of that process was recovered and how much of
   something else was pulled in.

A probabilistic grammar over morphology is generative over *trees*, so it fits
the second task: a join is judged by whether the growing tree stays well-formed,
which is a local decision with global context — you need the subtree you are
standing in (axon or basal dendrite, depth, caliber trend), not the whole cell.
That is how the global problem becomes local. EXP-073's constraints were
grammar-flavored but applied pairwise with no tree state, which is why they
pruned true links as fast as false ones.

Two consequences, both deliberate:

- **The box is a boundary the grammar is allowed to stop at.** The gold cell's
  axon leaves through +y, runs 90–223 µm outside and re-enters; inside the box
  those pieces have no connecting path. The rule "this is an axon; axons are long
  and leave" ends the process at the face. Matching what left to what re-entered
  is a grammar at the next scale — over cells — and a later problem. Not a
  bigger box: this cell needs a 1,285 µm cube.
- **Glia, vasculature and debris are out of scope by construction.** Nothing is
  a target unless a neuron's growth could reach it. That is a cleaner exclusion
  than any filter.

Implementation: `neuronauts/harness/box_truth.py` — `box_components` groups a
cell's fragments by in-box connectivity; `seeded_target(bt, seed)` returns the
seed's own component (empty when the seed is not a labelled in-box fragment, or
has nothing to grow into). `all_components` and `largest` remain for the
pairwise framing.

## Seed census, 100 µm harness cube

`nucleus_detection_v0`, spatial filter in the table's own 4 × 4 × 40 nm voxel
frame, every returned position re-checked inside the cube in nanometres (this
repo has three voxel frames; the count is validated, not trusted).

| | count |
|---|---:|
| Nuclei in the cube | **332** |
| …classified neuron (`nucleus_ref_neuron_svm`) | 256 |
| …not neuron: astrocyte 19, oligodendrocyte 13, microglia 8, OPC 6, pericyte 5, other 30 | **76** (23%) |
| Fine types (`aibs_metamodel_celltypes_v661`): 4P 116, 23P 112, BC 11, MC 6, BPC 4, 5P-IT 1, NGC 1 | |
| Soma supervoxel resolves to a v117 fragment | 332 |
| …fragment in the 279,075-atom population | 324 |
| **Evaluable seed** (fragment pure, owner proofread) | **103** |
| …owner is this nucleus's own v1822 root | 103 (all) |
| …excitatory / inhibitory | 86 / 17 |
| …non-neuron | **0** |

So 23% of cell bodies in the cube are non-neuronal and never become targets
under seeded growth, and every evaluable seed is a neuron, because proofread-
owner status is only granted to neurons.

A first attempt at the cell-type join returned nothing from every table: the
`cell_type_reference` tables are keyed on `target_id` (the nucleus id), not
`pt_root_id`, and 5,000 ids in one filter is also too many. Chunks of 200 on
`target_id` work. Recorded because the first reading was "the tables errored",
and the cause was the call.

## Re-cut of the truth, all 103 seeded cells

| | count |
|---|---:|
| Seeded cells with a level-2 graph and positions for every node | 103 (0 skipped) |
| Labelled fragments across them | 634 |
| …in the seed's own in-box component (the seeded target) | **299 (47%)** |
| …in the largest in-box component | 406 (64%) |
| Cells where the seed's component *is* the largest | **58 of 103** |

So in 45 of 103 cells the soma's piece is not the biggest piece of that cell in
the box — the larger piece is a re-entering arbor with no in-box path to the
soma. Under the seeded framing that larger piece is a different root process;
under `largest` it would have been the target and the soma a miss.

**A bug caught on the way, worth recording.** The first full re-cut reported the
seed's component largest in only 6 of 103 cells, with nearly every fragment its
own component. That was not a finding: the coordinate cache only covered the
connective nodes of the original 40 cells, so the 95 newly fetched graphs had
unpositioned nodes, those nodes fell out of "inside", and — because two v117
atoms never share a level-2 edge — every fragment isolated. `seeded_recut.py`
now refuses to score a cell with under 95% of its nodes positioned, and
`scripts/fetch_cell_l2_positions.py` fetches coordinates for every cached
graph first. The 8-cell table below predates the bug and stands.

### The first 8 (cached before the full fetch)

| cell | fragments | in-box components | seeded target | largest | seed is largest |
|---|---:|---:|---:|---:|---|
| …774581835 | 15 | 5 | 10 | 10 | yes |
| …197172556 | 13 | 1 | 13 | 13 | yes |
| …364362466 | 12 | 3 | 9 | 9 | yes |
| …968529870 | 12 | 4 | 6 | 6 | yes |
| …043283030 | 11 | 2 | 10 | 10 | yes |
| …011850926 (gold) | 10 | 5 | 3 | 3 | yes |
| …616533451 | 11 | 3 | **0** | 7 | — |
| …976202524 | 11 | 4 | **0** | 5 | — |

Over these 8: 95 fragments, 51 (54%) in the seeded target, 63 (66%) in the
largest component; the seed's component is the largest in 6 of 8.

**The two empty targets are not a bug.** Those soma fragments are 2,368 and
3,035 level-2 nodes (372 and 506 synapses) — the soma fragment already holds the
cell's in-box arbor, and the other labelled fragments are re-entries with no
in-box path back. For those cells the root process is already whole inside the
box; there is nothing to grow. That is a genuine category, and a grower should
score it as "complete, abstain" rather than as a miss.

## Pending

- Level-2 graphs for the remaining ~95 evaluable seed cells are being cached
  under `data/external/cell_l2_graphs/`; re-running
  `data/external/soma_viz/seeded_recut.py` then gives the full 103-seed truth.
- The seeded framing needs its own experiment (EXP-074): a soma-seeded grower
  scored against `seeded_target`, with "complete, abstain" as a recorded
  outcome. Not yet written.
