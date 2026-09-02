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

### Cell type: which labels are these?

The census first carried only model predictions — `nucleus_ref_neuron_svm` (a
support vector machine on nucleus volume, foldedness and cortical position) and
`aibs_metamodel_celltypes_v661` (a soma and nucleus metamodel). Checked against
the hand-typed tables that cover part of this volume —
`allen_v1_column_types_slanted_ref` (neurons; Casey Schneider-Mizell, Nuno da
Costa, Agnes Bodor) and `aibs_column_nonneuronal_ref` (non-neuronal; JoAnn
Buchanan):

| | |
|---|---:|
| Nuclei in the cube with a human label | 114 of 332 |
| Nuclei with both a human and a model fine label | 107 |
| **Exact agreement** | **99 (92.5%)** |
| Evaluable seeds with a human label | 88 of 103 (81 agree) |
| Human-labelled non-neuronal cells among the seeds | **0** |

Every one of the eight disagreements is 23P versus 4P — layer 2/3 against layer
4 pyramidal, a laminar boundary call — plus one "Unsure E". There is no
neuron/non-neuron disagreement. So the coarse class the challenge typing rests
on is safe, and anything resting on the pyramidal subtype is not.

The census and every card now carry `cell_type_final`, `cell_type_source`
("human" or "model") and both underlying labels, so a card cannot present a
prediction as a curated fact. Final seed types: 4P 44, 23P 41, BC 8, MC 5,
BPC 3, NGC 1, Unsure E 1 — 88 human-sourced, 15 model-sourced.

A first attempt at the cell-type join returned nothing from every table: the
`cell_type_reference` tables are keyed on `target_id` (the nucleus id), not
`pt_root_id`, and 5,000 ids in one filter is also too many. Chunks of 200 on
`target_id` work. Recorded because the first reading was "the tables errored",
and the cause was the call.

### What of this survives at v117 — the version the method actually runs on

Everything above is measured with v1822 proofread data in hand. A deployed
method has only v117, so it matters which inputs are version dependent. They
split three ways, and the split was checked rather than assumed:

| input | depends on the cell segmentation? |
|---|---|
| `nucleus_detection_v0` | No — the nucleus segmentation is separate |
| `nucleus_ref_neuron_svm` | No — features are nucleus volume, foldedness, position |
| `aibs_metamodel_celltypes_v661` (fine type) | **Yes** — "soma and nucleus features", soma from the cell segmentation at v661 |
| `evaluable` filter (pure + proofread owner) | **Yes** — v1822 ground truth, evaluation only |

So seeding is available at v117:

| | count |
|---|---:|
| Nuclei detected in the cube | 332 |
| Nucleus supervoxel resolves to a v117 object | **332 (100%)** |
| …object is in the 279,075-atom population | 324 (98%) |
| Support vector machine calls it a neuron | 256 |
| **v117-only seed set, no proofread data used** | **254** |
| Scoreable (pure + proofread owner at v1822) | 103 |

**Decision (with the user, 2026-09-02): assume the support vector machine labels
for now, and document the dependency.** The coarse neuron call is the only cell
type input the method uses, it is version independent, and it agrees with hand
labels wherever both exist (no neuron/non-neuron disagreement in 107 overlaps).

Two limitations follow, both to be repeated wherever a number from the 103 is
quoted:

1. **We can score 103 of the 254 seeds a deployed method would start from —
   41%.** The other 151 lack a proofread owner. They are not a random sample:
   they are the cells nobody chose to proofread, so any performance measured on
   the 103 carries that selection bias. Their predicted types are mostly
   pyramidal (72 layer 2/3, 67 layer 4).
2. **The support vector machine is the weak link, at about 2–3%.** Six of the
   254 v117 seeds are called glia by the fine model (3 astrocyte, 2 pericyte,
   1 oligodendrocyte) — contamination a v117 method inherits and cannot detect.
   One nucleus the machine rejects is called 23P by the fine model, so it errs
   in both directions. Of the 70 non-neuronal nuclei a v117 method must reject
   on the machine alone, the rejections are otherwise consistent with the fine
   model (microglia 8, astrocyte 16, oligodendrocyte 11, pericyte 7, OPC 6).

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

## What the 103 environments look like

All 103 cards built (`scripts/build_cell_cards.py`), each with the cell's box
structure, its split links with gap and compartment, its shared frankenmerged
objects, and its full proofreading change log. Aggregated by
`scripts/aggregate_cell_cards.py`.

### A third of the seeds have nothing to join

| challenge type | cells |
|---|---:|
| axon splits present | 40 |
| **already whole** | **36** |
| dendrite-only splits | 21 |
| frankenmerge-heavy | 3 |
| other | 3 |

For 36 of 103 cells the soma's fragment already holds the cell's entire in-box
arbor. They are the smaller-in-box cells (median 1,478 level-2 nodes against
1,811; 4 fragments against 6) — the cell simply does not extend far into the
cube beyond its soma piece. **"Complete, abstain" is therefore a third of the
task**, and a grower that cannot recognize it will invent joins for a third of
its seeds. It has to be a scored outcome, not an edge case.

### Compartment sets the scale, and one category is not what it looks like

232 split links across the 103 targets:

| link | count | gap median | p90 | max |
|---|---:|---:|---:|---:|
| axon–axon | 129 | 1,314 nm | 5,282 | 29,749 |
| dendrite–dendrite | 71 | **826 nm** | 1,846 | 4,319 |
| axon–dendrite | 32 | **7,574 nm** | 21,280 | 26,136 |

Dendrite continuations are tight — half under 826 nm, ninety percent under
1.85 µm. Axon continuations are about 1.6× wider. **This is the grammar's
case in one table:** what gap to expect is a function of which compartment you
are in, and no single distance threshold expresses that.

The third row is different in kind. Compartment-crossing links are **6.3× wider**
than same-compartment ones (7,574 nm against 1,202 nm) and make up 13.8% of the
target's links. A real continuation does not change compartment — an axon does
not become a dendrite — so these are almost certainly the minimum spanning tree
bridging two sub-arbors at their nearest approach rather than tracing a join
that exists. **They are probably not targets at all**, and a grower scored
against them is being asked to make a join no proofreader made. Excluding them
moves the median gap only from 1,274 to 1,202 nm, so the cost of dropping them
is small and the honesty gain is large. Recommended for EXP-074; not yet done,
because it changes the truth set and should be a recorded decision rather than a
quiet filter.

### Reach

| gap | share of the 232 links |
|---|---:|
| ≤ 0.5 µm | 19.0% |
| ≤ 1 µm | 34.9% |
| ≤ 2 µm | **70.7%** |
| ≤ 5 µm | 84.5% |
| ≤ 10 µm | 93.1% |

### Proofreading effort does not predict in-box difficulty

Edits per cell are substantial — median 514, p90 1,179, max 1,940, with a median
171 edit points falling inside the cube. But they are uncorrelated with local
difficulty: edits against number of split links r = +0.08, against shared
frankenmerged objects r = −0.00, against in-box size r = −0.10. Edit count
describes the whole cell, most of which is outside the cube, so it cannot serve
as a difficulty proxy for a box-local method. Useful to know before anyone
weights training data by it.

### By cell class

| class | n | already whole | links (median) | shared merged objects | edits (median) |
|---|---:|---:|---:|---:|---:|
| 4P | 44 | 19 | 1 | 0 | 744 |
| 23P | 41 | 12 | 2 | 0 | 340 |
| BC | 8 | 1 | 2 | 0 | 234 |
| MC | 5 | 1 | 1 | 1 | 244 |
| BPC | 3 | 2 | 0 | 0 | 61 |
| NGC | 1 | 0 | 6 | 6 | 275 |

Layer 4 pyramidal cells are most often already whole (19 of 44); interneurons
almost never are (2 of 14). The single neurogliaform cell is the hardest
environment in the cube — 6 split links and 6 shared frankenmerged objects, two
of them large (7,440 and 2,439 level-2 nodes, each shared with three other
cells) — which is what a dense space-filling axon plexus does to a segmentation.

### Representatives

| role | cell | class | fragments | components | target | links | shared merged | edits |
|---|---|---|---:|---:|---:|---:|---:|---:|
| already whole | …788476 | 23P | 4 | 3 | 0 | 0 | 0 | 647 |
| dendrite-only splits | …650282 | 4P | 5 | 2 | 4 | 3 | 0 | 791 |
| axon splits + frankenmerge | …445786 | BC | 21 | 5 | 15 | 14 | 4 | 1,175 |
| compartment-crossing | …581835 | 4P | 15 | 5 | 10 | 9 | 0 | 1,940 |
| dense plexus | …954248 | NGC | 12 | 6 | 7 | 6 | 6 | 275 |
