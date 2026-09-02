# EXP-074 specification — soma-seeded growth, the first experiment on the real task

*Written 2026-09-02 from the 103-cell environment census. Every bar below is
derived from a measured quantity in that census and is stated here before the
experiment is written, so the run can fail.*

## The question

Can a grower seeded at a cell body recover that cell's in-box root process — and
does it know when to stop?

This is the first experiment scored on the task a grammar performs
(`docs/threads/soma_seeded_targets.md`). Everything from EXP-060 through
EXP-073 scored pairwise join-finding, which collapsed at ~0.09% precision on
every substrate, radius, dust floor and read resolution tried.

## Substrate and truth

| | |
|---|---|
| Seeds | the 103 evaluable soma seeds (`data/external/soma_viz/seed_census.json`) |
| Geometry | object clouds, `data/substrate/c100um/object_clouds_mip5.npz` |
| Objects | `objects_v117_mip5.npz`, dust floor 0.041 µm³ on synapse-free objects |
| Target | `box_truth.seeded_target` — the seed's own in-box component |
| Links | the target's spanning links, compartment-crossing dropped |

**Two populations, scored separately, never pooled.**

- **67 cells that need joining** — 299 target fragments, median 3 per cell;
  200 scored links, gap median 1,202 nm, p90 3,417 nm.
- **36 cells already whole** — the soma fragment holds the entire in-box arbor.
  The correct output is *nothing added*.

## The baseline method

Deliberately simple, and deliberately not a grammar. Start from the soma
fragment; repeatedly take the nearest unclaimed object whose closest approach to
the current tree is under radius *r*; stop when none qualifies. No compartment,
no caliber, no tangent, nothing learned. Sweep *r* over 500 nm, 1 µm, 2 µm,
3 µm.

Its purpose is to establish what distance alone achieves on the seeded task, so
that a grammar's contribution is measurable against it rather than asserted. If
the baseline already clears the bars, the grammar has nothing to add here and
we should say so.

## Bars, and where each number comes from

Declared before the run.

**1. Recovery, on the 67 cells that need joining.** At *r* = 2 µm, recover at
least **60%** of target fragments, micro-averaged over cells.

*Derivation:* 80% of scored links are within 2 µm, and on 40 of the 67 cells
every scored link is. 80% is therefore the ceiling for any 2 µm distance-only
grower; 60% allows a quarter of the reachable links to be lost to ordering and
contention and still counts as working. Below 60% distance-only growth is not a
useful base.

**2. Purity, same cells.** At least **80%** of the objects a grower adds must
belong to the seeded target.

*Derivation:* this is the clause the pairwise line always failed — EXP-072
reached recall by collapsing, at 0.09% precision. 80% is where a proofreader
would still be corrected rather than misled. There is no measured precedent to
derive it from, which is stated rather than hidden: it is a judgement, and the
first run's value is the thing that calibrates it.

**3. Abstention, on the 36 already-whole cells.** Add nothing in at least
**70%** of them.

*Derivation:* a third of the seeds have nothing to join, so a grower that always
grows is wrong a third of the time. 70% is set below the recovery bar because
abstention is untested — no prior experiment measured it — and a first
measurement should be able to fail informatively rather than trivially.

**Reported, not gated:** recovery and purity per cell class (44 layer 4
pyramidal, 41 layer 2/3 pyramidal, 14 interneurons, 1 neurogliaform); recovery
against gap; how often the grower crosses into another cell's frankenmerged
object; and the radius sweep, so the trade is visible rather than a single
point.

## What would make this fail honestly

- Recovery under 60% at every radius → distance-only growth is not a base, and
  the grammar has to supply the join criterion, not merely refine it.
- Purity under 80% while recovery clears → the same collapse as the pairwise
  line, arriving by a different route. The seeded framing would then not be the
  fix, and that is worth knowing before a grammar is built on it.
- Abstention under 70% → the "complete" case needs an explicit stop rule, not
  an emergent one.

## Limitations to restate in the result

- **We score 103 of the 254 seeds a v117 method would start from (41%)**, and
  the unscoreable 151 are the cells nobody chose to proofread — a selection
  bias, not a random sample.
- **The neuron call is a support vector machine prediction** with roughly 2–3%
  contamination at v117 (six of the 254 seeds are called glia by the fine
  model).
- **Connectivity is decided on the proofread graph**, which a method does not
  have. The target says what is reachable in principle inside this box, not what
  a proposer had the information to find.
- **The box is a boundary.** 14.7% of nearest-sibling paths leave the cube, and
  those fragments are in separate components by construction.

## Not in scope

The grammar itself. This experiment establishes the task, the truth and the
distance-only base. A grammar-scored grower is EXP-075 and should be measured
against these numbers.
