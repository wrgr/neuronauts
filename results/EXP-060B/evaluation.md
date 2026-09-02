# EXP-060B — the recall/panel-size trade-off, correctly measured

## Result: passed (the curve is internally consistent) — and it reinstates most of what EXP-060 originally found

`results/EXP-060/CORRECTION.md` fixed two real errors (wrong denominator, L2/
endpoint units instead of object units) and predicted that a properly-reduced
panel would reach **~65% recall of the spanning links assembly needs**. That
number is confirmed exactly — **and it arrives at a panel size that cannot be
deployed**, which the correction did not check before writing "withdrawn."

## Tier ≥10 (20,826 atoms, 1,297 labelled, 350 MST links)

| panel size (objects) | recall at 2 µm | recall at 5 µm |
|---:|---:|---:|
| 20 | 12.0% | 12.0% |
| 50 | 15.7% | 15.7% |
| 100 | 22.9% | 22.9% |
| 300 | 34.6% | 38.6% |
| **uncapped** (median 1,467 / 3,870) | 42.9% | **64.6%** |

The 65% prediction is right at the uncapped limit. At a panel size a scorer
could actually use (≤20–100 candidates), recall is **12–23%** — close to
EXP-060's original, differently-computed 17.5%/24.6%, not the improvement the
correction implied.

## Tier ≥1 (238,966 atoms, 2,723 labelled, 1,899 MST links)

| panel size | recall at 2 µm | recall at 5 µm |
|---:|---:|---:|
| 20 | 14.3% | 14.7% |
| 50 | 16.3% | 20.2% |
| 100 | 16.4% | 23.0% |
| uncapped (median 32 / 293) | **16.4%** | **26.6%** |

Two things stand out. First, **recall saturates**: at 2 µm the uncapped median
panel is only 32 objects — most labelled atoms simply have no more than ~32
other atoms within 2 µm at all, so raising the cap further buys nothing.
Second, and more consequential: **the recall ceiling is lower at tier ≥1 than
at tier ≥10** (26.6% vs 64.6% uncapped at 5 µm), even though including small
fragments was the change your intuition said should bring partners closer.

## Why tier ≥1 is worse despite "closer" partners — a hypothesis, not yet confirmed

Two effects point in opposite directions and the second dominates:

- Nearest-partner distance genuinely tightens (median 1.3 µm at tier ≥10 →
  1.8 µm at tier ≥1 was the *all-tiers* figure in CORRECTION.md, which pools
  fragmentation effects; not a clean comparison to redo here).
- **The atoms that make tier ≥1 bigger are exactly the atoms with the fewest
  endpoints to be found by.** Tier ≥10 topology averages 245 endpoints per
  atom; tier ≥1 averages **5**. An atom with 5 endpoints exposes far less
  surface to an endpoint-based proximity search than one with 245, independent
  of how physically close its true partner is. The 5.4× jump in atoms needing
  linking (350 → 1,899 MST links) is mostly these small, sparsely-sampled
  atoms, and they may be structurally harder for *any* endpoint-based method
  to find — not only proximity.

This is stated as a hypothesis because it has not been isolated: distinguishing
"tier ≥1 partners are geometrically harder to find" from "tier ≥1 partners are
under-sampled by endpoint count" needs a comparison controlled for endpoint
count, which this experiment did not run.

## What this means for the program

**EXP-060/CORRECTION.md's "geometry cannot propose candidates" was withdrawn
too readily.** The denominator and units fixes were real and necessary — the
original 17.5%/6.5 µm numbers were wrong as stated — but once measured
correctly, geometric proximity still cannot deliver both usable recall and a
usable panel size at either tier. The trade-off is the finding, and it now
matches EXP-061's cone result in shape exactly: enrichment/recall is available,
but only by accepting a panel far too large to score.

Restated bottom line: **geometry (ball or cone, either tier) gives partial
credit — real but insufficient for a deployable proposer on its own.** That
still argues for embedding-based retrieval (EXP-057C) as the channel with no
radius and no panel-size trade-off, and it now also argues for combining
geometry with the free biological constraints CORRECTION.md's "what to run"
listed (polarity, one-soma, caliber continuity) to cut the panel size at fixed
recall, since geometry alone plainly cannot.

## Reproduce

```bash
uv run python -m neuronauts.experiments.exp060b_object_panel
```

Runtime ~2.5 minutes for both tiers, both radii, the full cap sweep — cheap
enough that this curve should be the default report, not a single number, for
every future candidate-generation measurement on this substrate.

---

## Correction, 2026-09-02 — "tier ≥1" above was mislabeled; here is the true comparison

The tier comparison in this file originally used `data/substrate/topology/k1.npz`
under the name "tier ≥1 (complete substrate)". **It was not that.** The tiered
fetch's own design has each tier "skip atoms already done" (`neuronauts/harness/geometry.py`),
so `shards/k1_*.npz` holds only the atoms *newly added* at the widen-to-≥1 step
— exactly the 1–4 synapse slice, since ≥10 and ≥5 already covered everything
above that. `k1.npz` was 238,966 atoms of 1–4 synapses each, not the 279,075-atom
union.

This was caught by a direct question about candidate synapse counts: filtering
candidates in that file to `n_synapses >= 5` returned **zero** candidates,
which is only possible if every atom present already had 1–4 synapses. Fixed
by adding `--tier all` to `scripts/build_atom_topology.py` (globs every shard),
rebuilding the true union (`kall.npz`, 279,075 atoms, verified against the
population count), and rerunning this experiment against it.

**Corrected tier comparison** (labelled/MST counts also correct now — the true
population has far more labelled atoms and links than the mislabeled slice
suggested):

| | tier ≥10 | **true full population** | *(mislabeled "tier≥1" — wrong)* |
|---|---:|---:|---:|
| atoms | 20,826 | 279,075 | ~~238,966~~ |
| labelled | 1,297 | **4,511** | ~~2,723~~ |
| MST links | 350 | **3,260** | ~~1,899~~ |
| uncapped recall @5µm | 64.6% | **47.4%** | ~~26.6%~~ |
| uncapped median panel @5µm | 3,870 | **1,172** | ~~293~~ |

The qualitative finding survives correction — recall on the full population is
still lower than the tier ≥10 slice, and the trade-off curve still holds — but
every number attributed to "tier ≥1" before this correction was wrong, and the
gap was smaller than claimed (47.4% vs 64.6%, not 26.6% vs 64.6%).

## The synapse-floor question, answered on the correct file

Prompted directly: given the population is entirely synapse-anchored, is
proximity's failure actually a symptom of the panel being flooded with
near-zero-synapse "junk" objects that a cheap floor would filter out?

Composition of the uncapped candidate crowd on the true full population, 5 µm:
**45.0% of nearby candidates have only 1–4 synapses**, 12.9% have 5–9, 42.1%
have ≥10. So the intuition that low-synapse objects dominate the local
neighbourhood is right, once measured on the correct file (at tier ≥10 the
figure is a tautological 0%, since that population excludes them by
construction — which is exactly what made the mislabeled file look deceptively
clean).

But filtering them out makes recall **worse**, not better, at every panel size
tested, capped or not:

| candidate floor | recall @ cap 20 | recall uncapped | uncapped median panel |
|---|---:|---:|---:|
| none | 10.5% | 47.4% | 1,172 |
| ≥5 synapses | 8.1% | 32.1% | 731 |
| ≥10 synapses | 6.7% | 26.6% | 605 |
| ≥30 synapses | 5.5% | 22.1% | 436 |

Raising the floor shrinks the panel, as expected — but recall falls faster
than the panel does, at every setting. The reason: a meaningful share of the
**true spanning partners themselves are low-synapse fragments** — a real
neuron's distal twig, with 1–4 synapses in this region, is exactly the kind of
object a floor removes along with the junk. An absolute synapse-count floor
cannot distinguish "irrelevant small fragment" from "real partner that happens
to be small," because both live in the same range.

**This refutes the specific mechanism** (filter by synapse count) while
confirming the premise that motivated it (the neighbourhood really is full of
low-synapse objects). The fix has to discriminate by *identity or connectivity
signature*, not by a raw count — which is what embedding retrieval (EXP-057C)
and the still-untested biological constraints (polarity, one-soma, caliber
continuity — same-owner fragments should agree with each other, not merely
clear a size threshold) are for.
