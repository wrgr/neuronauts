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
