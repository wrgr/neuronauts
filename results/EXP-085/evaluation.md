# EXP-085 — The grammar labels terminals at scale: 281,790 candidate split sites, no proofreading required

## Result: a neurite ends for one of three reasons, and the grammar names two of them without any human label. Of 8,183 sampled cut surfaces: 25.8% end at the object's own synapse (legitimate stop), 30.6% sit at the field boundary (not a decision), 43.6% are unexplained — cable that stops in the middle of tissue with no synapse and no edge to blame. Extrapolated: ~281,790 candidate split sites in this cube.

## Why this matters for the program

I had reported that the corpus holds 58 negative decision sites and cannot
certify a precision target. That was true of *proofread cells* and false of the
actual population. **Every cut surface in the segmentation is a site the
grammar can classify without a human ever having looked at it:**

- **Synaptic terminal.** A bouton or a spine head is where an axon or dendrite
  legitimately ends. The grammar says stop, and it needs no ground truth to say
  so — only that a synapse belongs to that object.
- **Field boundary.** The tip is truncated by our 100 µm box, not by biology.
  Not a decision either way.
- **Unexplained.** Neither. This is what a split looks like: real cable, ended
  for no reason a legitimate terminal would have.

The unexplained category is the negative population this program needs, at a
scale nothing else in the repository approaches.

## The measurement, and a bug caught before it shipped

First pass required only that a synapse lie within 1.5 µm of a tip, without
checking that the synapse belonged to the object. With 901,498 synapses in a
100 µm cube, mean spacing is about 1.04 µm, so that test passes almost
everywhere by chance — it returned 99.96% "explained," which is the artifact,
not a finding. Fixed to require the synapse be on the tip's own object.

| category | count (of 8,183 sampled) | share |
|---|---:|---:|
| synaptic terminal — grammar says STOP | 2,115 | 25.8% |
| field boundary — not a decision | 2,502 | 30.6% |
| **unexplained — candidate split** | **3,566** | **43.6%** |

Extrapolated over the 237,064 objects in this cube with enough cable to have an
end: **~167,130 labelled stop sites, ~281,790 candidate split sites.**

## What this changes

The precision target (EXP-081: false-positive rate must sit below roughly 2% at
a grower's real 1.6% base rate) can now be measured and trained against directly.
The unexplained population is the negative class; the synaptic-terminal
population is a second, distinct negative class (a legitimate stop that must
not be extended); live sites from `box_truth.seeded_target` remain the positive
class.

This is also where the user's framing lands precisely: production-rule grammar
supplies the terminal classification and the structural priors (EXP-084's
Murray's-law conservation, EXP-061's re-measured 44.7× directional enrichment
against the real candidate distribution) at population scale, with no electron
microscopy and no per-decision proofreading. Electron microscopy stays a
reserve instrument for whatever the grammar cannot resolve, not a default.

## Limits

Tip-finding runs on mip-5 centroid clouds, adequate at micron scale for locating
an ending but not for the fine geometry of what it ends at. The 1.5 µm synapse
radius is generous and untuned. "Unexplained" is not yet verified as "true
split" — some fraction may be truncation by our own dust floor or another
artifact of this pipeline rather than a real segmentation error, and that
distinction is the next thing to check before training on this population.
