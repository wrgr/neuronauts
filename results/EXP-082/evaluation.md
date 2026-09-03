# EXP-082 — Human proofreading history as a direct signal

**One line:** the edit log is a real, large-scale, held-out-transferable prior on
*where* to act (area under the curve 0.779 across held-out cells, versus 0.63 for
local pairwise geometry at the frontier), and it says two things that invalidate
the current candidate-generation design: **30% of human joins connect two v117
objects that do not touch near the join**, and **human proofreading order carries
no spatial locality at all** — it is statistically indistinguishable from doing
the same merges in random order.

Corpus: 103 gold-proofread cells, `data/external/edit_history/*.json`, joined to
the same cells' final skeletons (`data/external/cell_skeletons/*_skv4.npz`).
Substrate reads are CloudVolume `agglomerate=True, timestamp=V117_TS`, mip 2.

---

## The corpus

| | |
|---|---:|
| cells with a full change log | 103 |
| logged operations | 61,502 |
| merges / splits | 29,021 / 32,481 |
| merges **after** v117 (i.e. not already in our substrate) | **28,012** (96.5%) |
| distinct proofreaders on those merges | 65 (largest single share 10%) |
| merge endpoints landing within 2 µm of the cell's final skeleton | 55,223 / 58,042 (95.1%) |
| merges per cell | median 225 (range 5–895) |

Every operation carries two or more coordinates in nanometers, so this is a
*located* corpus, not just a lineage table.

---

## Q1. Where do humans actually edit? — **Yes, strongly predictable**

Denominator is real cable, not vertices: 1,282 mm of skeleton across the 103
cells. Rate = merges per mm of cable.

### By compartment

| compartment | cable | merges | per mm | enrichment |
|---|---:|---:|---:|---:|
| axon | 954 mm (74%) | 52,444 (95.0%) | 54.9 | **1.28×** |
| dendrite | 323 mm (25%) | 2,777 (5.0%) | 8.6 | 0.20× |
| soma | 4.7 mm | 2 | 0.4 | 0.01× |

Axon-versus-dendrite alone is a **6.4× rate difference**. Splits follow the same
shape (475/mm axon versus 117/mm dendrite).

### By caliber — the dominant variable

Within axon only, so this is not the compartment effect in disguise:

| skeleton radius | cable | merges | per mm | enrichment |
|---|---:|---:|---:|---:|
| < 110 nm | 35.6 mm | 5,704 | **160.3** | 2.92× |
| 110–125 | 200.8 mm | 20,265 | 100.9 | 1.84× |
| 125–140 | 230.8 mm | 16,914 | 73.3 | 1.33× |
| 140–160 | 134.2 mm | 4,994 | 37.2 | 0.68× |
| 160–200 | 172.3 mm | 2,853 | 16.6 | 0.30× |
| 200–300 | 163.2 mm | 1,605 | 9.8 | 0.18× |
| > 300 nm | 17.6 mm | 109 | 6.2 | 0.11× |

**26× top-to-bottom within axon.** Thin cable is where the segmentation breaks
and where humans spend their effort.

### By distance from the soma — real but secondary

Euclidean distance from the cell's own soma, all compartments: 6.7 merges/mm
inside 25 µm, rising to 64.0/mm at 200–400 µm, then falling to 49.2/mm beyond
400 µm. A 9.5× range, but most of the near-soma suppression is the thick-caliber
effect again.

### The actual predictive test

Unit = one skeleton vertex (mean spacing 1.97 µm, so ≈ one 2 µm segment of
cable). Label = a post-join merge endpoint landed here. Features: radius,
is-axon, path distance from soma, Euclidean distance from soma, degree, and the
three volume coordinates. **Gradient boosting, 5-fold grouped by cell, so every
score is on a cell the model never saw.**

- 650,200 vertices, 30,973 positive, **base rate 4.76%**
- **held-out-cell area under the curve 0.779**
- precision at top-30,973 = 0.184 → **3.86× lift**
- top 2% of all cable: precision 0.207, **4.35× lift**, captures 8.7% of merges
- top 10% of all cable: precision 0.157, 3.29× lift, captures **32.9%** of merges
- top 30% of all cable: captures **70.3%** of all merges

Ablations (area under the curve): drop radius → 0.722; drop anything else →
0.775–0.779. **Radius alone gives 0.750**; radius + is-axon 0.752.

Compare: local pairwise geometry at the grower's frontier reaches 0.63 and 0%
precision in the top 34 (EXP-081). This is a different and better regime, and it
is a property of *tissue* (caliber), not of the reconstruction, so it should read
off v117 fragments directly. **That transfer is not verified here** — the radius
used is from the final proofread skeleton — and verifying it on v117 fragments is
the first follow-on.

---

## Q4. Skip connections — **yes, and this is the headline**

For 400 sampled post-v117 merges, read v117 at mip 2 in a box padded 1 µm around
the two clicked points and ask what a candidate generator would have seen. The
uniform sub-sample (n = 266) is the unbiased estimate; a second stratum
oversamples the >1 µm click-span tail.

| | uniform (n=266) | >1 µm span stratum (n=134) |
|---|---:|---:|
| both clicks in the **same** v117 object | 7.1% | 0.7% |
| two **different** v117 objects | 91.0% | 98.5% |
| …of those, 6-adjacent within ±1 µm | 70.2% | 42.4% |
| …**not adjacent — a skip** | **29.8%** | **57.6%** |

Non-adjacency rises monotonically with how far apart the two clicks were:

| click span | n | not adjacent |
|---|---:|---:|
| 0–0.3 µm | 53 | 13.2% |
| 0.3–0.6 | 102 | 28.4% |
| 0.6–1.0 | 53 | 30.2% |
| 1.0–2.0 | 141 | **57.4%** |
| > 2.0 µm | 25 | **60.0%** |

Click-span distribution over all 29,021 merges: median 0.46 µm, p90 1.14 µm,
p99 2.67 µm, max 12.6 µm; 13.4% exceed 1 µm.

**Roughly 30% of human joins cannot be produced by any candidate generator that
proposes pairs from local contact.** The human saw a continuation across a gap
that the segmentation does not bridge, and acted on it.

### The same reads also kill adjacency as a *filter*

For the joined pairs, inside that same tiny ±1 µm box:

- objects 6-adjacent to one of the joined fragments: **median 23, p90 55, max 197**
- distinct v117 objects present in the box at all: median 168, p90 278

Adjacency in a 2 µm box already yields ~23 candidates. Over a whole fragment it
is far larger. Adjacency is neither necessary (30% of true joins lack it) nor
sufficient (23+ decoys per micron).

RECHECK_PLACEHOLDER

---

## Q2. What did humans do that geometry would not? — **they did not grow a frontier**

The frontier-grower framing assumes assembly proceeds outward from claimed
tissue. Test: take each cell's post-v117 merges in true timestamp order; for each
one, measure the path distance **along the final skeleton** from the new merge
site to the nearest site already touched (the soma counts as touched at t = 0).
Compare against the identical set of merges applied in shuffled order.

95 cells, 27,095 merge operations:

| distance to nearest already-touched site | real time order | shuffled null |
|---|---:|---:|
| median | 11.93 µm | 12.06 µm |
| p75 | 43.85 µm | 47.96 µm |
| > 5 µm | 64.9% | 65.0% |
| > 10 µm | **53.4%** | 53.6% |
| > 50 µm | 23.0% | 24.4% |
| > 100 µm | 12.6% | 14.1% |

**The observed order is indistinguishable from random.** Over half of human
merges are made more than 10 µm of cable away from anything that proofreader's
cell had touched before, and knowing the true order buys essentially nothing over
knowing the set.

Humans do not walk a frontier. They recognize a fragment as belonging to a cell
from its shape and context, wherever it is, and attach it; connectivity to the
soma is an emergent consequence, not the search procedure. That is the grammar
argument stated as a measurement.

The second half of Q2 — building the merge tree to ask whether humans assemble
subassemblies before attaching them — **was attempted and is not established.**
The before/after root-id chain in the cached change logs terminates 4 levels back
from the final root; only 847 of 28,012 post-v117 merges are reachable that way,
so any statistic over them is unrepresentative. I have not diagnosed why the
lineage is disconnected in this table, and I am not reporting a number from it.

---

## Q3. Is it usable as training signal at scale? — **yes**

| | |
|---|---:|
| post-v117 merge operations (usable positives) | **28,012** |
| merge endpoints | 56,024 |
| cells | 102 |
| distinct proofreaders | 65 |
| bounding volume | 0.48 mm³ (1,176 × 789 × 520 µm) |
| distinct 25 µm boxes containing ≥1 merge | **5,951** |
| distinct 50 µm boxes | 1,718 |
| endpoints inside the existing c100um harness cube | 2,410 (4.3%) |
| post-v117 merge endpoints per cell | median 446 (8–1,788) |

The distribution is broad, not a few hot spots: 5,951 occupied 25 µm boxes at a
mean of 9.4 merges each. Enough to fit on, enough to hold out by cell (which is
what the 0.779 above already does), and 2,410 endpoints sit inside the substrate
already built.

Splits are a second, larger corpus (32,481 operations, same coordinate quality)
that this experiment only touched in passing — a where-to-*cut* prior is directly
available from the same files.

---

## Which of these is a usable signal

| finding | usable? |
|---|---|
| **Where-to-edit prior (Q1)** | **Yes, now.** 0.779 held-out-cell, 3.9× lift, dominated by caliber. Needs one verification: recompute radius from v117 fragments instead of the proofread skeleton. |
| **Skip connections (Q4)** | **Yes, as a design constraint, not a model.** ~30% of true joins are non-adjacent at the seam; adjacency yields 23+ decoys per micron. Candidate generation must not be contact-gated. |
| **Non-local proofreading order (Q2)** | **Yes, as a design constraint.** Frontier growth from the soma is not the procedure that produced this ground truth. |
| **Scale (Q3)** | **Yes.** 28,012 located post-v117 merges, 102 cells, 65 proofreaders, 0.48 mm³. |
| **Merge-tree / subassembly structure** | **No.** Lineage chain is broken in the cached logs; not established, not reported. |

## Limits I am not papering over

- The 103 cells are gold-proofread and therefore atypical: they got human
  attention because they were worth it. The prior learned here is "where
  proofreaders acted on cells someone chose to finish," not "where the
  segmentation is wrong."
- Skeleton radius, compartment, and path distance all come from the *final*
  reconstruction. The Q1 model is measured, but its deployment on v117 fragments
  is assumed, not shown.
- "Not adjacent within ±1 µm of the clicks" is not "not adjacent anywhere." The
  recheck below bounds this; a pair could still touch far from the seam, which
  would not help a seam-local detector but would matter for a global one.
- 5.0% of sampled merges have both clicks inside the same v117 object — a
  post-v117 split followed by a re-merge, or a click landing on already-joined
  tissue. These are excluded from the adjacency rates.
- Growth order uses the *final* skeleton as the metric space; tissue merged and
  later split away is not represented.

## Files

- `results/EXP-082/{build_join,model,growth_order,probe_v117,recheck}.py` — the runs
- `results/EXP-082/v117_merge_probe.json` — 400 substrate probes
- `results/EXP-082/v117_recheck_4um.json` — the widened-box recheck
- `results/EXP-082/growth_order.npz` — observed and null distance arrays
- `data/external/edit_join_v082.npz` — 566,005 operation endpoints joined to skeletons
