# The abstention measurement, fourth attempt — the first valid one

## Result: local geometry carries a weak stop signal, AUC 0.642 (95% CI 0.54–0.75, and that interval is optimistic). Not anti-correlated, not chance, not sufficient.

Three previous attempts at this exact number each produced a confident answer
that dissolved on inspection. All three were defects in where the box sat or
which voxels reached the features, never in the method under test.

| attempt | reported | what was actually wrong |
|---|---|---|
| EXP-075 | 0.304, "anti-correlated" | whole-cell boxes on the soma, then at the cube face, then mid-cable; features on eroded objects |
| EXP-076 | 0.476, "chance" | same clipped terminals; classes separable by distality alone at AUC 0.054 |
| this run, unaudited | 0.652 | distality still separating at 0.858 after loose matching |
| **this run, audited** | **0.642 ± 0.104** | — |

## What was fixed

- **Identity.** Features now come from objects read with `agglomerate=True`
  rather than labelled through a mip-5 supervoxel map that knew 21.5% of the
  voxels present. Panels carry 2,000–2,750 candidates where the eroded build saw
  1,000–1,500.
- **Terminals.** Selected as genuine cable ends. Verified: **21 of 22 are clean
  tips** once branching is excluded — the earlier "17 of 22 still continue" was
  my audit counting seed cloud beyond a *plane*, which flags every point on a
  branch that extends further. The one real failure (40071996) is dropped.
- **Distality.** Terminals sit at 75–110 µm from the soma, cuts at 25–80 µm, and
  the classes separate on that alone at AUC 0.931. Matched with a 5 µm caliper
  inside the 71–106 µm overlap band, residual distality falls to 0.556.

## The number

Best-in-panel score (`along × collin × proximity`), cut sites against genuine
arbor terminals, 81 matched pairs from 21 terminals and 66 cuts:

**AUC 0.642, 95% CI [0.54, 0.75].**

The lower bound clears 0.5, so the signal is real. But the interval is computed
over matched *pairs*, which are not independent — the same 21 terminals recur
across pairs, so the effective sample is 21, not 81, and an interval respecting
that clustering would be materially wider. Treat 0.64 as "weak positive,
imprecise", not as a calibrated figure.

## What this means for the grammar

A stop rule built on local geometry alone would be right about 64% of the time
on a balanced two-class comparison. A soma-seeded grower makes many sequential
decisions, and 36 of 103 cells require it to decline every single time. That
error rate compounds far too quickly to carry the task.

So the conclusion that survives is close to the one the failed runs pointed at,
but for a defensible reason and with a different shape: local evidence is not
*silent* on when to stop — it is *weak*. Abstention needs evidence a single
contact panel does not contain, most plausibly tree context: whether joining
yields a morphologically valid arbor, which is a property of the tree rather
than of the pair.

## Limits

21 terminals is few, and they are the cells that both lack an in-box join and
possess an interior cable end — not a random sample of neurons. The comparison
is one decision per panel, not a grower's trajectory. And the score here is the
geometric combination EXP-077 validated at median per-panel AUC 0.919 for
*ranking* the true partner; ranking within a panel and deciding whether a panel
contains a partner at all are different questions, and this experiment only
answers the second.
