# EXP-057 — GT overlay and spatial split

## Result: failed, at 16.2% against a 30% bar

The 100 µm harness cube does not carry enough proofread label to support the
experiment program as scoped.

| Quantity | Synapse sides | Share of region |
|---|---:|---:|
| Total in region | 1,802,996 | 100% |
| **On single-lineage atoms owned by a proofread cell** | **291,931** | **16.2%** |
| …of which gold-owned | 111,028 | 6.2% |
| On mixed-lineage atoms (any lineage) | 194,966 | 10.8% |
| On mixed-lineage atoms whose roots are proofread | 4,559 | 0.25% |
| Unlabelled | 0 | 0% |

Every atom received a label row, so the shortfall is not a coverage bug in the
overlay: 83.8% of the region's synapse mass genuinely sits on objects no human
has verified.

## The seam-positive count is the harder problem

An atom is a usable seam positive only when it is mixed-lineage **and** the
roots it spans are proofread. There are **56** in the whole cube, and the
spatial split leaves **15 in train**:

| | train | val | buffer |
|---|---:|---:|---:|
| All atoms | 112,279 | 113,168 | 53,628 |
| Pure, proofread-owned | 1,157 | 2,747 | 898 |
| **Seam positives** | **15** | **22** | **19** |

This repo's own seam GNN was **net-negative at 150 objects** and first cleared
zero at **513**. Fifteen training positives is an order of magnitude below the
point where that model was merely useless. EXP-062 and EXP-063 cannot be run
as scoped.

A second, quieter problem: the split is balanced on atom *count* (112k vs
113k) but not on *label* — val holds 2.4× the labelled atoms of train, because
proofread cells are not uniformly distributed along the split axis. Any learned
scorer fitted on this split trains on the thinner half. A label-balanced split,
or a different axis, should be chosen before EXP-064.

## The criterion's prescribed remedy is wrong

The bar said "else widen the tier before proceeding". That remedy does not
apply, and saying so is more useful than following it:

- The **tier** (≥10 / ≥5 / ≥1 synapses) governs which atoms have *L2 geometry*
  fetched. The population and the label overlay already cover all 279,075
  atoms at ≥1 synapse.
- Label coverage is a property of **how much of this tissue has been
  proofread**, which no amount of geometry fetching changes.

So widening the tier adds arbors, not labels. The real options are:

1. **Accept 16.2% and scope every claim to the labelled subset**, reporting
   the unlabelled 83.8% as the known limit of what any number here describes.
   Cheapest and honest, but it leaves the seam experiments unrunnable.
2. **Import external labels — EXP-057B.** ConnectomeBench2 offers 716,485
   expert proofreading decisions. Its MICrONS split is the only realistic route
   to a seam-positive count in the hundreds. This experiment's failure is what
   promotes 057B from "worth a look" to the critical path.
3. **Move or enlarge the region.** This cube was already selected as the
   densest gold-proofread cube in the v1822 manifest, so a straight move is
   unlikely to help much; enlarging to 200 µm multiplies mass but not
   necessarily proofread *fraction*, and should be measured before it is
   assumed.

Option 2 is the recommendation, with option 1 applied to whatever runs before
it lands.

## What is safe to build on now

The 4,802 proofread-owned single-lineage atoms and their 291,931 synapse sides
are real, unambiguous, and enough for the **candidate-generation** work
(EXP-060/061), which needs same-cell continuation pairs rather than certified
false merges. That path is unaffected by this failure and should proceed.

```bash
uv run python -m neuronauts.experiments.exp057_gt_overlay
```
