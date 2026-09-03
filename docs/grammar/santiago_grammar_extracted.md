# The SANTIAGO morphological grammar, extracted

Pulled out of `attic/morpho_grammar/santiago_v2_grammar.py` so the grammar
itself survives without the code around it. The code was scored on skeletons cut
in software and, like 25 of the 26 engines in that directory, has no
checkpoint-loading path -- it was evaluated at initialization. **The production
rules below are the asset; its reported numbers are not.**

Why keep it: the current program has just measured that pairwise local questions
cannot carry the task. At a grower's frontier -- median 46 cut ends per cell,
one of which continues, a 1.6% base rate -- local geometry reaches AUC 0.63 and
**0% precision at the operating point** (EXP-081, results/EXP-081). A grammar
scores whole shapes and whole trees, which is the level this evidence says the
work has to move to.

## The production rules

```python
self.rules = {
            "<Volume>": [
                ("<Neuron>", 0.85),
                ("<Glia>", 0.12),
                ("<BloodVessel>", 0.03)
            ],
            "<Neuron>": [
                ("<PyramidalNeuron>", 0.75),
                ("<Interneuron>", 0.25)
            ],
            "<PyramidalNeuron>": [
                ("<Soma> <ApicalTree> <BasalTree> <AxonArbor>", 1.0)
            ],
            "<Interneuron>": [
                ("<Soma> <AspinyDendriteTree> <DenseAxonPlexus>", 1.0)
            ],
            "<Glia>": [
                ("<AstrocyteStar>", 0.60),
                ("<OligodendrocyteSheath>", 0.30),
                ("<MicrogliaProcess>", 0.10)
            ],
            "<ApicalTree>": [
                ("<ApicalTrunk> <ApicalFork>", 0.80),
                ("<ApicalTrunk> <ApicalTuft>", 0.20)
            ],
            "<ApicalFork>": [
                ("(<ApicalTree>, <ApicalTree>)", 0.85),
                ("<ApicalTuft>", 0.15)
            ],
            "<BasalTree>": [
                ("<BasalTrunk> <BasalFork>", 0.90),
                ("<BasalTerminal>", 0.10)
            ],
            "<AxonArbor>": [
                ("<AxonTrunk> <AxonCollateral>", 0.70),
                ("<AxonTerminal>", 0.30)
            ],
            "<Dendrite>": [
                ("<DendriteShaft> <PostSynapsePool>", 1.0)
            ],
            "<Axon>": [
                ("<AxonShaft> <PreSynapseBoutonPool>", 1.0)
            ]
        }
```

## Hard constraints the grammar asserts

- 1. Glial Exclusion Barrier
- 3. Axon-Dendrite Chimera Veto (Dendrite cannot merge into Axon, Axon cannot merge into Dendrite)
- 1. Zero-Synapse Glial Barrier

## What the module claimed for itself

```
SANTIAGO-v2 Complete Morphological Grammar, Half-Synapse Polarity, Hard Veto & Forensic Error Analyzer (EXP-040).
Extends SANTIAGO with:
  1. Immutable Biological Hard Polarity Veto: Prohibits Axon-Dendrite and Glia-Neuron merges (P = 0).
  2. Glial non-terminals and Zero-Synapse Exclusion Barrier.
  3. Half-Synapse Pre/Post Polarity from synapse table.
  4. Unsupervised Cell-Type Induction from observable morphology.
  5. Forensic Error Analyzer: Detailed root cause breakdown across distance and angles.
100% blind at inference without ground truth.
```

## What is worth carrying forward

1. **The polarity veto is a hard constraint, and it is independently confirmed.**
   Axon-dendrite and glia-neuron joins are prohibited outright. Our own
   measurement agrees that polarity is near-binary: of objects carrying any
   synapse, 175,715 are purely presynaptic and 97,797 purely postsynaptic, with
   only 5,563 mixed. The caveat we measured and this grammar does not state:
   only 279,075 of 910,888 objects carry a synapse at all, so the veto applies
   to about 31% of candidates. It can refuse a join; it cannot propose one.

2. **Glia as an explicit non-terminal.** 76 of 332 nuclei in our cube (23%) are
   non-neuronal. A grammar that can label something glial and stop is doing real
   work, not losing coverage.

3. **Cell-type-conditioned structure.** Separate expansions for pyramidal cells
   and interneurons. We have types for 88 of 103 seeds (hand-labelled) with the
   remainder predicted, so this is conditionable today.

4. **What our data contradicts.** A term this grammar leans on -- that
   polarity *agreement* indicates continuation -- is wrong at the soma. Every
   soma seed is postsynaptic-dominant while 20 of 39 true partners are purely
   presynaptic: the cell's own axon. The soma-to-axon transition is a REQUIRED
   polarity flip, and a grammar must expect it at that production rather than
   penalise it.

