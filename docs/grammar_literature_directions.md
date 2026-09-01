# Grammars of trees, elsewhere: literature and directions

> **Status: literature survey and direction ranking (2026-09-01).** Companion
> to `docs/pcfg_global_assembly_report.md`, which makes the case for a parsed,
> typed tree grammar as verifier and seam locator on the harness substrate.
> This document asks what the wider grammar, structured-inference, and
> transformer literature says about that plan, and which directions deserve a
> slot in the experimental program. Numbers quoted from papers are as reported
> by their authors; for several I read abstracts and summaries rather than the
> full text, and that is marked where it matters.

## TL;DR: ranked directions

| Rank | Direction | Why it ranks here | Feeds |
|---|---|---|---|
| **1** | **Hidden Markov tree / PCFG verifier and seam locator on `AtomTopo`** (companion report) | The exact formalism exists and is mature (Crouse 1998; Durand et al. 2005 on plant architecture; Bacciu et al. 2013 tree transductions). Low parameter count attacks the measured data-starvation wall. Cheapest thing that could move seam top-1 above 25%. | E1–E5 of the companion report |
| **2** | **A labelled corpus we did not have: ConnectomeBench2** (716,485 expert proofreading decisions across FlyWire, MICrONS, Fish1, H01) | The seam GNN crossed zero only at 513 objects. If the MICrONS split maps onto our substrate, it is orders of magnitude more merge/split decisions, and the benchmark's third task ("mask segmentation for merge error correction") is our seam-location problem by another name. | training for 3 and 4; a second evaluation set for E1/E2 |
| **3** | **Neural emissions inside the grammar** (torch-struct-style differentiable inside algorithm; compound-PCFG per-object latent) | Keeps the tree structure and the hard zeros, replaces hand-fitted emissions with a small learned encoder trained end to end on the seam objective. The compound latent is the principled fix for context dependence (cell class, layer). | E2 rerun with learned emissions |
| **4** | **Tree-structured transformer over contracted skeletons** (Transformer-Grammars masks, tree positional encodings, GraphDINO's AC-attention) | The strongest general-purpose learner for tree data, and GraphDINO already runs on MICrONS at 30,000-neuron scale. Only worth it once direction 2 supplies labels; otherwise it re-hits the wall. | seam benefit regression, object-level merge flag |
| **5** | **Existing MICrONS morphology embeddings** (Weis et al. 2025, GraphDINO over >30,000 excitatory neurons) as tree-DNA and class conditioning | Free if the per-root embeddings are released. Directly the "pooled DNA of large objects" the assembly design wants, and the class variable for the grammar mixture. | verifier battery, level-k stitch scorer |
| **6** | **Vision-language model as a verification channel** (ConnectomeBench protocol: three orthographic mesh renders) | Cheap to try, human-adjacent on multiple-choice merge identification (74% vs humans 74–80%), and it fails independently of geometry, lineage, and EM texture, which is what the independence conjunction needs. Not an assembler. | decoy panel / battery channel |
| **7** | **LLM as grammar-rule and feature inducer** (DreamCoder / ShapeLib pattern) | Our outer-loop idea (`program.md`) applied to a small, interpretable search space: propose productions and emission features, score by held-out log-likelihood and seam top-1, keep or revert. Needs direction 1 as the executor first. | grammar v1 |
| **8** | **Point-affinity clustering as atomiser** (Troidl et al. point affinity transformers) | Proven on FlyWire and MICrONS for merge correction via agglomerative clustering, but trained on simulated errors, the trap we already documented. Retrain on real v117→v1822 errors or skip. | E3 alternative |
| **9** | **Generative tree transformers for completion** (Latent L-systems; autoregressive hourglass trees) | Impressive tree generators, but our findings say completion is not the bottleneck and generation does not know identity. Keep as a plausibility-likelihood candidate only. | none now |
| **10** | **Graph-grammar form discovery** (Kemp & Tenenbaum; hyperedge replacement grammars) | Useful as an object-typing feature (tree vs bundle vs sheet), not as an assembler. | GLIA / bundle rejection feature |

---

## 1. The literature, by family, and what each says about our plan

### 1.1 Stochastic branching models of neurons (the emissions and transitions are already known)

Forty years of quantitative neuroanatomy has produced low-parameter stochastic
branching models that reproduce dendritic and axonal statistics:

- **BESTL** (Van Pelt & Uylings): branching probability as a function of a
  terminal segment's centrifugal order and the current number of terminals.
  This is exactly a typed branching process with `P(k | X, depth)`.
- **L-Neuron** (Ascoli & Krichmar 2000): stochastic L-systems whose parameters
  are sampled from measured distributions; "recursive rules that parsimoniously
  describe dendritic geometry and topology by locally inter-correlating
  morphological parameters (e.g. branch diameter and length)". That local
  inter-correlation is our caliber and length emission.
- **Cuntz et al. 2010, "One rule to grow them all"**: a neuron's tree is a
  minimum spanning tree over its spanning field under one balancing factor
  between material cost and conduction time. A global, one-parameter prior on
  branch geometry that a grammar can carry as a fork emission.
- **Samsonovich & Ascoli 2005**: a hidden Markov model of hippocampal pyramidal
  dendrites; the earliest use of latent states along dendritic paths.
- **Kanari et al. 2022 (TNS)**: topological neuron synthesis from the
  topological morphology descriptor; generates whole cortical regions from a
  few reference cells. Persistence barcodes are a compact summary of branching
  that could serve as a per-object plausibility feature.
- **NETMORPH** (Koene et al.): network-scale stochastic growth on the same
  principles.

**Reading for us.** The v0 grammar's emissions and transition families are not
guesses; they are the parameterisations this literature settled on. Use BESTL's
centrifugal-order dependence in `P(k | X)`, Rall/Murray caliber relations at
forks, and TMD barcodes as a sanity check on parsed atoms. None of these models
was ever used as an *error detector*; that is the gap.

### 1.2 Hidden Markov tree models on observed topology (the formalism we actually want)

- **Crouse, Nowak & Baraniuk 1998** introduced hidden Markov trees for wavelet
  coefficients: observed tree, latent state per node, upward-downward
  (inside-outside) inference and EM.
- **Durand, Guédon, Caraglio & Costes 2005** (New Phytologist) adapted HMTs to
  plant architecture: apple trees and bush willows measured at annual-shoot
  scale, with hidden states revealing "homogeneous zones and transitions
  between zones within tree-structured data". Their topology-description
  framework (Godin et al. 1999, multiscale tree graphs) is the botanical
  analogue of `AtomTopo`.
- **Bacciu, Micheli & Sperduti 2013**: input-output bottom-up hidden tree Markov
  models for tree transductions, i.e. labelling every node of an observed tree
  given per-node inputs. That is precisely "parse = label the observed
  skeleton".

**Reading for us.** The companion report's "the tree is observed, not latent"
is the standard HMT setting, with mature EM, forward-backward on trees, and
Viterbi. Implement it as an HMT with typed branching; call it a PCFG only for
the productions view. The botanical work also shows the model's natural
product is *zones and ruptures* along the tree, which is our seam.

### 1.3 Grammars with hard constraints in biology (SCFGs for RNA)

- **Eddy & Durbin's covariance models**, implemented in **Infernal** and
  maintaining Rfam: profile stochastic context-free grammars that score
  sequence *and* base-pairing structure, with hard structural constraints and
  bit-score thresholds for abstention.

**Reading for us.** The one place SCFGs became infrastructure in biology is
where structure is conserved, constraints are hard, and emissions are learned
per family. The analogue of "one covariance model per RNA family" is one
grammar per cell class, with the class marginalised when unknown.

### 1.4 Global structured inference for neuron reconstruction

- **Türetken et al. 2016** (TPAMI): curvilinear networks as a graph of candidate
  paths scored by discriminative path classifiers, then an integer program
  selecting a subset "subject to structural and topological constraints". The
  ILP is exact where Kruskal is greedy.
- **Matejek et al. 2019** (CVPR, "Biologically-Constrained Graphs"):
  geometric constraints from neuron morphology prune candidate nodes and edges,
  two networks learn neuronal shapes, and region merging is reformulated as
  graph partitioning (lifted multicut); average variation-of-information
  improvement of 21.3% on four datasets.
- **ViterBrain** (Athey et al. 2022, Communications Biology): an HMM over
  fragments whose states are endpoint pairs with tangents and whose transitions
  are a Boltzmann distribution over gap distance and curvature; maximum-
  probability reconstruction by dynamic programming. Light microscopy, single
  unbranched paths, 31% success vs 11% for the nearest competitor.

**Reading for us.** Assembly-as-constrained-optimisation is established, and
the grammar term drops into it as another edge or path weight. ViterBrain's
transition model is our `g(o_join)` emission. Türetken's ILP is the exact
alternative to constrained Kruskal inside a tile when tiles are small; worth
one comparison once `Δ_attach` exists, because the ILP can enforce tree-ness
and one-soma globally rather than greedily.

### 1.5 Graph grammars for network structure

- **Kemp & Tenenbaum 2008** (PNAS): structural forms (tree, chain, ring, grid)
  as graph grammars; joint inference of form and structure. Their rules were
  hand-drawn vertex-replacement productions with learned probabilities.
- **Aguinaga, Chiang & Weninger** (TPAMI 2018): hyperedge replacement grammars
  extracted automatically from a graph's clique tree, generating graphs with
  the original's properties.

**Reading for us.** Communication-graph grammars answer "what kind of object is
this?", not "where is the seam?". As a typing feature (a neuron parses as a
tree; a glial sheet or an axon bundle does not) it is a cheap complement to
the `GLIA` branch and the spatial-compactness cannot-link. Not a direction on
its own.

### 1.6 Neural grammars and structured transformers

- **Compound PCFG** (Kim, Dyer & Rush 2019): rule probabilities modulated by a
  per-sentence continuous latent, "inducing marginal dependencies beyond the
  traditional context-free assumptions"; inference by collapsed variational
  inference with the trees marginalised by dynamic programming.
- **Transformer Grammars** (Sartran et al. 2022, TACL): recursive syntactic
  composition inside a Transformer via a special attention mask over a
  linearised tree; outperforms strong baselines on syntax-sensitive metrics.
- **Torch-Struct** (Rush 2020): batched, differentiable inside algorithms for
  CFGs, HMMs and trees as PyTorch distributions.
- **Tree positional encodings** (Shiv & Quirk, NeurIPS 2019) and **hierarchical
  accumulation** (Nguyen et al., ICLR 2020) for transformers over trees.

**Reading for us.** Three transferable mechanisms. (i) The compound latent is
the principled answer to the companion report's context-dependence caveat:
one per-object vector modulates rule probabilities by cell class and layer
without breaking the tree DP. (ii) Torch-Struct means the grammar can be a
differentiable layer, so a small encoder can learn emissions from L2 segment
features while the transitions stay interpretable and the hard zeros stay
hard; train it on the seam objective directly. (iii) Transformer-Grammar-style
masks and tree positional encodings let a transformer respect the skeleton
tree if and when the labelled corpus justifies one.

### 1.7 Tree generators in graphics

- **Latent L-systems** (Lee, Li & Benes, ACM TOG 2023): a Transformer generates
  L-system strings for 3D trees; trained on 155k trees, 93.7% agreement in
  branching angles and 97.2% in branch lengths with the input distribution.
- **Autoregressive generation of static and growing trees** (Wang et al.,
  SIGGRAPH Asia 2025): an hourglass multi-resolution transformer over tree
  tokens, with image-to-tree and point-cloud-to-tree conditioning and 4D growth.
- **MorphGrower** (ICML 2024 oral): neuron morphologies grown layer by layer,
  sibling-branch pairs conditioned on the ancestor path and a GNN summary of
  the existing tree; beats MorphVAE, whose 3D-walk sequence model produced
  mostly topologically invalid trees. MorphGrower provides no explicit
  likelihood; plausibility is judged by a separate real-vs-generated
  classifier.

**Reading for us.** Transformers learn tree distributions at scale from
branch-hierarchy tokenisations, and point-cloud-to-tree conditioning is
literally "infill the missing continuation". Two cautions: nobody has shown
these likelihoods detect errors, and MorphVAE's failure is the same
"linearise the tree and lose topology" failure our autoregressive synapse
grammar had. Low priority until a plausibility use is demonstrated.

### 1.8 Neuron representation learning on MICrONS

- **GraphDINO** (Weis et al.): self-supervised transformer for spatially
  embedded graphs with AC-attention (global attention plus graph convolution)
  and augmentations for neuron skeletons.
- **Weis et al. 2025** (Nature Communications 16:3361): GraphDINO embeddings for
  more than 30,000 excitatory neurons in MICrONS V1/AL/RL, finding a morphological
  continuum rather than discrete types except in layers 5 and 6.
- **MorphRep** (Bioinformatics 2024): graph-transformer pretraining on
  single-neuron morphologies at scale.

**Reading for us.** These are the pooled-DNA vectors the tree-assembly design
wants for large objects, on our dataset, already computed. If per-root
embeddings are released with the code, they are a free source of prototypes for
the `NAMED` tier and of the class variable for the grammar mixture. Check the
release before building another encoder.

### 1.9 Connectomics proofreading with learning

- **NEURD** (Celii et al., Nature 2025): mesh → skeleton graph with rich
  features; heuristic graph rules for merge-error correction and feature
  extraction. A hand-written grammar in effect. ConnectomeBench reports NEURD
  "detected zero merge errors in tested cohort", a reminder that rule systems
  are brittle off their tuning distribution.
- **SyConn** (Schubert et al. 2019): cellular morphology networks on multi-view
  projections; glia detection used to resolve reconstruction errors.
- **Point affinity transformers** (Troidl et al., Harvard/Janelia, bioRxiv
  2024–25): a multi-neuron point cloud is embedded into a fixed-length feature
  set from which any pairwise affinity is decoded; agglomerative clustering on
  the affinities corrects *simulated* reconstruction errors; trained on FlyWire,
  works on MICrONS; beats GNNs, point transformers and unsupervised clustering.
- **Autoproof** (Huang, Katz, Berg & Scheffer, Janelia, 2025): learns from
  manual proofreading annotations on the Drosophila male CNS; 90% of a guided
  workflow's value at 20% of its cost; 200,000 fragments merged automatically
  (about four proofreader-years).
- **ConnectomeBench** (Brown et al., MIT, 2025): LLMs judge segment identity,
  split correction and merge identification from three orthographic
  1024×1024 mesh renders. Best multiple-choice merge identification 74.0% on
  FlyWire and 70.3% on MICrONS (o4-mini with descriptions) against a human
  74–80%; binary merge 62.8/61.5%; split multiple-choice 78.8/85.0% against a
  human 84–90%.
- **ConnectomeBench2** (Brown, Farkas, Razgar & Boyden, 2026): 716,485
  expert-labelled proofreading decisions across FlyWire, MICrONS, Fish1 and
  H01; tasks are split correction, merge classification and mask segmentation
  for merge-error correction; a ViT-B reaches 97.0% balanced accuracy on splits
  and 93.0% on merges against human 93.0/84.1%. Stated limits: 2D renders are
  lossy, only ViTs evaluated, species imbalance, and calibration degrades out
  of distribution, most for merges.

**Reading for us.**

1. The field's best merge *identification* is now at or above human level on
   rendered views, which matches our finding that detection is the easy half.
   The open task is seam *location*, which ConnectomeBench2 poses as mask
   segmentation. Our `S(e)` on the skeleton is a candidate answer that does not
   need a render.
2. ConnectomeBench2 is the corpus the seam GNN never had. The first action is
   to check whether its MICrONS decisions can be mapped to our v117 roots and
   L2 substrate (segmentation version, root ids, coordinates) and how many fall
   inside the harness cube.
3. Point affinity transformers repeat the synthetic-error training we already
   found does not transfer; the honest way to use the architecture is to
   retrain it on real lineage-derived errors.
4. VLM judgement of renders is a legitimate independent verification channel
   at about human level on multiple-choice merge questions, and cheap to run on
   a candidate panel. It is not an assembler and should never be the only
   channel.

---

## 2. What the literature says about the traps we have already hit

| Our finding | Same lesson elsewhere |
|---|---|
| Synthetic splices train a detector that transfers at chance (BiGRU 0.82 → 0.48); SSL seam nets net-negative zero-shot | Point affinity transformers are trained on simulated errors; ConnectomeBench2 reports calibration degrading out of distribution, most for merges |
| A linearised sequence grammar over synapses scores geometry, not topology (0.63 merge, chance split) | MorphVAE's 3D-walk sequence model generates mostly topologically invalid trees; MorphGrower fixed it by generating tree-structured units conditioned on the tree |
| Hand rules for frankenmerge oddness flag 85% of real fragments | NEURD found zero merge errors in ConnectomeBench's cohort |
| Detection is easy, the cut operator is hard | ConnectomeBench2 separates merge classification (93%) from mask segmentation for correction, the latter as its own task |
| Data-starved seam GNN (150 objects → net-negative, 513 → +4.6%) | ConnectomeBench2's motivation: the earlier benchmark had hundreds of samples, the new one 716k, "large enough to both train and evaluate on" |
| Proximity is not identity for axons (0/32) | ViterBrain and Türetken both put tangent and curvature, not distance alone, into the transition or path score |

---

## 3. Additions to the experimental process

Concrete, in order, each with the check that decides whether it stays. Slots
refer to the series in `docs/consolidation_plan.md` §6.3, which already maps
the companion report's E0–E5 onto EXP-060/062/063/066 (§6.4 there). The
v1822 overlay that the grammar experiments need now exists
(`results/atom_labels_v1822.json`: 2,444 mixed-lineage atoms in the cube, 56
of them among proofread cells).

1. **ConnectomeBench2 intake probe** (half a day; register beside EXP-057 in
   series A). Download the MICrONS split;
   record segmentation version, id space, coordinate frame, and task format;
   count decisions inside the harness cube and how many map to v117 roots via
   lineage. Keep if ≥ 1,000 mapped merge or split decisions land in or near the
   cube; otherwise note the version gap and move on.
2. **Embedding intake probe** (half a day). Check whether Weis et al. 2025
   release per-root GraphDINO embeddings; if so, join to the gold manifest and
   test cosine separation of same-cell vs different-cell fragment pairs on the
   harness, the same test tree-DNA has to pass.
3. **HMT/PCFG parser and E0–E2** (companion report; EXP-062/063 in the plan).
   Unchanged, priority one.
4. **Neural-emission grammar** (2–3 days after 3; a second grammar scorer in
   the EXP-064 bake-off). Replace fitted emissions with
   a small per-segment encoder under a Torch-Struct-style differentiable inside
   pass; train on the seam objective using lineage labels plus any
   ConnectomeBench2 decisions from step 1. Bar: seam top-1 and net pair-error
   above the fitted grammar on the same folds; hard zeros unchanged.
5. **VLM verifier probe** (1 day; a channel in EXP-064, then a battery member
   in EXP-066/067). Render candidate unions from E5 with the
   ConnectomeBench protocol (three orthographic views) plus our parse overlay;
   ask for a multiple-choice verdict with decoys; score against lineage. Bar:
   beats chance by the ConnectomeBench margin (≥ 70% multiple-choice) and is
   uncorrelated with the geometry channel's errors, so it can enter the
   conjunction.
6. **Tree transformer** only if step 1 delivers labels at scale. Architecture:
   Transformer-Grammar masks or tree positional encodings over contracted
   segments, GraphDINO-style augmentation; target the per-edge cut benefit.
   Bar: beats step 4 on held-out seams; report the axon side separately.
7. **LLM rule induction** only after step 3 exists as an executor: the LLM
   proposes candidate productions or emission features; the harness scores
   held-out log-likelihood and E2 top-1; keep or revert. This is `program.md`'s
   outer loop with an interpretable search space.

Not scheduled: generative tree transformers for completion, graph-grammar form
discovery beyond a typing feature, and point-affinity clustering unless
retrained on real errors.

---

## Sources

- Durand, Guédon, Caraglio & Costes 2005, New Phytologist, hidden Markov tree models of plant architecture: https://nph.onlinelibrary.wiley.com/doi/10.1111/j.1469-8137.2005.01405.x
- Bacciu, Micheli & Sperduti 2013, input–output hidden Markov model for tree transductions: https://www.sciencedirect.com/science/article/abs/pii/S0925231213001914
- Ascoli & Krichmar, L-Neuron: https://krasnow1.gmu.edu/cn3/L-Neuron/HTM/paper.htm
- Van Pelt & Uylings, growth models for topological binary trees (BESTL family): https://link.springer.com/article/10.1007/BF02459919
- Cuntz et al. 2010, One rule to grow them all: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000877
- Samsonovich & Ascoli 2005, hidden Markov model of dendritic morphology: https://www.wikidata.org/wiki/Q42468211
- Kanari et al. 2022, computational synthesis of cortical dendritic morphologies: https://www.cell.com/cell-reports/fulltext/S2211-1247(22)00330-8
- Infernal / covariance models: http://eddylab.org/infernal/
- Türetken et al. 2016, curvilinear networks with path classifiers and integer programming: https://infoscience.epfl.ch/record/201670
- Matejek et al. 2019, biologically-constrained graphs: https://openaccess.thecvf.com/content_CVPR_2019/html/Matejek_Biologically-Constrained_Graphs_for_Global_Connectomics_Reconstruction_CVPR_2019_paper.html and https://github.com/Rhoana/biologicalgraphs
- Athey et al. 2022, ViterBrain: https://pmc.ncbi.nlm.nih.gov/articles/PMC9038756/
- Kemp & Tenenbaum 2008, the discovery of structural form: https://www.pnas.org/doi/10.1073/pnas.0802631105
- Aguinaga, Chiang & Weninger, learning hyperedge replacement grammars: https://arxiv.org/abs/1802.08068
- Kim, Dyer & Rush 2019, compound PCFGs: https://aclanthology.org/P19-1228/
- Sartran et al. 2022, Transformer Grammars: https://aclanthology.org/2022.tacl-1.81/
- Rush 2020, Torch-Struct: https://arxiv.org/abs/2002.00876
- Shiv & Quirk 2019, tree positional encodings: https://www.microsoft.com/en-us/research/wp-content/uploads/2019/10/shiv_quirk_neurips_2019.pdf
- Nguyen et al. 2020, tree-structured attention with hierarchical accumulation: https://openreview.net/forum?id=HJxK5pEYvr
- Lee, Li & Benes 2023, Latent L-systems: https://dl.acm.org/doi/10.1145/3627101
- Wang et al. 2025, autoregressive generation of static and growing trees: https://arxiv.org/abs/2502.04762
- MorphGrower (ICML 2024): https://arxiv.org/abs/2401.09500
- MorphVAE (ICML 2021): https://proceedings.mlr.press/v139/laturnus21a.html
- GraphDINO: https://arxiv.org/abs/2112.12482
- Weis et al. 2025, unsupervised map of excitatory neuron dendritic morphology in MICrONS: https://www.nature.com/articles/s41467-025-58763-w and https://github.com/marissaweis/unsupervised_neuronal_map
- MorphRep (Bioinformatics 2024): https://academic.oup.com/bioinformatics/article/40/Supplement_2/ii128/7749074
- NEURD (Nature 2025): https://www.nature.com/articles/s41586-025-08660-5
- SyConn, learning cellular morphology with neural networks: https://www.nature.com/articles/s41467-019-10836-3
- Troidl et al., global neuron shape reasoning with point affinity transformers: https://www.biorxiv.org/content/10.1101/2024.11.24.625067v3 and https://github.com/jakobtroidl/neuron-shape-reasoning
- Huang, Katz, Berg & Scheffer 2025, Autoproof: https://arxiv.org/abs/2509.26585
- Brown et al. 2025, ConnectomeBench: https://arxiv.org/html/2511.05542
- Brown, Farkas, Razgar & Boyden 2026, ConnectomeBench2: https://arxiv.org/html/2606.21116v1
- Ellis et al. 2021, DreamCoder: https://dl.acm.org/doi/10.1145/3453483.3454080
- Jones, Guerrero, Mitra & Ritchie 2025, ShapeLib: https://arxiv.org/abs/2502.08884
- HDP-HMM-SCFG for trajectory grammars (2011): https://www.sciencedirect.com/science/article/pii/S1877705811016183
