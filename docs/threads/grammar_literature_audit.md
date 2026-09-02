# Audit of `docs/grammar_literature_directions.md`

> **Status: verification pass (2026-09-01).** This document checks every major
> claim and citation in `docs/grammar_literature_directions.md` §1.1–1.9
> against primary sources (fetched abstracts, DOI pages, or full text where
> accessible), and separately asks whether each cited family actually supports
> the formulation in `docs/pcfg_global_assembly_report.md` — a typed, parsed
> probabilistic grammar over the **observed** L2 skeleton tree, used to (i)
> score which single edge to cut inside a false merge (`S(e) = log P(A_e) +
> log P(B_e) - log P(O)`), (ii) validate assembled hypotheses via hard-zero
> productions for biologically impossible structure, and (iii) type endpoints
> as terminal/cut/clipped. Per this repo's working agreement, no claim below
> is marked CONFIRMED without a source actually fetched, and no external
> system or paper is called wrong without a citation to the source that shows
> it. Neither `docs/grammar_literature_directions.md` nor
> `docs/pcfg_global_assembly_report.md` was edited to produce this audit.

## Bottom line, up front

The survey's central formal claim — "the formalism we actually want already
exists and is mature" (§1.2) — is **partly right and partly overclaimed**.
Hidden Markov trees on observed topology are real and old (Crouse et al. 1998;
confirmed), and Durand et al. 2005 and Bacciu et al. 2013 are real, correctly
described precedents for "label every node of an observed tree." But none of
the three has anything resembling the hard-zero / biologically-impossible-is-
probability-zero mechanism that carries most of the weight in our validation
use case (ii) — that mechanism's actual literature precedent is §1.3 (RNA
covariance models), which the survey undersells relative to §1.2. The
branching-process literature (§1.1) is confirmed to exist essentially as
described, but every single one of its six citations is a **generative**
model (grows a plausible tree from a distribution) and not a **parser/verifier**
of an already-reconstructed tree — the survey says this itself and it holds up.

Several precise numbers check out exactly (Matejek's 21.3% VI improvement,
Weis et al.'s >30,000-neuron / V1-AL-RL / continuum finding, Lee-Li-Benes's
93.7%/97.2%, ConnectomeBench2's 716,485/97.0%/93.0% figures, Autoproof's
90%-value-at-20%-cost / 200,000-fragment figures). Several do not survive
contact with the source unchanged — see the per-family sections and the table
below, but the two most consequential are: **ConnectomeBench's quoted human
baselines are 4–12 points too low** (the survey used a 95%-CI lower bound
where the paper reports a materially higher point estimate), which quietly
makes automated systems look more competitive with humans than the paper
itself claims; and **the Troidl et al. point-affinity-transformer paper does
not evaluate on MICrONS at all** (its three benchmark datasets are all
Drosophila: FlyWire optic lobe, MANC, hemibrain) — the survey's "trained on
FlyWire, works on MICrONS" is not supported by anything found in the paper.
Neither error changes the survey's ranking, but both currently overstate how
close the field already is to something that transfers to this project's
MICrONS substrate.

The single most useful finding of this audit is in "what the survey missed"
below: two papers (Li et al. 2020, MICCAI; Dmitriev et al. 2018, BMVC) already
solve almost exactly this problem — skeleton-node compartment typing plus
skeleton-edge cut selection for merge-error correction — but with
discriminative CNNs and hand-built consistency heuristics, not a generative
grammar with a joint likelihood and hard structural zeros. That is the
precise gap the PCFG proposal claims to fill, and no work found closes it with
a grammar. The premise survives.

---

## Verification table

Status legend: **CONFIRMED** (matches a fetched primary source exactly),
**PARTLY CONFIRMED** (core claim right, a detail is off, unverifiable, or the
citation is to a weak secondary source), **MISSTATED** (a specific,
demonstrable error — wrong author/year/venue/term/number), **NOT FOUND**
(could not locate a source after a real search).

### §1.1 Stochastic branching models of neurons

| Claim / citation as stated | Status | Correction / evidence | Source |
|---|---|---|---|
| "**Van Pelt & Uylings**: branching probability as a function of centrifugal order and current terminal count" — cited via `10.1007/BF02459919` | MISSTATED (attribution) | That DOI resolves to **Van Pelt, J. & Verwer, R.W.H. (1985)**, "Growth models (including terminal and segmental branching) for topological binary trees," *Bulletin of Mathematical Biology* 47:323–336 — not Uylings. Uylings is a frequent Van Pelt co-author elsewhere in this literature but not on the paper the survey's own link points to. The mechanism description itself (branching probability ~ centrifugal order + current terminal count, the BE(S) model) is accurate. | https://link.springer.com/article/10.1007/BF02459919 ; https://pubmed.ncbi.nlm.nih.gov/4041665/ |
| **L-Neuron (Ascoli & Krichmar 2000)**, quoted: "recursive rules that parsimoniously describe dendritic geometry and topology by locally inter-correlating morphological parameters" | PARTLY CONFIRMED | Real paper: Ascoli, G.A. & Krichmar, J.L. (2000), "L-Neuron: A Modeling Tool for the Efficient Generation and Parsimonious Description of Dendritic Morphology," *Neurocomputing* 32–33:1003–1011. The quoted phrase is a **near-verbatim actual quote**. The cited page (`krasnow1.gmu.edu`) currently fails to load (TLS error) — content corroborated only via cached search snippets, not a direct fetch. | https://krasnow1.gmu.edu/cn3/L-Neuron/HTM/paper.htm (unreachable); corroborating cache and https://www.researchgate.net/publication/220550096 |
| **Cuntz et al. 2010**, "One rule to grow them all": neuron tree = MST over spanning field, one balancing factor between material cost and conduction time | CONFIRMED | Cuntz, Forstner, Borst & Häusser, *PLoS Computational Biology* 2010. Mechanism verified: greedy MST-like construction minimizing wiring cost + `bf`·(path length from root); single balancing factor `bf`. No omission found. | https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000877 |
| **Samsonovich & Ascoli 2005**: "a hidden Markov model of hippocampal pyramidal dendrites; the earliest use of latent states along dendritic paths" | PARTLY CONFIRMED | Real paper: "Statistical determinants of dendritic morphology in hippocampal pyramidal neurons: A hidden Markov model," *Hippocampus* 15(2):166–183 (2005) — the HMM claim is literally in the title, confirmed. The cited source is a bare Wikidata entry, not the paper — a weak citation. **"Earliest use of latent states along dendritic paths" is an unqualified priority claim with no supporting evidence**, and the survey's own §1.2 credits Crouse et al. 1998 as the general origin of HMTs, so "earliest" needs a "in dendritic morphology specifically" qualifier it doesn't have. | https://www.wikidata.org/wiki/Q42468211 (weak) ; PMID 15390156 |
| **Kanari et al. 2022 (TNS)**: topological neuron synthesis, "generates whole cortical regions from a few reference cells" | CONFIRMED | "Computational synthesis of cortical dendritic morphologies," *Cell Reports* 39(1):110586 (2022). TNS/TMD (topological morphology descriptor) terminology and the "reconstruct entire brain regions from few reference cells" capability both confirmed (paper reports ~10M cells in ~4 hours). | https://www.cell.com/cell-reports/fulltext/S2211-1247(22)00330-8 |
| **NETMORPH (Koene et al.)**: "network-scale stochastic growth on the same principles" | CONFIRMED (paper); gap confirmed (citation) | Real paper: Koene et al. (2009), "NETMORPH: A Framework for the Stochastic Generation of Large Scale Neuronal Networks With Realistic Neuron Morphologies," *Neuroinformatics* 7(3):195–210. **The survey's own "## Sources" list has no entry for NETMORPH at all** — a genuine missing citation, distinct from the other five which are at least linked (rightly or wrongly). | https://link.springer.com/article/10.1007/s12021-009-9052-3 |

**Congruence, §1.1.** All six works are confirmed to be real, and the survey's
own verdict — "None of these models was ever used as an error detector; that
is the gap" — holds up under fetch. Every one is a **generative growth
model**: it samples a plausible tree from a stochastic process (branching
events over simulated time, or a static MST/L-system construction), evaluated
by whether the *population* of generated trees matches measured statistics.
None parses, labels, or scores an *already-reconstructed, fixed* tree, and
none has ever been repurposed to flag a specific tree as anomalous. The
closest of the six architecturally is Samsonovich & Ascoli (an HMM with state
along the dendritic path, on real reconstructed data) — but even that was used
for statistical characterization and resampling, not merge/seam detection.
The honest reading is: this family gives usable **functional forms** for
emissions (`P(k | order, terminals)`, Rall/Murray caliber relations, TMD
barcodes as a feature) — a modest, correctly-scoped claim — not any evidence
that a grammar-as-verifier has precedent here.

### §1.2 Hidden Markov trees + §1.3 SCFGs for RNA

| Claim / citation | Status | Correction / evidence | Source |
|---|---|---|---|
| **Crouse, Nowak & Baraniuk 1998**: "observed tree, latent state per node, upward-downward (inside-outside) inference and EM" | CONFIRMED, with a terminology conflation | "Wavelet-Based Statistical Signal Processing Using Hidden Markov Models," *IEEE Trans. Signal Processing* 46(4):886–902, 1998. Their own name for the algorithm is the **"upward-downward algorithm"** (an EM/Baum-Welch analogue for trees). The survey's parenthetical "(inside-outside)" is its own gloss — inside-outside properly names the SCFG-parsing algorithm family in §1.3, a related but distinct derivation. Not wrong to draw the analogy; wrong to imply it's the same named object. | https://dl.acm.org/doi/10.1109/78.668544 |
| **Durand, Guédon, Caraglio & Costes 2005**, quoted: "homogeneous zones and transitions between zones within tree-structured data"; cites "Godin et al. 1999, multiscale tree graphs" | MISSTATED (secondary citation) | Fetched full PDF: *New Phytologist* 166:813–825 (2005). Title/authors/venue/year/quote all confirmed — the quoted phrase is near-verbatim from the abstract, and apple trees / bush willows / annual-shoot scale are exact. **But** Durand et al.'s own paper cites the topology framework as **Godin & Caraglio (1998)**, "A Multiscale Model of Plant Topological Structures," *J. Theor. Biol.* 191:1–46 — a **two-author** 1998 paper, not "Godin et al. 1999." Both the year and the implied author count are wrong. | https://nph.onlinelibrary.wiley.com/doi/10.1111/j.1469-8137.2005.01405.x (full text fetched) ; https://doi.org/10.1006/jtbi.1997.0561 |
| **Bacciu, Micheli & Sperduti 2013**: "input-output bottom-up hidden tree Markov models for tree transductions... labelling every node of an observed tree given per-node inputs" | CONFIRMED | Real title: "An Input-Output Hidden Markov Model for Tree Transductions," *Neurocomputing* 112:34–46, 2013 — matches the given URL. The survey's compound name blends this title with the "Bottom-up Hidden Tree Markov Model" name used elsewhere in the same authors' line; it is an accurate characterization of the model family, not a verbatim title, and the one-line task description is accurate. | https://www.sciencedirect.com/science/article/abs/pii/S0925231213001914 |
| **Eddy & Durbin / Infernal / Rfam**: "profile SCFGs that score sequence and base-pairing structure, with hard structural constraints and bit-score thresholds for abstention" | CONFIRMED | Eddy, S.R. & Durbin, R. (1994), "RNA sequence analysis using covariance models," *Nucleic Acids Research* 22(11):2079–2088 — correct origin. Infernal and Rfam's "gathering cutoff" (a literal bit-score threshold separating true from false family members) are both real and confirmed. "Abstention" is the survey's own gloss, not Rfam's vocabulary, but an accurate characterization of what the threshold does. | http://eddylab.org/infernal/ ; https://docs.rfam.org/en/latest/choosing-gathering-threshold.html ; PMID 8029015 |

**Congruence, §1.2–1.3 — the sharpest correction in this audit.** Crouse et
al. is a real but only *partial* precedent for "the formalism we actually
want." Two structural gaps are real, not cosmetic: (a) their trees are
fixed-arity, fixed-depth dyadic wavelet-decomposition trees — identical shape
across every instance — nothing like the variable-arity, variable-depth
biological skeletons this proposal parses; extending the recursions to
variable arity is straightforward algebra, but it is not what Crouse et al.
did or validated. (b) their latent state is a small (typically 2-state)
discrete mixture-component index controlling a Gaussian variance for one
wavelet coefficient's magnitude — a low-level nuisance label, categorically
different from a semantic compartment type (`SOMA`/`AXON`/`DENDRITE`/…)
carrying hard biological constraints. **More importantly: hard-zero
constraints are absent from the HMT literature entirely.** Durand et al.'s
plant HMT is fully soft/probabilistic (their own paper flags unmodeled
patterns as a limitation, with no hard-constraint mechanism), and Bacciu et
al.'s IO-HMM tree transduction is likewise fully probabilistic. "Hard zeros
for the impossible" — central to validation use case (ii) — is entirely the
survey's own addition bolted onto borrowed HMT machinery, not inherited from
either §1.2 source. **§1.3 (RNA SCFGs), not §1.2 (HMT), is the real precedent
for hard constraints + deployed abstention thresholds** — covariance models
bake base-pairing structure directly into grammar productions and Rfam's
gathering cutoff is a literal, shipped abstention threshold. The survey's own
§1.2 "reading for us" ("exactly the standard HMT setting... mature") overstates
its case relative to what §1.3 actually demonstrates for the validation half
of the proposal.

### §1.4 Global structured inference + §1.5 Graph grammars

| Claim / citation | Status | Correction / evidence | Source |
|---|---|---|---|
| **Türetken et al. 2016**: curvilinear networks, path classifiers + ILP, "subject to structural and topological constraints" | CONFIRMED | "Reconstructing Curvilinear Networks using Path Classifiers and Integer Programming," Türetken, Benmansour, Andres, Głowacki, Pfister, Fua, *IEEE TPAMI* 38(12):2515–2530, 2016. Quoted phrase matches the abstract. Note the method is **not tree-restricted** — it generalizes to loopy/cyclic curvilinear networks, a scope point the survey doesn't misstate but also doesn't flag. | https://infoscience.epfl.ch/record/201670 |
| **Matejek et al. 2019**: lifted multicut, "average VI improvement of 21.3% on four datasets" | CONFIRMED | "Biologically-Constrained Graphs for Global Connectomics Reconstruction," CVPR 2019, pp. 2089–2098. The 21.3% figure is **verbatim from the abstract**: "an average variation of information improvement of 21.3%" across four real-world connectomics datasets. Exact match, not an approximation. | https://openaccess.thecvf.com/content_CVPR_2019/html/Matejek_Biologically-Constrained_Graphs_for_Global_Connectomics_Reconstruction_CVPR_2019_paper.html |
| **ViterBrain (Athey et al. 2022)**: HMM over fragments, Boltzmann transitions on gap/curvature, "31% success vs 11% for the nearest competitor" | CONFIRMED, narrower scope than implied | "Hidden Markov modeling for maximum probability neuron reconstruction," *Communications Biology*, 2022. Across 35 MouseLight subvolumes: ViterBrain succeeded on 11/35 (31.4%); the best competitor, APP2, on 4/35 (11.4%) — matches "31% vs 11%." But **"success" specifically means completing an axon trace matching the first ten points of a ground-truth reconstruction from the soma**, a narrower claim than the survey's plain "31% success" phrasing suggests. HMM structure (endpoint+tangent states, Boltzmann transitions over gap distance and curvature) is accurate. | https://pmc.ncbi.nlm.nih.gov/articles/PMC9038756/ |
| **Kemp & Tenenbaum 2008**: "hand-drawn vertex-replacement productions with learned probabilities" | MISSTATED (term) | "The discovery of structural form," PNAS 105(31):10687–10692, 2008 — title/authors/venue/year all correct. But the paper's own term is a **"node-replacement graph grammar"** (each rule replaces a parent node with two children), not "vertex-replacement productions." "Vertex replacement grammar" names a distinct, unrelated formalism elsewhere in the graph-grammar literature, so this is a real terminology error, not just a synonym choice. | https://www.pnas.org/doi/10.1073/pnas.0802631105 |
| **Aguinaga, Chiang & Weninger (TPAMI 2018)**: hyperedge replacement grammars "extracted automatically from a graph's clique tree" | PARTLY CONFIRMED (year) | "Learning Hyperedge Replacement Grammars for Graph Generation," *IEEE TPAMI* 41(3):625–638. The canonical journal-issue year is **2019**; 2018 was only the early-access/arXiv-preprint date (DOI reflects a 2018 early-access stamp). Method description ("extracted from the clique tree") confirmed accurate. | https://arxiv.org/abs/1802.08068 ; dblp record (TPAMI 41(3), 2019) |

**Congruence, §1.4.** Real but partial resemblance. All three methods solve
**assembly/structure-discovery over an unresolved candidate space** — Türetken
selects a path subset from a super-graph of *candidates*; Matejek's lifted
multicut decides which supervoxels merge; ViterBrain's Viterbi search chains
fragments together. None of them takes a single, already-fixed tree and asks
"which one edge is the false-merge seam" — that presupposes the topology is
already resolved, which is precisely what these methods are still solving
for. They map onto the design report's *assembly* use case, not its cut/
validate/type core. Specifically: **ViterBrain's `Boltzmann(gap, curvature)`
transition is analogous in form to, but not the same mathematical object as,
our proposed `g(o_join)`** — both are Gibbs-form potentials penalizing large
gaps/sharp turns, but ViterBrain's potential governs a sequential-chain HMM
searching an exponential space of candidate connections to *build* topology,
while `g(o_join)` scores a junction *inside an already-instantiated tree*
under a hidden-Markov-tree model — no search over which edges exist, only
scoring of edges that already exist.

**Congruence, §1.5.** The survey's judgment holds up for Kemp & Tenenbaum: it
is a real precedent for typing/classifying an abstract graph's overall form,
never touching edge-level seam location. For Aguinaga et al., the survey's
"typing feature" framing is a genuine **repurposing beyond what the paper
demonstrates** — the paper's actual application is generative graph synthesis
(learn a grammar from one graph, generate statistically similar new graphs),
not classification of an object's type. That's an extrapolation, not a
misrepresentation, but it's worth flagging as such rather than presenting it
as a direct precedent for a typing feature.

### §1.6 Neural grammars and structured transformers

| Claim / citation | Status | Correction / evidence | Source |
|---|---|---|---|
| **Compound PCFG (Kim, Dyer & Rush 2019)**: latent per-sentence variable, "inducing marginal dependencies beyond the traditional context-free assumptions"; collapsed VI with trees marginalised by DP | CONFIRMED | "Compound Probabilistic Context-Free Grammars for Grammar Induction," ACL 2019. Quoted phrase is a real near-verbatim quote from the abstract. Inference description accurate. **Confirmed to operate in the classic NLP setting where the STRING is observed and the PARSE TREE IS LATENT**, inferred by CKY-style dynamic-programming marginalization over all bracketings — the opposite of this proposal's observed-tree setting. | https://aclanthology.org/P19-1228/ |
| **Transformer Grammars (Sartran et al. 2022, TACL)**: attention mask over a linearised tree | CONFIRMED | "Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale," TACL 2022. Abstract confirms the tree is **given/observed** (a pre-parsed tree is linearized and fed in via a deterministic attention mask), not inferred by the model. | https://aclanthology.org/2022.tacl-1.81/ |
| **Torch-Struct (Rush 2020)**: "batched, differentiable inside algorithms for CFGs, HMMs and trees" | PARTLY CONFIRMED (scope) | "Torch-Struct: Deep Structured Prediction Library." Published at ACL 2020 System Demonstrations (not arXiv-only, as the survey's citation implies by giving only the arXiv link). Confirmed scope, more specific than stated: CFG via CKY, dependency trees via Eisner/matrix-tree, HMM/HSMM via forward-backward — **all as distributions over the space of trees/parses consistent with a linear sequence**, i.e. still doing structure search, not a primitive for a single already-fixed arbitrary-arity tree. | https://arxiv.org/abs/2002.00876 ; https://aclanthology.org/2020.acl-demos.38 |
| **Shiv & Quirk 2019**, tree positional encodings | CONFIRMED | "Novel Positional Encodings to Enable Tree-Based Transformers," NeurIPS 2019, pp. 12058–12068. Generalizes sinusoidal positional encodings to tree paths; tree structure is given/target. | https://papers.nips.cc/paper/9376-novel-positional-encodings-to-enable-tree-based-transformers |
| **Nguyen et al. 2020**, hierarchical accumulation, ICLR | CONFIRMED | "Tree-Structured Attention with Hierarchical Accumulation," ICLR 2020. Encodes a **given** parse tree into self-attention by aggregating descendant states per nonterminal — operates on a supplied tree. | https://openreview.net/forum?id=HJxK5pEYvr |

**Congruence, §1.6 — the second-sharpest correction.** Compound PCFG and
Torch-Struct are presented as more directly reusable than they are.
**Compound PCFG's entire algorithmic apparatus (collapsed variational
inference, amortized posterior, DP-marginalized latent trees) exists to solve
a problem this proposal does not have** — recovering an unknown bracketing
from a string. Once the tree is observed, none of that machinery is needed; a
per-object continuous vector modulating emission/transition probabilities in
a single fixed bottom-up pass is a much simpler object with no marginalization
over structure. The survey borrows a modeling *idea* (a latent per-object
vector for context modulation) and calls it "the principled answer," implying
a technical provenance the idea doesn't actually carry once detached from
compound PCFG's machinery. Similarly, **Torch-Struct's tree/CFG modules are
built for structure search over sequences (CKY charts, Eisner parsing), not
sum-product over a single fixed arbitrary-arity tree like a biological
skeleton** — using it as "a differentiable layer" for this grammar would
require binarizing/reshaping the skeleton into Torch-Struct's supported chart
formats, or writing new fixed-tree DP code outside the library. That's real
implementation work, not the drop-in reuse the survey's phrasing suggests.
Transformer Grammars and tree positional encodings are, by contrast,
genuinely congruent: both take a tree as *given* input and require no search
over structure, matching this proposal's "tree is observed" setting far
better than either compound PCFG or Torch-Struct's tree/CFG modules.

### §1.7 Tree generators in graphics + §1.8 MICrONS representation learning

| Claim / citation | Status | Correction / evidence | Source |
|---|---|---|---|
| **Latent L-systems (Lee, Li & Benes, ACM TOG 2023)**: 155k trees, 93.7% branching-angle / 97.2% branch-length agreement | CONFIRMED | "Latent L-systems: Transformer-based Tree Generator," ACM TOG 43(1), Nov 2023. Abstract states verbatim: "trained on 155k tree geometries... agrees with the input by 93.7% in branching angles, 97.2% in branch lengths, and 92.3% in an extracted list of geometric features." Both cited numbers match exactly (the survey omits a third figure, 92.3%, but doesn't misstate what it does cite). | https://dl.acm.org/doi/10.1145/3627101 |
| **Wang et al. 2025**, autoregressive tree generation, SIGGRAPH Asia 2025 | PARTLY CONFIRMED (venue) | Authors, hourglass multi-resolution transformer over tree tokens, image-to-tree/point-cloud-to-tree conditioning, and "4D trees" (growth) all confirmed from the abstract. "SIGGRAPH Asia 2025" is not independently confirmable from the arXiv listing itself — plausible but unverified, not necessarily wrong. | https://arxiv.org/abs/2502.04762 |
| **MorphGrower (ICML 2024 oral)**: layer-by-layer growth, beats MorphVAE whose walks "produced mostly topologically invalid trees"; "no explicit likelihood; plausibility judged by a real-vs-generated classifier" | PARTLY CONFIRMED | "MorphGrower," Yang et al. — **ICML 2024 Oral status confirmed** via the official program and GitHub repo. Sibling-branch / ancestor-path / T-GNN mechanism confirmed. MorphVAE comparison confirmed **qualitatively** (MorphGrower's own text: "most of the generated samples are topologically invalid," citing multi-furcating nodes that violate the soma-only-branch-point rule) but **no percentage is given anywhere in either paper** — "mostly" is MorphGrower's own qualitative characterization, not a quantified figure the survey can cite as if numeric. The "real-vs-generated classifier" evaluation channel was not found in the sections reviewed (the paper instead emphasizes an electrophysiological-simulation evaluation) — flagged unconfirmed, not contradicted. | https://arxiv.org/abs/2401.09500 ; https://icml.cc/virtual/2024/oral/35513 |
| **MorphVAE (ICML 2021)** | MISATTRIBUTED framing | "MorphVAE: Generating Neural Morphologies from 3D-Walks using a Variational Autoencoder with Spherical Latent Space," Laturnus & Berens, PMLR 139:6021–6031 — real, correct citation. But **the "topologically invalid" claim is not self-reported by MorphVAE anywhere** — it is entirely MorphGrower's later characterization of MorphVAE's outputs. The survey's phrasing ("MorphVAE generating topologically invalid trees," listed as if a property of the source) invites a reader to take it as MorphVAE's own admitted limitation; it isn't, and there is no percentage behind "mostly" in either paper. | https://proceedings.mlr.press/v139/laturnus21a.html |
| **GraphDINO (Weis et al.)** | PARTLY CONFIRMED (venue) | "Self-Supervised Graph Representation Learning for Neuronal Morphologies," published at **TMLR 2023** (the survey cites only the bare arXiv link, so not wrong, just under-specified). AC-attention mechanism confirmed exactly as described. | https://arxiv.org/abs/2112.12482 |
| **Weis et al. 2025 (Nat Commun 16:3361)**: >30,000 excitatory neurons, V1/AL/RL, continuum except layers 5/6 | CONFIRMED | "An unsupervised map of excitatory neuron dendritic morphology in the mouse visual cortex," *Nature Communications* 16:3361 (2025). Full text (via PMC) confirms verbatim: "more than 30,000 excitatory neurons in mouse visual areas V1, AL, and RL" and a morphological landscape "better described as a continuum, with a few notable exceptions in layers 5 and 6." Both numbers repeated in the TL;DR ranked table (ranks 4–5) check out exactly. | https://www.nature.com/articles/s41467-025-58763-w ; full text https://pmc.ncbi.nlm.nih.gov/articles/PMC11982532/ |
| **MorphRep (Bioinformatics 2024)** | MISSTATED (title) | The paper's actual title is **"Learning meaningful representation of single-neuron morphology via large-scale pre-training"** (Fan, Li, Zhong, Hong, Li, Li), *Bioinformatics* 40(Suppl_2):ii128–ii136, 2024 — "MorphRep" is the method/tool name used inside the paper, not its title, which the Sources list implies. Venue/year/pages confirmed. Description ("graph-transformer pretraining at scale," >250,000 morphologies) is accurate — if anything an understatement. | https://academic.oup.com/bioinformatics/article/40/Supplement_2/ii128/7749074 |

**Congruence, §1.7.** All four are generative/completion models; none
repurposes its likelihood as an anomaly/error detector on an existing tree —
the survey's own "low priority... nobody has shown these likelihoods detect
errors" verdict holds up under fetch. Closest to our use case: Wang et al.'s
point-cloud-to-tree conditioning is structurally an "infill the missing
continuation" operation, adjacent to but not the same as validating or cutting
an existing skeleton. Furthest: MorphVAE/MorphGrower, which generate whole new
morphologies from a class-conditioned prior and judge plausibility by
reconstruction/classifier or downstream simulation, never by parsing or
scoring a specific pre-existing tree's structure.

**Congruence, §1.8.** Confirmed as whole-object (one vector per neuron)
embeddings for downstream clustering/classification — none operates per-
segment or per-edge, so none substitutes for a typed parse of tree structure.
The survey's own framing ("pooled DNA" / class-conditioning input to the
grammar, not a grammar substitute) is the correct characterization and is not
oversold; it is appropriately hedged on whether per-root embeddings are
actually released (unverified as of this audit).

---

### §1.9 Connectomics proofreading with learning

This is the highest-stakes section — the most number-dense, and the family
that most surprised the person who requested this audit.

| Claim / citation | Status | Correction / evidence | Source |
|---|---|---|---|
| **NEURD (Celii et al., Nature 2025)**: "heuristic graph rules... a hand-written grammar in effect"; elsewhere, "NEURD detected zero merge errors in ConnectomeBench's cohort" | PARTLY CONFIRMED / framing caveat | Real: Celii et al., "NEURD offers automated proofreading and feature extraction for connectomics," *Nature* 640:487–496 (9 Apr 2025). NEURD's own words are "heuristic proofreading rules implemented as graph filters" — threshold logic on scalar features (diameter jumps, branch angles), **not a probabilistic grammar with productions**; "a hand-written grammar in effect" is the survey's own analogy, not NEURD's self-description. The zero-merge-errors claim is correctly sourced to ConnectomeBench, but the test was **14** nucleus-backed MICrONS segments (chosen because they meet NEURD's soma-required design) at one detection radius — "ConnectomeBench's cohort" reads as the full benchmark (hundreds of items, two datasets); the actual n is 14. | https://www.nature.com/articles/s41586-025-08660-5 ; https://arxiv.org/html/2511.05542 |
| **SyConn (Schubert et al. 2019)**: multi-view projections, glia detection resolves errors | CONFIRMED | "Learning cellular morphology with neural networks," *Nature Communications* 10:2736 (2019). Multi-view-projection CNNs plus a glia-detection CNN (F1=0.979) feeding a splitting heuristic — matches. | https://www.nature.com/articles/s41467-019-10836-3 |
| **Point affinity transformers (Troidl et al.)**: "trained on FlyWire, works on MICrONS; beats GNNs, point transformers and unsupervised clustering" on *simulated* errors | MISSTATED (dataset claim) | Real: Troidl, Knittel, Li, Zhan, Pfister, Turaga (Harvard + HHMI Janelia), bioRxiv 2024.11.24.625067v3. Confirmed: point-cloud→embedding→pairwise-affinity method; beats GNN/point-transformer/unsupervised-clustering baselines; errors are confirmed **simulated** ("we employed simulated neuron reconstruction errors to evaluate..."). **But the "works on MICrONS" half is not supported** — a full-text search returns zero MICrONS mentions; the three benchmark datasets are FlyWire optic lobe, MANC, and hemibrain, all *Drosophila*. This matters because the survey uses this paper to argue relevance to a MICrONS-based project. | https://www.biorxiv.org/content/10.1101/2024.11.24.625067v3 |
| **Autoproof (Huang, Katz, Berg & Scheffer, 2025)**: 90% of value at 20% of cost; 200,000 fragments (~4 proofreader-years) | CONFIRMED | arXiv:2509.26585. Exact match: "90% of the value of a guided proofreading workflow while reducing required cost by 80%" (=20% of cost); "automatically attach 200 thousand fragments, equivalent to four proofreader years of manual work." | https://arxiv.org/abs/2509.26585 |
| **ConnectomeBench (Brown et al., MIT, 2025)**: 74.0%/70.3% merge MC (o4-mini with descriptions) vs human 74–80%; binary merge 62.8/61.5%; split MC 78.8/85.0% vs human 84–90% | MISSTATED (condition conflation + human baseline) | Real: Brown, Kirjner, Vivekananthan, Boyden (MIT), arXiv:2511.05542. Binary merge (62.8/61.5%) and split MC (78.8/85.0%) figures check out. Two problems: **(a)** the "74.0%/70.3%, o4-mini with descriptions" line conflates two different conditions — the paper's Table 7 shows 74.0% (FlyWire) is o4-mini **+ Null** (no description), while 70.3% (MICrONS) is o4-mini **+ Description**; only one of the two numbers is actually from the stated condition. **(b)** the human baselines are understated: actual point estimates are merge-MC human = **84.0% FlyWire / 79.6% MICrONS** (the survey's "74–80%" uses FlyWire's 95%-CI *lower bound*, 74.0, not the 84.0 point estimate) and split-MC human = **90.0% FlyWire / 92.0% MICrONS** (the survey's "84–90%" uses MICrONS's CI lower bound, not the 92.0 point estimate). Net effect: the survey's stated human baselines run 4–12 points below what the paper actually reports, making the automated systems look closer to human parity than the source supports. | https://arxiv.org/html/2511.05542 |
| **ConnectomeBench2 (Brown, Farkas, Razgar & Boyden, 2026)**: 716,485 decisions across four datasets; three named tasks; ViT-B 97.0%/93.0% vs human 93.0%/84.1%; stated limitations | CONFIRMED — real paper, not fabricated | Live at arXiv:2606.21116v1 (submitted 19 Jun 2026, Brown/Farkas/Razgar/Boyden, MIT/Mindspan Institute/McGovern Institute). All headline numbers match exactly: 716,485 expert-labeled decisions across mouse/MICrONS, human/H01, zebrafish/Fish1, fly/FlyWire; the three named tasks; ViT-B 97.0% (splits)/93.0% (merges) balanced accuracy; human baseline 93.0%/84.1%; and the stated limitations (2D-render lossiness, ViT-only coverage, species imbalance, calibration worse for merges/OOD) all check out. Given a June-2026 arXiv date is unusually recent, this was checked with real skepticism before confirming — it is a genuine, fetchable paper. | https://arxiv.org/html/2606.21116v1 |

**Congruence, §1.9.** None of the six is a parsed generative grammar with hard-
zero productions over a tree — all six are detectors/classifiers over
rendered images, point clouds, or hand-set thresholds. NEURD is the closest
superficially, but its own paper describes sequential heuristic graph filters
on scalar features: no production probabilities, no partition function over
trees, no hard-zero constraints, no inside-outside-style DP. "A hand-written
grammar in effect" should be read as the survey's metaphor, not as a
description of NEURD's actual mechanism. On the survey's own load-bearing
claim — that ConnectomeBench2's "mask segmentation for merge-error
correction" is "our seam-location problem by another name" — the audit finds
this needs a caveat the survey doesn't quite give: that task produces a 2D
pixel/image mask on an orthographic mesh render, not an edge selection on the
L2 skeleton graph. The survey's own hedge ("a candidate answer that does not
need a render") is honest about the representational difference, but no
result anywhere compares the two approaches numerically, so "our seam-location
problem by another name" is a structural analogy, not a demonstrated
equivalence.

**Housekeeping, outside §1.1–1.9 but sharing the same Sources list.** Two
citations that appear only in the TL;DR ranked table (direction 7) were
checked for completeness: **DreamCoder (Ellis et al. 2021, PLDI)** and
**ShapeLib (Jones, Guerrero, Mitra & Ritchie 2025, arXiv:2502.08884)** are both
CONFIRMED — real papers, correct authors/venue/year, and both genuinely match
the survey's "LLM proposes candidate structure, validated, kept or reverted"
pattern (DreamCoder's wake-sleep library learning; ShapeLib's guided LLM
workflow with validation against seed shapes). Separately, the Sources list
carries **"HDP-HMM-SCFG for trajectory grammars (2011)"** — a real paper
(*A Novel Model for Trajectory Representation and Classification*,
ScienceDirect, 2011, combining an SCFG with HDP-HMM emissions for human-
activity trajectory classification via bottom-up parsing) — but it is never
named, discussed, or connected to any claim anywhere in the body text of
sections 1–3. It is an orphan citation: real, but currently doing no work in
the document.

---

## What the survey missed

A dedicated search (roughly twenty queries across grammar/PCFG/SCFG ×
connectomics, hidden-Markov-tree × neuron/dendrite/skeleton, grammar × merge/
split error, tree-grammar × vasculature/airway-tree labeling, and rule-based
validity-checking tools in the NeuroMorpho/L-Measure/TREES-toolbox ecosystem)
found two papers close enough to change the novelty picture, and confirmed
several dead ends worth recording so nobody re-runs them.

**Close hits — worth reading before starting E0–E2:**

1. **Li, Januszewski, Jain & Li, "Neuronal Subcompartment Classification and
   Merge Error Correction," MICCAI 2020** (`10.1007/978-3-030-59722-1_9`;
   bioRxiv 2020.04.16.043398). Classifies skeleton nodes into axon/dendrite/
   soma via a 3D CNN (F1 0.972), then locates the merge-error cut edge by a
   heuristic "cut consistency score" — the edge whose removal maximizes
   label-consistency of the two resulting subgraphs — plus a separate rule for
   soma merges. **This is the nearest miss**: same problem shape as `S(e)`
   (compartment-typed skeleton, search over candidate cut edges, biological
   priors as hard constraints — one axon per soma, segregated compartments),
   but the mechanism is a discriminative classifier plus a hand-built
   consistency heuristic, not a generative grammar with a joint likelihood, no
   hard-zero production semantics, no inside-outside DP, and no calibrated tip
   posterior. https://link.springer.com/chapter/10.1007/978-3-030-59722-1_9

2. **Dmitriev, Parag, Matejek, Kaufman & Pfister, "Efficient Correction for EM
   Connectomics with Skeletal Representation," BMVC 2018** (Harvard VCG).
   Represents each EM segment by its skeleton, then runs CNNs at skeleton
   joints/endpoints to detect and correct both false-merge and false-split
   errors, cutting the correction search space by roughly five orders of
   magnitude versus voxel-level search. Same framing — skeleton joints as
   candidate seam locations — again via a discriminative per-joint classifier,
   not a structured model with a whole-object likelihood or hard-zero
   grammar.

Neither paper is cited anywhere in `docs/grammar_literature_directions.md`,
and both are closer to the seam-location use case than most of what §1.4 and
§1.9 do cite (Türetken, Matejek 2019, NEURD, SyConn). They should be added to
the survey and, more importantly, used as the baseline `S(e)`/`argmax_e S(e)`
has to beat in E1/E2 — a heuristic consistency score on CNN-typed nodes is a
cheaper, already-published thing to fail to beat.

**Searched and empty:** no combination of PCFG/SCFG/context-free-grammar with
connectomics, EM proofreading, or neuron reconstruction; no hidden-Markov-tree
(Crouse/Durand-style: latent state on an *observed* tree) applied to neuron,
dendrite, axon, or skeleton data — every "HMM + neuron" hit found is an HMM
over image-space paths in the ViterBrain mold, not a label model on
already-reconstructed topology; no grammar-based tree-labeling in the
vasculature/airway/coronary-tree literature (that field uses geodesic
matching and tree-LSTMs/GNNs instead); TREES toolbox (Cuntz) and NeuroMorpho/
L-Measure do generation and morphometric standardization, not grammar-based
error detection; no tool named "MorphoGrammar" or equivalent exists; a related
2026 bioRxiv preprint (Emissah, Tecuatl & Ascoli, "Automated Proofreading of
Digitally Reconstructed Neural Morphology") does rule-based deterministic
correction plus a GCN classifier and explicitly is not probabilistic or
grammar-based, per its own method section.

**Verdict on novelty.** The survey's implicit claim — that nobody has applied
a parsed, typed probabilistic grammar / hidden Markov tree to an *observed,
already-reconstructed* neural skeleton to locate a seam by likelihood-ratio
argmax, hard-zero invalid derivations, and produce a calibrated tip posterior
— holds up under a real search. The two near-misses above solve the same
*problem* with a heuristic score and a discriminative classifier instead of a
generative grammar; that is exactly the mechanism gap the PCFG proposal
claims to fill, and this audit did not find it closed elsewhere.

---

## Bottom line

**Does the PCFG proposal's premise survive contact with the literature?**
Yes, with three corrections and one genuine strengthening:

1. **The formalism exists but the survey overstates which half it covers.**
   HMTs on observed trees (Crouse et al. 1998; Durand et al. 2005; Bacciu et
   al. 2013) are real, correctly cited (modulo one wrong secondary citation —
   Godin & Caraglio 1998, not "Godin et al. 1999"), and genuinely establish
   that "observed tree, latent per-node label, tree-structured EM/Viterbi" is
   a mature setting. But none of the three has a hard-zero mechanism; that
   half of the proposal (the biologically-impossible-is-probability-zero
   productions that do most of the work in validation use case (ii)) has its
   real precedent in §1.3's RNA covariance models, not §1.2's HMTs. The survey
   should say this explicitly rather than filing hard zeros under "the
   formalism we actually want."
2. **§1.6's neural-mechanism borrows are idea-level, not machinery-level, for
   two of five citations.** Compound PCFG and Torch-Struct are held up as more
   directly reusable than a careful read supports — both are built for
   inferring/searching an *unknown* tree structure from a sequence, the
   opposite of this project's observed-tree setting. Transformer Grammars and
   tree positional encodings are the two that actually match the "tree is
   given" premise and should be weighted accordingly.
3. **Two 2025 numbers the survey leans on for "the field is near human-level"
   are off in the direction that flatters automation**: ConnectomeBench's
   quoted human baselines are 4–12 points below the paper's own point
   estimates, and Troidl et al.'s claimed MICrONS applicability is not
   supported by the paper (Drosophila-only). Both should be corrected before
   this survey is cited elsewhere, since both currently make prior work look
   closer to solving this project's problem than the sources support.
4. **The strengthening**: nobody has already built this. A real search for
   grammar/HMT-based seam location on reconstructed neural skeletons came back
   with two close-but-not-equivalent papers (Li et al. 2020; Dmitriev et al.
   2018) that solve the identical problem shape with discriminative
   heuristics instead of a generative grammar, and nothing closer. That is the
   best evidence available that the specific mechanism proposed — a joint
   likelihood over a typed, parsed tree, with hard zeros, used as a seam
   locator — is not redundant with existing work, only adjacent to it.

None of the corrections above change the survey's ranked recommendations
(§TL;DR) or the companion report's E0–E5 plan; they change what can be safely
asserted when this survey is cited elsewhere, and where the two nearest-miss
baselines (Li et al. 2020, Dmitriev et al. 2018) belong in the experimental
program as the bar E1/E2 should beat, not just the seam GNN and the global-
shape RF already named there.
