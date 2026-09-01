# neuronauts.coassign — Synapse Co-assignment Pipeline

## The problem

Given a set of synapses with noisy segment assignments (v117), partition them into neuron cliques: every synapse on the same neuron belongs to the same cluster.

**Synapses are the invariant nodes.** They are physical events that exist independent of any segmentation version. The partition we are learning is stable even as v117 → v1412 → future proofreading improves.

**Segments supply the evidence, not the nodes.** v117 segment IDs tell us which synapses are locally co-continuous (same-segment edges). DNA embeddings (SkeletonGNN over the segment's skeleton) tell us which segments share a neuron's morphological character.

## Architecture

```
Synapses  ──────────────────────────────────────────────────── Invariant nodes
    │
    │  edges: same-seg (v117) + spatial k-NN
    ▼
SynapseCoassigner
    │
    ├─ Encoder: LayerNorm([pos, dna]) → GNN → per-synapse embedding
    │     • LayerNorm: learned normalisation, no hardcoded pos_scale
    │     • GNN layers: message = Linear([h_src || same_seg])
    │       same_seg is a learned input feature, not a separate code path
    │
    └─ Scorer: MLP([h_u || h_v || |h_u−h_v| || same_seg]) → P(same neuron)
         • Calibrated binary cross-entropy loss
         • Hard negatives: spatial edges crossing neuron boundaries
    │
    ▼
Correlation clustering (greedy pivot, O(E))
    │
    ▼
K materializations — ranked candidate partitions
    │
    ▼
Metrics: pairwise P/R/F1 + coverage@K
```

Nothing is hardcoded. Position normalisation and edge-type weighting are learned from data.

## Files

| File | Responsibility |
|---|---|
| `graph.py` | `SynapseGraph` dataclass + `build_synapse_graph` |
| `model.py` | `SynapseCoassigner` (GNN encoder + edge scorer) |
| `cluster.py` | `greedy_cluster`, `materializations`, `pairwise_precision_recall`, `coverage_at_k` |
| `train.py` | `train()` — BCE loss with hard negative mining |

## Quick start

```python
from neuronauts.coassign import (
    build_synapse_graph,
    SynapseCoassigner,
    train,
    materializations,
    pairwise_precision_recall,
    coverage_at_k,
)
import numpy as np

# Build graph from synapse positions, segment IDs, ground-truth labels, and DNA
graph = build_synapse_graph(
    positions,    # [N, 3] nm
    seg_ids,      # [N] int64 — v117 segment assignments
    labels,       # [N] int64 — ground-truth neuron IDs (0 = unknown)
    seg_dna,      # dict: seg_id → DNA embedding [D]
)

# Train
model = SynapseCoassigner(node_dim=graph.node_dim)
history = train(model, [graph], n_epochs=60)

# Get K candidate partitions
import torch
node_feat  = torch.from_numpy(np.concatenate([graph.node_pos, graph.node_dna], 1)).float()
edge_src_t = torch.from_numpy(graph.edge_src).long()
edge_dst_t = torch.from_numpy(graph.edge_dst).long()
same_seg_t = torch.from_numpy(graph.same_seg).float()
probs = model.edge_probs(node_feat, edge_src_t, edge_dst_t, same_seg_t).numpy()

mats = materializations(graph.n_nodes, graph.edge_src, graph.edge_dst, probs, K=5)

# Evaluate
for labels, score in mats:
    r = pairwise_precision_recall(labels, graph.labels)
    print(f"P={r['precision']:.3f} R={r['recall']:.3f} score={score:.1f}")

covered = coverage_at_k(mats, graph.labels)
print(f"True partition in top-5: {covered}")
```

## Metrics

All metrics operate on synapse pairs — stable across segmentation versions.

**Pairwise precision**: of the co-assignments made, what fraction are correct.  
**Pairwise recall**: of the true co-assignments, what fraction were found.  
**coverage@K**: does any of the K materializations recover ≥90% of true co-assignments?

This is the primary quality signal for the probabilistic output: a well-calibrated model with large enough K should include the true partition (or near it) in the top-K candidates.

## Demo

```bash
python scripts/coassign_demo.py --n-neurons 20 --n-pieces 3
python scripts/coassign_demo.py --n-neurons 30 --n-pieces 3 --cell-type 23P  # within-type (harder)
```

## Next steps

See [NEXT_STEPS.md](../../docs/archive/2026-09/NEXT_STEPS.md) (archived; see
`docs/roadmap_global_assembly.md` for the current roadmap).

**Immediate improvements:**
1. Threshold tuning / learned threshold — the model learns good scores but the fixed 0.5 threshold is too conservative; a calibration pass on held-out data would improve recall without hurting precision
2. More GNN layers / larger d_model — the current 3-layer 64-dim model has limited receptive field; neurons that span hundreds of microns need longer-range message passing
3. **Endpoint-adjacent edges** — add a third edge type connecting skeleton endpoints that are spatially near each other across different segments; the principled cross-segment bridge signal. *Validated on real data* (`claude/tree-dna-phase-1-G1DNn`): endpoint edges (10 µm) boosted ARI from 0.088 → 0.418 and recovered 17/20 neurons correctly vs 5/20 without them.

**Medium term:**
4. Prototype-based assignment (EM-style) — instead of only pairwise similarity, maintain a running embedding per growing neuron hypothesis and assign synapses to prototypes; handles "see through the noise" cases where local pairwise evidence is weak but global neuron shape is recognisable
5. Within-type evaluation (`--cell-type 23P`) — all same-type negatives; harder and more honest than cross-type
6. Tree-topology constraint — neurons are trees; enforce that merged clusters are topologically consistent
