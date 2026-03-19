# Soma Graph Experiment

> **Status: experimental.** Distinct from the main box-level pipeline. Kept in
> `experiments/` to explore truly global connectome inference over the
> neuron × neuron graph.

## Idea

Instead of optimizing fragment merge / synapse-pair decisions within small
boxes, build a **soma-level graph** where:

- **Nodes** = neurons (root IDs from nucleus_detection)
- **Edges** = pre→post connections from the synapse table
- **Node features** = per-neuron embeddings (e.g. pooled from grammar)

For Minnie65: ~120k nodes, sparse edges (each neuron connects to hundreds–
thousands of others). The GAT runs over this global graph.

## Relation to main pipeline

| Main pipeline (neuronauts)     | Soma graph experiment         |
|--------------------------------|--------------------------------|
| Box-centric, 6–30 µm          | Neuron-centric, whole volume  |
| Nodes = MergedNeuron fragments | Nodes = root IDs (neurons)    |
| Edges = synapse connections   | Edges = pre→post from table    |
| Grammar + GAT per box          | GAT over global neuron graph  |

We reuse `GlobalAssemblyGAT` from `neuronauts.shared_grammar_model` — it
accepts generic node features and edge lists.

## Usage

```bash
# Smoke test: synthetic soma graph + GAT forward pass
python experiments/soma_graph/smoke_test.py

# Or via pytest
pytest tests/test_soma_graph_experiment.py -v
```

## Future

- Load real synapse table (pre_pt_root_id, post_pt_root_id), build graph
- Node features: pooled synapse embeddings from grammar (or placeholder stats)
- Train edge scorer for connectivity prediction / refinement
