"""Public package surface for neuronauts.

The exports below intentionally prioritize the active, no-EM training pipeline.
Legacy/experimental modules still exist in the repository but are no longer
re-exported here to keep top-level imports focused and maintainable.
"""

from .assembly import CandidateMerge, logit_to_probability, probability_to_log_odds
from .cell_graph import (
    CellGNN,
    CellGNNConfig,
    EDGE_FEATURE_NAMES,
    SynapseEdge,
    SynapseGraph,
    build_synapse_chain_paths,
    build_synapse_graph,
    cell_graph_edge_train_step,
    cell_graph_train_step,
    cell_gnn_assembly,
    infer_cells,
    infer_cells_edge,
    infer_cells_two_pass,
    load_seg_score_cache,
    load_self_skeleton_archive,
    load_skeleton_path_cache,
    partition_from_embeddings,
    precompute_seg_scores_fast,
    precompute_self_skeletons_for_cache,
    precompute_skeleton_paths_for_cache,
    save_seg_score_cache,
    save_skeleton_path_cache,
    subdivide_synapse_graph,
    train_cell_gnn,
)
from .line_graph import LineGraphMetrics, evaluate, evaluate_sampled
from .path_dataset import (
    extract_cell_chains,
    fetch_cave_edit_history,
    generate_path_examples,
    load_path_encoder,
    save_edit_pairs_tsv,
    train_path_encoder,
)
from .path_edge_encoder import PathEdgeEncoder, pad_path_sequences
from .data.fragments import extract_fragments_for_region, skeleton_to_fragment
from .represent.dna import (
    TreeDNAEncoder,
    encode_fragments,
    featurize_fragment,
    sample_tree_paths,
    train_dna_encoder,
)
from .represent.enrich import (
    build_synapse_dna_matrix,
    evaluate_dna_auc,
    spatial_proximity_scores,
    synapse_pair_dna_scores,
)
from .represent.skeleton_gnn import (
    SkeletonGNN,
    encode_fragments_gnn,
    fragment_to_tensors,
    train_skeleton_gnn,
)
from .assemble import (
    GlobalSynapseGraph,
    assemble_neurons,
    build_global_synapse_graph,
    run_global_gnn,
    train_global_gnn,
    HalfSynapseGraph,
    build_half_synapse_graph,
    HalfSynapseGNN,
    evaluate_partition_ari,
    partition_half_synapses,
    train_partition_gnn,
    assemble_fragments,
    build_fragment_graph,
    score_edge,
)

__all__ = [
    # CellGNN architecture & training
    "CellGNN",
    "CellGNNConfig",
    "EDGE_FEATURE_NAMES",
    "SynapseEdge",
    "SynapseGraph",
    "build_synapse_graph",
    "cell_graph_edge_train_step",
    "cell_graph_train_step",
    "cell_gnn_assembly",
    "build_synapse_chain_paths",
    "subdivide_synapse_graph",
    "infer_cells",
    "infer_cells_edge",
    "infer_cells_two_pass",
    "partition_from_embeddings",
    "train_cell_gnn",
    # Path-edge encoder
    "PathEdgeEncoder",
    "pad_path_sequences",
    # Path discrimination dataset
    "extract_cell_chains",
    "fetch_cave_edit_history",
    "save_edit_pairs_tsv",
    "generate_path_examples",
    "load_path_encoder",
    "train_path_encoder",
    # Edge-feature caches
    "load_seg_score_cache",
    "load_self_skeleton_archive",
    "load_skeleton_path_cache",
    "precompute_seg_scores_fast",
    "precompute_self_skeletons_for_cache",
    "precompute_skeleton_paths_for_cache",
    "save_seg_score_cache",
    "save_skeleton_path_cache",
    # evaluation
    "LineGraphMetrics",
    "evaluate",
    "evaluate_sampled",
    # probability / merge helpers
    "CandidateMerge",
    "logit_to_probability",
    "probability_to_log_odds",
    # Data stage: skeleton → Fragment
    "extract_fragments_for_region",
    "skeleton_to_fragment",
    # Represent stage: Fragment → tree-DNA embedding (path-sampling)
    "TreeDNAEncoder",
    "encode_fragments",
    "featurize_fragment",
    "sample_tree_paths",
    "train_dna_encoder",
    # Represent stage: DNA enrichment + ablation evaluation
    "build_synapse_dna_matrix",
    "evaluate_dna_auc",
    "spatial_proximity_scores",
    "synapse_pair_dna_scores",
    # Represent stage: GNN encoder (data-driven, no hand-crafted features)
    "SkeletonGNN",
    "encode_fragments_gnn",
    "fragment_to_tensors",
    "train_skeleton_gnn",
    # Assemble stage: global synapse graph + GNN
    "GlobalSynapseGraph",
    "assemble_neurons",
    "build_global_synapse_graph",
    "run_global_gnn",
    "train_global_gnn",
    # Assemble stage: half-synapse graph + partition GNN (Phase 2.1)
    "HalfSynapseGraph",
    "build_half_synapse_graph",
    "HalfSynapseGNN",
    "evaluate_partition_ari",
    "partition_half_synapses",
    "train_partition_gnn",
    # Assemble stage: fragment-proximity graph (endpoint-based stitching)
    "assemble_fragments",
    "build_fragment_graph",
    "score_edge",
]
