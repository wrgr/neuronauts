"""Dataset helpers for box-level assembly ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .line_graph import LineGraphMetrics, build_estimated_line_graph
from .merge import ConnectivityGraph

ASSEMBLY_FEATURE_NAMES = (
    "merge_threshold",
    "beam_width",
    "n_neurons",
    "n_edges",
    "n_unresolved",
    "n_resolved",
    "unresolved_fraction",
    "mean_synapses_per_neuron",
    "max_synapses_per_neuron",
    "edges_per_neuron",
    "estimated_line_edges",
)


@dataclass(frozen=True)
class HypothesisExample:
    box_id: str
    hypothesis_id: str
    features: np.ndarray
    f1: float
    is_best: int


def hypothesis_features(
    graph: ConnectivityGraph,
    *,
    merge_threshold: float,
    beam_width: int,
    n_synapses: int,
) -> np.ndarray:
    synapse_counts = [len(neuron.synapse_indices) for neuron in graph.neurons.values()]
    estimated_edges = len(build_estimated_line_graph(graph, n_synapses=n_synapses))
    n_unresolved = len(graph.unresolved_synapse_indices)
    n_resolved = max(0, int(n_synapses) - n_unresolved)
    n_neurons = len(graph.neurons)
    n_edges = len(graph.edges)
    return np.array(
        [
            float(merge_threshold),
            float(beam_width),
            float(n_neurons),
            float(n_edges),
            float(n_unresolved),
            float(n_resolved),
            float(n_unresolved / max(1, n_synapses)),
            float(np.mean(synapse_counts)) if synapse_counts else 0.0,
            float(np.max(synapse_counts)) if synapse_counts else 0.0,
            float(n_edges / max(1, n_neurons)),
            float(estimated_edges),
        ],
        dtype=np.float32,
    )


def build_hypothesis_examples(
    box_id: str,
    hypotheses: list[tuple[str, np.ndarray, LineGraphMetrics]],
) -> list[HypothesisExample]:
    if not hypotheses:
        return []
    best_f1 = max(metrics.f1 for _, _, metrics in hypotheses)
    return [
        HypothesisExample(
            box_id=box_id,
            hypothesis_id=hypothesis_id,
            features=features.astype(np.float32, copy=False),
            f1=float(metrics.f1),
            is_best=int(metrics.f1 >= best_f1 - 1e-9),
        )
        for hypothesis_id, features, metrics in hypotheses
    ]


def examples_to_arrays(examples: list[HypothesisExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        return (
            np.zeros((0, len(ASSEMBLY_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    x = np.stack([example.features for example in examples], axis=0).astype(np.float32)
    y_f1 = np.array([example.f1 for example in examples], dtype=np.float32)
    y_best = np.array([example.is_best for example in examples], dtype=np.int64)
    return x, y_f1, y_best


def save_hypothesis_examples_npz(path: str | Path, examples: list[HypothesisExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y_f1, y_best = examples_to_arrays(examples)
    box_ids = np.array([example.box_id for example in examples], dtype=object)
    hypothesis_ids = np.array([example.hypothesis_id for example in examples], dtype=object)
    np.savez(
        path,
        x=x,
        y_f1=y_f1,
        y_best=y_best,
        box_ids=box_ids,
        hypothesis_ids=hypothesis_ids,
        feature_names=np.array(ASSEMBLY_FEATURE_NAMES, dtype=object),
    )
