"""Fail-closed utilities for soma-seeded assembly of real dense fragments.

There is deliberately no synthetic-data fallback in this module. Callers must
provide real CAVE v117 ids, real observations, exact soma containment, and a
trained grammar model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np


@dataclass(frozen=True)
class Fragment:
    root_id: int
    vertices_nm: np.ndarray
    edges: np.ndarray
    soma_count: int = 0
    gt_label: int = 0
    gt_purity: float = 0.0

    @property
    def length_nm(self) -> float:
        if len(self.edges) == 0:
            return 0.0
        delta = self.vertices_nm[self.edges[:, 0]] - self.vertices_nm[self.edges[:, 1]]
        return float(np.linalg.norm(delta, axis=1).sum())


@dataclass(frozen=True)
class CandidateEdge:
    left_root: int
    right_root: int
    distance_nm: float
    grammar_logit: float
    tangent_alignment: float
    score: float


def true_merge_pair_count(fragments: Iterable[Fragment]) -> int:
    """Count labeled v117 fragment pairs that should join at the target version."""
    counts = Counter(
        int(fragment.gt_label)
        for fragment in fragments
        if int(fragment.gt_label) > 0
    )
    return int(sum(count * (count - 1) // 2 for count in counts.values()))


def assert_real_root_ids(root_ids: Iterable[int]) -> None:
    ids = list(root_ids)
    if not ids:
        raise ValueError("dense query returned no roots")
    if any(not isinstance(root, (int, np.integer)) for root in ids):
        raise TypeError("root ids must be integer CAVE ids")
    if any(int(root) <= 0 for root in ids):
        raise ValueError("root ids must be positive")
    if max(int(root) for root in ids) < (1 << 40):
        raise ValueError("ids do not look like CAVE root ids; refusing synthetic input")


def skeleton_from_observed_points(
    points_nm: np.ndarray,
    max_points: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic minimum-spanning path graph from real observations."""
    vertices = np.asarray(points_nm, dtype=np.float32)
    if len(vertices) < 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64)
    if len(vertices) > max_points:
        take = np.linspace(0, len(vertices) - 1, max_points, dtype=int)
        vertices = vertices[take]

    from scipy.spatial import cKDTree

    tree = cKDTree(vertices)
    k = min(6, len(vertices) - 1)
    distances, neighbors = tree.query(vertices, k=k + 1)
    weighted: dict[tuple[int, int], float] = {}
    for i in range(len(vertices)):
        for slot in range(1, k + 1):
            j = int(neighbors[i, slot])
            key = (min(i, j), max(i, j))
            weighted[key] = min(weighted.get(key, float("inf")), float(distances[i, slot]))

    parent = list(range(len(vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: list[tuple[int, int]] = []
    for (left, right), _distance in sorted(weighted.items(), key=lambda item: item[1]):
        p_left, p_right = find(left), find(right)
        if p_left == p_right:
            continue
        parent[p_left] = p_right
        edges.append((left, right))
        if len(edges) == len(vertices) - 1:
            break
    return vertices, np.asarray(edges, dtype=np.int64)


def endpoint_paths(
    fragment: Fragment,
    max_points: int = 64,
    max_paths: int = 8,
) -> list[np.ndarray]:
    """Return each leaf's longest inward tree path, beginning at the leaf."""
    n_vertices = len(fragment.vertices_nm)
    if n_vertices < 2 or len(fragment.edges) == 0:
        return []
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_vertices)]
    for left, right in fragment.edges:
        weight = float(np.linalg.norm(
            fragment.vertices_nm[left] - fragment.vertices_nm[right]))
        adjacency[int(left)].append((int(right), weight))
        adjacency[int(right)].append((int(left), weight))
    leaves = [i for i, neighbors in enumerate(adjacency) if len(neighbors) <= 1]
    if not leaves:
        leaves = list(range(n_vertices))

    paths: list[np.ndarray] = []
    for start in leaves:
        distance = np.full(n_vertices, -np.inf)
        parent = np.full(n_vertices, -1, dtype=np.int64)
        distance[start] = 0.0
        stack = [(start, -1)]
        while stack:
            current, previous = stack.pop()
            for neighbor, weight in adjacency[current]:
                if neighbor == previous:
                    continue
                parent[neighbor] = current
                distance[neighbor] = distance[current] + weight
                stack.append((neighbor, current))
        end = int(np.argmax(distance))
        reverse_path = [end]
        while reverse_path[-1] != start and parent[reverse_path[-1]] >= 0:
            reverse_path.append(int(parent[reverse_path[-1]]))
        indices = list(reversed(reverse_path))[:max_points]
        if len(indices) >= 2:
            paths.append(fragment.vertices_nm[indices])
    paths.sort(key=lambda path: -float(
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
    return paths[:max_paths]


def build_candidate_edges_batched(
    fragments: list[Fragment],
    model,
    featurize: Callable[[np.ndarray], np.ndarray],
    *,
    max_distance_nm: float = 2500.0,
    batch_size: int = 256,
) -> list[CandidateEdge]:
    """Score spatially adjacent real paths using batched learned inference."""
    import torch
    from scipy.spatial import cKDTree

    endpoint_records: list[tuple[int, np.ndarray]] = []
    for fragment in fragments:
        for path in endpoint_paths(fragment):
            if len(path) >= 3:
                endpoint_records.append((fragment.root_id, path))
    if not endpoint_records:
        return []

    xyz = np.stack([path[0] for _, path in endpoint_records])
    raw_pairs = cKDTree(xyz).query_pairs(float(max_distance_nm), output_type="set")
    pairs = [(i, j) for i, j in sorted(raw_pairs)
             if endpoint_records[i][0] != endpoint_records[j][0]]
    if not pairs:
        return []

    max_len = int(getattr(model.path_encoder, "max_len", 512))
    sequences: list[np.ndarray] = []
    for _root, path in endpoint_records:
        sequences.extend((featurize(path), featurize(path[::-1])))

    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = [seq[:max_len] for seq in sequences[start:start + batch_size]]
            length = max(len(seq) for seq in chunk)
            feature_dim = int(chunk[0].shape[1])
            x = np.zeros((len(chunk), length, feature_dim), dtype=np.float32)
            mask = np.ones((len(chunk), length), dtype=bool)
            for row, sequence in enumerate(chunk):
                x[row, :len(sequence)] = sequence
                mask[row, :len(sequence)] = False
            embeddings.append(model.path_encoder(
                torch.from_numpy(x), torch.from_numpy(mask)).cpu())
        embedding = torch.cat(embeddings, dim=0)

        logits: list[float] = []
        for start in range(0, len(pairs), batch_size * 4):
            chunk = pairs[start:start + batch_size * 4]
            left_reverse = embedding[torch.tensor([2 * i + 1 for i, _ in chunk])]
            right_forward = embedding[torch.tensor([2 * j for _, j in chunk])]
            right_reverse = embedding[torch.tensor([2 * j + 1 for _, j in chunk])]
            left_forward = embedding[torch.tensor([2 * i for i, _ in chunk])]
            score = 0.5 * (
                model.merge_scorer(left_reverse, right_forward)
                + model.merge_scorer(right_reverse, left_forward)
            )
            logits.extend(float(value) for value in score.cpu().numpy())

    best: dict[tuple[int, int], CandidateEdge] = {}
    for (i, j), grammar_logit in zip(pairs, logits):
        left_root, left = endpoint_records[i]
        right_root, right = endpoint_records[j]
        left_tangent = left[0] - left[1]
        right_tangent = right[1] - right[0]
        denominator = float(
            np.linalg.norm(left_tangent) * np.linalg.norm(right_tangent))
        alignment = (float(np.dot(left_tangent, right_tangent) / denominator)
                     if denominator > 0 else -1.0)
        distance = float(np.linalg.norm(left[0] - right[0]))
        combined = (grammar_logit + 1.5 * alignment
                    - distance / max(float(max_distance_nm), 1.0))
        key = (min(left_root, right_root), max(left_root, right_root))
        edge = CandidateEdge(
            left_root, right_root, distance, grammar_logit, alignment, combined)
        if key not in best or edge.score > best[key].score:
            best[key] = edge
    return sorted(
        best.values(),
        key=lambda edge: (-edge.score, edge.distance_nm, edge.left_root, edge.right_root),
    )


def soma_seeded_assemble(
    fragments: list[Fragment],
    edges: list[CandidateEdge],
    *,
    min_score: float = 0.0,
) -> dict[int, int]:
    """Grow competing trees from exact soma roots while enforcing <=1 soma."""
    by_root = {fragment.root_id: fragment for fragment in fragments}
    seeds = sorted(root for root, fragment in by_root.items()
                   if fragment.soma_count == 1)
    if not seeds:
        raise ValueError("no exact soma seed in candidate pool")
    if any(fragment.soma_count > 1 for fragment in fragments):
        raise ValueError("a v117 root contains multiple somata; atomic growth is invalid")

    incident: dict[int, list[CandidateEdge]] = {root: [] for root in by_root}
    for edge in edges:
        if edge.left_root in incident and edge.right_root in incident:
            incident[edge.left_root].append(edge)
            incident[edge.right_root].append(edge)

    owner = {seed: index for index, seed in enumerate(seeds)}
    queue: list[tuple[float, float, int, int, CandidateEdge]] = []
    for seed in seeds:
        for edge in incident[seed]:
            other = edge.right_root if edge.left_root == seed else edge.left_root
            heappush(queue, (-edge.score, edge.distance_nm, seed, other, edge))

    while queue:
        negative_score, _distance, source, target, _edge = heappop(queue)
        if -negative_score < min_score or source not in owner or target in owner:
            continue
        if by_root[target].soma_count:
            continue
        owner[target] = owner[source]
        for next_edge in incident[target]:
            other = (next_edge.right_root
                     if next_edge.left_root == target else next_edge.left_root)
            if other not in owner:
                heappush(queue, (
                    -next_edge.score, next_edge.distance_nm, target, other, next_edge))

    next_cluster = len(seeds)
    prediction: dict[int, int] = {}
    for root in sorted(by_root):
        if root in owner:
            prediction[root] = owner[root]
        else:
            prediction[root] = next_cluster
            next_cluster += 1
    return prediction


def partition_metrics(
    fragments: list[Fragment],
    prediction: dict[int, int],
) -> dict[str, float | int]:
    """Fragment ARI/pair metrics and merge-aware expected run length."""
    valid = [fragment for fragment in fragments if fragment.gt_label > 0]
    if not valid:
        return {"ari": 0.0, "merge_precision": 0.0, "merge_recall": 0.0,
                "erl_um": 0.0, "n_labeled_fragments": 0}
    truth = np.asarray([fragment.gt_label for fragment in valid], dtype=np.int64)
    predicted = np.asarray(
        [prediction[fragment.root_id] for fragment in valid], dtype=np.int64)

    def choose2(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2))

    _, truth_inverse = np.unique(truth, return_inverse=True)
    _, pred_inverse = np.unique(predicted, return_inverse=True)
    table = np.zeros((truth_inverse.max() + 1, pred_inverse.max() + 1), dtype=np.int64)
    np.add.at(table, (truth_inverse, pred_inverse), 1)
    true_positive = choose2(table)
    true_pairs = choose2(table.sum(axis=1))
    predicted_pairs = choose2(table.sum(axis=0))
    precision = true_positive / predicted_pairs if predicted_pairs else 1.0
    recall = true_positive / true_pairs if true_pairs else 1.0
    total_pairs = len(valid) * (len(valid) - 1) / 2
    expected = true_pairs * predicted_pairs / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (true_pairs + predicted_pairs)
    ari = ((true_positive - expected) / (maximum - expected)
           if maximum != expected else 1.0)

    lengths = np.asarray([max(fragment.length_nm, 1.0) for fragment in valid])
    total_length = float(lengths.sum())
    erl_numerator = 0.0
    for cluster in np.unique(predicted):
        mask = predicted == cluster
        if len(np.unique(truth[mask])) == 1:
            run_length = float(lengths[mask].sum())
            erl_numerator += run_length * run_length
    erl_nm = erl_numerator / total_length if total_length else 0.0
    return {
        "ari": float(ari),
        "merge_precision": float(precision),
        "merge_recall": float(recall),
        "erl_um": float(erl_nm / 1000.0),
        "n_labeled_fragments": len(valid),
    }


def single_soma_compliance(
    fragments: list[Fragment],
    prediction: dict[int, int],
) -> dict[str, float | int]:
    counts: dict[int, int] = {}
    for fragment in fragments:
        cluster = prediction[fragment.root_id]
        counts[cluster] = counts.get(cluster, 0) + fragment.soma_count
    soma_clusters = [count for count in counts.values() if count > 0]
    return {
        "n_pred_clusters": len(counts),
        "n_soma_clusters": len(soma_clusters),
        "single_soma_compliance": (
            float(sum(count == 1 for count in soma_clusters) / len(soma_clusters))
            if soma_clusters else 0.0
        ),
        "multi_soma_clusters": int(sum(count > 1 for count in soma_clusters)),
    }
