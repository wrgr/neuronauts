"""Main experiment runner."""

import argparse
import time
from dataclasses import asdict

import numpy as np

from .agent import AgentConfig
from .fields import compute_exploration_field, compute_membrane_field, compute_membrane_vectors
from .fetch import (
    RealBoxSpec,
    SyntheticBenchmarkConfig,
    fetch_synapses,
    fetch_volume,
    load_cached_membrane,
    make_test_volume,
)
from .line_graph import LineGraphMetrics, evaluate
from .merge import ConnectivityGraph, MergedNeuron, cKDTree
from .vectorized import run_agents_vectorized

# ============================================================
# EXPERIMENT CONFIG -- autoresearch edits this block
# ============================================================

# Tuned baseline after local sweep:
# fixed_validation F1 ~= 0.558
# random-batch mean F1 ~= 0.468 over short confirmation loops
AGENT_CONFIG = AgentConfig(
    w_membrane_repulsion=2.0,
    w_wall_follow=0.3,
    w_exploration=1.0,
    w_synapse_attraction=1.0,
    w_inertia=0.6,
    max_speed=1.5,
    noise_scale=0.1,
    membrane_threshold=0.4,
    exploration_decay=0.15,
    exploration_decay_end=0.05,
    exploration_radius=2,
    synapse_capture_radius=1.30,
    spawn_jitter_scale=1.0,
    max_steps=450,
    respawn_on_boundary=True,
    kill_on_membrane=False,
)

# Swarm / merge controls
N_AGENTS = 700
SYNAPSE_SPAWN_FRACTION = 0.25
MERGE_RADIUS = 3.0
MERGE_OVERLAP_THRESHOLD = 0.65
ROLE_MERGE_MIN_SHARED_HITS = 1
MAX_SYNAPSES_PER_NEURON = 32
MIN_PATH_LENGTH = 5
WAYPOINTS_PER_AGENT = 20

# Field / assignment controls
MEMBRANE_SIGMA = 1.0
MEMBRANE_VECTOR_SIGMA = 1.5
POLARITY_CAPTURE_R = 3.5
PRE_POST_OWNER_TOPK = 3
OWNER_MARGIN = 0.0

# Synthetic benchmark policy
BENCHMARK_CONFIG = SyntheticBenchmarkConfig(
    shape=(96, 96, 96),
    n_synapses=30,
    membrane_planes=10,
    min_neuron_groups=6,
    max_neuron_groups=15,
    anchor_margin=12,
    pre_cluster_std=4.0,
    post_cluster_std=4.0,
)
BENCHMARK_CASES = 5
BENCHMARK_MODE = "random"

# Real-data benchmark policy
REAL_MIN_SYNAPSES = 50
REAL_BOXES = [
    RealBoxSpec(center_nm=(1_153_592, 793_592, 655_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(733_592, 513_592, 595_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(1_213_592, 333_592, 975_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(473_592, 433_592, 1_095_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(893_592, 973_592, 915_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(1_333_592, 633_592, 975_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(773_592, 533_592, 795_640), side_um=6.0, mip=2),
]
REAL_BOXES_PER_EVAL = 3
MEMBRANE_CACHE_DIR = "cache/membranes"

# ============================================================
# END CONFIG
# ============================================================


def _valid_agent_indices(path_arr: np.ndarray) -> np.ndarray:
    if path_arr.shape[1] <= MIN_PATH_LENGTH:
        return np.array([], dtype=np.int32)
    valid_mask = (path_arr[:, MIN_PATH_LENGTH, :] != 0).any(axis=1)
    return np.where(valid_mask)[0].astype(np.int32)


def _agent_points(path_arr: np.ndarray, agent_idx: int) -> np.ndarray:
    path = path_arr[agent_idx]
    return path[np.any(path != 0, axis=1)]


def _subsample_points(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    step_idx = np.linspace(0, len(path) - 1, min(WAYPOINTS_PER_AGENT, len(path)), dtype=int)
    return path[step_idx].astype(np.float32)


def _merge_role_groups(
    path_arr: np.ndarray,
    role_hits: np.ndarray,
    role_name: str,
    next_neuron_id: int,
) -> tuple[dict[int, MergedNeuron], dict[int, list[int]], dict[int, cKDTree], int]:
    role_agent_ids = np.where(role_hits.any(axis=1))[0].astype(np.int32)
    if len(role_agent_ids) == 0:
        return {}, {}, {}, next_neuron_id

    parent = {int(agent_id): int(agent_id) for agent_id in role_agent_ids.tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    sub_pts_list = []
    sub_labels = []
    for agent_id in role_agent_ids.tolist():
        pts = _subsample_points(_agent_points(path_arr, agent_id))
        if len(pts) == 0:
            continue
        sub_pts_list.append(pts)
        sub_labels.extend([agent_id] * len(pts))

    if not sub_pts_list:
        return {}, {}, {}, next_neuron_id

    sub_pts = np.vstack(sub_pts_list)
    sub_labels_arr = np.array(sub_labels, dtype=np.int32)
    pairs = cKDTree(sub_pts).query_pairs(r=MERGE_RADIUS, output_type="ndarray")
    for a, b in pairs:
        agent_a = int(sub_labels_arr[a])
        agent_b = int(sub_labels_arr[b])
        hits_a = role_hits[agent_a]
        hits_b = role_hits[agent_b]
        shared_count = int(np.count_nonzero(hits_a & hits_b))
        if shared_count < ROLE_MERGE_MIN_SHARED_HITS:
            continue
        overlap = shared_count / max(1, min(int(np.count_nonzero(hits_a)), int(np.count_nonzero(hits_b))))
        if overlap >= MERGE_OVERLAP_THRESHOLD:
            union(agent_a, agent_b)

    grouped_agents: dict[int, list[int]] = {}
    for agent_id in role_agent_ids.tolist():
        grouped_agents.setdefault(find(int(agent_id)), []).append(int(agent_id))

    neurons = {}
    synapse_owner = {}
    trees = {}
    for members in grouped_agents.values():
        pts = np.vstack([_agent_points(path_arr, agent_id) for agent_id in members if len(_agent_points(path_arr, agent_id)) > 0])
        synapse_indices = sorted(np.flatnonzero(role_hits[members, :].any(axis=0)).tolist())
        neuron_id = next_neuron_id
        next_neuron_id += 1
        neurons[neuron_id] = MergedNeuron(
            neuron_id=neuron_id,
            agent_ids=members,
            path_points=pts,
            synapse_indices=synapse_indices,
            role=role_name,
        )
        trees[neuron_id] = cKDTree(pts)
        for syn_idx in synapse_indices:
            synapse_owner.setdefault(int(syn_idx), []).append(neuron_id)

    return neurons, synapse_owner, trees, next_neuron_id


def _nearest_owner(
    syn_idx: int,
    pt: np.ndarray,
    owners: dict[int, list[int]],
    trees: dict[int, cKDTree],
) -> tuple[int | None, float]:
    candidates = []
    for neuron_id in owners.get(syn_idx, [])[:PRE_POST_OWNER_TOPK]:
        dist, _ = trees[neuron_id].query(pt)
        candidates.append((neuron_id, float(dist)))

    if not candidates:
        return None, float("inf")

    candidates.sort(key=lambda item: item[1])
    best_id, best_dist = candidates[0]
    if len(candidates) > 1 and (candidates[1][1] - best_dist) < OWNER_MARGIN:
        return None, float("inf")
    return best_id, best_dist


def _build_graph(
    path_arr: np.ndarray,
    path_lengths: np.ndarray,
    synapse_hits: np.ndarray,
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
) -> ConnectivityGraph:
    del path_lengths
    n_syn = len(pre_pts)
    valid_idx = _valid_agent_indices(path_arr)
    if len(valid_idx) == 0:
        return ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=list(range(n_syn)))

    role_hits = synapse_hits[valid_idx]
    pre_hits = np.zeros_like(synapse_hits[:, :n_syn], dtype=bool)
    post_hits = np.zeros_like(synapse_hits[:, n_syn:], dtype=bool)
    pre_hits[valid_idx] = role_hits[:, :n_syn]
    post_hits[valid_idx] = role_hits[:, n_syn:]

    pre_neurons, pre_owners, pre_trees, next_id = _merge_role_groups(path_arr, pre_hits, "pre", 0)
    post_neurons, post_owners, post_trees, next_id = _merge_role_groups(path_arr, post_hits, "post", next_id)
    del next_id

    neurons = {}
    neurons.update(pre_neurons)
    neurons.update(post_neurons)
    assigned_synapses = {neuron_id: [] for neuron_id in neurons}

    edges = []
    unresolved = []
    for syn_idx in range(n_syn):
        pre_neuron, pre_dist = _nearest_owner(syn_idx, pre_pts[syn_idx], pre_owners, pre_trees)
        post_neuron, post_dist = _nearest_owner(syn_idx, post_pts[syn_idx], post_owners, post_trees)

        if (
            pre_neuron is not None
            and post_neuron is not None
            and pre_dist < POLARITY_CAPTURE_R
            and post_dist < POLARITY_CAPTURE_R
            and len(assigned_synapses[pre_neuron]) < MAX_SYNAPSES_PER_NEURON
            and len(assigned_synapses[post_neuron]) < MAX_SYNAPSES_PER_NEURON
        ):
            edges.append((pre_neuron, post_neuron, syn_idx))
            assigned_synapses[pre_neuron].append(syn_idx)
            assigned_synapses[post_neuron].append(syn_idx)
        else:
            unresolved.append(syn_idx)

    for neuron_id, neuron in neurons.items():
        neuron.synapse_indices = sorted(set(assigned_synapses[neuron_id]))

    return ConnectivityGraph(neurons=neurons, edges=edges, unresolved_synapse_indices=unresolved)


def run(
    volume: np.ndarray,
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
    seed: int = 42,
    verbose: bool = True,
    membrane_field_override: np.ndarray | None = None,
) -> LineGraphMetrics:
    rng = np.random.default_rng(seed)
    volume_shape = np.array(volume.shape)
    all_syn_pts = np.vstack([pre_pts, post_pts])
    t0 = time.time()

    if verbose:
        print("Computing membrane fields...")
    if membrane_field_override is not None:
        mf = membrane_field_override.astype(np.float32, copy=False)
    else:
        mf = compute_membrane_field(volume, sigma=MEMBRANE_SIGMA)
    mv = compute_membrane_vectors(mf, sigma=MEMBRANE_VECTOR_SIGMA)
    ef = compute_exploration_field(volume.shape)
    if verbose:
        source = "cached membrane" if membrane_field_override is not None else "Sobel EM"
        print(f"  {time.time() - t0:.2f}s | {source} | vol={volume.shape} synapses={len(pre_pts)}")

    if verbose:
        print(f"Running {N_AGENTS} agents x {AGENT_CONFIG.max_steps} steps...")
    t1 = time.time()
    path_arr, synapse_hits, alive = run_agents_vectorized(
        volume_shape=volume_shape,
        n_agents=N_AGENTS,
        synapse_pts=all_syn_pts,
        membrane_field=mf,
        membrane_vectors=mv,
        exploration_field=ef,
        config=AGENT_CONFIG,
        rng=rng,
        synapse_fraction=SYNAPSE_SPAWN_FRACTION,
        verbose=verbose,
    )
    if verbose:
        hit_count = synapse_hits.any(axis=0).sum()
        print(f"  {time.time() - t1:.2f}s | {hit_count}/{len(all_syn_pts)} sites hit, {alive.sum()} alive")

    path_lengths = np.array([AGENT_CONFIG.max_steps] * N_AGENTS)
    t2 = time.time()
    graph = _build_graph(path_arr, path_lengths, synapse_hits, pre_pts, post_pts)
    if verbose:
        print(
            f"  merge+graph {time.time() - t2:.2f}s | "
            f"{len(graph.neurons)} neurons, {len(graph.edges)} edges, "
            f"{len(graph.unresolved_synapse_indices)} unresolved"
        )

    metrics = evaluate(graph, pre_root_ids, post_root_ids)
    if verbose:
        print(f"\nTotal: {time.time() - t0:.2f}s")
        print(f"Result: {metrics}")
    return metrics


def evaluate_synthetic_case(
    benchmark_config: SyntheticBenchmarkConfig,
    volume_seed: int | None = None,
    run_seed: int | None = None,
    verbose: bool = True,
) -> LineGraphMetrics:
    chunk, synapses = make_test_volume(config=benchmark_config, seed=volume_seed)
    return run(
        volume=chunk.data,
        pre_pts=synapses.pre_pt,
        post_pts=synapses.post_pt,
        pre_root_ids=synapses.pre_root_id,
        post_root_ids=synapses.post_root_id,
        seed=42 if run_seed is None else run_seed,
        verbose=verbose,
    )


def evaluate_synthetic_batch(
    benchmark_config: SyntheticBenchmarkConfig,
    cases: int,
    mode: str,
    base_seed: int | None = None,
    verbose: bool = True,
) -> tuple[LineGraphMetrics, list[dict[str, float | int | None]]]:
    if cases < 1:
        raise ValueError("cases must be >= 1")

    batch_rng = np.random.default_rng(base_seed)
    case_summaries = []
    metrics_list = []

    for case_idx in range(cases):
        if mode == "fixed_validation":
            volume_seed = case_idx
            run_seed = case_idx
        elif mode == "random":
            volume_seed = int(batch_rng.integers(0, 2**31 - 1))
            run_seed = int(batch_rng.integers(0, 2**31 - 1))
        else:
            raise ValueError(f"unsupported benchmark mode: {mode}")

        if verbose:
            print(f"\n--- Case {case_idx + 1}/{cases} volume_seed={volume_seed} run_seed={run_seed} ---")
        metrics = evaluate_synthetic_case(
            benchmark_config=benchmark_config,
            volume_seed=volume_seed,
            run_seed=run_seed,
            verbose=verbose,
        )
        metrics_list.append(metrics)
        case_summaries.append(
            {
                "case": case_idx + 1,
                "volume_seed": volume_seed,
                "run_seed": run_seed,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
            }
        )

    agg = LineGraphMetrics(
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        precision=float(np.mean([m.precision for m in metrics_list])),
        recall=float(np.mean([m.recall for m in metrics_list])),
        f1=float(np.mean([m.f1 for m in metrics_list])),
        n_true_edges=sum(m.n_true_edges for m in metrics_list),
        n_estimated_edges=sum(m.n_estimated_edges for m in metrics_list),
        n_synapses=sum(m.n_synapses for m in metrics_list),
    )
    return agg, case_summaries


def evaluate_real_box(
    box: RealBoxSpec,
    min_synapses: int = REAL_MIN_SYNAPSES,
    seed: int = 42,
    verbose: bool = True,
    membrane_source: str = "auto",
    membrane_cache_dir: str = MEMBRANE_CACHE_DIR,
) -> tuple[LineGraphMetrics | None, dict[str, int | float | tuple]]:
    synapses = fetch_synapses(box.bbox_nm, mip=box.mip)
    summary = {
        "center_nm": box.center_nm,
        "side_um": box.side_um,
        "mip": box.mip,
        "synapses": int(len(synapses.pre_pt)),
    }
    if len(synapses.pre_pt) < min_synapses:
        return None, summary

    chunk = fetch_volume(box.bbox_nm, mip=box.mip)
    membrane = None
    membrane_status = "sobel"
    if membrane_source in {"auto", "cache"}:
        membrane = load_cached_membrane(box, membrane_cache_dir)
        if membrane is not None:
            membrane_status = "cache"
        elif membrane_source == "cache":
            raise FileNotFoundError(f"missing cached membrane for {box.center_nm} in {membrane_cache_dir}")
    summary["membrane_source"] = membrane_status
    metrics = run(
        volume=chunk.data,
        pre_pts=synapses.pre_pt,
        post_pts=synapses.post_pt,
        pre_root_ids=synapses.pre_root_id,
        post_root_ids=synapses.post_root_id,
        seed=seed,
        verbose=verbose,
        membrane_field_override=membrane,
    )
    return metrics, summary


def evaluate_real_box_set(
    boxes: list[RealBoxSpec],
    boxes_per_eval: int,
    min_synapses: int = REAL_MIN_SYNAPSES,
    seed: int = 42,
    verbose: bool = True,
    membrane_source: str = "auto",
    membrane_cache_dir: str = MEMBRANE_CACHE_DIR,
) -> tuple[LineGraphMetrics, list[dict[str, int | float | tuple]]]:
    summaries = []
    metrics_list = []

    for box in boxes:
        metrics, summary = evaluate_real_box(
            box=box,
            min_synapses=min_synapses,
            seed=seed,
            verbose=verbose,
            membrane_source=membrane_source,
            membrane_cache_dir=membrane_cache_dir,
        )
        if metrics is None:
            summaries.append({**summary, "status": "skip_low_synapses"})
            continue
        summaries.append({**summary, "status": "used", "f1": metrics.f1})
        metrics_list.append(metrics)
        if len(metrics_list) >= boxes_per_eval:
            break

    if len(metrics_list) < boxes_per_eval:
        raise RuntimeError(
            f"only found {len(metrics_list)} real boxes with >= {min_synapses} synapses; "
            f"need {boxes_per_eval}"
        )

    agg = LineGraphMetrics(
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        precision=float(np.mean([m.precision for m in metrics_list])),
        recall=float(np.mean([m.recall for m in metrics_list])),
        f1=float(np.mean([m.f1 for m in metrics_list])),
        n_true_edges=sum(m.n_true_edges for m in metrics_list),
        n_estimated_edges=sum(m.n_estimated_edges for m in metrics_list),
        n_synapses=sum(m.n_synapses for m in metrics_list),
    )
    return agg, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Whether to evaluate on synthetic benchmark cases or real MICrONS boxes.",
    )
    parser.add_argument("--cases", type=int, default=BENCHMARK_CASES, help="Synthetic cases per evaluation.")
    parser.add_argument(
        "--benchmark-mode",
        choices=["random", "fixed_validation"],
        default=BENCHMARK_MODE,
        help="Whether to sample fresh synthetic cases or use a fixed validation set.",
    )
    parser.add_argument("--eval-seed", type=int, default=None, help="Optional seed for reproducible batch sampling.")
    parser.add_argument("--volume-seed", type=int, default=None, help="Optional debug override for a single synthetic case.")
    parser.add_argument("--run-seed", type=int, default=None, help="Optional debug override for a single agent simulation.")
    parser.add_argument("--real-boxes-per-eval", type=int, default=REAL_BOXES_PER_EVAL, help="Real boxes to average per evaluation.")
    parser.add_argument("--real-min-synapses", type=int, default=REAL_MIN_SYNAPSES, help="Minimum synapses required for a real box to be used.")
    parser.add_argument(
        "--membrane-source",
        choices=["auto", "cache", "sobel"],
        default="auto",
        help="Use cached learned membranes when available, require them, or always use Sobel.",
    )
    parser.add_argument(
        "--membrane-cache-dir",
        default=MEMBRANE_CACHE_DIR,
        help="Directory containing cached membrane .npy volumes for real boxes.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step benchmark logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_mode == "synthetic":
        print("=== Synthetic benchmark ===")
        print(
            f"mode={args.benchmark_mode} cases={args.cases} "
            f"eval_seed={args.eval_seed} benchmark_config={asdict(BENCHMARK_CONFIG)}"
        )

        if args.volume_seed is not None or args.run_seed is not None:
            print(f"debug_replay volume_seed={args.volume_seed} run_seed={args.run_seed}")
            metrics = evaluate_synthetic_case(
                benchmark_config=BENCHMARK_CONFIG,
                volume_seed=args.volume_seed,
                run_seed=args.run_seed,
                verbose=not args.quiet,
            )
            print(f"Result: {metrics}")
            print(f"\nval_f1 = {metrics.f1:.4f}")
            return

        metrics, case_summaries = evaluate_synthetic_batch(
            benchmark_config=BENCHMARK_CONFIG,
            cases=args.cases,
            mode=args.benchmark_mode,
            base_seed=args.eval_seed,
            verbose=not args.quiet,
        )
        for summary in case_summaries:
            print(
                "case_result "
                f"case={summary['case']} "
                f"volume_seed={summary['volume_seed']} "
                f"run_seed={summary['run_seed']} "
                f"f1={summary['f1']:.4f} "
                f"p={summary['precision']:.3f} "
                f"r={summary['recall']:.3f}"
            )
        print(f"Result: {metrics}")
        print(f"\nval_f1 = {metrics.f1:.4f}")
        return

    print("=== Real MICrONS benchmark ===")
    print(
        f"boxes_per_eval={args.real_boxes_per_eval} "
        f"real_min_synapses={args.real_min_synapses} "
        f"candidate_boxes={len(REAL_BOXES)}"
    )
    metrics, box_summaries = evaluate_real_box_set(
        boxes=REAL_BOXES,
        boxes_per_eval=args.real_boxes_per_eval,
        min_synapses=args.real_min_synapses,
        seed=42 if args.run_seed is None else args.run_seed,
        verbose=not args.quiet,
        membrane_source=args.membrane_source,
        membrane_cache_dir=args.membrane_cache_dir,
    )
    for idx, summary in enumerate(box_summaries, start=1):
        print(
            "box_result "
            f"idx={idx} center_nm={summary['center_nm']} side_um={summary['side_um']} "
            f"synapses={summary['synapses']} status={summary['status']} membrane={summary.get('membrane_source', 'n/a')}"
            + (f" f1={summary['f1']:.4f}" if 'f1' in summary else "")
        )
    print(f"Result: {metrics}")
    print(f"\nval_f1 = {metrics.f1:.4f}")


if __name__ == "__main__":
    main()
