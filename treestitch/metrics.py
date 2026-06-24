"""Aggregate partition metrics and human-readable dashboard.

The primary entry points are:

    metrics = compute_full_metrics(pred_labels, graph, region, root_label_map)
    print_dashboard(metrics, title="T3 pre cc_bias=-2")

``compute_full_metrics`` calls every available metric function in one shot and
returns a flat dict with consistent key names.  ``print_dashboard`` /
``format_dashboard`` render it as a compact terminal table showing cluster
quality, merge/split confusion, fragment completeness confusion, and
connectome edge accuracy — with raw counts alongside ratios so you can see
exactly how many merges/splits were right or wrong.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _population_stats(
    pred_labels: np.ndarray,
    graph,
    region,
    root_label_map: dict,
    *,
    ignore_label: int = 0,
) -> dict:
    """Compute neuron-count and synapse-coverage population statistics."""
    # --- fragment/neuron counts ------------------------------------------
    # v117 fragments present in the eval set
    frag_ids = graph.fragment_id
    v117_frags = np.unique(frag_ids[frag_ids != ignore_label])
    n_v117 = len(v117_frags)

    # GT v1718 neurons (all v1718 roots that appear in root_label_map for
    # the fragments we're evaluating)
    v1718_set: set = set()
    for fr in v117_frags.tolist():
        v1718_set.update(root_label_map.get(int(fr), set()))
    n_v1718 = len(v1718_set)

    # predicted clusters
    active_pred = pred_labels[pred_labels != ignore_label]
    n_pred = int(np.unique(active_pred).size) if len(active_pred) else 0

    # GT: how many v117 frags per v1718 neuron
    v1718_to_v117s: dict[int, list] = {}
    for fr in v117_frags.tolist():
        for v18 in root_label_map.get(int(fr), set()):
            v1718_to_v117s.setdefault(v18, []).append(fr)
    frags_per_neuron_gt = np.array([len(v) for v in v1718_to_v117s.values()],
                                   dtype=np.float32)
    merges_needed_gt = int(n_v117 - n_v1718)  # > 0 means there are fragments to merge

    # Pred: how many v117 frags per predicted cluster
    cluster_to_frags: dict[int, set] = {}
    for fid, clid in zip(frag_ids.tolist(), pred_labels.tolist()):
        if int(clid) == ignore_label:
            continue
        cluster_to_frags.setdefault(int(clid), set()).add(int(fid))
    frags_per_cluster_pred = np.array([len(v) for v in cluster_to_frags.values()],
                                      dtype=np.float32)
    merges_pred = int(n_v117 - n_pred)

    # Pred: how many clusters a single v1718 neuron was split into (splits per GT neuron)
    # For each v1718 neuron, collect the set of predicted clusters for its v117 frags
    v1718_to_clusters: dict[int, set] = {}
    frag_to_cluster = {}
    for fid, clid in zip(frag_ids.tolist(), pred_labels.tolist()):
        if int(clid) != ignore_label:
            frag_to_cluster[int(fid)] = int(clid)
    for v18, frgs in v1718_to_v117s.items():
        cls = {frag_to_cluster[f] for f in frgs if f in frag_to_cluster}
        if cls:
            v1718_to_clusters[v18] = cls
    clusters_per_v1718 = np.array([len(c) for c in v1718_to_clusters.values()],
                                  dtype=np.float32)

    # --- synapse coverage ------------------------------------------------
    pre_ids  = getattr(region, "pre_root_id",  None)
    post_ids = getattr(region, "post_root_id", None)

    n_total = len(pred_labels)
    n_pre_labelled  = int((pre_ids  > 0).sum()) if pre_ids  is not None else -1
    n_post_labelled = int((post_ids > 0).sum()) if post_ids is not None else -1
    n_both_labelled = (
        int(((pre_ids > 0) & (post_ids > 0)).sum())
        if pre_ids is not None and post_ids is not None else -1
    )
    n_pre_only  = n_pre_labelled  - n_both_labelled if n_both_labelled >= 0 else -1
    n_post_only = n_post_labelled - n_both_labelled if n_both_labelled >= 0 else -1
    n_neither   = n_total - n_pre_labelled - n_post_labelled + n_both_labelled \
                  if n_both_labelled >= 0 else -1

    def _safe_stat(arr):
        if len(arr) == 0:
            return float("nan"), float("nan"), float("nan")
        return float(arr.mean()), float(np.median(arr)), float(arr.max())

    fpn_mean, fpn_med, fpn_max = _safe_stat(frags_per_neuron_gt)
    fpc_mean, fpc_med, fpc_max = _safe_stat(frags_per_cluster_pred)
    cpv_mean, cpv_med, cpv_max = _safe_stat(clusters_per_v1718)

    return {
        # neuron-count triangle
        "pop_n_v117_frags":       n_v117,
        "pop_n_v1718_neurons":    n_v1718,
        "pop_n_pred_clusters":    n_pred,
        "pop_merges_needed_gt":   merges_needed_gt,
        "pop_merges_predicted":   merges_pred,
        # GT fragmentation per neuron
        "pop_frags_per_neuron_mean":   fpn_mean,
        "pop_frags_per_neuron_median": fpn_med,
        "pop_frags_per_neuron_max":    fpn_max,
        # pred: frags per cluster (merge aggressiveness)
        "pop_frags_per_cluster_mean":   fpc_mean,
        "pop_frags_per_cluster_median": fpc_med,
        "pop_frags_per_cluster_max":    fpc_max,
        # pred: clusters per GT neuron (split rate)
        "pop_clusters_per_neuron_mean":   cpv_mean,
        "pop_clusters_per_neuron_median": cpv_med,
        "pop_clusters_per_neuron_max":    cpv_max,
        # synapse coverage
        "pop_n_obs_total":       n_total,
        "pop_n_pre_labelled":    n_pre_labelled,
        "pop_n_post_labelled":   n_post_labelled,
        "pop_n_both_labelled":   n_both_labelled,
        "pop_n_pre_only":        n_pre_only,
        "pop_n_post_only":       n_post_only,
        "pop_n_neither_labelled":n_neither,
    }


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------

def compute_full_metrics(
    pred_labels: np.ndarray,
    graph,
    region,
    root_label_map: dict,
    *,
    side: str = "pre",
    min_syn: int = 1,
    ignore_label: int = 0,
) -> dict:
    """Compute all partition metrics in one call and return a flat dict.

    Parameters
    ----------
    pred_labels:
        [N] predicted cluster per observation node.
    graph:
        ``ObservationGraph`` — must have ``.labels``, ``.fragment_id``,
        ``.edge_type``, ``.edge_feat``, ``.src``, ``.dst``.
    region:
        ``Region`` — must have ``.pre_root_id``, ``.post_root_id``.
    root_label_map:
        ``{v117_root: set[v1718_root]}`` from world-building.
    side:
        Which side to use for connectome accuracy (``"pre"`` or ``"post"``).
    min_syn:
        Minimum synapse count to count as a connectome edge.
    ignore_label:
        Cluster label treated as "unassigned" (default 0).

    Returns
    -------
    Flat dict with keys grouped by prefix:

    * (no prefix) — cluster quality: ``ari``, ``homogeneity``, ``v_measure``,
      ``n_clusters_pred``, ``n_clusters_true``, ``n_nodes``
    * ``merge_*`` — edge merge/split: ``merge_precision``, ``merge_recall``,
      ``merge_f1``, ``over_merge_rate``, ``under_merge_rate``,
      ``tp_merges``, ``fp_merges``, ``fn_merges``, ``tn_splits``,
      ``n_merges_pred``, ``n_splits_pred``, ``n_true_merges``, ``n_edges_eval``
    * ``cmpl_*`` — fragment completeness: ``cmpl_precision``, ``cmpl_recall``,
      ``cmpl_f1``, ``cmpl_accuracy``, ``cmpl_n_complete_gt``,
      ``cmpl_n_fragments``, ``cmpl_tp``, ``cmpl_fp``, ``cmpl_fn``, ``cmpl_tn``
    * ``conn_*`` — connectome: ``conn_edge_f1``, ``conn_edge_precision``,
      ``conn_edge_recall``, ``conn_edge_f1_undir``, ``n_true_edges``,
      ``n_pred_edges``, ``synapse_attr_acc``, ``n_synapses_labelled``
    """
    from treestitch.partition import (
        evaluate_partition,
        merge_metrics,
        completeness_metrics,
        pred_fragment_completeness,
    )

    m: dict = {}

    # 0. Population counts
    _pop = _population_stats(pred_labels, graph, region, root_label_map,
                             ignore_label=ignore_label)
    m.update(_pop)

    # 1. Cluster quality
    ev = evaluate_partition(pred_labels, graph.labels, ignore_label=ignore_label)
    m.update(ev)

    # 2. Edge merge / split counts
    mm = merge_metrics(graph, pred_labels, ignore_label=ignore_label)
    m.update(mm)

    # 3. Fragment completeness
    pred_cmpl = pred_fragment_completeness(
        graph.fragment_id, pred_labels, ignore_label=-1)
    cm = completeness_metrics(root_label_map, pred_cmpl)
    m["cmpl_precision"] = cm["precision"]
    m["cmpl_recall"]    = cm["recall"]
    m["cmpl_f1"]        = cm["f1"]
    m["cmpl_accuracy"]  = cm["accuracy"]
    m["cmpl_n_complete_gt"] = cm["n_complete_gt"]
    m["cmpl_n_fragments"]   = cm["n_fragments"]
    m["cmpl_tp"] = cm.get("tp_complete", float("nan"))
    m["cmpl_fp"] = cm.get("fp_complete", float("nan"))
    m["cmpl_fn"] = cm.get("fn_complete", float("nan"))
    m["cmpl_tn"] = cm.get("tn_complete", float("nan"))

    # 4. Connectome accuracy (may be unavailable if region has no post labels)
    try:
        from treestitch.connectivity import connectome_accuracy
        conn = connectome_accuracy(
            pred_labels, region, min_syn=min_syn, ignore_label=ignore_label)
        m["conn_edge_precision"]      = conn.get("conn_edge_precision", float("nan"))
        m["conn_edge_recall"]         = conn.get("conn_edge_recall", float("nan"))
        m["conn_edge_f1"]             = conn.get("conn_edge_f1", float("nan"))
        m["conn_edge_precision_undir"]= conn.get("conn_edge_precision_undir", float("nan"))
        m["conn_edge_recall_undir"]   = conn.get("conn_edge_recall_undir", float("nan"))
        m["conn_edge_f1_undir"]       = conn.get("conn_edge_f1_undir", float("nan"))
        m["n_true_edges"]             = conn.get("n_true_edges", 0)
        m["n_pred_edges"]             = conn.get("n_pred_edges", 0)
        m["n_true_edges_undir"]       = conn.get("n_true_edges_undir", 0)
        m["n_pred_edges_undir"]       = conn.get("n_pred_edges_undir", 0)
        m["synapse_attr_acc"]         = conn.get("synapse_attr_acc", float("nan"))
        m["n_synapses_labelled"]      = conn.get("n_synapses_labelled", 0)
    except Exception:
        pass

    return m


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

def _fmt(v, pct: bool = False, digits: int = 3) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "  n/a"
    if pct:
        return f"{v * 100:5.1f}%"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _bar(label: str, width: int = 60) -> str:
    return "  " + ("─" * width)


def format_dashboard(
    metrics: dict,
    *,
    title: str = "",
    width: int = 66,
) -> str:
    """Return a formatted multi-section metrics dashboard string.

    Sections: Cluster quality · Merge/split confusion · Fragment
    completeness confusion · Connectome accuracy.
    """
    W = width
    lines: list[str] = []

    def rule(char="═"):
        lines.append("  " + char * W)

    def sec(label: str):
        lines.append(f"  {'─'*W}")
        lines.append(f"  {label}")

    def row(*cols, indent=4):
        lines.append(" " * indent + "  ".join(str(c) for c in cols))

    rule()
    if title:
        lines.append(f"  {title}")
        rule()

    # ── 0. Population counts ─────────────────────────────────────────────
    n_v117  = metrics.get("pop_n_v117_frags",    "?")
    n_v1718 = metrics.get("pop_n_v1718_neurons", "?")
    n_pred  = metrics.get("pop_n_pred_clusters", "?")
    mg_gt   = metrics.get("pop_merges_needed_gt",  0)
    mg_pr   = metrics.get("pop_merges_predicted",  0)

    fpn_m   = metrics.get("pop_frags_per_neuron_mean",   float("nan"))
    fpn_md  = metrics.get("pop_frags_per_neuron_median", float("nan"))
    fpn_mx  = metrics.get("pop_frags_per_neuron_max",    float("nan"))
    cpv_m   = metrics.get("pop_clusters_per_neuron_mean",   float("nan"))
    cpv_md  = metrics.get("pop_clusters_per_neuron_median", float("nan"))
    cpv_mx  = metrics.get("pop_clusters_per_neuron_max",    float("nan"))

    n_obs     = metrics.get("pop_n_obs_total",       0)
    n_pre_lb  = metrics.get("pop_n_pre_labelled",    -1)
    n_post_lb = metrics.get("pop_n_post_labelled",   -1)
    n_both    = metrics.get("pop_n_both_labelled",   -1)
    n_neither = metrics.get("pop_n_neither_labelled",-1)

    def _pct(n, d):
        return f"{n/d:.0%}" if d and d > 0 else "?"

    lines.append(f"  POPULATION  ({n_obs} observations)")
    lines.append(f"    fragments v117 (before edit):  {n_v117:6}  ← input objects")
    lines.append(f"    neurons   v1718 (GT):          {n_v1718:6}  ← target")
    lines.append(f"    clusters  predicted:            {n_pred:6}  ← model output")
    lines.append(f"    merges needed (GT):   {mg_gt:5}   "
                 f"merges predicted:  {mg_pr:5}")
    lines.append(f"    GT frags/neuron  mean {fpn_m:.2f}  median {fpn_md:.0f}  max {fpn_mx:.0f}")
    lines.append(f"    pred clusters/GT-neuron  "
                 f"mean {_fmt(cpv_m)}  median {cpv_md:.0f}  max {cpv_mx:.0f}  "
                 f"(splits)")

    if n_obs > 0 and n_pre_lb >= 0:
        lines.append(f"    observations linked to pre-neuron:   {n_pre_lb:6}  "
                     f"({_pct(n_pre_lb, n_obs)})  pre unlinked: {n_obs - n_pre_lb}")
    if n_obs > 0 and n_post_lb >= 0:
        lines.append(f"    observations linked to post-neuron:  {n_post_lb:6}  "
                     f"({_pct(n_post_lb, n_obs)})  post unlinked: {n_obs - n_post_lb}")
    if n_both >= 0:
        lines.append(f"    both sides linked: {n_both}  "
                     f"pre-only: {metrics.get('pop_n_pre_only', '?')}  "
                     f"post-only: {metrics.get('pop_n_post_only', '?')}  "
                     f"neither: {n_neither}")

    # ── 1. Cluster quality ───────────────────────────────────────────────
    n_nodes      = metrics.get("n_nodes", "?")
    n_pred_cl    = metrics.get("n_clusters_pred", "?")
    n_true_cl    = metrics.get("n_clusters_true", "?")
    ari          = _fmt(metrics.get("ari", float("nan")))
    hom          = _fmt(metrics.get("homogeneity", float("nan")))
    vm           = _fmt(metrics.get("v_measure", float("nan")))

    lines.append(f"  CLUSTER QUALITY"
                 f"   nodes {n_nodes}  clusters pred/true {n_pred_cl}/{n_true_cl}")
    lines.append(f"    ARI {ari}   homogeneity {hom}   v-measure {vm}")

    # ── 2. Merge / split confusion ───────────────────────────────────────
    n_edges   = metrics.get("n_edges_eval", 0)
    tp_m      = metrics.get("tp_merges",    0)
    fp_m      = metrics.get("fp_merges",    0)
    fn_m      = metrics.get("fn_merges",    0)
    tn_s      = metrics.get("tn_splits",    0)
    n_true_m  = metrics.get("n_true_merges",  0)
    n_pred_m  = metrics.get("n_merges_pred",  0)
    n_true_s  = n_edges - n_true_m
    n_pred_s  = n_edges - n_pred_m
    merge_p   = _fmt(metrics.get("merge_precision",   float("nan")))
    merge_r   = _fmt(metrics.get("merge_recall",      float("nan")))
    merge_f1  = _fmt(metrics.get("merge_f1",          float("nan")))
    over_r    = _fmt(metrics.get("over_merge_rate",   float("nan")))
    under_r   = _fmt(metrics.get("under_merge_rate",  float("nan")))
    gt_m_pct  = f"{n_true_m/n_edges*100:.1f}%" if n_edges else "?"

    sec(f"MERGE / SPLIT  ({n_edges} edges, {n_true_m} GT merges = {gt_m_pct})")
    lines.append(f"    {'':20s}  {'GT merge':>12}  {'GT split':>12}  {'total':>8}")
    lines.append(f"    {'pred merge':20s}  {tp_m:>8} (TP)  {fp_m:>8} (FP)  {n_pred_m:>8}")
    lines.append(f"    {'pred split':20s}  {fn_m:>8} (FN)  {tn_s:>8} (TN)  {n_pred_s:>8}")
    lines.append(f"    {'total':20s}  {n_true_m:>12}  {n_true_s:>12}  {n_edges:>8}")
    lines.append(f"    precision {merge_p}   recall {merge_r}   F1 {merge_f1}")
    lines.append(f"    over-merge {over_r} (FP/total)   under-merge {under_r} (FN/total)")

    # ── 3. Completeness confusion ────────────────────────────────────────
    n_frags   = metrics.get("cmpl_n_fragments",   0)
    n_gt_cmpl = metrics.get("cmpl_n_complete_gt", 0)
    tp_c      = metrics.get("cmpl_tp", float("nan"))
    fp_c      = metrics.get("cmpl_fp", float("nan"))
    fn_c      = metrics.get("cmpl_fn", float("nan"))
    tn_c      = metrics.get("cmpl_tn", float("nan"))
    cmpl_p    = _fmt(metrics.get("cmpl_precision", float("nan")))
    cmpl_r    = _fmt(metrics.get("cmpl_recall",    float("nan")))
    cmpl_f1   = _fmt(metrics.get("cmpl_f1",        float("nan")))
    cmpl_acc  = _fmt(metrics.get("cmpl_accuracy",  float("nan")))
    n_gt_inc  = n_frags - n_gt_cmpl

    def _ci(v):
        return int(v) if not (isinstance(v, float) and math.isnan(v)) else "n/a"

    n_pred_cmpl = (_ci(tp_c) + _ci(fp_c)) if isinstance(tp_c, int) and isinstance(fp_c, int) else "n/a"
    n_pred_inc  = (_ci(fn_c) + _ci(tn_c)) if isinstance(fn_c, int) and isinstance(tn_c, int) else "n/a"
    gt_cmpl_pct = f"{n_gt_cmpl/n_frags*100:.0f}%" if n_frags else "?"

    sec(f"COMPLETENESS  ({n_frags} fragments, {n_gt_cmpl} GT complete = {gt_cmpl_pct})")
    lines.append(f"    {'':22s}  {'GT complete':>11}  {'GT merged':>11}  {'total':>8}")
    lines.append(f"    {'pred complete':22s}  {_ci(tp_c):>7} (TP)  {_ci(fp_c):>7} (FP)  {n_pred_cmpl!s:>8}")
    lines.append(f"    {'pred needs merge':22s}  {_ci(fn_c):>7} (FN)  {_ci(tn_c):>7} (TN)  {n_pred_inc!s:>8}")
    lines.append(f"    {'total':22s}  {n_gt_cmpl:>11}  {n_gt_inc:>11}  {n_frags:>8}")
    lines.append(f"    precision {cmpl_p}   recall {cmpl_r}   F1 {cmpl_f1}   acc {cmpl_acc}")

    # ── 4. Connectome ────────────────────────────────────────────────────
    if "conn_edge_f1" in metrics:
        n_te  = metrics.get("n_true_edges", 0)
        n_pe  = metrics.get("n_pred_edges", 0)
        n_teu = metrics.get("n_true_edges_undir", 0)
        n_peu = metrics.get("n_pred_edges_undir", 0)
        cf1   = _fmt(metrics.get("conn_edge_f1",             float("nan")))
        cp    = _fmt(metrics.get("conn_edge_precision",      float("nan")))
        cr    = _fmt(metrics.get("conn_edge_recall",         float("nan")))
        uf1   = _fmt(metrics.get("conn_edge_f1_undir",       float("nan")))
        up    = _fmt(metrics.get("conn_edge_precision_undir",float("nan")))
        ur    = _fmt(metrics.get("conn_edge_recall_undir",   float("nan")))
        sa    = _fmt(metrics.get("synapse_attr_acc",         float("nan")))
        ns    = metrics.get("n_synapses_labelled", 0)

        sec("CONNECTOME  (pre→post directed)")
        lines.append(f"    directed:    true edges {n_te:4d}  pred {n_pe:4d}"
                     f"   F1 {cf1} (prec {cp} / rec {cr})")
        lines.append(f"    undirected:  true edges {n_teu:4d}  pred {n_peu:4d}"
                     f"   F1 {uf1} (prec {up} / rec {ur})")
        lines.append(f"    synapse attribution acc {sa}   ({ns} labelled synapses)")

    rule()
    return "\n".join(lines)


def print_dashboard(metrics: dict, *, title: str = "", width: int = 66) -> None:
    """Print a human-readable metrics dashboard to stdout."""
    print(format_dashboard(metrics, title=title, width=width))


def boundary_clip_stats(
    pos: np.ndarray,
    frag_ids: np.ndarray,
    bbox_nm: tuple,
    *,
    margin_nm: float = 10_000.0,
) -> dict:
    """Quantify how many fragments are clipped by bbox boundaries.

    A fragment is "boundary-clipped" if its L2 nodes are concentrated
    near the bbox edge — a sign that the neuron continues outside the box
    and the in-box nodes are only a partial view.  This is the root cause
    of "invisible merge paths": two v117 sub-fragments of the same v1718
    neuron whose connecting L2 path exits the bbox can never be merged by
    the GNN because there's no edge linking them inside the volume.

    Parameters
    ----------
    pos:
        [N, 3] float array of L2 node positions in nm.
    frag_ids:
        [N] int array mapping each node to its v117 fragment root.
    bbox_nm:
        ``((x0,y0,z0), (x1,y1,z1))`` in nm.
    margin_nm:
        Fragments whose centroid is within this distance of ANY bbox face
        are flagged as "boundary-adjacent".

    Returns
    -------
    dict with keys:

    * ``n_fragments``           — total fragments in bbox
    * ``n_boundary_adjacent``   — fragments with centroid within margin_nm of a face
    * ``frac_boundary``         — n_boundary_adjacent / n_fragments
    * ``n_single_node``         — fragments with only 1 L2 node (maximally clipped)
    * ``frac_single_node``      — n_single_node / n_fragments
    * ``median_nodes_per_frag`` — median L2 node count across all fragments
    * ``pct10_nodes_per_frag``  — 10th-percentile L2 node count
    * ``boundary_frag_ids``     — array of v117 root ids flagged as boundary-adjacent
    """
    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    faces = np.array([[x0, y0, z0], [x1, y1, z1]], dtype=np.float64)

    frags = np.unique(frag_ids)
    n_frags = len(frags)
    n_boundary = 0
    n_single = 0
    counts = []
    boundary_frag_ids = []

    for fr in frags:
        mask = frag_ids == fr
        pts = pos[mask]
        n = len(pts)
        counts.append(n)
        if n == 1:
            n_single += 1

        cx = pts.mean(0)
        # distance from centroid to nearest bbox face along each axis
        dist_lo = cx - np.array([x0, y0, z0])
        dist_hi = np.array([x1, y1, z1]) - cx
        min_dist = min(dist_lo.min(), dist_hi.min())
        if min_dist < margin_nm:
            n_boundary += 1
            boundary_frag_ids.append(int(fr))

    counts_arr = np.array(counts)
    return {
        "n_fragments":           n_frags,
        "n_boundary_adjacent":   n_boundary,
        "frac_boundary":         n_boundary / n_frags if n_frags else 0.0,
        "n_single_node":         n_single,
        "frac_single_node":      n_single / n_frags if n_frags else 0.0,
        "median_nodes_per_frag": float(np.median(counts_arr)) if n_frags else 0.0,
        "pct10_nodes_per_frag":  float(np.percentile(counts_arr, 10)) if n_frags else 0.0,
        "boundary_frag_ids":     np.array(boundary_frag_ids, dtype=np.int64),
    }


def print_boundary_report(
    pos: np.ndarray,
    frag_ids: np.ndarray,
    bbox_nm: tuple,
    root_label_map: dict | None = None,
    *,
    margin_nm: float = 10_000.0,
    title: str = "",
) -> None:
    """Print a boundary-clip diagnostic to stdout.

    If ``root_label_map`` is provided, also reports how many of the
    boundary-clipped fragments are GT-incomplete (i.e. boundary clipping
    may be hiding the merge path that would make them complete).
    """
    from treestitch.partition import fragment_completeness

    s = boundary_clip_stats(pos, frag_ids, bbox_nm, margin_nm=margin_nm)
    hdr = f"  BOUNDARY CLIP DIAGNOSTIC  {title}"
    print(f"\n{'─'*len(hdr)}")
    print(hdr)
    print(f"{'─'*len(hdr)}")
    print(f"    margin: {margin_nm/1000:.0f} µm from each bbox face")
    print(f"    total fragments:          {s['n_fragments']:5d}")
    print(f"    boundary-adjacent (≤{margin_nm/1000:.0f}µm): "
          f"{s['n_boundary_adjacent']:5d}  ({s['frac_boundary']:.1%})")
    print(f"    single-node fragments:    {s['n_single_node']:5d}  "
          f"({s['frac_single_node']:.1%})  ← maximally clipped")
    print(f"    median nodes / fragment:  {s['median_nodes_per_frag']:.0f}")
    print(f"    10th-pct nodes / frag:    {s['pct10_nodes_per_frag']:.0f}")

    if root_label_map is not None:
        gt_cmpl = fragment_completeness(root_label_map)
        bids = set(s["boundary_frag_ids"].tolist())
        # among boundary fragments, how many are GT-incomplete?
        n_bnd_incomplete = sum(
            1 for f in bids if f in gt_cmpl and not gt_cmpl[f])
        n_bnd_known = sum(1 for f in bids if f in gt_cmpl)
        print(f"    boundary frags GT-incomplete: "
              f"{n_bnd_incomplete}/{n_bnd_known}"
              f"  ({n_bnd_incomplete/n_bnd_known:.1%} of boundary-known)"
              if n_bnd_known else "    (no GT data for boundary frags)")
        # invisible merges: GT-incomplete frags that are ONLY on the boundary
        # (their merge partner may be outside the bbox)
        print(f"    → these {n_bnd_incomplete} boundary-clipped incomplete "
              f"fragments may have their merge path hidden outside the bbox")
    print()


__all__ = [
    "compute_full_metrics",
    "format_dashboard",
    "print_dashboard",
    "boundary_clip_stats",
    "print_boundary_report",
]
