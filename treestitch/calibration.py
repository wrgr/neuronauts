"""Temperature scaling calibration for EdgePartitionGNN.

Fits a single scalar temperature T on held-out edges from a training graph.
After scaling, sigmoid((logit + bias) / T) is a calibrated 0-1 per-edge
probability: a predicted confidence of 0.8 should be correct ~80% of the time.

Per-observation confidence is the mean calibrated edge probability over all
edges touching that observation — a direct 0-1 score suitable for review queues.

Usage
-----
    from treestitch.calibration import fit_temperature, calibrated_obs_confidence
    from treestitch.calibration import reliability_diagram

    T = fit_temperature(model, graph, bias=-2.0, val_frac=0.2)
    conf = calibrated_obs_confidence(model, graph, T, bias=-2.0)
    diag = reliability_diagram(model, graph, T, bias=-2.0, n_bins=10)

References
----------
Guo et al. (2017) "On Calibration of Modern Neural Networks", ICML.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _edge_logits(model, graph, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return raw logits and co-neuron labels for all edges in graph."""
    import torch

    model.eval()
    node_feat = torch.tensor(graph.node_feat, dtype=torch.float32, device=device)
    edge_src  = torch.tensor(graph.edge_src,  dtype=torch.long,    device=device)
    edge_dst  = torch.tensor(graph.edge_dst,  dtype=torch.long,    device=device)
    edge_type = torch.tensor(graph.edge_type, dtype=torch.long,    device=device)
    edge_feat = torch.tensor(graph.edge_feat, dtype=torch.float32, device=device) \
                if graph.edge_feat is not None else None

    with torch.no_grad():
        logits = model(node_feat, edge_src, edge_dst, edge_type,
                       edge_feat=edge_feat).squeeze(-1).cpu().numpy()

    # co-neuron label: 1 if labels[src] == labels[dst] and both > 0
    labels = graph.labels
    y = ((labels[graph.edge_src] == labels[graph.edge_dst]) &
         (labels[graph.edge_src] > 0)).astype(np.float32)

    return logits, y


def fit_temperature(
    model,
    graph,
    *,
    bias: float = -2.0,
    val_frac: float = 0.2,
    n_steps: int = 200,
    lr: float = 0.05,
    device: str = "cpu",
) -> float:
    """Fit temperature T on a held-out fraction of graph edges.

    Minimises NLL of sigmoid((logit + bias) / T) on val edges via SGD.
    Returns T (scalar float). T > 1 means the model is overconfident;
    T < 1 means underconfident (rare in practice).

    Parameters
    ----------
    val_frac:
        Fraction of edges held out for temperature fitting (default 0.2).
        The remainder are treated as training edges (ignored here).
    """
    logits, y = _edge_logits(model, graph, device=device)
    shifted = logits + bias                      # same shift used at inference

    rng = np.random.default_rng(0)
    n_val = max(int(len(logits) * val_frac), 32)
    idx = rng.choice(len(logits), n_val, replace=False)
    val_logits = torch.tensor(shifted[idx], dtype=torch.float32)
    val_y      = torch.tensor(y[idx],       dtype=torch.float32)

    log_T = torch.tensor(0.0, requires_grad=True)   # T = exp(log_T) > 0
    opt   = torch.optim.LBFGS([log_T], lr=lr, max_iter=n_steps)

    def closure():
        opt.zero_grad()
        T   = torch.exp(log_T)
        nll = F.binary_cross_entropy_with_logits(val_logits / T, val_y)
        nll.backward()
        return nll

    opt.step(closure)
    T = float(torch.exp(log_T).item())
    return max(T, 1e-3)   # safety clamp


def calibrated_obs_confidence(
    model,
    graph,
    T: float,
    *,
    bias: float = -2.0,
    device: str = "cpu",
) -> np.ndarray:
    """Per-observation calibrated confidence (0-1).

    For each observation i, returns the mean calibrated edge probability over
    all edges touching i.  An observation with no edges gets confidence 0.5.

    Parameters
    ----------
    T:
        Temperature from `fit_temperature`.

    Returns
    -------
    conf : np.ndarray [N] float — per-observation calibrated confidence.
    """
    logits, _ = _edge_logits(model, graph, device=device)
    probs = 1.0 / (1.0 + np.exp(-((logits + bias) / T)))   # calibrated sigmoid

    N = graph.n_nodes
    sum_p = np.zeros(N, dtype=np.float64)
    count = np.zeros(N, dtype=np.int64)
    for e, (src, dst, p) in enumerate(
            zip(graph.edge_src, graph.edge_dst, probs)):
        sum_p[src] += p
        sum_p[dst] += p
        count[src] += 1
        count[dst] += 1

    conf = np.where(count > 0, sum_p / count, 0.5)
    return conf.astype(np.float32)


def reliability_diagram(
    model,
    graph,
    T: float,
    *,
    bias: float = -2.0,
    n_bins: int = 10,
    device: str = "cpu",
) -> dict:
    """Compute reliability diagram data (calibration plot).

    Returns a dict with:
        bin_centers : [n_bins] — centre of each confidence bin
        mean_conf   : [n_bins] — mean predicted confidence in each bin
        frac_pos    : [n_bins] — fraction of true positive edges in each bin
        counts      : [n_bins] — number of edges in each bin

    Plot `frac_pos` vs `mean_conf`; a perfectly calibrated model lies on
    the diagonal y = x.
    """
    logits, y = _edge_logits(model, graph, device=device)
    probs = 1.0 / (1.0 + np.exp(-((logits + bias) / T)))

    edges_lo = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (edges_lo[:-1] + edges_lo[1:])
    mean_conf = np.zeros(n_bins)
    frac_pos  = np.zeros(n_bins)
    counts    = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = edges_lo[b], edges_lo[b + 1]
        mask = (probs >= lo) & (probs < hi)
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() > 0:
            mean_conf[b] = probs[mask].mean()
            frac_pos[b]  = y[mask].mean()
            counts[b]    = mask.sum()
        else:
            mean_conf[b] = (lo + hi) / 2
            frac_pos[b]  = float('nan')

    return {
        "bin_centers": bin_centers,
        "mean_conf":   mean_conf,
        "frac_pos":    frac_pos,
        "counts":      counts,
        "T":           T,
    }


def expected_calibration_error(diag: dict) -> float:
    """Weighted mean absolute calibration error (ECE) from a reliability diagram."""
    total = diag["counts"].sum()
    if total == 0:
        return float("nan")
    valid = ~np.isnan(diag["frac_pos"])
    ece = (np.abs(diag["frac_pos"][valid] - diag["mean_conf"][valid])
           * diag["counts"][valid]).sum() / total
    return float(ece)
