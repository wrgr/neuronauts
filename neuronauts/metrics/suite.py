"""One call that runs every applicable metric on a predicted partition.

:func:`evaluate_partition_suite` takes the per-item prediction and truth and
whatever side information the experiment has, and returns one flat dict whose
sections switch on by what was passed:

======================  ==========================================  ===============
inputs                  section                                     key prefix
======================  ==========================================  ===============
``pred, true``          partition (ARI, pair P/R, VI, purity)       (none)
``+ weights``           cable-weighted pairs, ERL                   ``wpair_``, ``erl``
``+ src, dst``          candidate-edge merge/split confusion        ``merge_``, ``*_merges``
``+ fragment_id``       frankenmerge separation, naive baseline     ``fk_``, ``naive_``
``+ root_label_map``    fragment completeness                       ``cmpl_``
``+ true_post``         connectome edge F1, line-graph F1           ``conn_``, ``lg_``
======================  ==========================================  ===============

Every key is documented in :data:`neuronauts.metrics.report.KEY_DOCS`; a test
keeps that registry and this function in sync. Undefined ratios are NaN.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .completeness import completeness_metrics, fragment_completeness, pred_fragment_completeness
from .connectome import connectome_metrics
from .edges import edge_merge_metrics
from .frankenmerge import frankenmerge_metrics
from .line_graph import evaluate_suite as line_graph_suite
from .partition import partition_metrics


def _singletonize(labels, ignore) -> np.ndarray:
    """Give every ``ignore`` item a unique label so it pairs with nothing."""
    labels = np.asarray(labels)
    if ignore is None:
        return labels
    unknown = labels == ignore
    if not unknown.any():
        return labels
    _, inv = np.unique(labels, return_inverse=True)
    inv = inv.reshape(-1).astype(np.int64)
    inv[unknown] = inv.max() + 1 + np.arange(int(unknown.sum()), dtype=np.int64)
    return inv


def evaluate_partition_suite(
    pred,
    true,
    *,
    ignore=0,
    pred_ignore=None,
    weights: Optional[np.ndarray] = None,
    src: Optional[np.ndarray] = None,
    dst: Optional[np.ndarray] = None,
    same_fragment: Optional[np.ndarray] = None,
    fragment_id: Optional[np.ndarray] = None,
    root_label_map: Optional[dict] = None,
    true_post: Optional[np.ndarray] = None,
    pred_post: Optional[np.ndarray] = None,
    min_syn: int = 1,
    naive_baseline: bool = True,
) -> dict:
    """Run every metric the inputs support and return one flat dict.

    Parameters
    ----------
    pred, true:
        ``[N]`` predicted cluster and true label per item (synapse observation,
        fragment, atom).
    ignore:
        True-label value meaning unknown (default 0). Such items are dropped
        from partition, edge, frankenmerge and connectome metrics and made
        singletons in the line graph.
    pred_ignore:
        Predicted value meaning abstained; abstained items are singletons.
    weights:
        ``[N]`` non-negative per-item weights (cable length in µm) for the
        weighted pair block.
    src, dst:
        ``[E]`` candidate-edge endpoints for the edge-level block.
    same_fragment:
        ``[E]`` bool: edge endpoints share an input object (frankenmerge cut
        candidates). Only used with ``src, dst``.
    fragment_id:
        ``[N]`` input-object id per item. Enables the frankenmerge block, the
        naive "no edits" baseline and, with ``root_label_map``, completeness.
    root_label_map:
        ``{fragment_id: set(true_labels)}`` ground truth for completeness.
    true_post, pred_post:
        ``[N]`` true post-neuron (and optional predicted post-side cluster)
        per synapse for the connectome and line-graph blocks. ``pred`` and
        ``true`` are then the pre side.
    min_syn:
        Minimum synapses for a directed connection to count as a connectome
        edge.
    naive_baseline:
        Also score ``fragment_id`` itself as the prediction (``naive_*``), the
        do-nothing reference every model must beat.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    m: dict = {}

    m.update(partition_metrics(pred, true, ignore=ignore, pred_ignore=pred_ignore,
                               weights=weights))

    if src is not None or dst is not None:
        if src is None or dst is None:
            raise ValueError("src and dst must be given together")
        abstain = (pred == pred_ignore) if pred_ignore is not None else None
        m.update(edge_merge_metrics(src, dst, pred, true, ignore=ignore,
                                    same_fragment=same_fragment, abstain=abstain))

    if fragment_id is not None:
        fragment_id = np.asarray(fragment_id)
        fk = frankenmerge_metrics(pred, true, fragment_id, ignore=ignore, pred_ignore=pred_ignore)
        fk.pop("fk_parents")
        m.update(fk)

        if naive_baseline:
            naive = partition_metrics(fragment_id, true, ignore=ignore, weights=weights)
            for k in ("ari", "pair_precision", "pair_recall", "pair_f1", "vi_split", "vi_merge"):
                m[f"naive_{k}"] = naive[k]
            if weights is not None:
                for k in ("wpair_precision", "wpair_recall", "wpair_f1", "erl"):
                    m[f"naive_{k}"] = naive[k]

        if root_label_map is not None:
            pred_cmpl = pred_fragment_completeness(fragment_id, pred, ignore_label=pred_ignore)
            cm = completeness_metrics(root_label_map, pred_cmpl)
            for k in ("precision", "recall", "f1", "accuracy", "n_complete_gt", "n_fragments"):
                m[f"cmpl_{k}"] = cm[k]
            for k in ("tp", "fp", "fn", "tn"):
                m[f"cmpl_{k}"] = cm[f"{k}_complete"]
            if naive_baseline:
                all_complete = {f: True for f in fragment_completeness(root_label_map)}
                ncm = completeness_metrics(root_label_map, all_complete)
                for k in ("precision", "recall", "f1"):
                    m[f"naive_cmpl_{k}"] = ncm[k]

    if true_post is not None:
        true_post = np.asarray(true_post)
        m.update(connectome_metrics(pred, true, true_post, min_syn=min_syn, ignore=ignore))
        lg = line_graph_suite(
            _singletonize(pred, pred_ignore),
            _singletonize(true, ignore),
            _singletonize(true_post, ignore),
            None if pred_post is None else _singletonize(pred_post, pred_ignore),
        )
        m.update(lg.to_dict())

    return m


__all__ = ["evaluate_partition_suite"]
