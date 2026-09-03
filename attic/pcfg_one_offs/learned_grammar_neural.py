#!/usr/bin/env python3
"""Learned, end-to-end self-supervised neural grammar of neurite shape from RAW points.

Thesis: 15 years of hand-engineered connectomics features (F/B/L/R tokens, gap, turn-angle,
curvature) have not solved error correction.  Here we feed a sequence encoder ONLY raw 3D
synapse coordinates (centered + globally scaled) and raw per-step displacement vectors
(p_{i+1}-p_i, nm).  NO per-step angles / lengths / curvature are computed by hand.  The net
learns "what a coherent neurite continuation looks like" from a single self-supervised
objective:

    COHERENT vs SPLICED contrastive classification.
      positive = a real v1718 clean arbor's ordered point sequence.
      negative = first half of arbor A spliced onto the second half of a DIFFERENT,
                 spatially-near arbor B (an adversarial fake continuation).

Trained on clean v1718 arbors only (label-free w.r.t. proofreading edits), grouped-by-cell
so no cell straddles train/val.  Then the FROZEN encoder is scored on the REAL proofreading
errors, exactly like the existing hand-feature scripts:

  * Anchor gate (de-merge / split): within each v117 root, a "seam" step is a consecutive
    (PCA-ordered) synapse pair whose root_later differs.  Score each interior junction by the
    model's local incoherence (drop in coherence prob when the sequence is split AT that
    junction vs. kept whole).  AUC(seam | score), grouped so a whole cell is held out.
    Baselines on this exact cache: bigram-token=0.539, 2-D Gaussian(log-len,turn)=0.658,
    kNN 5-feat=0.587, gap-after alone=0.813, supervised geometry ceiling=0.85.

  * De-split / merge (the prize): a real de-split = two different root_v117 fragments sharing
    the same root_later.  Take spatially-adjacent cross-v117-root fragment tips; positive if
    same root_later, negative if different (adjacent-but-distinct cells).  Score each pair by
    the model's coherence-of-the-concatenation (concatenate the two tip neighbourhoods into
    one ordered sequence; read off P(coherent)).  AUC, grouped by cell.  Gap is useless here
    (a gap exists in BOTH join and leave cases), so a learned continuation grammar should win.

Group-by-cell CV everywhere (union-find over v117<->later co-occurrence); within-group
permutation nulls; seeded; CPU-fast.

Usage:
    python -m experiments.pcfg.learned_grammar_neural \
        --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001  unbuffered progress

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import (  # noqa: E402
    SideTable,
    cell_components,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# -----------------------------------------------------------------------------
# Raw-point sequence construction.  The ONLY normalization is: center by the
# sequence centroid and divide by a single global scale constant (nm).  We feed
# raw centered coords (3) and raw per-step displacement vectors (3) -- NO hand
# features (no angle/length/curvature/gap).  A leading zero-displacement pads
# step 0 so the displacement channel aligns with the coord channel.
# -----------------------------------------------------------------------------
GLOBAL_SCALE_NM = 1000.0  # one global constant; not per-step, not learned


def pca_order(pts: np.ndarray) -> np.ndarray:
    """Order indices along the principal axis (ORDERING ONLY -- no features kept)."""
    c = pts - pts.mean(axis=0)
    if len(pts) < 2:
        return np.arange(len(pts))
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return np.argsort(c @ Vt[0])


def seq_from_points(pts: np.ndarray, centroid: np.ndarray | None = None) -> np.ndarray:
    """Raw (centered coord || displacement) features for an ALREADY-ORDERED point seq.

    centroid: if given, center by it (so a spliced/concatenated sequence is centered
    consistently); else center by this sequence's own centroid.
    """
    pts = pts.astype(np.float64)
    if centroid is None:
        centroid = pts.mean(axis=0)
    coord = (pts - centroid) / GLOBAL_SCALE_NM
    disp = np.zeros_like(pts)
    disp[1:] = (pts[1:] - pts[:-1]) / GLOBAL_SCALE_NM
    return np.concatenate([coord, disp], axis=1).astype(np.float32)  # (n, 6)


# -----------------------------------------------------------------------------
# Tiny GRU encoder + coherence head.  CPU-trainable in minutes.
# -----------------------------------------------------------------------------
class CoherenceNet(nn.Module):
    def __init__(self, d_in: int = 6, d_model: int = 64, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=layers,
                          batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # masked mean-pool over time
        mask = (torch.arange(out.size(1))[None, :] < lengths[:, None]).float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)  # logit P(coherent)


MAXLEN = 64  # cap sequence length: continuation is local, and splices sit mid-sequence,
             # so a centered window keeps the junction while keeping the BiGRU CPU-fast.


def _center_crop(s: np.ndarray) -> np.ndarray:
    if len(s) <= MAXLEN:
        return s
    mid = len(s) // 2
    lo = max(0, mid - MAXLEN // 2)
    return s[lo:lo + MAXLEN]


def collate(seqs: list[np.ndarray]):
    seqs = [_center_crop(s) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    L = int(lengths.max())
    x = torch.zeros(len(seqs), L, seqs[0].shape[1], dtype=torch.float32)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.from_numpy(s)
    return x, lengths


# -----------------------------------------------------------------------------
# Splice negatives: first half of arbor A onto second half of a spatially-near
# arbor B (different cell).  This is the adversarial "fake continuation".
# -----------------------------------------------------------------------------
def make_spliced(ptsA: np.ndarray, ptsB: np.ndarray, rng) -> np.ndarray | None:
    """A_ordered[:half] ++ B_ordered[half:], re-centered, displacement re-derived.

    We deliberately translate B so its retained half starts near where A's half ends,
    so the splice is geometrically plausible (a hard negative): the net must detect the
    DIRECTIONAL incoherence (caliber/continuation), not a trivial teleport gap.
    """
    oa = pca_order(ptsA)
    ob = pca_order(ptsB)
    a = ptsA[oa]
    b = ptsB[ob]
    na, nb = len(a), len(b)
    if na < 4 or nb < 4:
        return None
    ha = na // 2
    hb = nb // 2
    head = a[:ha]
    tail = b[hb:]
    if len(head) < 2 or len(tail) < 2:
        return None
    # plausible join: shift B's tail so it begins one median-A-step past A's head end,
    # along A's local direction.  Removes the teleport so the net can't cheat on gap.
    a_dir = head[-1] - head[-2]
    n = np.linalg.norm(a_dir)
    a_dir = a_dir / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    med_step = np.median(np.linalg.norm(np.diff(a, axis=0), axis=1)) if na > 1 else 500.0
    target_start = head[-1] + a_dir * med_step
    tail = tail - tail[0] + target_start
    spliced = np.concatenate([head, tail], axis=0)
    return seq_from_points(spliced)


def build_selfsup_dataset(by_v1718, pt, comp_v_to_cell, rl_to_v117, min_syn, max_arbors,
                          rng):
    """Positives = clean arbor sequences; negatives = spatially-near A|B splices.

    Returns lists of (seq, label, cell) where cell groups the POSITIVE's anchor cell
    (for splices we use A's cell) so train/val never share a cell.
    """
    from scipy.spatial import cKDTree

    # eligible clean arbors and their centroids
    arbors = [(lr, idxs) for lr, idxs in by_v1718.items() if len(idxs) >= min_syn]
    rng.shuffle(arbors)
    if max_arbors and len(arbors) > max_arbors:
        arbors = arbors[:max_arbors]
    cents = np.array([pt[idxs].mean(axis=0) for _, idxs in arbors])
    tree = cKDTree(cents)

    def cell_of_later(lr):
        # a later root's cell = the cell of any v117 root it co-occurs with
        v = rl_to_v117.get(lr)
        return comp_v_to_cell.get(v, -1) if v is not None else -1

    data = []  # (seq, label, cell)
    for ai, (lr, idxs) in enumerate(arbors):
        ptsA = pt[idxs]
        cellA = cell_of_later(lr)
        # positive
        oa = pca_order(ptsA)
        data.append((seq_from_points(ptsA[oa]), 1, cellA))
        # negative: pick a spatially-near DIFFERENT-cell arbor B
        kq = min(8, len(arbors))
        _, nn_idx = tree.query(cents[ai], k=kq)
        nn_idx = np.atleast_1d(nn_idx)
        cand = [j for j in nn_idx if j != ai and cell_of_later(arbors[j][0]) != cellA]
        if not cand:
            cand = [j for j in range(len(arbors)) if j != ai]
        bj = int(rng.choice(cand))
        ptsB = pt[arbors[bj][1]]
        spl = make_spliced(ptsA, ptsB, rng)
        if spl is not None:
            data.append((spl, 0, cellA))
    return data


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_encoder(train_data, val_data, epochs, batch, lr, device, seed):
    torch.manual_seed(seed)
    net = CoherenceNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    n = len(train_data)
    rng = np.random.default_rng(seed)
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            seqs = [train_data[i][0] for i in idx]
            ys = torch.tensor([train_data[i][1] for i in idx], dtype=torch.float32, device=device)
            x, lengths = collate(seqs)
            x = x.to(device)
            logit = net(x, lengths)
            loss = lossf(logit, ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        va = eval_encoder_acc(net, val_data, device)
        if va > best_auc:
            best_auc, best_state = va, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  epoch {ep+1:2d}  train_loss={tot/n:.4f}  val_AUC(coh/splice)={va:.3f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_auc


@torch.no_grad()
def eval_encoder_acc(net, data, device, batch=256):
    net.eval()
    logits, ys = [], []
    for s in range(0, len(data), batch):
        chunk = data[s:s + batch]
        seqs = [c[0] for c in chunk]
        x, lengths = collate(seqs)
        logits.append(net(x.to(device), lengths).cpu().numpy())
        ys += [c[1] for c in chunk]
    logits = np.concatenate(logits)
    ys = np.array(ys)
    if len(np.unique(ys)) < 2:
        return float("nan")
    return roc_auc_score(ys, logits)


@torch.no_grad()
def score_sequences(net, seqs, device, batch=256):
    """Return P(coherent) logit for each raw sequence."""
    net.eval()
    out = []
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        x, lengths = collate(chunk)
        out.append(net(x.to(device), lengths).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# -----------------------------------------------------------------------------
# Eval 1: seam anchor-gate (de-merge / split)
# -----------------------------------------------------------------------------
def eval_seam(net, by_v117, pt, rl, comp, device, min_syn, win, rng, n_perm,
              max_junctions=60000):
    """Score each interior junction by local incoherence = coherence(whole window) minus
    coherence(window split at the junction, halves pushed apart along principal axis).

    Higher score = more incoherent = more likely a seam.  We build a LOCAL window of
    +-`win` synapses around junction k so big arbors don't wash out the local signal, and
    the score is the directional drop in P(coherent) when we cut at k.
    """
    # collect junctions; keep ALL seams (rare positives), cap NON-seams for CPU speed.
    pos_j, neg_j = [], []  # each: (whole_seq, cut_seq, group)
    for rv, idxs in by_v117.items():
        if len(idxs) < min_syn:
            continue
        pts = pt[idxs]
        lab = rl[idxs]
        order = pca_order(pts)
        ps, lb = pts[order], lab[order]
        n = len(ps)
        g = comp.get(int(rv), -1)
        for k in range(1, n - 1):  # interior junction after synapse k
            lo, hi = max(0, k - win), min(n, k + win + 1)
            window = ps[lo:hi]
            kk = k - lo  # junction position within window (between kk-1 and kk)
            if kk < 2 or (len(window) - kk) < 2:
                continue
            cent = window.mean(axis=0)
            whole = seq_from_points(window, centroid=cent)
            # CUT at kk: separate the two halves along principal axis so the encoder sees
            # them as two lobes (this is the "splice/mask at junction" perturbation).
            left = window[:kk].copy()
            right = window[kk:].copy()
            axis = window - cent
            _, _, Vt = np.linalg.svd(axis, full_matrices=False)
            pax = Vt[0]
            push = 3.0 * GLOBAL_SCALE_NM  # 3 um separation marker, in the SAME raw units
            left = left - pax * push
            right = right + pax * push
            cut = seq_from_points(np.concatenate([left, right]), centroid=cent)
            is_seam = bool(lb[k] != lb[k + 1])
            (pos_j if is_seam else neg_j).append((whole, cut, g))
    # subsample non-seams to keep total <= max_junctions (all seams retained)
    cap_neg = max(0, max_junctions - len(pos_j))
    if len(neg_j) > cap_neg:
        sel = rng.choice(len(neg_j), cap_neg, replace=False)
        neg_j = [neg_j[i] for i in sel]
    allj = pos_j + neg_j
    whole_seqs = [a[0] for a in allj]
    cut_seqs = [a[1] for a in allj]
    seams = np.array([True] * len(pos_j) + [False] * len(neg_j))
    groups = np.array([a[2] for a in allj])
    cw = score_sequences(net, whole_seqs, device)
    cc = score_sequences(net, cut_seqs, device)
    score = cw - cc  # incoherence: how much coherence is LOST keeping it whole vs cutting
    auc = roc_auc_score(seams, score)
    null = grouped_perm_null(seams, score, groups, rng, n_perm)
    return auc, null, int(seams.sum()), len(seams)


# -----------------------------------------------------------------------------
# Eval 2: de-split / merge.  Reuse the exact pos/neg construction from
# build_correction_pairs' merge stratum, but score by the LEARNED coherence of the
# concatenated tip neighbourhoods.
# -----------------------------------------------------------------------------
def _tip_neighbourhood(pt_local, root_pts, anchor_pt, k):
    """The k nearest same-fragment points to the anchor, ordered along their PCA axis."""
    if len(root_pts) <= k:
        sub = root_pts
    else:
        d = np.linalg.norm(root_pts - anchor_pt, axis=1)
        sub = root_pts[np.argsort(d)[:k]]
    o = pca_order(sub)
    return sub[o]


def build_desplit_pairs(tab, comp, rng, cross_k=12, radius=6000.0, neg_ratio=3.0,
                        tip_k=8):
    from scipy.spatial import cKDTree
    pos, neg = [], []  # each: (ra, rb)
    for side_code in (0, 1):
        sel = (tab.side == side_code) & (tab.root_later > 0)
        rows = np.nonzero(sel)[0]
        if len(rows) < 2:
            continue
        sub = tab.mask(sel)
        by_root = defaultdict(list)
        for li, rv in enumerate(sub.root_v117.tolist()):
            by_root[int(rv)].append(li)
        by_later = defaultdict(list)
        for li in range(len(sub)):
            by_later[int(sub.root_later[li])].append(li)

        pos_pairs = []
        for members in by_later.values():
            if len(members) < 2:
                continue
            if len({int(sub.root_v117[m]) for m in members}) < 2:
                continue
            mpts = sub.pt[members]
            tree = cKDTree(mpts)
            kq = min(cross_k + 1, len(members))
            dnn, inn = tree.query(mpts, k=kq, workers=-1)
            seen = set()
            for a in range(len(members)):
                ra = members[a]; rva = int(sub.root_v117[ra])
                for slot in range(1, kq):
                    if np.atleast_1d(dnn[a])[slot] > radius:
                        break
                    rb = members[int(np.atleast_1d(inn[a])[slot])]
                    if int(sub.root_v117[rb]) == rva:
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    if key in seen:
                        continue
                    seen.add(key)
                    pos_pairs.append((ra, rb))

        neg_pairs = []
        if pos_pairs:
            gtree = cKDTree(sub.pt)
            anchors = list({p[0] for p in pos_pairs} | {p[1] for p in pos_pairs})
            kq = min(cross_k + 1, len(sub))
            seen = set()
            for ra in anchors:
                rva = int(sub.root_v117[ra]); la = int(sub.root_later[ra])
                dnn, inn = gtree.query(sub.pt[ra], k=kq, workers=-1)
                for slot in range(1, kq):
                    if dnn[slot] > radius:
                        break
                    rb = int(inn[slot])
                    if int(sub.root_v117[rb]) == rva or int(sub.root_later[rb]) == la:
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    if key in seen:
                        continue
                    seen.add(key)
                    neg_pairs.append((ra, rb))
        n_neg = min(len(neg_pairs), max(1, int(max(1, len(pos_pairs)) * neg_ratio)))
        if len(neg_pairs) > n_neg:
            neg_pairs = [neg_pairs[p] for p in rng.choice(len(neg_pairs), n_neg, replace=False)]

        # materialize tip-neighbourhood sequences + labels + groups
        root_pts_cache = {rv: sub.pt[idxs] for rv, idxs in by_root.items()}
        for (ra, rb), lbl in ([(p, 1) for p in pos_pairs] + [(p, 0) for p in neg_pairs]):
            rva, rvb = int(sub.root_v117[ra]), int(sub.root_v117[rb])
            na = _tip_neighbourhood(None, root_pts_cache[rva], sub.pt[ra], tip_k)
            nb = _tip_neighbourhood(None, root_pts_cache[rvb], sub.pt[rb], tip_k)
            # concatenate the two tips into ONE ordered sequence: order A toward the join,
            # then B away from the join, so the encoder reads a candidate continuation.
            seq_pts = np.concatenate([na, nb], axis=0)
            cent = seq_pts.mean(axis=0)
            seq = seq_from_points(seq_pts, centroid=cent)
            (pos if lbl else neg).append((seq, lbl, comp.get(rva, -1)))
    return pos, neg


def eval_desplit(net, tab, comp, device, rng, n_perm):
    pos, neg = build_desplit_pairs(tab, comp, rng)
    data = pos + neg
    if len(data) < 4 or len({d[1] for d in data}) < 2:
        return float("nan"), np.zeros(0), 0, len(data)
    seqs = [d[0] for d in data]
    ys = np.array([d[1] for d in data])
    groups = np.array([d[2] for d in data])
    score = score_sequences(net, seqs, device)  # P(coherent) -> high for true continuations
    auc = roc_auc_score(ys, score)
    null = grouped_perm_null(ys, score, groups, rng, n_perm)
    return auc, null, int(ys.sum()), len(ys)


# -----------------------------------------------------------------------------
# Within-group permutation null: permute labels only WITHIN each cell group so the
# null respects the group structure (the project's leakage discipline).
# -----------------------------------------------------------------------------
def grouped_perm_null(y, score, groups, rng, n_perm):
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.nonzero(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_perm):
        yp = y.copy()
        for g, idx in idx_by_g.items():
            yp[idx] = rng.permutation(y[idx])
        if len(np.unique(yp)) < 2:
            continue
        out.append(roc_auc_score(yp, score))
    return np.array(out)


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--side", choices=["pre", "post", "both"], default="both")
    ap.add_argument("--min-syn", type=int, default=6)
    ap.add_argument("--max-arbors", type=int, default=12000,
                    help="subsample clean arbors for CPU-fast training (0=all)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seam-win", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)  # v117 root -> cell id (union-find)

    side_codes = {"pre": [0], "post": [1], "both": [0, 1]}[args.side]
    sel = np.isin(tab.side, side_codes) & (tab.root_later > 0)
    pt = tab.pt[sel]
    rv = tab.root_v117[sel]
    rl = tab.root_later[sel]

    by_v117 = defaultdict(list)
    by_v1718 = defaultdict(list)
    rl_to_v117 = {}  # a representative v117 for each later root (for cell lookup)
    for i in range(len(pt)):
        by_v117[int(rv[i])].append(i)
        by_v1718[int(rl[i])].append(i)
        rl_to_v117.setdefault(int(rl[i]), int(rv[i]))

    print(f"[data] side={args.side}  synapse-sides={len(pt):,}  "
          f"v117 roots={len(by_v117):,}  v1718 arbors={len(by_v1718):,}")

    # ---- build self-supervised dataset (coherent vs spliced), grouped by cell ----
    t0 = time.time()
    data = build_selfsup_dataset(by_v1718, pt, comp, rl_to_v117,
                                 args.min_syn, args.max_arbors, rng)
    cells = sorted({c for _, _, c in data})
    rng.shuffle(cells)
    n_val = max(1, int(len(cells) * args.val_frac))
    val_cells = set(cells[:n_val])
    train_data = [x for x in data if x[2] not in val_cells]
    val_data = [x for x in data if x[2] in val_cells]
    ntr_pos = sum(1 for x in train_data if x[1] == 1)
    print(f"[selfsup] sequences={len(data):,}  train={len(train_data):,} "
          f"(pos {ntr_pos:,})  val={len(val_data):,}  "
          f"cells train/val={len(cells)-n_val}/{n_val}  build={time.time()-t0:.1f}s")
    # leakage assert: no cell in both splits
    assert not ({c for _, _, c in train_data} & val_cells), "cell leakage train/val!"

    # ---- train ----
    print("[train] tiny BiGRU coherence encoder (raw coords+displacements only)")
    tT = time.time()
    net, val_auc = train_encoder(train_data, val_data, args.epochs, args.batch,
                                 args.lr, device, args.seed)
    train_secs = time.time() - tT
    print(f"[train] done in {train_secs:.1f}s  best val_AUC(coherent/spliced)={val_auc:.3f}")

    # ---- eval 1: seam anchor gate ----
    print("\n[eval] seam anchor-gate (de-merge / split)")
    auc_s, null_s, npos_s, ntot_s = eval_seam(net, by_v117, pt, rl, comp, device,
                                              args.min_syn, args.seam_win, rng, args.n_perm)
    p_s = (null_s >= auc_s).mean() if len(null_s) else float("nan")
    print(f"    junctions={ntot_s:,}  seams={npos_s:,} ({npos_s/max(1,ntot_s):.2%})")
    print(f"    AUC(seam | learned incoherence) = {auc_s:.3f}   "
          f"null={null_s.mean():.3f}±{null_s.std():.3f}  p={p_s:.3f}")

    # ---- eval 2: de-split / merge ----
    print("\n[eval] de-split / merge (coherence of concatenated tips)")
    auc_m, null_m, npos_m, ntot_m = eval_desplit(net, tab, comp, device, rng, args.n_perm)
    p_m = (null_m >= auc_m).mean() if len(null_m) else float("nan")
    print(f"    pairs={ntot_m:,}  positives(same later)={npos_m:,} "
          f"({npos_m/max(1,ntot_m):.2%})")
    print(f"    AUC(same-cell | learned coherence) = {auc_m:.3f}   "
          f"null={null_m.mean():.3f}±{null_m.std():.3f}  p={p_m:.3f}")

    # ---- summary table vs hand-feature baselines ----
    print("\n" + "=" * 72)
    print("RESULTS  (learned end-to-end, RAW coords only -- no hand features)")
    print("=" * 72)
    print(f"{'task':<26}{'learned AUC':<14}{'null mean':<12}{'baselines':<28}")
    print(f"{'seam (de-merge/split)':<26}{auc_s:<14.3f}{null_s.mean():<12.3f}"
          f"gap=0.813 gauss=0.658 ceil=0.85")
    print(f"{'de-split (merge)':<26}{auc_m:<14.3f}{null_m.mean():<12.3f}"
          f"RF-handfeat=0.979 (caveated)")
    print("=" * 72)
    print(f"val coherent/spliced AUC={val_auc:.3f}  train_time={train_secs:.1f}s")


if __name__ == "__main__":
    main()
