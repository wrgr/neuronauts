"""PathGrammarReranker: cross-attention path-pair affinity model.

Motivation
----------
SkeletonGNN produces one DNA vector per fragment (fast, ANN-indexable).  For
hard cases — same-cell-type fragments with similar global shape — a second-
stage cross-encoder that sees *both* fragments simultaneously can resolve
ambiguity that pairwise embedding comparison cannot.

The key idea: two paths from the same neuron are *complementary*, not just
*similar*.  A basal dendrite path and an apical trunk path look very different
in embedding space yet belong to the same cell.  A cross-encoder can learn
that pattern; a bi-encoder cannot.

Architecture
------------
Stage 1 — intrinsic path features (rotation- and translation-invariant):
  At each vertex t along a sampled path, compute 4 scalars:
    step_log_nm   = log(||v_{t+1} - v_t|| + 1)    local step length
    turn_rad      = arccos(tangent_t · tangent_{t-1})  local curvature
    log_radius    = log(r_t + 1)                    calibre
    d_log_radius  = log(r_{t+1}+1) - log(r_t+1)    calibre change rate

  No global orientation (dx/dy/dz) — orientation is uninformative within a
  cell type where all apical dendrites point the same direction.

Stage 2 — per-path encoding (shared weights for both fragments):
  Linear(4, d_model) → Transformer encoder (n_layers, n_heads)
  Mean-pool over path tokens → one [d_model] vector per path.
  A fragment with K sampled paths → [K, d_model] path-set representation.

Stage 3 — cross-attention reranking:
  n_cross_layers of: A attends to B (MHA), B attends to A (MHA), residuals.
  Mean-pool both sides → concat → 2-layer MLP → sigmoid → affinity in [0,1].

Integration
-----------
  1. SkeletonGNN → DNA[D] per fragment (ANN search → top-K candidates)
  2. PathGrammarReranker reranks top-K with full cross-encoder pass (K ~ 20)
  3. Merge decisions use reranker affinity, not DNA cosine distance

Training
--------
  train_path_grammar_reranker() — BCE loss over balanced pos/neg fragment pairs.
  Positives: two fragments from bisected halves of the same neuron.
  Negatives: two fragments from different neurons of the same cell type.
  Shares the hard-negative structure of the within-type ablation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..schemas import Fragment
from .dna import sample_tree_paths


# ---------------------------------------------------------------------------
# Intrinsic path features
# ---------------------------------------------------------------------------

def path_to_intrinsic(
    vertices_nm: np.ndarray,  # [T, 3] float32 — path vertices in order
    radii_nm: np.ndarray,     # [T] float32
) -> np.ndarray:
    """Compute 4 rotation/translation-invariant features per path step.

    Returns [T, 4] float32:
      col 0: log(step_length_nm + 1)
      col 1: turn angle (radians) between consecutive steps  [0 at t=0]
      col 2: log(radius_nm + 1)
      col 3: log(r_{t+1}+1) - log(r_t+1)  [0 at last step]
    """
    T = len(vertices_nm)
    feat = np.zeros((T, 4), dtype=np.float32)

    if T < 2:
        feat[:, 2] = np.log(radii_nm + 1.0)
        return feat

    # Step vectors and lengths
    deltas = np.diff(vertices_nm, axis=0)                  # [T-1, 3]
    lens = np.linalg.norm(deltas, axis=1, keepdims=True)   # [T-1, 1]
    lens_safe = np.maximum(lens, 1e-6)
    tangents = deltas / lens_safe                           # [T-1, 3] unit

    # Feature 0: log step length (at step t = edge t→t+1, assigned to vertex t)
    feat[:-1, 0] = np.log(lens[:, 0] + 1.0)

    # Feature 1: turning angle at vertex t (between step t-1→t and t→t+1)
    if T >= 3:
        cos_turns = np.einsum('ij,ij->i', tangents[:-1], tangents[1:])
        cos_turns = np.clip(cos_turns, -1.0, 1.0)
        turns = np.arccos(cos_turns)             # [T-2]
        feat[1:-1, 1] = turns

    # Feature 2: log radius at each vertex
    log_r = np.log(radii_nm + 1.0)
    feat[:, 2] = log_r

    # Feature 3: log-radius change (d_log_r at vertex t = r at t+1 minus r at t)
    feat[:-1, 3] = log_r[1:] - log_r[:-1]

    return feat


def fragment_to_intrinsic_paths(
    fragment: Fragment,
    *,
    n_paths: int = 8,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Sample n_paths through fragment's skeleton tree; return intrinsic features.

    Returns
    -------
    List of [T_k, 4] float32 arrays, one per sampled path.
    """
    # sample_tree_paths returns [T, 3] vertex coordinate arrays.
    # We recover per-vertex radii by nearest-neighbour lookup into the fragment.
    path_verts_list = sample_tree_paths(
        fragment.vertices_nm,
        fragment.edges,
        n_paths=n_paths,
        rng=rng,
    )
    all_verts = fragment.vertices_nm          # [V, 3]
    all_radii = fragment.radius_nm            # [V]

    results = []
    for path_verts in path_verts_list:
        path_verts = np.asarray(path_verts, dtype=np.float32)
        if len(path_verts) == 0:
            results.append(np.zeros((0, 4), dtype=np.float32))
            continue
        # Nearest-vertex lookup to recover radii along this path.
        # (path_verts are exact rows of all_verts, so L2-NN gives exact indices.)
        diffs = all_verts[:, None, :] - path_verts[None, :, :]  # [V, T, 3]
        dists = np.einsum('vti,vti->vt', diffs, diffs)           # [V, T]
        nearest = np.argmin(dists, axis=0)                        # [T]
        radii = all_radii[nearest]
        results.append(path_to_intrinsic(path_verts, radii))
    return results


# ---------------------------------------------------------------------------
# PathGrammarReranker
# ---------------------------------------------------------------------------

class PathGrammarReranker:
    """Cross-attention reranker: (fragment_A, fragment_B) → same-neuron affinity.

    Factory pattern (returns nn.Module) consistent with SkeletonGNN.

    Parameters
    ----------
    input_dim : int
        Intrinsic feature dim per path step (default 4).
    d_model : int
        Hidden width throughout.
    n_heads : int
        Attention heads (must divide d_model).
    n_path_layers : int
        Transformer layers for per-path encoding.
    n_cross_layers : int
        Cross-attention layers between the two fragments.
    dropout : float
    n_paths : int
        Paths sampled per fragment at train/inference time.
    """

    def __new__(
        cls,
        *,
        input_dim: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_path_layers: int = 2,
        n_cross_layers: int = 2,
        dropout: float = 0.1,
        n_paths: int = 8,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("pip install torch") from exc

        n_paths_ = n_paths

        class _CrossAttnBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_ab = nn.MultiheadAttention(d_model, n_heads,
                                                     dropout=dropout, batch_first=True)
                self.attn_ba = nn.MultiheadAttention(d_model, n_heads,
                                                     dropout=dropout, batch_first=True)
                self.norm_a1 = nn.LayerNorm(d_model)
                self.norm_b1 = nn.LayerNorm(d_model)
                self.ff_a = nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
                )
                self.ff_b = nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
                )
                self.norm_a2 = nn.LayerNorm(d_model)
                self.norm_b2 = nn.LayerNorm(d_model)
                self.drop = nn.Dropout(dropout)

            def forward(self, a, b):
                # a: [1, K_A, d_model], b: [1, K_B, d_model]
                a2, _ = self.attn_ab(a, b, b)
                a = self.norm_a1(a + self.drop(a2))
                a = self.norm_a2(a + self.ff_a(a))

                b2, _ = self.attn_ba(b, a, a)
                b = self.norm_b1(b + self.drop(b2))
                b = self.norm_b2(b + self.ff_b(b))
                return a, b

        class _PathGrammarReranker(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_paths = n_paths_
                self.d_model = d_model

                self.input_proj = nn.Linear(input_dim, d_model)

                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout,
                    batch_first=True,
                )
                self.path_encoder = nn.TransformerEncoder(enc_layer,
                                                          num_layers=n_path_layers)

                self.cross_attn = nn.ModuleList(
                    [_CrossAttnBlock() for _ in range(n_cross_layers)]
                )

                self.head = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                    nn.Sigmoid(),
                )

            def _encode_paths(self, path_list: list) -> "torch.Tensor":
                """Encode a list of [T_k, 4] tensors → [1, K, d_model].

                Each path is independently encoded; mean-pooled to one vector.
                """
                import torch
                vecs = []
                for path_feat in path_list:
                    # path_feat: [T, input_dim]
                    if path_feat.size(0) == 0:
                        vecs.append(torch.zeros(d_model, device=path_feat.device,
                                                dtype=path_feat.dtype))
                        continue
                    x = self.input_proj(path_feat).unsqueeze(0)  # [1, T, d_model]
                    x = self.path_encoder(x)                     # [1, T, d_model]
                    vecs.append(x.squeeze(0).mean(dim=0))        # [d_model]
                return torch.stack(vecs, dim=0).unsqueeze(0)     # [1, K, d_model]

            def forward(
                self,
                paths_a: list,  # list of [T_k, input_dim] tensors
                paths_b: list,  # list of [T_k, input_dim] tensors
            ) -> "torch.Tensor":
                """
                Parameters
                ----------
                paths_a, paths_b : list of [T_k, input_dim] tensors

                Returns
                -------
                Tensor [] — scalar affinity in [0, 1].
                """
                a = self._encode_paths(paths_a)  # [1, K_A, d_model]
                b = self._encode_paths(paths_b)  # [1, K_B, d_model]

                for cross in self.cross_attn:
                    a, b = cross(a, b)

                a_pool = a.squeeze(0).mean(dim=0)  # [d_model]
                b_pool = b.squeeze(0).mean(dim=0)  # [d_model]

                return self.head(
                    torch.cat([a_pool, b_pool], dim=-1)
                ).squeeze(-1)  # scalar

        return _PathGrammarReranker()


# ---------------------------------------------------------------------------
# Fragment pair → tensors
# ---------------------------------------------------------------------------

def fragment_pair_to_tensors(
    frag_a: Fragment,
    frag_b: Fragment,
    *,
    n_paths: int = 8,
    device: str = "cpu",
    rng: np.random.Generator | None = None,
) -> tuple[list, list]:
    """Convert a fragment pair to lists of intrinsic-feature tensors.

    Returns (paths_a, paths_b) where each is a list of [T_k, 4] torch tensors.
    """
    import torch

    def _to_tensors(frag: Fragment, rng_: np.random.Generator) -> list:
        raw = fragment_to_intrinsic_paths(frag, n_paths=n_paths, rng=rng_)
        return [torch.from_numpy(p).to(device) for p in raw]

    rng = rng or np.random.default_rng(0)
    rng_a = np.random.default_rng(int(rng.integers(1 << 32)))
    rng_b = np.random.default_rng(int(rng.integers(1 << 32)))
    return _to_tensors(frag_a, rng_a), _to_tensors(frag_b, rng_b)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def score_fragment_pairs(
    reranker,
    pairs: Sequence[tuple[Fragment, Fragment]],
    *,
    n_paths: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """Score a list of (fragment_A, fragment_B) pairs.

    Returns [N] float32 affinity scores in [0, 1].
    Higher = more likely same neuron.
    """
    import torch

    reranker.eval()
    reranker = reranker.to(device)
    rng = np.random.default_rng(0)
    scores = []

    with torch.no_grad():
        for fa, fb in pairs:
            pa, pb = fragment_pair_to_tensors(fa, fb, n_paths=n_paths,
                                              device=device, rng=rng)
            s = reranker(pa, pb)
            scores.append(float(s.item()))

    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_path_grammar_reranker(
    reranker,
    fragment_lists: list[list[Fragment]],
    *,
    n_epochs: int = 40,
    lr: float = 1e-3,
    n_paths: int = 8,
    n_pairs_per_epoch: int = 256,
    device: str = "cpu",
    root_label_map: dict[int, set[int]] | None = None,
    log_every: int = 10,
) -> dict:
    """Train PathGrammarReranker with BCE loss on fragment pairs.

    Positives: fragment pairs sharing the same label root.
    Negatives: fragment pairs with different label roots (same call, implies
               same cell type if you built the world with within_type_ablation).
    Loss: binary cross-entropy.

    Returns history dict: loss/epoch, pos_score/epoch, neg_score/epoch.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    reranker = reranker.to(device)
    opt = torch.optim.Adam(reranker.parameters(), lr=lr)

    all_frags = [f for fl in fragment_lists for f in fl]

    # Group by label root (same logic as train_skeleton_gnn)
    group_to_frags: dict[int, list[Fragment]] = {}
    for frag in all_frags:
        rid = frag.base_root_id
        if root_label_map is not None:
            labels = root_label_map.get(rid, set())
            if len(labels) != 1:
                continue
            key = next(iter(labels))
        else:
            key = rid
        group_to_frags.setdefault(key, []).append(frag)

    groups = [(k, v) for k, v in group_to_frags.items() if len(v) >= 1]
    if len(groups) < 2:
        raise ValueError("Need ≥2 neuron groups with ≥1 fragment each")

    pos_groups = [(k, v) for k, v in groups if len(v) >= 2]
    history: dict[str, list[float]] = {"loss": [], "pos_score": [], "neg_score": []}

    for epoch in range(1, n_epochs + 1):
        reranker.train()
        opt.zero_grad()

        n_each = n_pairs_per_epoch // 2
        pos_affinities, neg_affinities = [], []
        targets = []

        # Positive pairs
        for _ in range(n_each):
            if not pos_groups:
                break
            _, frags = pos_groups[int(rng.integers(len(pos_groups)))]
            ia, ib = rng.choice(len(frags), size=2, replace=False)
            pa, pb = fragment_pair_to_tensors(
                frags[int(ia)], frags[int(ib)],
                n_paths=n_paths, device=device, rng=rng,
            )
            score = reranker(pa, pb)
            pos_affinities.append(score)
            targets.append(torch.ones(1, device=device))

        # Negative pairs
        for _ in range(n_each):
            ga_i, gb_i = rng.choice(len(groups), size=2, replace=False)
            ga_f = groups[int(ga_i)][1]
            gb_f = groups[int(gb_i)][1]
            ia = int(rng.integers(len(ga_f)))
            ib = int(rng.integers(len(gb_f)))
            pa, pb = fragment_pair_to_tensors(
                ga_f[ia], gb_f[ib],
                n_paths=n_paths, device=device, rng=rng,
            )
            score = reranker(pa, pb)
            neg_affinities.append(score)
            targets.append(torch.zeros(1, device=device))

        if not pos_affinities and not neg_affinities:
            history["loss"].append(0.0)
            history["pos_score"].append(0.0)
            history["neg_score"].append(0.0)
            continue

        all_scores = torch.stack(pos_affinities + neg_affinities).squeeze(-1)
        all_targets = torch.cat(targets)
        loss = F.binary_cross_entropy(all_scores, all_targets)

        loss.backward()
        opt.step()

        lv = float(loss.item())
        ps = float(np.mean([s.item() for s in pos_affinities])) if pos_affinities else 0.0
        ns = float(np.mean([s.item() for s in neg_affinities])) if neg_affinities else 0.0
        history["loss"].append(lv)
        history["pos_score"].append(ps)
        history["neg_score"].append(ns)

        if log_every > 0 and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}: loss={lv:.4f}  pos={ps:.3f}  neg={ns:.3f}")

    reranker.eval()
    return history
