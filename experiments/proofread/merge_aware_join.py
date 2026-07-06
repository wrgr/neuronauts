"""Merge-aware constrained joining: recognize merges, defer that subtree.

The follower's per-join ranking is good, but greedy union-find cascades — one wrong join
(axon fused to a dendrite, two somata merged) fuses whole neurons and destroys many
synapse pairs.  Here we commit joins **most-confident-first**, but `union(a, b)` only if
the *resulting component* stays grammatical:

* **A↔D veto** — reject if one component is axon-typed and the other dendrite-typed and
  neither contains a soma (a local axon↔dendrite fusion is a merge signature).  Fragment
  type comes from **synapse polarity** (pre side = axon, post side = dendrite), which is
  cheap and robust on sparse fragments.
* **2-soma veto** — reject if the merged component would hold ≥ 2 soma-scale fragments.
* **caliber veto** — reject if the join's cross-section area ratio > ``caliber_ratio``.
* **quarantine** — reject if either fragment is itself merge-contaminated (carries strong
  *both* pre and post away from a soma).

A vetoed edge is **deferred** — the subtree stays split (a cheap missed split) rather than
risk a catastrophic cascade merge.  Each veto is individually toggleable for ablation.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

AXON, DEND, UNKNOWN = "axon", "dend", "unknown"


def fragment_types(pre_l2, post_l2, *, dom=0.6, min_syn=2, contam_min=2):
    """Type each L2 fragment from synapse polarity; flag merge-contaminated ones.

    ``pre_l2``  = L2 id of every synapse *pre* side (axonal contribution).
    ``post_l2`` = L2 id of every synapse *post* side (dendritic contribution).

    Returns ``(types, pre_count, post_count, contaminated)``:
    ``types[frag] in {axon, dend, unknown}`` (unknown if < ``min_syn`` synapses or mixed);
    ``contaminated`` = fragments with ≥ ``contam_min`` of *both* pre and post (an internal
    axon↔dendrite graft) — candidates to quarantine.
    """
    pre_c = Counter(int(x) for x in pre_l2 if x > 0)
    post_c = Counter(int(x) for x in post_l2 if x > 0)
    frags = set(pre_c) | set(post_c)
    types, contaminated = {}, set()
    for f in frags:
        p, q = pre_c.get(f, 0), post_c.get(f, 0)
        tot = p + q
        if tot < min_syn:
            types[f] = UNKNOWN
        else:
            af = p / tot
            types[f] = AXON if af >= dom else DEND if af <= 1 - dom else UNKNOWN
        if p >= contam_min and q >= contam_min:
            contaminated.add(f)
    return types, dict(pre_c), dict(post_c), contaminated


class _CDSU:
    """Union-find that carries per-component aggregates for grammar vetoes."""
    def __init__(self, pre_count, post_count, soma_frags):
        self.p = {}
        self.pre = dict(pre_count); self.post = dict(post_count)
        self.nsoma = {}                       # root -> # soma-scale fragments
        self._soma0 = set(soma_frags)

    def find(self, x):
        self.p.setdefault(x, x)
        if x not in self.nsoma:
            self.nsoma[x] = 1 if x in self._soma0 else 0
            self.pre.setdefault(x, 0); self.post.setdefault(x, 0)
        r = x
        while self.p[r] != r: r = self.p[r]
        while self.p[x] != r: self.p[x], x = r, self.p[x]
        return r

    def kind(self, root, *, dom=0.6, min_syn=2):
        p, q = self.pre.get(root, 0), self.post.get(root, 0)
        if p + q < min_syn: return UNKNOWN
        af = p / (p + q)
        return AXON if af >= dom else DEND if af <= 1 - dom else UNKNOWN

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        self.p[ra] = rb
        self.pre[rb] = self.pre.get(rb, 0) + self.pre.get(ra, 0)
        self.post[rb] = self.post.get(rb, 0) + self.post.get(ra, 0)
        self.nsoma[rb] = self.nsoma.get(rb, 0) + self.nsoma.get(ra, 0)


def constrained_union_find(edges, *, pre_count, post_count, soma_frags, contaminated,
                           area_of, threshold=0.0, caliber_ratio=2.5,
                           use_ad=True, use_soma=True, use_caliber=True,
                           use_quarantine=True, dom=0.6, min_syn=2):
    """Confident-first union-find with per-veto toggles.  Returns ``(dsu, rejects)``.

    ``rejects`` counts vetoes fired by cause; ``dsu.find`` gives the merged partition.
    """
    dsu = _CDSU(pre_count, post_count, soma_frags)
    contaminated = set(contaminated)
    rejects = {"ad": 0, "soma": 0, "caliber": 0, "quarantine": 0,
               "committed": 0, "committed_correct": 0}
    for w, a, b, _correct in sorted(edges, key=lambda e: -e[0]):
        if w < threshold:
            break
        if use_quarantine and (a in contaminated or b in contaminated):
            rejects["quarantine"] += 1; continue
        if use_caliber:
            aa, ab = area_of.get(a, 0.0), area_of.get(b, 0.0)
            if min(aa, ab) > 0 and max(aa, ab) / (min(aa, ab) + 1e-9) > caliber_ratio:
                rejects["caliber"] += 1; continue
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        if use_soma and (dsu.nsoma.get(ra, 0) + dsu.nsoma.get(rb, 0)) >= 2:
            rejects["soma"] += 1; continue
        if use_ad:
            ka, kb = dsu.kind(ra, dom=dom, min_syn=min_syn), dsu.kind(rb, dom=dom, min_syn=min_syn)
            has_soma = dsu.nsoma.get(ra, 0) + dsu.nsoma.get(rb, 0) > 0
            if not has_soma and {ka, kb} == {AXON, DEND}:
                rejects["ad"] += 1; continue
        dsu.union(a, b); rejects["committed"] += 1
        rejects["committed_correct"] += int(_correct)
    return dsu, rejects


def apply_partition(dsu, l2_ids):
    """Map synapse L2 ids through the merged partition (untouched frags map to self)."""
    return np.array([dsu.find(int(x)) if x > 0 else 0 for x in l2_ids])
