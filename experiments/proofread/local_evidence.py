"""Pillar 2 — local *ultrastructure* evidence at a candidate edit site.

The global grammar (Pillar 1) parses the *shape* of the reconstructed neuron.  It
is blind to the residual case a human resolves by looking at the raw EM at the
ambiguous point: two same-compartment processes that meet — is this one continuous
neurite (a false split to join) or two touching neurites (a false merge to cut)?
The trained proofreader reads two local cues there:

1. **cross-section / cytoplasm match** — do the two faces look like the same
   process (same caliber, same intra-axonal ultrastructure)?  We reuse the
   committed contrastive **cut-face encoder** (``cutface_encoder*.pt`` via
   ``neuronauts.em_corridor.cross_section_patch``) — the proven substrate
   (P=1.0 @ 11% coverage on held-out re-ID).
2. **membrane barrier** — is there a dark membrane plane cutting across the line
   between the two points?  A continuous process has no barrier; two touching
   neurites have one.  We sample mean EM intensity in thin discs perpendicular to
   the axis and score the darkest interior disc relative to the endpoints.  This
   is a *first-cut* approximation of the human's membrane read, not the RoboEM
   flight method — labelled as such (CLAUDE.md).

Both cues come from *one* bulk EM+seg fetch per site.  ``cutface_sim`` high and
``barrier`` low ⇒ local EM supports a join; low ``cutface_sim`` / high ``barrier``
⇒ supports a cut.  The combiner (Pillar 3) learns the weighting against the edit
log; here we keep the channels raw and interpretable.

No torch is needed to *import* this module; the encoder is supplied by the caller
as ``embed_fn`` (build once via
``experiments.fingerprints.cutface.learned_cutface_encoder.load_encoder`` +
``make_embed_fn``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuronauts.em_corridor import CUTFACE_PATCH, CUTFACE_SLAB, cross_section_patch
from neuronauts.fetch import MICRONS_EM_PATH, MICRONS_SEG_PATH


@dataclass
class LocalEvidence:
    """Interpretable local-EM channels at one candidate edit site."""
    cutface_sim: float    # cosine of cut-face embeddings in [-1, 1]; high => same process
    barrier: float        # membrane-barrier score in [0, 1]; high => a dark plane separates them
    axis_len_nm: float    # distance between the two points
    ok: bool              # both endpoints had a non-background cross-section footprint

    def as_dict(self) -> dict:
        return {"cutface_sim": self.cutface_sim, "barrier": self.barrier,
                "axis_len_nm": self.axis_len_nm, "ok": self.ok}

    @property
    def continuation(self) -> float:
        """A single ``[0, 1]`` "these are one continuous process" score.

        Convenience combiner-free summary: high cut-face similarity and a low
        membrane barrier both argue for continuation.  The learned combiner in
        Pillar 3 should *replace* this hand-set average — it exists only so the
        stream can be inspected on its own.
        """
        sim01 = 0.5 * (float(self.cutface_sim) + 1.0)
        return float(np.clip(0.5 * sim01 + 0.5 * (1.0 - self.barrier), 0.0, 1.0))


def _membrane_barrier(em_vol, pos_a_nm, pos_b_nm, *, disc_radius_nm=250.0,
                      thick_nm=32.0, n_steps=25, end_frac=0.16) -> float:
    """Darkest interior cross-disc, relative to the endpoints — a membrane proxy.

    Sample ``n_steps`` discs *perpendicular* to the axis along the segment: each
    disc is a thin slab (half-thickness ``thick_nm`` along the axis) of radius
    ``disc_radius_nm`` across it — wide enough to average the process cross-section,
    thin enough to localise a membrane plane.  A continuous neurite is uniformly
    dark cytoplasm; a membrane between two touching processes shows as a *local*
    intensity dip in the interior darker than the two endpoint neighbourhoods.
    Returns ``(end_mean - interior_min) / end_mean`` clipped to ``[0, 1]``
    (0 = no barrier, larger = sharper dark plane).

    Handles oblique edits (the disc is defined by the axis direction, not axis-
    aligned).  First-cut approximation (CLAUDE.md): a crude intensity read, not a
    trained membrane detector; used only as one raw channel for the combiner.
    """
    vox = np.asarray(em_vol.voxel_size_nm, dtype=np.float64)
    origin = np.asarray(em_vol.bbox_voxels[0], dtype=np.float64)
    data = em_vol.data
    shape = np.asarray(data.shape)
    a = np.asarray(pos_a_nm, float)
    b = np.asarray(pos_b_nm, float)
    axis = b - a
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return 0.0
    u = axis / axis_len
    # half-extent (in voxels) of the local box we scan around each step point
    half_vox = np.maximum(np.ceil(disc_radius_nm / vox).astype(int), 1)

    means = np.full(n_steps, np.nan)
    for s in range(n_steps):
        t = s / (n_steps - 1)
        p = a + t * axis
        idx = np.round(p / vox - origin - 0.5).astype(int)
        lo = np.maximum(idx - half_vox, 0)
        hi = np.minimum(idx + half_vox + 1, shape)
        if np.any(hi <= lo):
            continue
        gx = np.arange(lo[0], hi[0]); gy = np.arange(lo[1], hi[1]); gz = np.arange(lo[2], hi[2])
        # voxel-centre nm coords, vector from step point, decomposed along / across axis
        cx = (origin[0] + gx + 0.5) * vox[0]; cy = (origin[1] + gy + 0.5) * vox[1]
        cz = (origin[2] + gz + 0.5) * vox[2]
        dx = cx - p[0]; dy = cy - p[1]; dz = cz - p[2]
        DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing="ij")
        along = DX * u[0] + DY * u[1] + DZ * u[2]
        perp2 = (DX * DX + DY * DY + DZ * DZ) - along * along
        disc = (np.abs(along) <= thick_nm) & (perp2 <= disc_radius_nm ** 2)
        block = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if disc.any():
            means[s] = float(block[disc].mean())

    if np.isnan(means).all():
        return 0.0
    k = max(1, int(round(end_frac * n_steps)))
    ends = np.concatenate([means[:k], means[-k:]])
    end_mean = np.nanmean(ends)
    interior = means[k:n_steps - k]
    if not np.isfinite(end_mean) or end_mean <= 1e-6 or np.isnan(interior).all():
        return 0.0
    interior_min = np.nanmin(interior)
    return float(np.clip((end_mean - interior_min) / end_mean, 0.0, 1.0))


def local_evidence(pos_a_nm, pos_b_nm, embed_fn, *, mip=1,
                   patch=CUTFACE_PATCH, slab=CUTFACE_SLAB, margin_nm=1000.0,
                   disc_radius_nm=250.0, em_path=MICRONS_EM_PATH,
                   seg_path=MICRONS_SEG_PATH, em_vol=None, seg_vol=None) -> LocalEvidence:
    """Local-EM evidence for the edit between ``pos_a_nm`` and ``pos_b_nm``.

    One bulk EM+seg fetch (unless ``em_vol``/``seg_vol`` are supplied) covering
    both points; extracts each point's cut-face cross-section, embeds both via
    ``embed_fn``, and measures the membrane barrier along the connecting line.
    """
    a = np.asarray(pos_a_nm, float)
    b = np.asarray(pos_b_nm, float)
    if em_vol is None or seg_vol is None:
        from neuronauts.fetch import fetch_volume, fetch_seg_volume
        lo = np.minimum(a, b) - margin_nm
        hi = np.maximum(a, b) + margin_nm
        bbox = ((float(lo[0]), float(lo[1]), float(lo[2])),
                (float(hi[0]), float(hi[1]), float(hi[2])))
        em_vol = fetch_volume(bbox, mip=mip, em_path=em_path)
        seg_vol = fetch_seg_volume(bbox, mip=mip, seg_path=seg_path)

    pa = cross_section_patch(em_vol, seg_vol, a, patch=patch, slab=slab)
    pb = cross_section_patch(em_vol, seg_vol, b, patch=patch, slab=slab)
    ok = bool(pa.any() and pb.any())
    if ok:
        emb = np.asarray(embed_fn(np.stack([pa, pb]).astype(np.float32)))
        na = np.linalg.norm(emb[0]) + 1e-9
        nb = np.linalg.norm(emb[1]) + 1e-9
        cutface_sim = float(emb[0] @ emb[1] / (na * nb))
    else:
        cutface_sim = 0.0
    barrier = _membrane_barrier(em_vol, a, b, disc_radius_nm=disc_radius_nm)
    return LocalEvidence(cutface_sim=cutface_sim, barrier=barrier,
                         axis_len_nm=float(np.linalg.norm(b - a)), ok=ok)


def batch_local_evidence(sites, embed_fn, *, mip=1, patch=CUTFACE_PATCH,
                         slab=CUTFACE_SLAB, margin_nm=1000.0, disc_radius_nm=250.0,
                         max_span_nm=40_000.0, em_path=MICRONS_EM_PATH,
                         seg_path=MICRONS_SEG_PATH, verbose=False) -> list[LocalEvidence]:
    """Local evidence for many ``(pos_a_nm, pos_b_nm)`` sites with one bulk fetch.

    All sites must lie in a shared bounding box no larger than ``max_span_nm`` on
    any axis (keep a batch spatially local, e.g. one neuron / one column tile);
    otherwise the single fetch is huge.  Raises if the span is exceeded so we never
    silently pull a giant volume (CLAUDE.md: no accidental heavy queries).
    """
    from neuronauts.fetch import fetch_volume, fetch_seg_volume
    pts = np.asarray([p for site in sites for p in site], float)
    if len(pts) == 0:
        return []
    lo = pts.min(axis=0) - margin_nm
    hi = pts.max(axis=0) + margin_nm
    span = hi - lo
    if np.any(span > max_span_nm):
        raise ValueError(
            f"batch span {span.astype(int).tolist()} nm exceeds max_span_nm="
            f"{max_span_nm:.0f}; split the batch into spatially local tiles.")
    bbox = ((float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2])))
    if verbose:
        print(f"[local] fetching MIP{mip} EM+seg for {len(sites)} sites, "
              f"span {span.astype(int).tolist()} nm ...")
    em_vol = fetch_volume(bbox, mip=mip, em_path=em_path)
    seg_vol = fetch_seg_volume(bbox, mip=mip, seg_path=seg_path)
    return [local_evidence(a, b, embed_fn, mip=mip, patch=patch, slab=slab,
                           disc_radius_nm=disc_radius_nm, em_vol=em_vol, seg_vol=seg_vol)
            for (a, b) in sites]
