"""Validate Pillar 2 local evidence on REAL MICrONS EM.

Re-ID sanity: two points on the SAME neurite (separated in z, a synthetic "cut")
should score high cut-face similarity and low membrane barrier; a point pair
spanning TWO DIFFERENT neurites should score lower similarity. This exercises
cross_section_patch + committed encoder + barrier on real EM end-to-end.
"""
import numpy as np
from neuronauts.fetch import fetch_volume, fetch_seg_volume
from experiments.fingerprints.cutface.learned_cutface_encoder import load_encoder, make_embed_fn
from experiments.proofread.local_evidence import local_evidence

enc = load_encoder("experiments/fingerprints/cutface_encoder.pt")
embed = make_embed_fn(enc)

c = np.array([700000.0, 700000.0, 800000.0])
half = np.array([2000.0, 2000.0, 1200.0])          # ~4x4x2.4 um box
bbox = (tuple(c - half), tuple(c + half))
em = fetch_volume(bbox, mip=1); seg = fetch_seg_volume(bbox, mip=1)
vox = np.asarray(seg.voxel_size_nm, float); origin = np.asarray(seg.bbox_voxels[0], float)

def to_nm(idx):
    return (origin + np.asarray(idx, float) + 0.5) * vox

# pick neurites that span the z-range (so we can sample two z-separated faces)
sd = seg.data; nz = sd.shape[2]
ids, counts = np.unique(sd[sd > 0], return_counts=True)
order = ids[np.argsort(counts)[::-1]]
zlo, zhi = 2, nz - 3

def sample_point(sid, z):
    xs, ys = np.nonzero(sd[:, :, z] == sid)
    if len(xs) == 0:
        return None
    m = len(xs) // 2
    return to_nm([xs[m], ys[m], z])

# find two big neurites each present at both zlo and zhi
usable = [s for s in order[:20]
          if (sd[:, :, zlo] == s).any() and (sd[:, :, zhi] == s).any()][:6]
print(f"usable neurites (present at z={zlo} and z={zhi}): {len(usable)}")

print("\n=== SAME neurite (two z-separated faces): want high sim, low barrier ===")
same = []
for s in usable:
    pa, pb = sample_point(s, zlo), sample_point(s, zhi)
    if pa is None or pb is None:
        continue
    ev = local_evidence(pa, pb, embed, em_vol=em, seg_vol=seg)
    same.append(ev.cutface_sim)
    print(f"  seg {int(s)}: sim={ev.cutface_sim:+.3f} barrier={ev.barrier:.3f} "
          f"cont={ev.continuation:.3f} len={ev.axis_len_nm:.0f}nm ok={ev.ok}")

print("\n=== DIFFERENT neurites (face of A vs face of B): want lower sim ===")
diff = []
for i in range(min(4, len(usable))):
    for j in range(i + 1, min(4, len(usable))):
        pa, pb = sample_point(usable[i], zlo), sample_point(usable[j], zhi)
        if pa is None or pb is None:
            continue
        ev = local_evidence(pa, pb, embed, em_vol=em, seg_vol=seg)
        diff.append(ev.cutface_sim)
        print(f"  {int(usable[i])} vs {int(usable[j])}: sim={ev.cutface_sim:+.3f} "
              f"barrier={ev.barrier:.3f} cont={ev.continuation:.3f}")

if same and diff:
    print(f"\nmean sim  SAME={np.mean(same):+.3f}  DIFFERENT={np.mean(diff):+.3f}  "
          f"-> {'SEPARABLE (same>diff)' if np.mean(same) > np.mean(diff) else 'NOT separable'}")
