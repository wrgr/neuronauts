"""Offline assembly test for learned-band eval (no CAVE / no torch)."""
import numpy as np
import pytest


def test_evaluate_bands_learned_assembles(monkeypatch):
    import experiments.fingerprints.cutface.train_band_encoders as tb
    from experiments.fingerprints.cutface import v117_error_relink as v
    P = 48
    site = v.ErrorSite(root=1, pos_main_nm=(0, 0, 0), pos_frag_nm=(0, 0, 0), gap_nm=500.0, frag_l2=3)
    monkeypatch.setattr(v, "sites_from_l2_graph", lambda *a, **k: [site])
    rng = np.random.default_rng(0)
    base_lo = rng.normal(size=(P, P)).astype(np.float32)
    base_hi = rng.normal(size=(P, P)).astype(np.float32)

    def fake_faces(cl, ts, s, **k):
        lows = np.stack([base_lo + 0.02 * rng.normal(size=(P, P)),
                         rng.normal(size=(P, P)), rng.normal(size=(P, P))]).astype(np.float32)
        highs = np.stack([base_hi + 0.02 * rng.normal(size=(P, P)),
                          rng.normal(size=(P, P)), rng.normal(size=(P, P))]).astype(np.float32)
        return {"q_low": base_lo, "q_high": base_hi, "low": lows, "high": highs,
                "is_true": np.array([True, False, False]),
                "geom_dist": np.array([100.0, 50.0, 200.0])}   # geom picks idx1 -> miss

    monkeypatch.setattr(tb, "site_faces_bands", fake_faces)
    emb = lambda X: np.asarray(X).reshape(len(X), -1)
    ranks, ncand, recov = tb.evaluate_bands_learned(None, None, [1, 1], emb, emb, max_sites=3)
    for key in ("geom", "bio", "art", "bio+art", "geom+bio+art", "gated_geom_top3+bioart"):
        assert key in ranks and len(ranks[key]) == 2
    # bio band matches the true partner -> bio should rank it top-1
    assert np.mean([r == 0 for r in ranks["bio"]]) == 1.0
    # geometry misses here (nearest is a distractor)
    assert np.mean([r == 0 for r in ranks["geom"]]) == 0.0
