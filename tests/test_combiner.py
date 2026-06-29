"""Offline test for the learned confidence combiner (no CAVE)."""
import numpy as np
import pytest

from experiments.fingerprints import train_combiner as tc


def _make_site(rng, signal="art"):
    C = int(rng.integers(4, 8))
    it = np.zeros(C, np.float32); t = int(rng.integers(C)); it[t] = 1
    gd = rng.uniform(200, 2000, C)
    arts = rng.uniform(-0.2, 0.2, C)
    bios = rng.uniform(-0.2, 0.2, C)
    if signal == "art":
        arts[t] += 0.6          # true candidate is the art-band favourite
    gz = -tc._z(gd)
    X = np.stack([gz, tc._z(arts), tc._z(bios), arts, bios,
                  (gd == gd.min()).astype(np.float32),
                  (arts == arts.max()).astype(np.float32)], 1).astype(np.float32)
    return X, it, gd, arts


def test_combiner_learns_to_follow_signal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    train = [_make_site(rng) for _ in range(150)]
    test = [_make_site(rng) for _ in range(60)]
    net = tc.train_mlp(train, epochs=150, verbose=False)
    res = tc.evaluate(test, net)
    # the true candidate is the art favourite, not the geom-nearest:
    assert res["combiner_top1"] >= 0.8
    assert res["combiner_top1"] > res["geom_top1"]


def test_z_helper():
    z = tc._z([1.0, 2.0, 3.0])
    assert abs(float(z.mean())) < 1e-6
