"""Offline tests for the Pillar-1 grammar energy (no network)."""
import numpy as np

from experiments.proofread.grammar_energy import (
    grammar_energy, cut_delta_energy, join_delta_energy)


def _line(n=11, step=1000.0):
    V = np.array([[i * step, 0.0, 0.0] for i in range(n)])
    E = np.array([[i, i + 1] for i in range(n - 1)])
    R = np.full(n, 200.0)
    return V, E, R


def test_single_soma_low_energy():
    V, E, R = _line()
    R[0:3] = 5000.0  # one soma
    e = grammar_energy(V, E, R)
    assert e.soma == 0.0          # n_soma == 1 -> (1-1)+ == 0
    assert e.disconnect == 0.0


def test_two_soma_high_energy():
    # two big-radius blobs far apart -> a merge
    V = np.array([[0, 0, 0], [1000, 0, 0], [60_000, 0, 0], [61_000, 0, 0]], float)
    E = np.array([[0, 1], [1, 2], [2, 3]])
    R = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    assert grammar_energy(V, E, R).soma == 1.0   # 2 somas -> energy 1


def test_join_two_somata_is_rejected():
    # each object is a single-soma cell; joining them creates a 2-soma object
    Va, Ea, Ra = _line(); Ra[0:3] = 5000.0
    Vb, Eb, Rb = _line(); Vb = Vb + np.array([0, 60_000, 0]); Rb[0:3] = 5000.0
    jde = join_delta_energy(Va, Ea, Ra, Vb, Eb, Rb)
    assert jde < 0                # joining two somata is ungrammatical -> reject


def test_cut_removes_multisoma():
    # a bridged two-soma object (somata >8um apart); cutting the bridge should help
    V = np.array([[0, 0, 0], [1000, 0, 0], [60_000, 0, 0], [61_000, 0, 0]], float)
    E = np.array([[0, 1], [2, 3], [1, 2]])   # last edge is the bridge
    R = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    assert grammar_energy(V, E, R).soma == 1.0        # 2 spatial somata
    assert cut_delta_energy(V, E, R, (1, 2)) > 0      # cutting the bridge helps
