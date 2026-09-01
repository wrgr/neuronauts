"""Spatial train/val split with a seam buffer -- the leak-prevention primitive."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.harness.spatial_split import (
    SPLIT_BUFFER, SPLIT_TRAIN, SPLIT_VAL, assign_split, describe, pair_split,
)


def test_assign_split_three_zones():
    centroid = np.array([[0, 0, 0], [40000, 0, 0], [100000, 0, 0]], np.float64)
    split = assign_split(centroid, axis=0, centre_nm=50000, buffer_nm=20000)
    assert split.tolist() == [SPLIT_TRAIN, SPLIT_BUFFER, SPLIT_VAL]


def test_assign_split_uses_requested_axis():
    centroid = np.array([[0, 0, 0], [0, 100000, 0]], np.float64)
    split = assign_split(centroid, axis=1, centre_nm=50000, buffer_nm=1000)
    assert split.tolist() == [SPLIT_TRAIN, SPLIT_VAL]


def test_pair_split_requires_both_sides_to_agree():
    a = np.array([SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_VAL])
    b = np.array([SPLIT_TRAIN, SPLIT_VAL, SPLIT_VAL])
    got = pair_split(a, b)
    assert got.tolist() == [SPLIT_TRAIN, SPLIT_BUFFER, SPLIT_VAL]


def test_pair_split_buffer_atom_poisons_the_pair():
    a = np.array([SPLIT_BUFFER, SPLIT_TRAIN])
    b = np.array([SPLIT_TRAIN, SPLIT_TRAIN])
    got = pair_split(a, b)
    assert got.tolist() == [SPLIT_BUFFER, SPLIT_TRAIN]


def test_describe_counts():
    split = np.array([SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_VAL, SPLIT_BUFFER])
    d = describe(split)
    assert d == {"n_train": 2, "n_val": 1, "n_buffer": 1}
