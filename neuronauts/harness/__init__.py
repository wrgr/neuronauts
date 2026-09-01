"""Offline experiment harness: cached real substrate + well-posed evaluation.

The point of this package is to separate the slow, networked, once-only work
(pulling real geometry and lineage out of CAVE) from the fast, offline,
repeatable work (scoring candidate merges and assembling arbors). Earlier
experiment rounds mixed the two, so every idea paid the full fetch cost and
most runs died on timeouts before producing a comparable number.

Layout:
  substrate.py -- build/load the cached region substrate
  split.py     -- spatial train/val split with a seam buffer
  metrics.py   -- evaluation that stays meaningful at realistic base rates
"""

from neuronauts.harness.substrate import Substrate, load_substrate

__all__ = ["Substrate", "load_substrate"]
