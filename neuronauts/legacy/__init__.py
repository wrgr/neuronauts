"""Legacy v1 modules — the agent/membrane simulation pipeline.

Everything under ``neuronauts.legacy`` predates the tree-DNA / global-assembly
direction and is **not** part of the active import surface (``import neuronauts``
does not load it). It is retained for historical reference, the v1 synthetic
benchmark / GAT-from-simulation path, and existing tests. New code should not
import from here. See ``docs/stage_ownership.md`` for the quarantine plan.
"""
