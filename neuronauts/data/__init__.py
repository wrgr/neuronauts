"""Data stage: seg-volume + synapses → Fragment artifacts."""

from .fragments import extract_fragments_for_region, skeleton_to_fragment

__all__ = [
    "extract_fragments_for_region",
    "skeleton_to_fragment",
]
