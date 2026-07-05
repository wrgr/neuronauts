"""Orchestrator for the two-cue abstaining auto-proofreader.

Ties the three pillars into one flow over a cached column ``SideTable``
(ground truth from the v117->later proofreading divergence):

    Pillar 1  grammar_energy   — global shape grammar as ΔEnergy per edit
    Pillar 2  local_evidence   — local EM ultrastructure at the edit site
    Pillar 3  complementarity  — calibrated combiner over both cues (leakage-safe)
              queue            — ranked, abstaining edit/review worklist + ngl links

This module is deliberately thin: the real logic lives in the pillar modules so
each is independently testable.  See ``run_complementarity.py`` for the CLI.

    from experiments.proofread.pipeline import run_pipeline
    res, items = run_pipeline("cache/sidetable/col_n1_v1718.npz")
"""
from __future__ import annotations

import numpy as np

from experiments.pcfg.synapse_correction import SideTable, summarize_edits
from experiments.fingerprints.cutface.learned_cutface_encoder import (
    load_encoder, make_embed_fn)
from experiments.proofread.complementarity import run_complementarity
from experiments.proofread.queue import build_queue, queue_summary


def load_side_table(path: str) -> SideTable:
    d = np.load(path)
    return SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])


def run_pipeline(sidetable_path: str, *,
                 encoder_path: str = "experiments/fingerprints/cutface_encoder.pt",
                 max_candidates: int = 80, max_pair_nm: float = 6000.0,
                 mip: int = 1, seed: int = 0, with_urls: bool = True,
                 verbose: bool = True):
    """Run the full two-cue eval + ranked queue; return ``(res, queue_items)``."""
    tab = load_side_table(sidetable_path)
    if verbose:
        print("edit summary:", summarize_edits(tab))
    embed = make_embed_fn(load_encoder(encoder_path))
    res = run_complementarity(tab, embed, max_candidates=max_candidates,
                              max_pair_nm=max_pair_nm, mip=mip, seed=seed, verbose=verbose)
    if "error" in res:
        return res, []
    items = build_queue(res, with_urls=with_urls)
    if verbose:
        print("\n" + queue_summary(items))
    return res, items
