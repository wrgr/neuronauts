"""
Unit tests for 3D Morphological Grammar, Cajal Priors, and Tree-Beam MCTS Assembler.
"""

import pytest
import numpy as np

from neuronauts.morpho_grammar.blind_pcfg_morphology import BlindMorphologicalPCFG
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors
from neuronauts.morpho_grammar.blind_geodesic_em_tracer import BlindGeodesicEMTracer
from neuronauts.morpho_grammar.mcts_handshake_engine import TreeBeamMCTSAssembler
from neuronauts.morpho_grammar.synapse_segment_typer import (
    type_segment_from_synapses,
    compute_full_pairwise_confusion_matrix
)


def test_pcfg_derivation_rules():
    pcfg = BlindMorphologicalPCFG()
    lhs_soma = pcfg.derive_expected_lhs_from_parent("[SOMA]")
    assert len(lhs_soma) > 0
    assert any("ApicalTree" in lhs or "BasalTree" in lhs for lhs in lhs_soma)

    lhs_axon = pcfg.derive_expected_lhs_from_parent("[AXON_TRUNK]")
    assert len(lhs_axon) > 0
    assert any("Axon" in lhs for lhs in lhs_axon)


def test_cajal_space_and_time_priors():
    cajal = SantiagoCajalPriors()
    p_angle = cajal.compute_bifurcation_angle_prior(r_mother=300.0, r_d1=220.0, r_d2=180.0, observed_angle_rad=0.85)
    assert 0.0 <= p_angle <= 1.0

    p_time = cajal.compute_conduction_time_prior(centrifugal_order=2, dist_from_soma_nm=15000.0, is_axon=False)
    assert 0.0 < p_time <= 1.0


def test_synapse_segment_typing():
    # Soma by caliber
    t_soma = type_segment_from_synapses(n_pre=2, n_post=5, mean_radius_nm=800.0, max_radius_nm=1500.0)
    assert t_soma == "Soma"

    # Axon by pre-dominance
    t_axon = type_segment_from_synapses(n_pre=10, n_post=1, mean_radius_nm=80.0, max_radius_nm=100.0)
    assert t_axon == "Axon"

    # Dendrite by post-dominance
    t_dend = type_segment_from_synapses(n_pre=1, n_post=12, mean_radius_nm=250.0, max_radius_nm=350.0)
    assert t_dend == "Dendrite"


def test_mcts_handshake_engine():
    engine = TreeBeamMCTSAssembler(emb_dim=32, seed=42)

    parent_tok = {
        "symbol": "[SOMA]",
        "fragment_id": "frag_001_0",
        "coord_nm": [10000.0, 10000.0, 1000.0],
        "radius_nm": 1500.0,
        "tangent": [0.0, -1.0, 0.0],
        "syn_partners": [101, 102, 103],
        "n_syn_pre": 1,
        "n_syn_post": 8
    }

    mask_tok = {
        "symbol": "[MASK_FRAGMENT]",
        "fragment_id": "mask_001_1",
        "coord_nm": [10000.0, 8500.0, 1000.0],
        "radius_nm": 450.0,
        "tangent": [0.0, -1.0, 0.0],
        "syn_partners": [101, 102],
        "n_syn_pre": 0,
        "n_syn_post": 5
    }

    child_tok = {
        "symbol": "[APICAL_TRUNK]",
        "fragment_id": "frag_001_1",
        "coord_nm": [10000.0, 6000.0, 1000.0],
        "radius_nm": 400.0,
        "tangent": [0.0, -1.0, 0.0],
        "syn_partners": [101, 102, 104],
        "n_syn_pre": 0,
        "n_syn_post": 9
    }

    cand_pool = [
        {"token": child_tok, "fragment_id": "frag_001_1"},
        {"token": parent_tok, "fragment_id": "frag_001_0"}
    ]

    res = engine.run_tree_beam_mcts(
        parent_token=parent_tok,
        mask_token=mask_tok,
        candidate_pool=cand_pool
    )

    assert res["predicted_id"] == "frag_001_1"
    assert res["accepted"] is True
    assert res["p_handshake"] > 0.50
