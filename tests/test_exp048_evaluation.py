import importlib.util
from pathlib import Path

import numpy as np

from neuronauts.assemble import HungarianBipartiteAssembler
from neuronauts.morpho_grammar.santiago_v2_grammar import derive_expected_lhs_v2


def _script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hungarian_scores_full_local_pool_without_grammar_beam():
    class Engine:
        def __init__(self):
            self.calls = 0

        def evaluate_bidirectional_handshake(self, left, right):
            self.calls += 1
            return 0.9

    engine = Engine()
    tokens = [{"fragment_id": str(i), "coord_nm": np.array([i * 10.0, 0, 0])} for i in range(4)]
    links, meta = HungarianBipartiteAssembler(engine, max_search_dist_nm=100, verbose=False).assemble_volume_bipartite(tokens)
    assert engine.calls == 12
    assert meta["n_cands_scored"] == 12
    assert len(links) == 4


def test_interneuron_soma_has_interneuron_productions():
    lhs = derive_expected_lhs_v2("[SOMA]", {"n_pre": 9, "n_post": 1, "max_radius_nm": 800, "bouton_density": .2})
    assert lhs == ("<AspinyDendriteTree>", "<DenseAxonPlexus>")


def test_population_tokens_are_blind_and_stratified():
    exp050 = _script("benchmark_exp050_interneuron_stratified.py")
    tokens, _, subtype, _ = exp050.build_population()
    assert not any({"gt_cell_type", "cell_type", "subtype", "gt_label"} & token.keys() for token in tokens)
    assert {value for value in subtype.values()} == {"Pyramidal", "Basket", "Martinotti", "VIP", "Glia"}


def test_train_test_boxes_are_spatially_disjoint():
    exp049 = _script("benchmark_exp049_dense_subvolume.py")
    train = exp049.make_cube_bbox_nm(exp049.DEFAULT_TRAIN_CENTER_NM, 100)
    test = exp049.make_cube_bbox_nm(exp049.DEFAULT_TEST_CENTER_NM, 30)
    assert not exp049.boxes_overlap(train, test)
