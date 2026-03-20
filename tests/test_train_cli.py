from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _import_train():
    spec = importlib.util.spec_from_file_location(
        "train_script_cli",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TrainCliProofreadCoreTest(unittest.TestCase):
    def setUp(self):
        self.mod = _import_train()

    def test_parse_args_accepts_proofread_core_strategy(self):
        args = self.mod.parse_args(
            [
                "build-dataset",
                "--strategy",
                "proofread-core",
            ]
        )
        self.assertEqual(args.strategy, "proofread-core")

    def test_cmd_build_dataset_uses_roots_tsv_when_provided(self):
        calls = {}

        def _fake_build_root_neighborhood_cache(**kwargs):
            calls["build"] = kwargs

        def _fake_sample_proofread_roots(**kwargs):
            calls["sample"] = kwargs
            return [111, 222]

        fake_module = types.ModuleType("experiments.root_neighborhood_dataset")
        fake_module.build_root_neighborhood_cache = _fake_build_root_neighborhood_cache
        fake_module.sample_proofread_roots = _fake_sample_proofread_roots

        with tempfile.TemporaryDirectory() as tmpdir:
            roots_tsv = Path(tmpdir) / "roots.tsv"
            with roots_tsv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow(["root_id"])
                writer.writerow([101])
                writer.writerow([202])

            args = types.SimpleNamespace(
                cache_dir=str(Path(tmpdir) / "cache"),
                counts_tsv=None,
                nucleus_csv=None,
                n_boxes=50,
                min_synapses=15,
                max_synapses=20000,
                box_side_um=6.0,
                seed=7,
                strategy="proofread-core",
                cave_token="tok",
                cave_version=117,
                no_em=False,
                proofread_datastack="minnie65_public",
                proofread_n_roots=25,
                proofread_roots_tsv=str(roots_tsv),
                proofread_radius_um=40.0,
                proofread_anchor_side="both",
                proofread_min_anchor_synapses=50,
                proofread_per_root_timeout_s=180,
                proofread_require_dendrite=True,
                proofread_require_axon=False,
            )

            with mock.patch.dict(sys.modules, {"experiments.root_neighborhood_dataset": fake_module}):
                rc = self.mod.cmd_build_dataset(args)

        self.assertEqual(rc, 0)
        self.assertNotIn("sample", calls)
        self.assertEqual(calls["build"]["root_ids"], [101, 202])
        self.assertEqual(calls["build"]["version"], 117)
        self.assertEqual(calls["build"]["cache_dir"], args.cache_dir)
        self.assertEqual(calls["build"]["token"], "tok")

    def test_cmd_build_dataset_samples_roots_when_tsv_missing(self):
        calls = {}

        def _fake_build_root_neighborhood_cache(**kwargs):
            calls["build"] = kwargs

        def _fake_sample_proofread_roots(**kwargs):
            calls["sample"] = kwargs
            return [303, 404]

        fake_module = types.ModuleType("experiments.root_neighborhood_dataset")
        fake_module.build_root_neighborhood_cache = _fake_build_root_neighborhood_cache
        fake_module.sample_proofread_roots = _fake_sample_proofread_roots

        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                cache_dir=str(Path(tmpdir) / "cache"),
                counts_tsv=None,
                nucleus_csv=None,
                n_boxes=50,
                min_synapses=15,
                max_synapses=20000,
                box_side_um=6.0,
                seed=9,
                strategy="proofread-core",
                cave_token=None,
                cave_version=117,
                no_em=True,
                proofread_datastack="minnie65_public",
                proofread_n_roots=12,
                proofread_roots_tsv=None,
                proofread_radius_um=55.0,
                proofread_anchor_side="pre",
                proofread_min_anchor_synapses=33,
                proofread_per_root_timeout_s=90,
                proofread_require_dendrite=False,
                proofread_require_axon=True,
            )

            with mock.patch.dict(sys.modules, {"experiments.root_neighborhood_dataset": fake_module}):
                rc = self.mod.cmd_build_dataset(args)

        self.assertEqual(rc, 0)
        self.assertEqual(calls["sample"]["version"], 117)
        self.assertEqual(calls["sample"]["n_roots"], 12)
        self.assertFalse(calls["sample"]["require_dendrite"])
        self.assertTrue(calls["sample"]["require_axon"])
        self.assertEqual(calls["build"]["root_ids"], [303, 404])
        self.assertEqual(calls["build"]["radius_um"], 55.0)
        self.assertEqual(calls["build"]["anchor_side"], "pre")
        self.assertEqual(calls["build"]["min_anchor_synapses"], 33)


class TrainCliSkeletonGraphTest(unittest.TestCase):
    def setUp(self):
        self.mod = _import_train()

    def test_parse_args_accepts_skeleton_graph_source(self):
        args = self.mod.parse_args(
            [
                "train",
                "--graph-source",
                "skeleton",
                "--skeleton-version",
                "117",
            ]
        )
        self.assertEqual(args.graph_source, "skeleton")
        self.assertEqual(args.skeleton_version, 117)

    def test_normalize_graph_source_defaults_skeleton_version_to_base(self):
        args = types.SimpleNamespace(
            graph_source="skeleton",
            base_version=117,
            target_version=1412,
            skeleton_version=None,
        )
        out = self.mod._normalize_graph_source_args(args)
        self.assertEqual(out.skeleton_version, 117)

    def test_normalize_graph_source_rejects_target_materialization_skeletons(self):
        args = types.SimpleNamespace(
            graph_source="skeleton",
            base_version=117,
            target_version=1412,
            skeleton_version=1412,
        )
        with self.assertRaises(SystemExit):
            self.mod._normalize_graph_source_args(args)
