"""Tests for CAVE synapse/degree modules (cave_synapse_degrees_v1412,
cave_synapse_counts_v1412) using mocked CAVEclient.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class BuildRootTableDegreesTest(unittest.TestCase):
    """Test build_root_table in cave_synapse_degrees_v1412 (pure logic)."""

    def test_build_root_table_sums_in_out_and_flags_soma(self):
        from neuronauts.cave_synapse_degrees_v1412 import build_root_table

        deg_df = pd.DataFrame({
            "root_id": [100, 200, 300],
            "in_synapse_count": [5, 10, 0],
            "out_synapse_count": [3, 7, 12],
        })
        soma_roots = {100, 300}
        result = build_root_table(deg_df, soma_roots)

        # Sorted by total_synapse_count descending: 200(17), 300(12), 100(8)
        self.assertEqual(list(result["root_id"]), [200, 300, 100])
        self.assertEqual(list(result["total_synapse_count"]), [17, 12, 8])
        self.assertEqual(list(result["has_soma"]), [False, True, True])

    def test_build_root_table_sorts_by_total_descending(self):
        from neuronauts.cave_synapse_degrees_v1412 import build_root_table

        deg_df = pd.DataFrame({
            "root_id": [1, 2, 3],
            "in_synapse_count": [1, 100, 50],
            "out_synapse_count": [1, 0, 0],
        })
        result = build_root_table(deg_df, set())
        self.assertEqual(list(result["root_id"]), [2, 3, 1])


class BuildRootTableCountsTest(unittest.TestCase):
    """Test build_root_table in cave_synapse_counts_v1412 (pure logic)."""

    def test_build_root_table_combines_pre_post_counts(self):
        from neuronauts.cave_synapse_counts_v1412 import build_root_table

        pre_counts = {10: 50, 20: 30}
        post_counts = {10: 20, 30: 5}
        soma_roots = pd.Series([10])

        result = build_root_table(pre_counts, post_counts, soma_roots)

        self.assertEqual(set(result["root_id"]), {10, 20, 30})
        self.assertEqual(int(result[result["root_id"] == 10]["pre_synapse_count"].iloc[0]), 50)
        self.assertEqual(int(result[result["root_id"] == 10]["post_synapse_count"].iloc[0]), 20)
        self.assertEqual(int(result[result["root_id"] == 30]["pre_synapse_count"].iloc[0]), 0)
        self.assertEqual(int(result[result["root_id"] == 30]["post_synapse_count"].iloc[0]), 5)
        self.assertTrue(bool(result[result["root_id"] == 10]["has_soma"].iloc[0]))
        self.assertFalse(bool(result[result["root_id"] == 20]["has_soma"].iloc[0]))


class FetchDegreeTableMockedTest(unittest.TestCase):
    """Test fetch_degree_table with mocked CAVE client."""

    def test_fetch_degree_table_normalizes_pt_root_id_column(self):
        from neuronauts.cave_synapse_degrees_v1412 import fetch_degree_table

        mock_df = pd.DataFrame({
            "pt_root_id": [100, 200],
            "in_degree": [5, 10],
            "out_degree": [3, 7],
        })
        mock_client = MagicMock()
        mock_client.version = 1412
        mock_client.materialize.query_table.return_value = mock_df

        result = fetch_degree_table(mock_client)

        self.assertEqual(list(result.columns), ["root_id", "in_synapse_count", "out_synapse_count"])
        self.assertEqual(list(result["root_id"]), [100, 200])
        self.assertEqual(list(result["in_synapse_count"]), [5, 10])


class FetchSomaRootsMockedTest(unittest.TestCase):
    """Test fetch_soma_roots with mocked client."""

    def test_fetch_soma_roots_uses_soma_counts_when_available(self):
        from neuronauts.cave_synapse_degrees_v1412 import fetch_soma_roots

        mock_client = MagicMock()
        mock_client.materialize.get_tables.return_value = ["soma_counts", "other"]
        mock_client.materialize.query_table.return_value = pd.DataFrame({
            "pt_root_id": [100, 200, 0],
        })

        result = fetch_soma_roots(mock_client)

        self.assertEqual(result, {100, 200})


class GetClientTest(unittest.TestCase):
    """Smoke test get_client (no network if we patch CAVEclient)."""

    def test_get_client_returns_client_with_version(self):
        with patch("neuronauts.cave_synapse_degrees_v1412.CAVEclient") as MockCAVE:
            mock_inst = MagicMock()
            MockCAVE.return_value = mock_inst

            from neuronauts.cave_synapse_degrees_v1412 import get_client

            client = get_client(version=1412)
            self.assertIs(client, mock_inst)
            self.assertEqual(mock_inst.version, 1412)


if __name__ == "__main__":
    unittest.main()
