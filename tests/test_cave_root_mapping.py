"""Tests for neuronauts.cave_root_mapping (mocked CAVEclient).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from neuronauts.cave_root_mapping import map_roots_between_versions


class MapRootsBetweenVersionsTest(unittest.TestCase):
    """Test map_roots_between_versions with mocked CAVE client."""

    def test_returns_correct_mapping_for_valid_ids(self):
        """When get_latest_roots returns known values, mapping is correct."""
        root_ids = [100, 200, 300]
        latest_responses = [[101, 202, 303]]  # batch response

        mock_chunkedgraph = MagicMock()
        mock_chunkedgraph.get_latest_roots.side_effect = latest_responses

        mock_client = MagicMock()
        mock_client.chunkedgraph = mock_chunkedgraph

        with patch("neuronauts.cave_root_mapping.get_client", return_value=mock_client):
            result = map_roots_between_versions(
                root_ids,
                old_version=117,
                new_version=1412,
                chunk_size=10,
            )

        self.assertEqual(result, {100: 101, 200: 202, 300: 303})
        mock_chunkedgraph.get_latest_roots.assert_called_once_with([100, 200, 300])

    def test_chunking_respects_chunk_size(self):
        """Multiple batches are requested when root count exceeds chunk_size."""
        root_ids = list(range(1, 26))  # 25 IDs
        chunk_size = 10

        call_results = [
            list(range(101, 111)),   # batch 1: 1-10 -> 101-110
            list(range(111, 121)),   # batch 2: 11-20 -> 111-120
            list(range(121, 126)),   # batch 3: 21-25 -> 121-125
        ]
        mock_chunkedgraph = MagicMock()
        mock_chunkedgraph.get_latest_roots.side_effect = call_results

        mock_client = MagicMock()
        mock_client.chunkedgraph = mock_chunkedgraph

        with patch("neuronauts.cave_root_mapping.get_client", return_value=mock_client):
            result = map_roots_between_versions(
                root_ids,
                old_version=100,
                new_version=1412,
                chunk_size=chunk_size,
            )

        self.assertEqual(mock_chunkedgraph.get_latest_roots.call_count, 3)
        self.assertEqual(result[1], 101)
        self.assertEqual(result[10], 110)
        self.assertEqual(result[21], 121)
        self.assertEqual(result[25], 125)

    def test_zero_in_input_mapped_to_zero_without_api_call(self):
        """Input root_id 0 is always mapped to 0 and never sent to the API."""
        root_ids = [0, 100, 200]

        mock_chunkedgraph = MagicMock()
        mock_chunkedgraph.get_latest_roots.return_value = [101, 202]

        mock_client = MagicMock()
        mock_client.chunkedgraph = mock_chunkedgraph

        with patch("neuronauts.cave_root_mapping.get_client", return_value=mock_client):
            result = map_roots_between_versions(
                root_ids,
                old_version=117,
                new_version=1412,
                chunk_size=10,
            )

        self.assertEqual(result[0], 0)
        self.assertEqual(result[100], 101)
        self.assertEqual(result[200], 202)
        # Only 100 and 200 should be sent (0 and negative filtered)
        mock_chunkedgraph.get_latest_roots.assert_called_once_with([100, 200])

    def test_negative_ids_filtered_out(self):
        """Negative root IDs are dropped and never sent to the API."""
        root_ids = [-1, 50, 100]

        mock_chunkedgraph = MagicMock()
        mock_chunkedgraph.get_latest_roots.return_value = [51, 101]

        mock_client = MagicMock()
        mock_client.chunkedgraph = mock_chunkedgraph

        with patch("neuronauts.cave_root_mapping.get_client", return_value=mock_client):
            result = map_roots_between_versions(
                root_ids,
                old_version=117,
                new_version=1412,
            )

        self.assertEqual(result[50], 51)
        self.assertEqual(result[100], 101)
        self.assertNotIn(-1, result)
        mock_chunkedgraph.get_latest_roots.assert_called_once_with([50, 100])

    def test_empty_valid_ids_still_calls_get_client(self):
        """When all IDs are 0 or negative, no get_latest_roots call is made."""
        root_ids = [0, -5, -10]

        mock_chunkedgraph = MagicMock()
        mock_client = MagicMock()
        mock_client.chunkedgraph = mock_chunkedgraph

        with patch("neuronauts.cave_root_mapping.get_client", return_value=mock_client):
            result = map_roots_between_versions(
                root_ids,
                old_version=117,
                new_version=1412,
            )

        self.assertEqual(result, {0: 0})
        mock_chunkedgraph.get_latest_roots.assert_not_called()


if __name__ == "__main__":
    unittest.main()
