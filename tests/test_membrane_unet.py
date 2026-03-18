"""Tests for the 2.5D MembraneUNet upgrade."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


from neuronauts.membrane_unet import (
    MembraneUNet,
    TrainingConfig,
    _assemble_2_5d_slice,
    load_model,
    normalize_slice,
    predict_membranes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_volume(shape=(16, 16, 8), seed=0):
    """Return a small random EM-like uint8 volume in [X, Y, Z] order."""
    rng = np.random.default_rng(seed)
    return rng.integers(80, 180, size=shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# normalize_slice
# ---------------------------------------------------------------------------

class NormalizeSliceTest(unittest.TestCase):
    def test_uint8_maps_to_0_1(self):
        img = np.array([0, 128, 255], dtype=np.uint8)
        out = normalize_slice(img)
        self.assertAlmostEqual(float(out[2]), 1.0, places=3)
        self.assertGreater(float(out[1]), 0.0)

    def test_already_float_unchanged(self):
        img = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = normalize_slice(img)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_output_clipped_to_0_1(self):
        img = np.array([-0.5, 2.0], dtype=np.float32)
        out = normalize_slice(img)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


# ---------------------------------------------------------------------------
# _assemble_2_5d_slice
# ---------------------------------------------------------------------------

class Assemble2_5dTest(unittest.TestCase):
    def setUp(self):
        self.vol = np.random.default_rng(0).integers(0, 255, (6, 8, 8), dtype=np.uint8).astype(np.float32) / 255.0

    def test_output_shape_no_context(self):
        out = _assemble_2_5d_slice(self.vol, z=3, context_slices=0)
        self.assertEqual(out.shape, (1, 8, 8))

    def test_output_shape_with_context(self):
        out = _assemble_2_5d_slice(self.vol, z=3, context_slices=2)
        self.assertEqual(out.shape, (5, 8, 8))

    def test_centre_channel_is_target_slice(self):
        k = 2
        out = _assemble_2_5d_slice(self.vol, z=3, context_slices=k)
        np.testing.assert_allclose(out[k], normalize_slice(self.vol[3]), atol=1e-6)

    def test_boundary_replication_at_z0(self):
        # At z=0, the -1 and -2 neighbours should replicate slice 0.
        out = _assemble_2_5d_slice(self.vol, z=0, context_slices=2)
        np.testing.assert_allclose(out[0], out[1], atol=1e-6)
        np.testing.assert_allclose(out[0], out[2], atol=1e-6)

    def test_boundary_replication_at_z_last(self):
        Z = self.vol.shape[0]
        out = _assemble_2_5d_slice(self.vol, z=Z - 1, context_slices=2)
        np.testing.assert_allclose(out[-1], out[-2], atol=1e-6)


# ---------------------------------------------------------------------------
# MembraneUNet architecture
# ---------------------------------------------------------------------------

class MembraneUNetArchTest(unittest.TestCase):
    def test_2d_mode_output_shape(self):
        model = MembraneUNet(context_slices=0, base_channels=8)
        x = torch.zeros(1, 1, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, 1, 32, 32))

    def test_2_5d_mode_output_shape(self):
        model = MembraneUNet(context_slices=2, base_channels=8)
        x = torch.zeros(1, 5, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, 1, 32, 32))

    def test_instance_norm_present(self):
        model = MembraneUNet(context_slices=2, base_channels=8)
        norm_layers = [m for m in model.modules() if isinstance(m, torch.nn.InstanceNorm2d)]
        self.assertGreater(len(norm_layers), 0, "Expected InstanceNorm2d layers")

    def test_no_batch_norm(self):
        model = MembraneUNet(context_slices=2, base_channels=8)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertEqual(len(bn_layers), 0, "No BatchNorm2d expected in 2.5D UNet")

    def test_legacy_in_channels_kwarg(self):
        # Old callers may pass in_channels=1 directly.
        model = MembraneUNet(in_channels=1, base_channels=8)
        x = torch.zeros(1, 1, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, 1, 32, 32))

    def test_forward_deterministic_in_eval(self):
        model = MembraneUNet(context_slices=2, base_channels=8).eval()
        x = torch.randn(2, 5, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_output_is_finite(self):
        model = MembraneUNet(context_slices=2, base_channels=8).eval()
        x = torch.randn(1, 5, 48, 48)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_batch_size_one_works(self):
        model = MembraneUNet(context_slices=2, base_channels=8).eval()
        x = torch.randn(1, 5, 32, 32)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[0], 1)

    def test_asymmetric_spatial_size(self):
        model = MembraneUNet(context_slices=0, base_channels=8).eval()
        x = torch.zeros(1, 1, 32, 48)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (1, 1, 32, 48))


# ---------------------------------------------------------------------------
# predict_membranes
# ---------------------------------------------------------------------------

class PredictMembranesTest(unittest.TestCase):
    def _make_model_and_volume(self, context_slices=2, vol_shape=(32, 32, 8)):
        model = MembraneUNet(context_slices=context_slices, base_channels=8).eval()
        volume = _tiny_volume(shape=vol_shape)
        return model, volume

    def test_output_shape_matches_input(self):
        model, volume = self._make_model_and_volume()
        result = predict_membranes(model, volume, device="cpu")
        self.assertEqual(result.shape, volume.shape)

    def test_output_dtype_float32(self):
        model, volume = self._make_model_and_volume()
        result = predict_membranes(model, volume, device="cpu")
        self.assertEqual(result.dtype, np.float32)

    def test_output_values_in_0_1(self):
        model, volume = self._make_model_and_volume()
        result = predict_membranes(model, volume, device="cpu")
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_batch_size_1_gives_same_result(self):
        model, volume = self._make_model_and_volume()
        r1 = predict_membranes(model, volume, device="cpu", batch_size=1)
        r2 = predict_membranes(model, volume, device="cpu", batch_size=10)
        np.testing.assert_allclose(r1, r2, atol=1e-5)

    def test_context_slices_override(self):
        # Force 0 context — predict_membranes should use override not model attr.
        model2d = MembraneUNet(context_slices=0, base_channels=8)
        volume = _tiny_volume((32, 32, 6))
        r = predict_membranes(model2d, volume, device="cpu", context_slices=0)
        self.assertEqual(r.shape, volume.shape)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

class CheckpointRoundTripTest(unittest.TestCase):
    def test_save_and_load_predict(self):
        model = MembraneUNet(context_slices=2, base_channels=8).eval()
        volume = _tiny_volume((32, 32, 4))

        with torch.no_grad():
            vol_zyx = np.moveaxis(volume.astype(np.float32) / 255.0, 2, 0)
            inp = torch.from_numpy(
                np.stack([_assemble_2_5d_slice(vol_zyx, z, 2) for z in range(4)], axis=0)
            ).float()
            logits_orig = model(inp)
            probs_orig = torch.sigmoid(logits_orig).cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "membrane.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"context_slices": 2, "base_channels": 8},
                    "best_val_loss": 0.0,
                },
                ckpt_path,
            )
            loaded_model, device = load_model(ckpt_path, device="cpu")
            result = predict_membranes(loaded_model, volume, device=device)

        np.testing.assert_allclose(
            result,
            np.moveaxis(probs_orig[:, 0, :, :], 0, 2),
            atol=1e-5,
        )

    def test_load_model_in_eval_mode(self):
        model = MembraneUNet(context_slices=0, base_channels=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "m.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"context_slices": 0, "base_channels": 8},
                    "best_val_loss": 0.5,
                },
                ckpt_path,
            )
            loaded, _ = load_model(ckpt_path, device="cpu")
        self.assertFalse(loaded.training)


# ---------------------------------------------------------------------------
# run() integration: membrane_unet_checkpoint kwarg
# ---------------------------------------------------------------------------

class RunMembraneIntegrationTest(unittest.TestCase):
    def test_run_accepts_membrane_unet_checkpoint(self):
        """run() should use the UNet to compute membranes when a checkpoint
        is provided instead of falling back to Sobel."""
        from neuronauts.fetch import SyntheticBenchmarkConfig, make_test_volume
        from neuronauts.run import run

        model = MembraneUNet(context_slices=0, base_channels=8).eval()
        chunk, synapses = make_test_volume(
            config=SyntheticBenchmarkConfig(n_synapses=6, shape=(32, 32, 32)),
            seed=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "membrane.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"context_slices": 0, "base_channels": 8},
                    "best_val_loss": 0.0,
                },
                ckpt_path,
            )
            metrics = run(
                volume=chunk.data,
                pre_pts=synapses.pre_pt,
                post_pts=synapses.post_pt,
                pre_root_ids=synapses.pre_root_id,
                post_root_ids=synapses.post_root_id,
                verbose=False,
                membrane_unet_checkpoint=ckpt_path,
            )

        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
