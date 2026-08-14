from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "reference_temporal_offset.py"
SPEC = importlib.util.spec_from_file_location("reference_temporal_offset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
OFFSET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OFFSET)


class ReferenceTemporalOffsetTest(unittest.TestCase):
    def test_default_matches_standard_frame_zero_placement(self):
        kind, options = OFFSET.reference_temporal_offset_input()
        self.assertEqual(kind, "INT")
        self.assertEqual(options["default"], 0)
        self.assertEqual(OFFSET.reference_temporal_pixel_offset(0, 8), 0)
        self.assertEqual(OFFSET.reference_temporal_pixel_offset(-1, 8), -8)

    def test_shift_changes_only_temporal_axis_without_mutating_input(self):
        positions = torch.tensor(
            [[[0.0, 8.0], [10.0, 20.0], [30.0, 40.0]]],
            dtype=torch.float32,
        )
        shifted = OFFSET.shift_reference_temporal_positions(positions, -1, 8)

        torch.testing.assert_close(positions[:, 0], torch.tensor([[0.0, 8.0]]))
        torch.testing.assert_close(shifted[:, 0], torch.tensor([[-8.0, 0.0]]))
        torch.testing.assert_close(shifted[:, 1:], positions[:, 1:])
        self.assertIsNot(shifted, positions)

    def test_zero_is_exact_noop_for_legacy_specs(self):
        positions = torch.randn(1, 3, 4, 2)
        shifted = OFFSET.shift_reference_temporal_positions(positions, 0, 8)
        self.assertIs(shifted, positions)

    def test_multiple_controls_exposes_only_identity_offset(self):
        source = (ROOT / "ltx_multiple_controls.py").read_text(encoding="utf-8")
        self.assertIn(
            '"identity_temporal_offset_latents": reference_temporal_offset_input()',
            source,
        )
        self.assertNotIn('"guide_temporal_offset_latents"', source)
        self.assertNotIn('"mask_temporal_offset_latents"', source)
        self.assertNotIn('"identity_mask_temporal_offset_latents"', source)

    def test_edit_anything_offsets_reference_but_keeps_guide_at_zero(self):
        package_name = "bfsnodes_offset_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        sys.modules[package_name] = package

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda _kind: []
        sys.modules.setdefault("folder_paths", folder_paths)

        comfy_extras = types.ModuleType("comfy_extras")
        comfy_extras.__path__ = []
        nodes_lt = types.ModuleType("comfy_extras.nodes_lt")
        frame_indices = []

        class AddGuide:
            @staticmethod
            def append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                _guiding_latent,
                _strength,
                _scale_factors,
                causal_fix,
            ):
                self.assertTrue(causal_fix)
                frame_indices.append(frame_idx)
                return positive, negative, latent_image, noise_mask

        nodes_lt.LTXVAddGuide = AddGuide
        nodes_lt._append_guide_attention_entry = (
            lambda positive, negative, *args, **kwargs: (
                positive,
                negative,
            )
        )
        comfy_extras.nodes_lt = nodes_lt
        sys.modules["comfy_extras"] = comfy_extras
        sys.modules["comfy_extras.nodes_lt"] = nodes_lt

        module_path = ROOT / "ltxv_editanything.py"
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.ltxv_editanything",
            module_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        target = torch.zeros(1, 128, 2, 1, 1)
        mask = torch.ones(1, 1, 2, 1, 1)
        reference = torch.zeros(1, 128, 1, 1, 1)
        guide = torch.zeros(1, 128, 2, 1, 1)
        module.LTXVEditAnythingLoopingSampler._build_chunk_conds(
            "positive",
            "negative",
            target,
            mask,
            reference,
            guide,
            (8, 32, 32),
            1.0,
            1.0,
            -1,
            True,
        )

        self.assertEqual(frame_indices, [-8, 0])

    def test_all_reference_injection_nodes_expose_the_control(self):
        expected_fields = {
            "ltx_identity_overlap.py": "reference_temporal_offset_latents",
            "ltx_identity_multiangle.py": "reference_temporal_offset_latents",
            "ltx_multiple_controls.py": "identity_temporal_offset_latents",
            "ltxv_editanything.py": "ref_temporal_offset_latents",
        }
        for filename, field in expected_fields.items():
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
