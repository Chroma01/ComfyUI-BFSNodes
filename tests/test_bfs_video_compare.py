from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="bfs_compare_test_")
_fp = types.ModuleType("folder_paths")
_fp.get_temp_directory = lambda: _TMP
sys.modules.setdefault("folder_paths", _fp)

SPEC = importlib.util.spec_from_file_location("bfs_video_compare", ROOT / "bfs_video_compare.py")
VC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VC)


class DifferenceTest(unittest.TestCase):
    def test_equal_frames_are_black_and_score_zero(self):
        x = torch.rand(3, 8, 8, 3)
        m, mae, mse = VC.difference(x, x.clone())
        self.assertEqual(float(m.abs().max()), 0.0)
        self.assertEqual(float(mae.max()), 0.0)
        self.assertEqual(VC.psnr(float(mse.mean())), float("inf"))

    def test_map_lights_up_only_where_it_changed(self):
        g = torch.zeros(1, 8, 8, 3)
        t = g.clone()
        t[:, :4, :4] = 1.0
        m, mae, _ = VC.difference(g, t, gain=1.0)
        self.assertEqual(m[0, 0, 0].tolist(), [1.0, 1.0, 1.0])     # fully different: white
        self.assertEqual(m[0, 7, 7].tolist(), [0.0, 0.0, 0.0])     # unchanged: black
        self.assertAlmostEqual(float(mae[0]), 255 / 4, places=3)    # a quarter of the frame

    def test_heat_scale(self):
        self.assertEqual(VC.heat(torch.tensor(1 / 3)).tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(VC.heat(torch.tensor(2 / 3)).tolist()[:2], [1.0, 1.0])


class CompareNodeTest(unittest.TestCase):
    def test_compares_at_the_guide_size_and_writes_three_previews(self):
        import cv2
        guide = torch.rand(20, 32, 48, 3)
        target = torch.rand(18, 64, 96, 3)            # bigger and shorter, like the Join's images vs its guide
        audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
        out = VC.BFSVideoCompare().compare(guide, target, fps=24.0, audio=audio, preview_height=32)
        diff, report, mean = out["result"]
        self.assertEqual(tuple(diff.shape), (18, 32, 48, 3))
        self.assertGreater(mean, 0)
        self.assertIn("lengths differ", report)
        self.assertIn("target scaled from 96x64 to 48x32", report)
        ui = out["ui"]["bfs_compare"][0]
        self.assertEqual((ui["frames"], len(ui["mae"]), ui["width"], ui["height"]), (18, 18, 48, 32))
        for k in ("guide", "target", "difference"):
            path = os.path.join(_TMP, ui[k]["subfolder"], ui[k]["filename"])
            cap = cv2.VideoCapture(path)
            n = 0
            while cap.read()[0]:
                n += 1
            self.assertEqual(n, 18, k)


if __name__ == "__main__":
    unittest.main()
