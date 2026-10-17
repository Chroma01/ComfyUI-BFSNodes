"""Source blur modes: '<region> (<what stays sharp>)', SAM 3.1 or Sapiens2, blur or inverted colours.
The segmenters are replaced by fixed masks, so this checks which pixels end up blurred, not the models."""
from __future__ import annotations

import os
import sys
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# only what the blur uses of the shot loop: the frame key and SAM 3 (a fixed 'mouth' / part mask here)
_SL = types.ModuleType("bfs_shot_loop")
_SL._image_key = lambda t: (tuple(t.shape), float(t.float().sum()))
_SL.DEFAULT_MASK = {}
_SL.SAM_CALLS = []


def _sam(frames, spec):
    _SL.SAM_CALLS.append(spec["text"])
    m = torch.zeros(frames.shape[:3])
    if spec["text"] == "mouth":
        m[:, 22:25, 28:36] = 1
    else:
        m[:, 10:30, 20:44] = 1
    return m


_SL.segment_frames = _sam
sys.modules.setdefault("bfs_shot_loop", _SL)
import bfs_sapiens as SP  # noqa: E402
import bfs_source_blur as B  # noqa: E402

H, W = 48, 64
FACE = (slice(10, 30), slice(20, 44))
LIPS = (slice(22, 24), slice(28, 36))
TEETH = (slice(24, 25), slice(28, 36))


def fake_classes(frames, out_size=None, batch=4):
    c = torch.zeros(frames.shape[0], H, W, dtype=torch.uint8)
    c[:, 5:45, 10:54] = SP.ID["Upper_Clothing"]
    c[(slice(None),) + FACE] = SP.ID["Face_Neck"]
    c[(slice(None),) + LIPS] = SP.ID["Upper_Lip"]
    c[(slice(None),) + TEETH] = SP.ID["Upper_Teeth"]
    return c


class SourceBlurTest(unittest.TestCase):
    def setUp(self):
        B._CACHE.clear()
        self._seg = SP.segment
        SP.segment = fake_classes
        self.frames = torch.rand(3, H, W, 3)

    def tearDown(self):
        SP.segment = self._seg

    def mask(self, mode, **kw):
        return B.blur_mask(self.frames, mode, segmenter=kw.pop("segmenter", "Sapiens2"), **kw)

    def test_saved_workflows_keep_their_values(self):
        for old in ("off", "face (keep mouth)", "head + hands (keep mouth)", "person (keep mouth)"):
            self.assertIn(old, B.SOURCE_BLUR)
        self.assertEqual(B.parse("masked area (mouth too)"), ("masked area", "mouth too"))
        self.assertIsNone(B.parse("off"))

    def test_keep_lips_and_tongue_blurs_the_teeth(self):
        m = self.mask("face (keep lips + tongue)")
        self.assertEqual(float(m[0, 15, 25]), 1.0)                 # face
        self.assertEqual(float(m[0, 23, 32]), 0.0)                 # lips stay sharp
        self.assertEqual(float(m[0, 24, 32]), 0.0)                 # teeth are inside the lips' small grow here...
        big = fake_classes(self.frames)
        big[:, 26:29, 28:36] = SP.ID["Lower_Teeth"]               # ...so check teeth away from the lips
        SP.segment = lambda *a, **k: big
        B._CACHE.clear()
        m = self.mask("face (keep lips + tongue)")
        self.assertEqual(float(m[0, 28, 32]), 1.0)
        m = self.mask("face (keep mouth)")
        self.assertEqual(float(m[0, 28, 32]), 0.0)                 # keep mouth keeps the teeth too

    def test_mouth_too_blurs_everything_in_the_region(self):
        m = self.mask("face (mouth too)")
        self.assertEqual(float(m[0, 23, 32]), 1.0)
        self.assertEqual(float(m[0, 2, 2]), 0.0)                   # background untouched

    def test_masked_area_uses_the_shot_mask_without_segmenting(self):
        SP.segment = lambda *a, **k: self.fail("masked area + mouth too must not segment")
        area = torch.zeros(3, H, W)
        area[:, 0:10, 0:10] = 1
        m = self.mask("masked area (mouth too)", area=area)
        self.assertEqual(float(m[0, 5, 5]), 1.0)
        self.assertEqual(float(m[0, 20, 30]), 0.0)

    def test_masked_area_without_a_mask_blurs_the_whole_person(self):
        m = self.mask("masked area (mouth too)")
        self.assertEqual(float(m[0, 40, 15]), 1.0)                 # clothing = person

    def test_invert_colours_inside_the_mask_only(self):
        x = torch.full((3, H, W, 3), 0.2)
        out = B.blur_source(x, "face (mouth too)", 1.0, segmenter="Sapiens2", effect="invert colours")
        self.assertAlmostEqual(float(out[0, 18, 30, 0]), 0.8, places=4)
        self.assertAlmostEqual(float(out[0, 2, 2, 0]), 0.2, places=4)

    def test_sam_keeps_the_old_behaviour(self):
        SP.segment = lambda *a, **k: self.fail("SAM 3.1 + keep mouth must not run Sapiens2")
        _SL.SAM_CALLS.clear()
        m = self.mask("face (keep mouth)", segmenter="SAM 3.1")
        self.assertEqual(_SL.SAM_CALLS, ["face", "mouth"])
        self.assertEqual(float(m[0, 15, 25]), 1.0)
        self.assertEqual(float(m[0, 23, 32]), 0.0)


class SapiensSizeTest(unittest.TestCase):
    def test_input_keeps_the_area_and_the_aspect(self):
        for h, w in ((1080, 1920), (1920, 1080), (512, 512)):
            ih, iw = SP.input_size(h, w)
            self.assertEqual((ih % 16, iw % 16), (0, 0))
            self.assertAlmostEqual(iw / ih, w / h, delta=0.03)          # never stretched to 1024x768
            self.assertAlmostEqual(ih * iw / SP.PIXELS, 1.0, delta=0.04)


if __name__ == "__main__":
    unittest.main()
