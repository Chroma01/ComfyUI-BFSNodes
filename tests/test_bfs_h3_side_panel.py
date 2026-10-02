from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class _Nested:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors


# stub the two comfy modules the node uses, so the tests run without ComfyUI
_comfy = types.ModuleType("comfy")
_nt = types.ModuleType("comfy.nested_tensor")
_nt.NestedTensor = _Nested
_ut = types.ModuleType("comfy.utils")
_ut.reshape_mask = lambda m, shape: torch.nn.functional.interpolate(m.reshape(1, 1, *m.shape[-3:]).float(), size=shape[2:])
_comfy.nested_tensor, _comfy.utils = _nt, _ut
for name, mod in (("comfy", _comfy), ("comfy.nested_tensor", _nt), ("comfy.utils", _ut)):
    sys.modules.setdefault(name, mod)

SPEC = importlib.util.spec_from_file_location("bfs_h3_side_panel", ROOT / "bfs_h3_side_panel.py")
SP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SP)


class FakeVAE:
    """[N,H,W,3] -> [1,24,T,H/16,W/16] with the H3 frame grid; the mean colour in every channel."""

    def encode(self, px):
        n = px.shape[0]
        t = 1 if n == 1 else ((n - 5) // 17) * 5 + 2
        x = px.mean(-1)[None, None]                       # [1,1,N,H,W]
        x = torch.nn.functional.interpolate(x, size=(t, px.shape[1] // 16, px.shape[2] // 16))
        return x.repeat(1, 24, 1, 1, 1)

    def decode(self, z):
        t, h, w = z.shape[2:]
        return torch.full((1, SP.frame_count_of(t) if t > 1 else 1, h * 16, w * 16, 3), float(z.mean()))


def _latent(w=576, h=1024, frames=22):
    t = ((frames - 5) // 17) * 5 + 2
    return {"samples": _Nested((torch.zeros(1, 24, t, h // 16, w // 16), torch.zeros(1, 32, 2, 37)))}


class SidePanelTest(unittest.TestCase):
    def test_strip_sizes_snap_to_patches(self):
        self.assertEqual(SP.strip_size(576, 1024, "left", 1.0, 0), (576, 1024, 576, 1024))
        self.assertEqual(SP.strip_size(576, 1024, "top", 0.33, 32), (576, 384, 576, 352))

    def test_mask_holds_only_the_strip(self):
        info = {"position": "left", "h": 4, "w": 6, "strip_h": 0, "strip_w": 2}
        m = SP.panel_mask(info, 3, "all frames", panel_noise=0.1)
        self.assertEqual(tuple(m.shape), (1, 1, 3, 4, 8))
        self.assertTrue(torch.allclose(m[..., :2], torch.tensor(0.1)))
        self.assertTrue(bool((m[..., 2:] == 1).all()))
        first = SP.panel_mask(info, 3, "first latent frame")
        self.assertEqual(float(first[0, 0, 1, 0, 0]), 1.0)

    def test_layout_text_follows_the_share(self):
        half = SP.layout_text({"position": "left", "h": 64, "w": 36, "strip_h": 0, "strip_w": 36})
        self.assertIn("the LEFT half is the kept footage", half)
        narrow = SP.layout_text({"position": "left", "h": 64, "w": 72, "strip_h": 0, "strip_w": 36})
        self.assertIn("narrow strip", narrow)
        self.assertIn("LARGER", narrow)

    def test_apply_and_crop_round_trip(self):
        lat = _latent()
        panel = torch.full((22, 512, 512, 3), 0.9)
        guide = torch.full((22, 1024, 576, 3), 0.2)
        pos, out, info, preview, text = SP.BFSH3SidePanel().apply(
            [[torch.zeros(1), {}]], lat, FakeVAE(), panel, "left", 1.0, "contain", 0, "all frames", 0.0, guide, 0)
        v = out["samples"].tensors[0]
        self.assertEqual(tuple(v.shape), (1, 24, 7, 64, 72))
        kf = pos[0][1]["minimax_keyframes"][0]
        self.assertEqual(tuple(kf["latent"].shape[-2:]), (64, 72))
        self.assertEqual(tuple(preview.shape), (1, 1024, 1152, 3))
        cropped = SP.BFSH3SidePanelCrop().crop(info, latent=out, images=torch.zeros(5, 1024, 1152, 3))
        self.assertEqual(tuple(cropped[0]["samples"].tensors[0].shape), (1, 24, 7, 64, 36))
        self.assertNotIn("noise_mask", cropped[0])
        self.assertEqual(tuple(cropped[1].shape), (5, 1024, 576, 3))

    def test_existing_guides_are_moved_onto_the_canvas(self):
        lat = _latent()
        kf = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 64, 36)}
        pos = SP.BFSH3SidePanel().apply([[torch.zeros(1), {"minimax_keyframes": [kf]}]], lat, FakeVAE(),
                                        torch.ones(1, 64, 64, 3), "top", 0.5, "cover", 1, "all frames")[0]
        self.assertEqual(tuple(pos[0][1]["minimax_keyframes"][0]["latent"].shape[-2:]), (64 + 34, 36))


if __name__ == "__main__":
    unittest.main()
