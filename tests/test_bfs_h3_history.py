import importlib.util
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bfs_h3_history", ROOT / "bfs_h3_history.py")
HI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HI)


class Layout:
    """text 4 rows | guide cond 6 rows | history cond 4 rows | audio 3 | video 6, all at the target origin 4.0"""

    def __init__(self):
        self.segments = [(0, 4, "text"), (4, 10, "cond"), (10, 14, "cond"), (14, 17, "audio"), (17, 23, "video")]
        p = torch.zeros(23, 3)
        p[:4, 0] = torch.arange(4.)
        p[4:, 0] = 4.0
        self.position_ids = p


class HistoryTest(unittest.TestCase):
    def setUp(self):
        HI.history_span = lambda t: 10.0 * t

    def test_target_moves_past_the_history_only(self):
        lay = Layout()
        kfs = [{"latent": torch.zeros(1, 24, 3, 2, 2)}, {"latent": torch.zeros(1, 24, 2, 2, 2), "anchor": "history"}]
        self.assertTrue(HI.relocate(lay, kfs))
        t = lay.position_ids[:, 0]
        self.assertTrue(torch.equal(t[:4], torch.arange(4.)))        # text untouched
        self.assertTrue(bool((t[10:14] == 4.0).all()))              # history stays at the origin
        self.assertTrue(bool((t[4:10] == 24.0).all()))              # the other guide moves with the video
        self.assertTrue(bool((t[14:] == 24.0).all()))               # audio and video move by the span (2 steps)
        self.assertFalse(HI.relocate(lay, kfs))                     # already moved: no double shift

    def test_no_history_no_change(self):
        lay = Layout()
        before = lay.position_ids.clone()
        self.assertFalse(HI.relocate(lay, [{"latent": torch.zeros(1, 24, 3, 2, 2)}]))
        self.assertTrue(torch.equal(before, lay.position_ids))


if __name__ == "__main__":
    unittest.main()
