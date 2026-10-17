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

    def test_reference_pictures_go_to_the_player_no_taller_than_it(self):
        from PIL import Image
        guide, target = torch.rand(4, 32, 48, 3), torch.rand(4, 32, 48, 3)
        ref = torch.rand(6, 80, 40, 3)                 # taller than the preview, more than the 4 shown
        ui = VC.BFSVideoCompare().compare(guide, target, preview_height=32, reference=ref)["ui"]["bfs_compare"][0]
        self.assertEqual([r["filename"] for r in ui["refs"]], ["ref_0.png", "ref_1.png", "ref_2.png", "ref_3.png"])
        im = Image.open(os.path.join(_TMP, ui["refs"][0]["subfolder"], "ref_0.png"))
        self.assertEqual(im.size, (16, 32))
        none = VC.BFSVideoCompare().compare(guide, target, preview_height=32)["ui"]["bfs_compare"][0]
        self.assertEqual(none["refs"], [])


@unittest.skipUnless(VC.VIO.ffmpeg_bin(), "needs an ffmpeg program")
class ExportTest(unittest.TestCase):
    """The player's recording (WebM from MediaRecorder) converted for sharing."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        cls.src = os.path.join(_TMP, "rec_src.webm")
        # 20 s of noise at 720p with sound: far above 10 MB at the browser's bitrate
        subprocess.run([VC.VIO.ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=1280x720:r=30:d=20,geq=random(1)*255:128:128",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-c:v", "libvpx", "-b:v", "16M",
                        "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus", "-shortest", cls.src], check=True)

    def probe(self, path):
        import json
        import subprocess
        r = subprocess.run([VC.VIO.ffmpeg_bin().replace("ffmpeg", "ffprobe"), "-v", "error", "-show_streams", "-of", "json", path],
                           capture_output=True, text=True)
        return {st["codec_type"]: st for st in json.loads(r.stdout)["streams"]}

    def test_small_mp4_is_h264_aac_faststart_under_the_limit(self):
        self.assertGreater(os.path.getsize(self.src), VC.LIMIT_BYTES)
        out = VC.encode(self.src, "small", 20.0, os.path.join(_TMP, "exp_small"))
        self.assertLessEqual(os.path.getsize(out), VC.LIMIT_BYTES)
        st = self.probe(out)
        self.assertEqual((st["video"]["codec_name"], st["video"]["pix_fmt"], st["audio"]["codec_name"]), ("h264", "yuv420p", "aac"))
        head = open(out, "rb").read(256 * 1024)
        self.assertLess(head.find(b"moov"), head.find(b"mdat") if b"mdat" in head else 1 << 30)   # faststart: moov first

    def test_gif_fits_the_limit(self):
        out = VC.encode(self.src, "gif", 20.0, os.path.join(_TMP, "exp_gif"))
        self.assertTrue(out.endswith(".gif"))
        self.assertLessEqual(os.path.getsize(out), VC.LIMIT_BYTES)

    def test_bad_ids_and_formats_are_refused(self):
        with self.assertRaises(ValueError):
            VC.write_part("../../etc", 0, b"x")
        with self.assertRaises(ValueError):
            VC.encode(self.src, "avi", 1.0, os.path.join(_TMP, "x"))
        rid = "abcdef0123"
        VC.write_part(rid, 0, b"12345")
        VC.write_part(rid, 5, b"678")
        self.assertEqual(open(os.path.join(_TMP, "bfs_compare", "exports", rid + ".rec"), "rb").read(), b"12345678")


class NoFfmpegTest(unittest.TestCase):
    """Windows portable installs often have no ffmpeg program: everything must go through PyAV."""

    def setUp(self):
        self._saved = dict(VC.VIO._BIN)
        VC.VIO._BIN.update(ffmpeg=None, ffprobe=None)

    def tearDown(self):
        VC.VIO._BIN.clear()
        VC.VIO._BIN.update(self._saved)

    def test_writer_and_reader_work_without_ffmpeg(self):
        import av
        path = os.path.join(_TMP, "noffmpeg.mp4")
        w = VC.VIO.Mp4Writer(path, 33, 21, 24.0, {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000},
                             metadata={"prompt": {"1": 2}})
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 21), torch.linspace(0, 1, 33), indexing="ij")
        x = torch.stack([torch.stack([xx, yy, (xx + yy + t / 30) % 1], -1) for t in range(30)])   # smooth, like video
        w.write(x[:16])
        w.write(x[16:])
        w.close()
        with av.open(path) as c:
            self.assertEqual(len(c.streams.audio), 1)
            self.assertEqual((c.streams.video[0].codec_context.width, c.streams.video[0].codec_context.height), (34, 22))
            self.assertEqual(c.metadata.get("prompt"), '{"1": 2}')
        frames, s = VC.VIO.read_video_scaled(path, 1e9)
        self.assertEqual((tuple(frames.shape), s), ((30, 22, 34, 3), 1.0))
        self.assertLess(float((frames[:, :21, :33] - x).abs().mean()), 0.05)   # lossy, but the same pictures
        small, s = VC.VIO.read_video_scaled(path, 30 * 22 * 34 * 3 * 4 / 4)   # a quarter of the bytes
        self.assertLess(small.shape[1], 22)

    def test_compare_node_without_ffmpeg(self):
        out = VC.BFSVideoCompare().compare(torch.rand(10, 32, 48, 3), torch.rand(10, 32, 48, 3), preview_height=32)
        ui = out["ui"]["bfs_compare"][0]
        self.assertTrue(os.path.getsize(os.path.join(_TMP, ui["guide"]["subfolder"], ui["guide"]["filename"])) > 0)


if __name__ == "__main__":
    unittest.main()
