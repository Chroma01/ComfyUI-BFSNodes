from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="bfs_shotloop_test_")

# The module only needs a few folder_paths calls; stub them so the tests run without ComfyUI.
_fp = types.ModuleType("folder_paths")
_fp.get_temp_directory = lambda: _TMP
_fp.get_input_directory = lambda: _TMP
_fp.get_annotated_filepath = lambda name: str(Path(_TMP) / name)
sys.modules.setdefault("folder_paths", _fp)


class _Blocker:
    def __init__(self, message):
        self.message = message


_ce = types.ModuleType("comfy_execution")
_gu = types.ModuleType("comfy_execution.graph_utils")
_gu.ExecutionBlocker = _Blocker
sys.modules.setdefault("comfy_execution", _ce)
sys.modules.setdefault("comfy_execution.graph_utils", _gu)

SPEC = importlib.util.spec_from_file_location("bfs_shot_loop", ROOT / "bfs_shot_loop.py")
assert SPEC is not None and SPEC.loader is not None
SL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SL)
H3 = "H3 (17n+5)"


def _shot(i, start, end, cut=False, gen=None, count=1, queue=False, run_id=""):
    length = end - start
    return {"index": i, "count": count, "start": start, "end": end, "length": length,
            "gen_length": gen or SL.snap_up(length, H3), "fps": 24.0, "cut_before": cut,
            "queue": queue, "run_id": run_id}


class GridTest(unittest.TestCase):
    def test_h3_grid(self):
        self.assertEqual([SL.snap_up(x, H3) for x in (1, 5, 6, 22, 23, 107, 108)], [5, 5, 22, 22, 39, 107, 124])
        self.assertEqual([SL.snap_down(x, H3) for x in (4, 5, 21, 22, 106, 107, 110)], [5, 5, 5, 22, 90, 107, 107])

    def test_other_grids(self):
        self.assertEqual(SL.snap_up(50, "LTX / Wan (8n+1)"), 57)
        self.assertEqual(SL.snap_down(50, "LTX / Wan (8n+1)"), 49)
        self.assertEqual(SL.snap_up(6, "Wan (4n+1)"), 9)
        self.assertEqual(SL.snap_up(37, "any"), 37)

    def test_generation_size_is_on_the_multiple(self):
        w, h = SL.generation_size(1280, 720, 0.15, 32)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertGreater(w, h)


class PlanTest(unittest.TestCase):
    def test_fixed_mode_splits_evenly_under_the_limit(self):
        segs = SL.plan_segments(300, [], "fixed", 107, 24)
        self.assertEqual([s["end"] - s["start"] for s in segs], [100, 100, 100])

    def test_shots_split_at_cuts_and_long_shots_are_divided(self):
        segs = SL.plan_segments(300, [50, 80], "shots", 107, 10)
        bounds = [(s["start"], s["end"]) for s in segs]
        self.assertEqual(bounds[:2], [(0, 50), (50, 80)])
        self.assertTrue(all(e - s <= 107 for s, e in bounds))
        self.assertEqual(bounds[-1][1], 300)
        self.assertTrue(segs[1]["cut_before"])

    def test_short_shots_merge_into_a_neighbour_that_still_fits(self):
        segs = SL.plan_segments(200, [10, 100], "shots", 107, 24)
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0, 100), (100, 200)])

    def test_limits_and_manual_bounds(self):
        self.assertEqual(len(SL.plan_segments(1000, [], "fixed", 100, 10, max_parts=3)), 3)
        self.assertEqual(SL.plan_segments(1000, [], "fixed", 100, 10, max_total=250)[-1]["end"], 250)
        manual = SL.plan_segments(100, [], "manual", 107, 10, manual=[30, 70])
        self.assertEqual([(s["start"], s["end"]) for s in manual], [(0, 30), (30, 70), (70, 100)])

    def test_builtin_detector_finds_spikes_and_respects_the_gap(self):
        score = [1.0] * 100
        raw = [0.01] * 100
        for i in (20, 22, 60):
            score[i], raw[i] = 50.0, 0.5
        self.assertEqual(SL.detect_cuts(score, raw, 0.5, 24.0), [20, 60])


class JoinTest(unittest.TestCase):
    def _frames(self, n, value):
        return torch.full((n, 8, 8, 3), float(value))

    def test_trims_to_true_lengths_in_order(self):
        shots = [_shot(1, 30, 50, cut=True, gen=22), _shot(0, 0, 30, gen=39)]
        imgs = [self._frames(22, 0.8), self._frames(39, 0.2)]
        video, audio, fps = SL.BFSShotJoin()._join_all(imgs, shots, [0])
        self.assertEqual(video.shape[0], 50)
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), 0.2)
        self.assertAlmostEqual(float(video[-1, 0, 0, 0]), 0.8)
        self.assertEqual(fps, 24.0)

    def test_crossfade_only_across_soft_joins(self):
        imgs = [self._frames(39, 0.0), self._frames(39, 1.0)]
        soft = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 30, gen=39), _shot(1, 30, 60, gen=39)], [4])[0]
        hard = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 30, gen=39), _shot(1, 30, 60, cut=True, gen=39)], [4])[0]
        self.assertGreater(float(soft[31, 0, 0, 0]), 0.0)
        self.assertLess(float(soft[31, 0, 0, 0]), 1.0)
        self.assertEqual(float(hard[31, 0, 0, 0]), 1.0)

    def test_audio_is_trimmed_to_the_video(self):
        audio = {"waveform": torch.zeros(1, 2, 48000 * 10), "sample_rate": 48000}
        imgs = [self._frames(39, 0.5)]
        _, a, _ = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 24, gen=39)], [0], [audio])
        self.assertEqual(a["waveform"].shape[-1], 48000)

    def test_queue_mode_blocks_until_every_shot_is_stored(self):
        join = SL.BFSShotJoin()
        shots = [_shot(0, 0, 20, count=2, queue=True, run_id="t1"), _shot(1, 20, 45, cut=True, count=2, queue=True, run_id="t1")]
        first = join.join([self._frames(22, 0.1)], [shots[0]], [0])
        self.assertIsInstance(first[0], _Blocker)
        video, _, _, _ = join.join([self._frames(39, 0.9)], [shots[1]], [0])
        self.assertEqual(video.shape[0], 45)
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), 0.1, places=2)


class FilterTest(unittest.TestCase):
    STATS = {"persons": 1, "person_area": 0.2, "person_frames": 6, "faces": 1, "brightness": 0.4, "motion": 0.02, "sampled": 6}

    def test_no_filter_keeps_everything(self):
        self.assertEqual(SL.skip_reason(self.STATS, 50, dict(SL.DEFAULT_FILTERS)), "")
        self.assertFalse(SL.filters_active(dict(SL.DEFAULT_FILTERS)))

    def test_each_filter(self):
        f = dict(SL.DEFAULT_FILTERS)
        self.assertEqual(SL.skip_reason(dict(self.STATS, persons=0), 50, dict(f, person=True)), "no person")
        self.assertIn("smaller", SL.skip_reason(dict(self.STATS, person_area=0.01), 50, dict(f, person=True, min_person_area=0.03)))
        self.assertIn("more than", SL.skip_reason(dict(self.STATS, persons=5), 50, dict(f, max_persons=2)))
        self.assertEqual(SL.skip_reason(dict(self.STATS, faces=0), 50, dict(f, face=True)), "no face")
        self.assertEqual(SL.skip_reason(dict(self.STATS, brightness=0.02), 50, dict(f, skip_dark=True)), "dark / fade")
        self.assertEqual(SL.skip_reason(dict(self.STATS, motion=0.0), 50, dict(f, skip_static=True)), "static")
        self.assertIn("shorter", SL.skip_reason(self.STATS, 5, dict(f, min_frames=10)))


class TimelineJoinTest(unittest.TestCase):
    def _tl(self, fill):
        return {"path": "", "fps": 24.0, "width": 8, "height": 8, "fill": fill, "audio": None, "n": 60,
                "src": list(range(60)),
                "segs": [{"start": 0, "end": 20, "run": True, "cut_before": False, "run_index": 0},
                         {"start": 20, "end": 40, "run": False, "cut_before": True, "run_index": -1},
                         {"start": 40, "end": 60, "run": True, "cut_before": True, "run_index": 1}]}

    def test_drop_removes_skipped_shots(self):
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 40, 60, cut=True, gen=22)]
        imgs = [torch.zeros(22, 8, 8, 3), torch.ones(22, 8, 8, 3)]
        video = SL.BFSShotJoin()._join_all(imgs, shots, [0], None, self._tl("drop"))[0]
        self.assertEqual(video.shape[0], 40)
        self.assertEqual(float(video[25, 0, 0, 0]), 1.0)

    def test_original_fill_reads_the_source_frames(self):
        shots = [_shot(0, 0, 20, gen=22), _shot(1, 40, 60, cut=True, gen=22)]
        imgs = [torch.zeros(22, 8, 8, 3), torch.ones(22, 8, 8, 3)]
        orig = SL._read_frames
        SL._read_frames = lambda path, idx, size: [__import__("numpy").full((8, 8, 3), 128, "uint8") for _ in idx]
        try:
            video = SL.BFSShotJoin()._join_all(imgs, shots, [0], None, self._tl("original"))[0]
        finally:
            SL._read_frames = orig
        self.assertEqual(video.shape[0], 60)
        self.assertAlmostEqual(float(video[30, 0, 0, 0]), 128 / 255, places=3)


class CastTest(unittest.TestCase):
    # person 0 on frames 0-47 (big face), person 1 on 48-95, both small on 96-119
    CAST = {"step": 6, "people": [{"id": 0}, {"id": 1}], "samples":
            [{"f": f, "faces": [{"pid": 0, "area": 0.1}]} for f in range(0, 48, 6)]
            + [{"f": f, "faces": [{"pid": 1, "area": 0.1}]} for f in range(48, 96, 6)]
            + [{"f": f, "faces": []} for f in range(96, 120, 6)]}

    def test_shot_people_and_main(self):
        self.assertEqual(SL.shot_people(self.CAST, 0, 60), {"people": [0, 1], "main": 0})
        self.assertEqual(SL.shot_people(self.CAST, 96, 120), {"people": [], "main": -1})

    def test_change_points_need_a_long_enough_run(self):
        self.assertEqual(SL.person_change_points(self.CAST, 0, 120, 24), [48])
        self.assertEqual(SL.person_change_points(self.CAST, 0, 60, 24), [])

    def test_plan_splits_assigns_and_skips_by_person(self):
        path = str(Path(_TMP) / "cast.mp4"); Path(path).write_bytes(b"x")
        a = {"fps": 24.0, "n": 120, "score": [0.0] * 120, "raw": [0.0] * 120}
        SL._CAST_CACHE[SL._cast_key(path, a)] = self.CAST
        orig = SL.find_cuts
        SL.find_cuts = lambda *args, **kw: ([], "test")
        try:
            plan = dict(SL.DEFAULT_PLAN, max_s=10, min_s=0.5, cast={"0": {"ref": "a.png"}}, cast_split=True, cast_only=True)
            segs = SL.apply_filters(plan, a, path, SL.resolve_plan(plan, a, path))
        finally:
            SL.find_cuts = orig
        self.assertEqual([(s["start"], s["end"]) for s in segs], [(0, 48), (48, 120)])
        self.assertEqual(segs[0]["ref"], "a.png")
        self.assertTrue(segs[0]["run"])
        self.assertEqual(segs[1]["skip_reason"], "no linked person")


class ComparisonTest(unittest.TestCase):
    def test_join_returns_a_labelled_side_by_side(self):
        shots = [dict(_shot(0, 0, 20, gen=22), frames=torch.zeros(22, 32, 48, 3), ref=torch.ones(1, 40, 30, 3),
                      ref2=None, prompt="a test prompt"),
                 dict(_shot(1, 20, 40, cut=True, gen=22), frames=torch.zeros(22, 32, 48, 3), ref=None, ref2=None,
                      prompt="")]
        imgs = [torch.full((22, 32, 48, 3), 0.5), torch.full((22, 32, 48, 3), 0.7)]
        video, _, _, comp = SL.BFSShotJoin().join(imgs, shots, [0], comparison=[True], label=["steps 20"])
        self.assertEqual(video.shape[0], 40)
        self.assertEqual(comp.shape[0], 40)
        self.assertEqual(comp.shape[2] % 16, 0)
        self.assertEqual(comp.shape[1] % 16, 0)
        self.assertGreaterEqual(comp.shape[2], 48 * 2 + 32)
        self.assertGreater(comp.shape[1], 32)
        off = SL.BFSShotJoin().join(imgs, shots, [0])[3]
        self.assertEqual(off.shape[0], 1)


class ContinuityTest(unittest.TestCase):
    def test_pick_frame_ignores_the_overlap(self):
        frames = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 2, 2, 3)
        self.assertEqual(float(SL.pick_frame(frames, 20, "first")[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(SL.pick_frame(frames, 20, "middle")[0, 0, 0, 0]), 10.0)
        self.assertEqual(float(SL.pick_frame(frames, 20, "last")[0, 0, 0, 0]), 19.0)

    def test_chain_uses_the_previous_result_only_when_asked(self):
        frames = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 2, 2, 3)
        SL.remember_result({"index": 0, "count": 2}, frames)
        shot = {"index": 1, "count": 2, "chain": "reference", "chain_frame": "last", "prev_length": 20}
        self.assertEqual(float(SL.chain_image(shot)[0, 0, 0, 0]), 19.0)
        self.assertIsNone(SL.chain_image(dict(shot, chain="off")))
        self.assertIsNone(SL.chain_image(dict(shot, index=0)))
        self.assertIsNone(SL.chain_image(dict(shot, index=3, count=4)))   # not the shot right after
        stored = torch.ones(1, 2, 2, 3)
        self.assertIs(SL.chain_image(dict(shot, chain_image=stored)), stored)

    def test_plan_carries_the_per_shot_setting(self):
        a = {"fps": 24.0, "n": 100, "score": [0.0] * 100, "raw": [0.0] * 100}
        plan = dict(SL.DEFAULT_PLAN, mode="fixed", max_s=2, bounds=[50],
                    segs=[{}, {"chain": "first frame", "chain_frame": "middle"}])
        segs = SL.resolve_plan(plan, a)
        self.assertEqual((segs[0]["chain"], segs[1]["chain"], segs[1]["chain_frame"]), ("off", "first frame", "middle"))


class MaskCropTest(unittest.TestCase):
    def test_spec_takes_target_from_shot_and_settings_from_global(self):
        spec = SL.mask_spec({"text": "person", "padding": 0.9}, {"padding": 0.3, "text": "ignored"})
        self.assertEqual((spec["text"], spec["padding"], spec["fill_holes"]), ("person", 0.3, True))

    def test_shape_fills_holes_and_holds_in_time(self):
        m = torch.zeros(5, 9, 9, dtype=torch.uint8)
        m[2, 2:7, 2:7] = 1
        m[2, 4, 4] = 0                                 # a hole
        out = SL.shape_mask(m, dict(SL.DEFAULT_MASK, temporal_expand=1))
        self.assertEqual(int(out[2, 4, 4]), 1)
        self.assertEqual(int(out[1, 3, 3]), 1)          # held one frame back
        self.assertEqual(int(out[0, 3, 3]), 0)
        blocks = SL.shape_mask(m, dict(SL.DEFAULT_MASK, temporal_expand=0, blockify=4))
        self.assertEqual(tuple(blocks.shape), (5, 9, 9))

    def test_box_is_the_union_plus_padding(self):
        m = torch.zeros(2, 10, 10, dtype=torch.uint8)
        m[0, 2:4, 2:4] = 1
        m[1, 6:8, 6:8] = 1
        self.assertEqual(SL.crop_box(m, 0.0), [0.2, 0.2, 0.8, 0.8])
        self.assertIsNone(SL.crop_box(torch.zeros(2, 10, 10), 0.1))

    def test_uncrop_pastes_only_inside_the_mask(self):
        full = torch.zeros(3, 20, 20, 3)
        mask = torch.zeros(3, 20, 20)
        mask[:, 5:15, 5:15] = 1
        shot = {"full_frames": full, "crop": {"box": [0.25, 0.25, 0.75, 0.75], "mask": mask, "paste": "mask",
                                              "expand": 0, "feather": 0}}
        out = SL.uncrop(torch.ones(3, 8, 8, 3), shot)
        self.assertEqual(tuple(out.shape), (3, 20, 20, 3))
        self.assertEqual(float(out[0, 10, 10, 0]), 1.0)
        self.assertEqual(float(out[0, 1, 1, 0]), 0.0)


class DuetPanelCropTest(unittest.TestCase):
    def test_join_side_cuts_the_panel_off(self):
        shot = {"width": 64, "height": 32, "panel": {"position": "left", "h": 2, "w": 4, "strip_h": 0, "strip_w": 4}}
        canvas = torch.zeros(3, 32, 128, 3)
        canvas[:, :, 64:] = 1.0
        out = SL.crop_panel(canvas, shot)
        self.assertEqual(tuple(out.shape), (3, 32, 64, 3))
        self.assertEqual(float(out.min()), 1.0)
        self.assertIs(SL.crop_panel(out, shot), out)            # already the video size
        self.assertIs(SL.crop_panel(canvas, {"width": 64}), canvas)  # no panel


class RepackTest(unittest.TestCase):
    def test_replaces_only_connected_pieces_and_fits_the_guide(self):
        shot = dict(_shot(0, 0, 20, gen=22), width=8, height=8, frames=torch.zeros(22, 8, 8, 3),
                    ref=torch.zeros(1, 4, 4, 3), ref2=None, prompt="old", audio=None)
        ref = torch.ones(1, 6, 6, 3)
        out = SL.BFSShotRepack().repack(shot, ref_image=ref)[0]
        self.assertTrue(torch.equal(out["ref"], ref))
        self.assertEqual(out["prompt"], "old")
        self.assertIs(out["frames"], shot["frames"])
        out = SL.BFSShotRepack().repack(shot, guide_frames=torch.ones(30, 16, 16, 3), prompt="new")[0]
        self.assertEqual(tuple(out["frames"].shape), (22, 8, 8, 3))
        self.assertEqual(out["prompt"], "new")
        self.assertEqual(out["length"], 20)


if __name__ == "__main__":
    unittest.main()
