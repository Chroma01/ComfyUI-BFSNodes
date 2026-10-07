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


class SilentAudioTest(unittest.TestCase):
    def test_join_without_audio_returns_a_silent_track(self):
        imgs = [torch.full((39, 8, 8, 3), 0.5)]
        video, a, fps = SL.BFSShotJoin()._join_all(imgs, [_shot(0, 0, 24, gen=39)], [0])
        self.assertEqual(tuple(a["waveform"].shape[:2]), (1, 2))
        self.assertEqual(a["waveform"].shape[-1], 44100)        # 24 frames at 24 fps = 1 s
        self.assertEqual(float(a["waveform"].abs().max()), 0.0)
        self.assertFalse(SL._usable({"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100}))


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


class ConditioningWriterTest(unittest.TestCase):
    """The optional prompt writer of BFS Shot H3 Conditioning (template path, no models)."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import bfs_h3_side_panel as SP
        self.SP, self.seen = SP, {}
        seen = self.seen

        class R2V:
            @staticmethod
            def execute(**kw):
                seen["prompt"] = kw["prompt"]
                return types.SimpleNamespace(args=("pos", "lat"))

        mod = types.ModuleType("comfy_extras.nodes_minimax_h3")
        mod.MiniMaxH3ReferenceToVideo, mod.MiniMaxH3AddGuide = R2V, R2V
        self._old = {k: sys.modules.get(k) for k in ("comfy_extras", "comfy_extras.nodes_minimax_h3")}
        sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
        sys.modules["comfy_extras.nodes_minimax_h3"] = mod
        self._apply = SP.BFSH3SidePanel.apply
        SP.BFSH3SidePanel.apply = lambda self, pos, lat, *a, **k: (pos, lat, {"position": "left"}, None, None)
        self._patch = SP.patch_model_rope
        SP.patch_model_rope = lambda m, info, gap: "patched"

    def tearDown(self):
        self.SP.BFSH3SidePanel.apply, self.SP.patch_model_rope = self._apply, self._patch
        for k, v in self._old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def _shot(self):
        return dict(prompt="my own {layout} prompt", width=448, height=800, gen_length=22,
                    frames=torch.zeros(22, 8, 8, 3), ref=torch.zeros(1, 4, 4, 3), ref2=None, audio=None)

    def _run(self, **kw):
        c = SL.BFSShotH3Conditioning()
        return c.condition(self._shot(), "clip", "vae", "none", True, "none", "match", **kw)

    def test_planner_prompt_is_default_and_layout_filled(self):
        out = self._run()
        self.assertEqual(out[3], "my own prompt")
        out = self._run(duet="canvas")
        self.assertIn("LEFT half is the kept footage", out[3])
        self.assertEqual(self.seen["prompt"], out[3])

    def test_task_template_follows_the_duet_mode(self):
        out = self._run(duet="canvas", task="character swap")
        self.assertIn("split screen", out[3])
        self.assertIn("<Picture 1>", out[3])
        out = self._run(duet="shifted RoPE", model="m", task="character swap")
        self.assertNotIn("split screen", out[3])
        self.assertEqual(out[2], "patched")

    def test_setting_picture_is_the_last_reference_with_static(self):
        seen = {}
        orig = sys.modules["comfy_extras.nodes_minimax_h3"].MiniMaxH3ReferenceToVideo.execute

        def capture(**kw):
            seen.update(kw)
            return orig(**kw)
        sys.modules["comfy_extras.nodes_minimax_h3"].MiniMaxH3ReferenceToVideo.execute = staticmethod(capture)
        shot = self._shot()
        shot["frames"] = torch.full((22, 8, 8, 3), 0.5)
        mask = torch.zeros(22, 8, 8)
        mask[:, 2:4, 2:4] = 1
        out = SL.BFSShotH3Conditioning().condition(shot, "clip", "vae", "none", True, "none", "match", duet="canvas",
                                                    task="character swap", setting_ref="on (generation size)",
                                                    setting_mask=mask)
        refs = list(seen["ref_images"].values())
        self.assertEqual(len(refs), 2)
        pic = refs[-1][0]
        self.assertTrue(((pic[3, 3] == 0) | (pic[3, 3] == 1)).all())      # static inside the (grown) mask
        self.assertTrue(torch.allclose(pic[7, 7], torch.tensor(0.5)))   # the place outside it
        self.assertIn("<Picture 2> shows the setting", out[3])
        self.assertEqual(SL.add_setting("a {setting} b", 3, True), "a <Picture 3> b")

    def test_task_needs_duet(self):
        with self.assertRaises(ValueError):
            self._run(task="style", instruction="anime")


class TargetTest(unittest.TestCase):
    def test_target_placeholder(self):
        self.assertEqual(SL.fill_target("Replace {target} with <Subject 1>", "the man in a red coat"),
                         "Replace the man in a red coat with <Subject 1>")
        self.assertEqual(SL.fill_target("Replace {target} with <Subject 1>", ""), "Replace the person with <Subject 1>")
        self.assertEqual(SL.fill_target("no placeholder", "x"), "no placeholder")


class MaskOverlayTest(unittest.TestCase):
    def test_overlay_marks_mask_and_box(self):
        o = torch.zeros(3, 32, 32, 3)
        m = torch.zeros(3, 32, 32); m[:, 8:16, 8:16] = 1
        out = SL._mask_overlay(o, {"mask": m, "box": [0.1, 0.1, 0.9, 0.9]}, 3)
        self.assertGreater(float(out[0, 12, 12, 0]), 0.3)          # red inside the mask
        self.assertEqual(float(out[0, 20, 20].sum()), 0.0)          # untouched outside mask and box
        self.assertGreater(float(out[0, 3, 16, 1]), 0.5)            # yellow box edge


class StitchFinishTest(unittest.TestCase):
    def _shot(self):
        full = torch.full((4, 64, 64, 3), 0.5)
        mask = torch.zeros(4, 64, 64); mask[:, 20:44, 24:40] = 1
        return {"full_frames": full, "crop": {"box": [0.125, 0.125, 0.875, 0.875], "mask": mask, "paste": "mask",
                                              "expand": 0, "feather": 2}}

    def test_ring_colour_match_removes_drift_and_keeps_subject_contrast(self):
        shot = self._shot()
        gen = torch.full((4, 48, 48, 3), 0.62)          # the model brightened the whole crop (+0.12)
        gen[:, 12:36, 16:32] = 0.2                       # a darker new subject
        plain = SL.uncrop(gen, shot)
        fixed = SL.uncrop(gen, shot, match_colors=1.0)
        # around the subject the drift is gone, inside it the new subject stays darker than the scene
        self.assertGreater(abs(float(plain[0, 21, 23].mean()) - 0.5), abs(float(fixed[0, 21, 23].mean()) - 0.5))
        self.assertLess(float(fixed[0, 30, 30].mean()), 0.45)

    def test_edge_hardness(self):
        a = torch.tensor([[[0.1, 0.5, 0.9]]])
        h = SL._alpha_hardness(a, 1.0)
        self.assertEqual([round(float(x), 2) for x in h[0, 0]], [0.0, 0.5, 1.0])
        self.assertTrue(torch.equal(SL._alpha_hardness(a, 0.0), a))


class InpaintInCropTest(unittest.TestCase):
    def test_latent_starts_from_the_shot_and_only_the_mask_is_generated(self):
        sys.path.insert(0, str(ROOT))
        nested = sys.modules.get("comfy.nested_tensor")
        if nested is None:
            class _N:
                is_nested = True

                def __init__(self, t):
                    self.tensors = list(t)
            comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
            nested = types.ModuleType("comfy.nested_tensor"); nested.NestedTensor = _N
            comfy.nested_tensor = nested
            sys.modules["comfy.nested_tensor"] = nested
            sys.modules.setdefault("comfy.utils", types.ModuleType("comfy.utils"))

        class VAE:
            def encode(self, px):
                t = ((px.shape[0] - 5) // 17) * 5 + 2
                x = px.mean(-1)[None, None]
                return torch.nn.functional.interpolate(x, size=(t, px.shape[1] // 16, px.shape[2] // 16)).repeat(1, 24, 1, 1, 1)

        lat = {"samples": nested.NestedTensor((torch.zeros(1, 24, 7, 8, 8), torch.zeros(1, 32, 2, 37)))}
        frames = torch.full((22, 128, 128, 3), 0.4)
        mask = torch.zeros(22, 128, 128); mask[:, 48:80, 48:80] = 1
        pos, out = SL.inpaint_latent(lat, [], VAE(), frames, mask)
        video = out["samples"].tensors[0]
        self.assertAlmostEqual(float(video.mean()), 0.4, places=4)        # starts from the shot's own frames
        vm, am = out["noise_mask"].tensors
        self.assertEqual(tuple(vm.shape), (1, 1, 7, 8, 8))
        self.assertEqual(float(vm[0, 0, 3, 4, 4]), 1.0)                   # the person: generated
        self.assertEqual(float(vm[0, 0, 3, 0, 0]), 0.0)                   # the rest of the crop: kept
        self.assertEqual(float(vm[0, 0, 3, 2, 4]), 1.0)                   # grown by one latent cell
        self.assertTrue(bool((am == 1).all()))                            # audio fully generated
        _, out2 = SL.inpaint_latent(lat, [], VAE(), frames, mask, strength=0.8)
        vm2 = out2["noise_mask"].tensors[0]
        self.assertAlmostEqual(float(vm2[0, 0, 3, 4, 4]), 0.8, places=5)     # partly regenerated inside the mask
        self.assertEqual(float(vm2[0, 0, 3, 0, 0]), 0.0)                       # still kept outside


    def test_generation_mask_with_and_without_crop(self):
        m = torch.zeros(5, 64, 64); m[:, 20:40, 20:40] = 1
        cropped = SL.generation_mask({"crop": {"crop_mask": m, "expand": 4}, "frames": torch.zeros(5, 64, 64, 3)})
        self.assertEqual(float(cropped[0, 17, 30]), 1.0)                  # grown by expand
        self.assertEqual(float(cropped[0, 10, 30]), 0.0)
        with self.assertRaises(ValueError):
            SL.generation_mask({"crop": None, "mask_src": None, "frames": torch.zeros(5, 64, 64, 3)})
        orig = SL.shot_mask
        SL.shot_mask = lambda *a: {"masks": torch.nn.functional.interpolate(m[:, None], size=(32, 32))[:, 0].to(torch.uint8)}
        try:
            full = SL.generation_mask({"crop": None, "frames": torch.zeros(5, 64, 64, 3),
                                       "mask_src": {"path": "", "analysis": {}, "start": 0, "length": 5,
                                                    "spec": {"expand": 0}}})
        finally:
            SL.shot_mask = orig
        self.assertEqual(tuple(full.shape), (5, 64, 64))
        self.assertTrue(torch.equal(full, m))


    def test_comparison_overlay_for_mask_only_shots(self):
        m = torch.zeros(5, 32, 32); m[:, 8:24, 8:24] = 1
        ms = {"path": __file__, "analysis": {"fps": 24.0}, "start": 0, "length": 5, "spec": dict(SL.DEFAULT_MASK, expand=0)}
        shot = {"crop": None, "frames": torch.zeros(5, 64, 64, 3), "mask_src": ms, "inpaint": True}
        orig = SL.shot_mask
        SL.shot_mask = lambda *a: {"masks": m.to(torch.uint8)}
        try:
            ov = SL.overlay_of(shot)
            self.assertEqual(tuple(ov["mask"].shape), (5, 64, 64))
            self.assertIsNone(SL.overlay_of(dict(shot, inpaint=False)))      # not used and not segmented yet: no SAM 3
            self.assertIsNone(SL.overlay_of({"crop": None, "frames": shot["frames"]}))
        finally:
            SL.shot_mask = orig
        o = SL._mask_overlay(torch.full((5, 32, 32, 3), 0.5), ov, 5)
        self.assertGreater(float(o[0, 16, 16, 0]), float(o[0, 16, 16, 1]))     # red inside the mask
        self.assertAlmostEqual(float(o[0, 2, 2, 0]), 0.5)                        # untouched outside


class ExternalMaskTest(unittest.TestCase):
    def _video(self, name, n=48, fps=24.0, w=64, h=36, box=(10, 5, 30, 25)):
        import cv2
        import numpy as np
        path = str(Path(_TMP) / name)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            f = np.zeros((h, w, 3), np.uint8)
            x0, y0, x1, y1 = box
            f[y0:y1, x0 + i % 4:x1 + i % 4] = 255
            vw.write(f)
        vw.release()
        return path

    def setUp(self):
        self.an = {"fps": 24.0, "n": 48, "n_src": 48, "fps_src": 24.0, "width": 64, "height": 36}

    def test_precedence(self):
        p = {"mask_video": "global.mp4"}
        self.assertEqual(SL.plan_spec(p, {"video": "shot.mp4", "text": "man"})["ext"]["scope"], "shot")
        self.assertNotIn("ext", SL.plan_spec(p, {"text": "man"}))                  # the shot's own SAM 3 text wins
        self.assertEqual(SL.plan_spec(p, {})["ext"], {"file": "global.mp4", "scope": "video"})
        SL._EXT_MASK["v.mp4"] = torch.ones(48, 36, 64)
        try:
            self.assertEqual(SL.plan_spec(p, {}, "v.mp4")["ext"], {"tensor": "v.mp4"})   # the node input beats the panel file
        finally:
            SL._EXT_MASK.pop("v.mp4")
        self.assertFalse(SL.spec_has_mask(SL.plan_spec({}, {})))

    def test_file_and_tensor_masks_replace_sam(self):
        path = self._video("roto_whole.mp4")
        self._video("roto_shot.mp4", n=12)
        orig = SL.segment_frames
        SL.segment_frames = lambda *a, **k: self.fail("SAM 3 must not run with an external mask")
        try:
            r = SL.shot_mask(path, self.an, 24, 12, SL.plan_spec({"mask_video": "roto_whole.mp4"}, {}))
            self.assertEqual(r["masks"].shape[0], 12)
            self.assertTrue(r["box"] is not None)
            cov = float(r["masks"].float().mean())
            self.assertTrue(0.1 < cov < 0.4, cov)
            r2 = SL.shot_mask(path, self.an, 24, 12, SL.plan_spec({}, {"video": "roto_shot.mp4"}))
            self.assertEqual(r2["masks"].shape[0], 12)
            t = torch.zeros(48, 36, 64); t[:, 0:18, :] = 1                         # top half
            SL._EXT_MASK[path] = t
            r3 = SL.shot_mask(path, self.an, 0, 10, SL.plan_spec({}, {}, path))
            m = r3["masks"].float()
            self.assertGreater(float(m[:, :8].mean()), 0.9)
            self.assertLess(float(m[:, -8:].mean()), 0.1)
        finally:
            SL.segment_frames = orig
            SL._EXT_MASK.pop(path, None)


class FramePasteTest(unittest.TestCase):
    def test_paste_back_keeps_the_background_and_takes_the_new_outline(self):
        base = torch.full((5, 64, 64, 3), 0.2)                       # the original frames
        result = torch.full((5, 64, 64, 3), 0.9)                     # generated: everything changed
        old = torch.zeros(5, 64, 64); old[:, 20:44, 20:36] = 1       # the original person
        new = torch.zeros(5, 64, 64); new[:, 16:48, 18:44] = 1       # the new one is bigger
        ms = {"path": "", "analysis": {}, "start": 0, "length": 5, "spec": dict(SL.DEFAULT_MASK, expand=0, feather=0)}
        shot = {"index": 0, "frames": base, "mask_src": ms, "paste": True}
        o1, o2 = SL.shot_mask, SL.segment_frames
        SL.shot_mask = lambda *a: {"masks": old.to(torch.uint8)}
        SL.segment_frames = lambda imgs, sp: torch.nn.functional.interpolate(new[:, None], size=imgs.shape[1:3])[:, 0]
        try:
            out = SL.paste_back(result, shot, True)
            only_old = SL.paste_back(result, shot, False)
        finally:
            SL.shot_mask, SL.segment_frames = o1, o2
        self.assertAlmostEqual(float(out[0, 2, 2, 0]), 0.2, places=4)      # background: the original
        self.assertAlmostEqual(float(out[0, 30, 28, 0]), 0.9, places=4)    # the person: generated
        self.assertAlmostEqual(float(out[0, 30, 41, 0]), 0.9, places=4)    # the bigger new outline is pasted too
        self.assertAlmostEqual(float(only_old[0, 30, 41, 0]), 0.2, places=4)


class MaskGuideTest(unittest.TestCase):
    def test_masked_only_greys_everything_outside_the_mask(self):
        fr = torch.full((4, 32, 32, 3), 0.9)
        m = torch.zeros(4, 16, 16); m[:, 4:12, 4:12] = 1
        g = SL.masked_only(fr, m)
        self.assertEqual(tuple(g.shape), (4, 32, 32, 3))
        self.assertAlmostEqual(float(g[0, 16, 16, 0]), 0.9)      # the masked region
        self.assertAlmostEqual(float(g[0, 2, 2, 0]), 0.5)        # grey elsewhere
        self.assertEqual(SL.MASK_GUIDES[0], "off")
        fr2 = torch.rand(3, 64, 64, 3)
        m2 = torch.zeros(3, 64, 64); m2[:, 10:50, 10:50] = 1
        old_det = dict(SL._DET); SL._DET["pose"] = None                  # no YOLO pose model: a clear error
        try:
            with self.assertRaisesRegex(ValueError, "YOLO pose model"):
                SL.masked_only(fr2, m2, look="pose (people)")
        finally:
            SL._DET.clear(); SL._DET.update(old_det)
        for look in [x for x in SL.MASK_GUIDE_LOOKS if not x.startswith("pose")]:
            g2 = SL.masked_only(fr2, m2, look=look)
            self.assertEqual(tuple(g2.shape), (3, 64, 64, 3), look)
            self.assertAlmostEqual(float(g2[0, 2, 2, 0]), 0.5, msg=look)                 # grey outside, every look
            if look != "colour":                                                       # no colour inside
                self.assertLess(float((g2[..., 0] - g2[..., 1]).abs().max()), 1e-5, look)
        t = SL.add_mask_video("x subject_definitions: <Subject 1> is ...", 2)
        self.assertIn("<Video 2> shows only the region", t)
        self.assertEqual(SL.add_mask_video("follow {mask_video}", 1), "follow <Video 1>")


class ConditionFlowTest(unittest.TestCase):
    """BFS Shot H3 Conditioning with the native H3 nodes replaced by recorders."""

    def setUp(self):
        sys.path.insert(0, str(ROOT))
        calls = self.calls = {"r2v": [], "guides": []}

        class Out:
            def __init__(self, *a):
                self.args = a

        class R2V:
            @staticmethod
            def execute(**kw):
                calls["r2v"].append(kw)
                N = sys.modules["comfy.nested_tensor"].NestedTensor
                return Out([[torch.zeros(1), {}]], {"samples": N((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 37)))})

        class Guide:
            @staticmethod
            def execute(positive, latent, frame_idx, vae, image=None, **kw):
                calls["guides"].append(image)
                return Out(positive)

        mod = types.ModuleType("comfy_extras.nodes_minimax_h3")
        mod.MiniMaxH3ReferenceToVideo, mod.MiniMaxH3AddGuide = R2V, Guide
        sys.modules.setdefault("comfy_extras", types.ModuleType("comfy_extras"))
        self._old = sys.modules.get("comfy_extras.nodes_minimax_h3")
        sys.modules["comfy_extras.nodes_minimax_h3"] = mod
        if "comfy.nested_tensor" not in sys.modules:
            class _N:
                is_nested = True

                def __init__(self, t):
                    self.tensors = list(t)
            comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
            nt = types.ModuleType("comfy.nested_tensor"); nt.NestedTensor = _N
            comfy.nested_tensor = nt; sys.modules["comfy.nested_tensor"] = nt

    def tearDown(self):
        if self._old is not None:
            sys.modules["comfy_extras.nodes_minimax_h3"] = self._old
        else:
            sys.modules.pop("comfy_extras.nodes_minimax_h3", None)

    def _shot(self, **kw):
        m = torch.zeros(22, 64, 64); m[:, 16:48, 16:40] = 1
        shot = {"index": 0, "count": 1, "frames": torch.full((22, 64, 64, 3), 0.7), "ref": torch.zeros(1, 32, 32, 3),
                "ref2": None, "prompt": "subject_definitions: <Subject 1> is ...", "width": 64, "height": 64,
                "gen_length": 22, "audio": None, "crop": {"crop_mask": m, "expand": 0} if kw.get("crop") else None,
                "inpaint": kw.get("inpaint", True)}
        if not kw.get("crop"):
            shot["mask_src"] = {"path": __file__, "analysis": {"fps": 24.0}, "start": 0, "length": 22, "spec": dict(SL.DEFAULT_MASK, expand=0)}
        return shot, m

    def run_cond(self, shot, m, **kw):
        class VAE:
            def encode(self, px):
                return torch.zeros(1, 24, 2, px.shape[1] // 16, px.shape[2] // 16)
        orig = SL.shot_mask
        SL.shot_mask = lambda *a: {"masks": m.to(torch.uint8)}
        try:
            return SL.BFSShotH3Conditioning().condition(shot, None, VAE(), SL.BFSShotH3Conditioning.GUIDE_MODES[0], False,
                                                        "none", "match", **kw)
        finally:
            SL.shot_mask = orig

    def test_reference_video_of_the_masked_region(self):
        shot, m = self._shot()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3], mask_ref_size="full",
                      mask_guide_look="colour")
        kw = self.calls["r2v"][0]
        v = kw["ref_videos"]["ref_video_1"]
        self.assertAlmostEqual(float(v[0, 2, 2, 0]), 0.5)          # grey outside the mask
        self.assertAlmostEqual(float(v[0, 30, 30, 0]), 0.7)        # the subject
        self.assertIn("<Video 1> shows only the region", kw["prompt"])
        self.assertEqual(len(self.calls["guides"]), 1)             # the normal aligned guide is still there
        self.calls["r2v"].clear()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3], mask_ref_size="1/2")
        self.assertEqual(tuple(self.calls["r2v"][0]["ref_videos"]["ref_video_1"].shape[1:3]), (32, 32))

    def test_extra_and_instead_guides(self):
        shot, m = self._shot()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[1])
        self.assertEqual(len(self.calls["guides"]), 2)
        self.calls["guides"].clear()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[2])
        self.assertEqual(len(self.calls["guides"]), 1)
        self.assertAlmostEqual(float(self.calls["guides"][0][0, 2, 2, 0]), 0.5)   # it is the masked one

    def test_ignored_outside_mask_only(self):
        shot, m = self._shot(inpaint=False)
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3])
        self.assertNotIn("ref_videos", self.calls["r2v"][0])
        shot, m = self._shot(crop=True)
        self.calls["r2v"].clear()
        self.run_cond(shot, m, inpaint=SL.INPAINT_MODES[0], mask_guide=SL.MASK_GUIDES[3])
        self.assertNotIn("ref_videos", self.calls["r2v"][0])


if __name__ == "__main__":
    unittest.main()
