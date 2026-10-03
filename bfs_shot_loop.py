"""BFS Shot Loop: plan a long video into model-sized shots, run any workflow once per shot, join.

A video model only follows a guide reliably inside the clip length it was trained on (an H3 LoRA
trained on 107 frames loses the scene at 243). This splits the source into shots that fit, at
camera cuts when there are any, and hands them to the rest of the graph as a ComfyUI *list*.
ComfyUI then runs every node that receives a list once per item, pairing the lists by index, so
shot ``i`` always gets guide ``i``, reference ``i`` and prompt ``i``. ``BFS Shot Join`` takes the
decoded list back, trims each shot to its true length and concatenates them in order.

Every shot is generated at a length the model accepts (for example 17n+5 frames for MiniMax H3).
When a shot is shorter than that, the extra guide frames are taken from the video that follows it,
so the guide never repeats or stretches; the join then cuts the result back to the shot's own
frames, which keeps the timing identical to the source. Those extra frames also give the join a
real overlap to cross-fade across boundaries that are not camera cuts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
from typing import Any

import numpy as np
import torch

import folder_paths

# ---------------------------------------------------------------------------- frame grids

GRIDS: dict[str, tuple[int, int]] = {
    "H3 (17n+5)": (17, 5),
    "LTX / Wan (8n+1)": (8, 1),
    "Wan (4n+1)": (4, 1),
    "any": (1, 0),
}
DEFAULT_GRID = "H3 (17n+5)"
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def snap_up(n: int, grid: str) -> int:
    """Smallest valid clip length >= n."""
    step, off = GRIDS.get(grid, GRIDS[DEFAULT_GRID])
    n = max(1, int(n))
    if step == 1:
        return n
    if n <= off:
        return off
    return off + step * math.ceil((n - off) / step)


def snap_down(n: int, grid: str) -> int:
    """Largest valid clip length <= n (never below the grid's smallest length)."""
    step, off = GRIDS.get(grid, GRIDS[DEFAULT_GRID])
    n = max(1, int(n))
    if step == 1:
        return n
    if n <= off:
        return off
    return off + step * ((n - off) // step)


# ---------------------------------------------------------------------------- video io

def _input_path(name: str) -> str:
    path = folder_paths.get_annotated_filepath(name) if name else ""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"BFS Shot Loop: file not found in the input folder: {name!r}")
    return path


def _probe(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"BFS Shot Loop: cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or n <= 0:
        raise ValueError(f"BFS Shot Loop: could not read fps/frame count from {path}")
    return {"fps_src": float(fps), "n_src": n, "width": w, "height": h, "duration": n / fps}


def _timeline(n_src: int, fps_src: float, fps: float) -> np.ndarray:
    """Source frame index for each frame of the resampled timeline (nearest frame, real time)."""
    n = max(1, int(math.floor(n_src / fps_src * fps + 1e-6)))
    idx = np.round(np.arange(n) * fps_src / fps).astype(np.int64)
    return np.clip(idx, 0, n_src - 1)


_STATUS_T = [0.0]


def _status(stage: str, done: int = 0, total: int = 0, force: bool = False) -> None:
    """Progress for the planner panel's loading bar (throttled)."""
    import time
    now = time.time()
    if not force and now - _STATUS_T[0] < 0.25 and done < total:
        return
    _STATUS_T[0] = now
    _notify("bfs-shotloop-status", {"stage": stage, "done": int(done), "total": int(total)})


def _read_frames(path: str, src_indices: np.ndarray, size: tuple[int, int] | None,
                 stage: str | None = None) -> list[np.ndarray]:
    """Decode the requested source frames (sorted, may repeat) as RGB uint8, sequentially."""
    import cv2
    want = sorted(set(int(i) for i in src_indices))
    top = want[-1] + 1 if want else 0
    got: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    i, k = 0, 0
    last = None
    while k < len(want):
        ok = cap.grab()
        if not ok:
            break
        if stage and i % 24 == 0:
            _status(stage, i, top)
        if i == want[k]:
            ok, fr = cap.retrieve()
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                if size is not None:
                    fr = _fit(fr, size)
                last = fr
                got[i] = fr
            while k < len(want) and want[k] == i:
                k += 1
        i += 1
    cap.release()
    if not got:
        raise ValueError(f"BFS Shot Loop: no frames decoded from {path}")
    out = []
    for s in src_indices:
        s = int(s)
        out.append(got.get(s, last if s > max(got) else got[min(got, key=lambda x: abs(x - s))]))
    return out


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Cover-resize then center-crop to (w, h)."""
    import cv2
    w, h = size
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def generation_size(src_w: int, src_h: int, megapixels: float, multiple: int) -> tuple[int, int]:
    """Aspect-preserving size with the given pixel area, both sides on the multiple grid."""
    multiple = max(8, int(multiple))
    area = max(0.05, float(megapixels)) * 1e6
    ar = src_w / max(1, src_h)
    h = math.sqrt(area / ar)
    w = h * ar
    w = max(multiple, int(round(w / multiple)) * multiple)
    h = max(multiple, int(round(h / multiple)) * multiple)
    return w, h


def audio_info(path: str) -> dict:
    """Whether the video has a usable audio track (PyAV, no ffmpeg binary needed)."""
    try:
        import av
        with av.open(path) as c:
            if not c.streams.audio:
                return {"has_audio": False, "reason": "no audio track"}
            st = c.streams.audio[0]
            cc = st.codec_context
            ch = getattr(cc, "channels", 0) or (len(cc.layout.channels) if getattr(cc, "layout", None) else 0)
            return {"has_audio": True, "sample_rate": int(cc.sample_rate or 0), "channels": int(ch or 0),
                    "codec": cc.name}
    except Exception as exc:  # noqa: BLE001
        return {"has_audio": False, "reason": f"audio unreadable ({type(exc).__name__})"}


def silence(seconds: float, sample_rate: int = 44100, channels: int = 2) -> dict:
    """A silent AUDIO of the given length: Create Video and friends always get a valid track."""
    n = max(1, int(round(max(0.0, seconds) * sample_rate)))
    return {"waveform": torch.zeros(1, channels, n), "sample_rate": sample_rate}


def _usable(audio: dict | None) -> bool:
    try:
        return audio is not None and audio["waveform"].ndim == 3 and audio["waveform"].shape[-1] > 0
    except Exception:  # noqa: BLE001
        return False


def _read_audio_av(path: str, sample_rate: int) -> dict | None:
    """PyAV fallback when the ffmpeg binary is missing."""
    import av
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        st = c.streams.audio[0]
        layout = "stereo" if (getattr(st.codec_context, "channels", 2) or 2) >= 2 else "mono"
        rs = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        chunks = []
        for fr in c.decode(st):
            for r in rs.resample(fr):
                chunks.append(r.to_ndarray())
        for r in rs.resample(None):
            chunks.append(r.to_ndarray())
    if not chunks:
        return None
    a = np.concatenate(chunks, axis=1).astype(np.float32)
    return {"waveform": torch.from_numpy(a)[None], "sample_rate": sample_rate}


def _read_audio(path: str, sample_rate: int = 44100) -> dict | None:
    """Decode the whole soundtrack as float32 [1, C, T] (ffmpeg, else PyAV); None when there is none."""
    a = _read_audio_ffmpeg(path, sample_rate)
    if not _usable(a):
        try:
            a = _read_audio_av(path, sample_rate)
        except Exception:  # noqa: BLE001 - audio is optional
            a = None
    return a if _usable(a) else None


def _read_audio_ffmpeg(path: str, sample_rate: int = 44100) -> dict | None:
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=channels", "-of", "csv=p=0", path],
                               capture_output=True, text=True, timeout=60)
        ch = int((probe.stdout.strip() or "0").split(",")[0] or 0)
        if ch <= 0:
            return None
        ch = min(ch, 2)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", str(ch), "-ar", str(sample_rate),
                              "-f", "f32le", "-"], capture_output=True, timeout=600).stdout
        if not raw:
            return None
        a = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).T.copy()
        return {"waveform": torch.from_numpy(a)[None], "sample_rate": sample_rate}
    except Exception:  # noqa: BLE001 - audio is optional
        return None


def _slice_audio(audio: dict | None, start_s: float, dur_s: float) -> dict | None:
    if audio is None:
        return None
    sr = audio["sample_rate"]
    wf = audio["waveform"]
    a, n = int(round(start_s * sr)), int(round(dur_s * sr))
    seg = wf[..., a:a + n]
    if seg.shape[-1] < n:
        seg = torch.nn.functional.pad(seg, (0, n - seg.shape[-1]))
    return {"waveform": seg.contiguous(), "sample_rate": sr}


# ---------------------------------------------------------------------------- analysis

_ANALYSIS_CACHE: dict[tuple, dict] = {}


def analyze(path: str, fps: float, thumbs: int = 120, thumb_h: int = 72) -> dict:
    """Probe, per-frame cut score on the resampled timeline, and a strip of thumbnails."""
    import cv2
    key = (path, os.path.getmtime(path), round(float(fps), 4), thumbs, thumb_h)
    if key in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[key]
    info = _probe(path)
    src = _timeline(info["n_src"], info["fps_src"], fps)
    n = len(src)
    _status("Reading the video", 0, info["n_src"], force=True)
    # small frames straight from the decoder: the full-size frames of a long video do not fit in memory
    sw = 64
    sh = max(8, int(round(sw * info["height"] / max(1, info["width"]))))   # keep the aspect: nothing cropped
    small = _read_frames(path, src, (sw, sh), stage="Reading the video")
    hsv = [cv2.calcHist([cv2.cvtColor(s, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]) for s in small]
    hsv = [cv2.normalize(h, h).flatten() for h in hsv]
    raw = np.zeros(n, np.float32)
    for i in range(1, n):
        pix = np.abs(small[i].astype(np.float32) - small[i - 1].astype(np.float32)).mean() / 255.0
        hist = cv2.compareHist(hsv[i - 1], hsv[i], cv2.HISTCMP_BHATTACHARYYA)
        raw[i] = 0.5 * pix + 0.5 * float(hist)
    # local normalisation: a cut stands out from the motion around it
    score = np.zeros(n, np.float32)
    w = max(4, int(round(fps)))
    for i in range(1, n):
        lo, hi = max(1, i - w), min(n, i + w + 1)
        neigh = np.concatenate([raw[lo:i], raw[i + 1:hi]])
        base = float(np.median(neigh)) if len(neigh) else 0.0
        score[i] = raw[i] / (base + 0.02)
    k = max(1, n // max(1, thumbs))
    th = []
    tw = max(16, int(round(thumb_h * info["width"] / max(1, info["height"]))))
    frames_for_thumbs = _read_frames(path, src[::k], (tw, thumb_h), stage="Making thumbnails")
    for j, fr in enumerate(frames_for_thumbs):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        th.append({"f": int(j * k), "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})
    out = dict(info, fps=float(fps), n=n, thumbs=th, thumb_w=tw, thumb_h=thumb_h, audio=audio_info(path),
               raw=[round(float(x), 4) for x in raw], score=[round(float(x), 3) for x in score])
    _ANALYSIS_CACHE[key] = out
    return out


_PSD_CACHE: dict[tuple, list[float]] = {}


def scenedetect_cut_times(path: str, detector: str, sensitivity: float) -> list[float] | None:
    """Cut times in seconds from PySceneDetect, or None when the package is missing."""
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError:
        return None
    sensitivity = float(min(1.0, max(0.0, sensitivity)))
    key = (path, os.path.getmtime(path), detector, round(sensitivity, 3))
    if key in _PSD_CACHE:
        return _PSD_CACHE[key]
    _status("Detecting camera cuts (PySceneDetect)", force=True)
    video = open_video(path)
    min_len = max(3, int(round(video.frame_rate * 0.25)))
    sm = SceneManager()
    if detector == "content":
        sm.add_detector(ContentDetector(threshold=45.0 - 33.0 * sensitivity, min_scene_len=min_len))
    else:
        sm.add_detector(AdaptiveDetector(adaptive_threshold=5.0 - 3.5 * sensitivity, min_scene_len=min_len))
    sm.detect_scenes(video, show_progress=False)
    times = [float(a.get_seconds()) for a, _ in sm.get_scene_list()[1:]]
    _PSD_CACHE[key] = times
    return times


def find_cuts(path: str, analysis: dict, detector: str, sensitivity: float) -> tuple[list[int], str]:
    """Cut frames on the resampled timeline and which detector produced them."""
    fps = float(analysis["fps"])
    if detector in ("adaptive", "content"):
        times = scenedetect_cut_times(path, detector, sensitivity)
        if times is not None:
            n = analysis["n"]
            return sorted({int(round(t * fps)) for t in times if 0 < round(t * fps) < n}), f"PySceneDetect {detector}"
    return detect_cuts(analysis["score"], analysis["raw"], sensitivity, fps), "builtin"


def detect_cuts(score: list[float], raw: list[float], sensitivity: float, fps: float) -> list[int]:
    """Frame indices that start a new shot (built-in detector). Higher sensitivity finds more cuts."""
    sensitivity = float(min(1.0, max(0.0, sensitivity)))
    thr = 12.0 - 10.0 * sensitivity          # local ratio threshold: 12 (strict) .. 2 (loose)
    abs_min = 0.12 - 0.09 * sensitivity      # ignore tiny global changes
    gap = max(3, int(round(fps * 0.25)))      # no two cuts closer than a quarter second
    cand = [i for i in range(1, len(score)) if score[i] >= thr and raw[i] >= abs_min]
    cuts: list[int] = []
    for i in sorted(cand, key=lambda j: -score[j]):
        if all(abs(i - c) >= gap for c in cuts):
            cuts.append(i)
    return sorted(cuts)


# ---------------------------------------------------------------------------- planning

def plan_segments(n: int, cuts: list[int], mode: str, max_len: int, min_len: int,
                  max_parts: int = 0, max_total: int = 0, manual: list[int] | None = None) -> list[dict]:
    """Split [0, n) into shots. Returns [{start, end, cut_before}]. Lengths are in frames."""
    if max_total and max_total > 0:
        n = min(n, int(max_total))
    max_len = max(1, int(max_len))
    min_len = max(1, min(int(min_len), max_len))
    cuts = sorted(c for c in set(cuts or []) if 0 < c < n)
    cutset = set(cuts)
    if mode == "manual" and manual:
        bounds = sorted(set([0, n] + [b for b in manual if 0 < b < n]))
    elif mode == "fixed":
        k = max(1, math.ceil(n / max_len))
        bounds = sorted(set(round(n * j / k) for j in range(k + 1)))
    else:  # shots
        bounds = [0] + cuts + [n]
        # merge shots that are too short into a neighbour, as long as the merge still fits
        changed = True
        while changed:
            changed = False
            for j in range(len(bounds) - 1):
                if bounds[j + 1] - bounds[j] >= min_len or len(bounds) <= 2:
                    continue
                left = bounds[j + 1] - bounds[j - 1] if j > 0 else None
                right = bounds[j + 2] - bounds[j] if j + 2 < len(bounds) else None
                opts = [(v, side) for v, side in ((left, "l"), (right, "r")) if v is not None and v <= max_len]
                if not opts:
                    continue
                side = min(opts)[1]
                bounds.pop(j if side == "l" else j + 1)
                changed = True
                break
        # split shots that are too long into equal parts
        out = [0]
        for a, b in zip(bounds[:-1], bounds[1:]):
            k = max(1, math.ceil((b - a) / max_len))
            out += [a + round((b - a) * j / k) for j in range(1, k + 1)]
        bounds = sorted(set(out))
    segs = [{"start": a, "end": b, "cut_before": a in cutset} for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if max_parts and max_parts > 0:
        segs = segs[:int(max_parts)]
    return segs


DEFAULT_PLAN = {
    "video": "", "fps": 24.0, "grid": DEFAULT_GRID, "mode": "shots", "max_s": 4.5, "min_s": 1.0,
    "sensitivity": 0.5, "max_parts": 0, "max_total_s": 0.0, "bounds": [], "segs": [],
    "global_ref": "", "global_ref2": "", "global_prompt": "", "megapixels": 0.15, "multiple": 32,
    "detector": "adaptive", "run": "auto", "filters": {}, "skip_fill": "original",
    "cast": {}, "cast_assign": True, "cast_split": False, "cast_only": False, "mask_cfg": {}, "vlm_cfg": {}, "audio_mode": "auto", "ref_details": {},
}


def _load_plan(plan_json: str) -> dict:
    try:
        p = json.loads(plan_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"BFS Shot Planner: the plan is not valid JSON ({exc})") from exc
    out = dict(DEFAULT_PLAN)
    out.update({k: v for k, v in p.items() if v is not None})
    return out


def resolve_plan(plan: dict, analysis: dict, path: str | None = None) -> list[dict]:
    """Shots for this plan: the boundaries edited in the panel, or an automatic split."""
    fps = float(analysis["fps"])
    grid = plan["grid"]
    max_len = snap_down(int(round(float(plan["max_s"]) * fps)), grid)
    min_len = max(1, int(round(float(plan["min_s"]) * fps)))
    max_total = int(round(float(plan.get("max_total_s") or 0) * fps))
    if path:
        cuts, _ = find_cuts(path, analysis, plan.get("detector", "adaptive"), float(plan["sensitivity"]))
    else:
        cuts = detect_cuts(analysis["score"], analysis["raw"], float(plan["sensitivity"]), fps)
    manual = plan.get("bounds") or None
    mode = "manual" if manual else plan["mode"]
    cast = cached_cast(path, analysis) if path else None   # only once the people were analysed
    if cast is not None and plan.get("cast_split") and not manual:
        extra = []
        bounds = [0] + cuts + [analysis["n"]]
        for a, b in zip(bounds[:-1], bounds[1:]):
            extra += person_change_points(cast, a, b, max(min_len, int(round(fps))))
        cuts = sorted(set(cuts) | set(extra))
    segs = plan_segments(analysis["n"], cuts, mode, max_len, min_len,
                         int(plan.get("max_parts") or 0), max_total, manual)
    meta = plan.get("segs") or []
    for i, s in enumerate(segs):
        m = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
        s["enabled"] = bool(m.get("enabled", True))
        s["ref"] = m.get("ref") or ""
        s["ref2"] = m.get("ref2") or ""
        s["prompt"] = m.get("prompt") or ""
        s["chain"] = m.get("chain") or "off"
        s["chain_frame"] = m.get("chain_frame") or "first"
        s["crop"] = bool(m.get("crop"))
        s["mask"] = m.get("mask") or {}
        s["cut_before"] = bool(s.get("cut_before")) or (s["start"] in cuts)
        s["gen_len"] = snap_up(s["end"] - s["start"], grid)
        s["people"], s["main"] = [], -1
        if cast is not None:
            sp = shot_people(cast, s["start"], s["end"])
            s["people"], s["main"] = sp["people"], sp["main"]
            entry = (plan.get("cast") or {}).get(str(sp["main"])) or {}
            if plan.get("cast_assign", True) and not entry.get("ignore"):
                s["ref"] = s["ref"] or entry.get("ref", "")
                s["ref2"] = s["ref2"] or entry.get("ref2", "")
            linked = [p for p in sp["people"] if (plan.get("cast") or {}).get(str(p), {}).get("ref")
                      and not (plan.get("cast") or {}).get(str(p), {}).get("ignore")]
            s["cast_skip"] = bool(plan.get("cast_only")) and not linked
    return segs


# ---------------------------------------------------------------------------- content filters

DEFAULT_FILTERS = {
    "person": False, "min_person_area": 0.0, "max_persons": 0, "face": False,
    "skip_dark": False, "dark_level": 0.06, "skip_static": False, "static_level": 0.004,
    "min_frames": 0, "samples": 6,
}
_DET: dict[str, Any] = {}
_STATS_CACHE: dict[tuple, dict] = {}


def _yolo(kind: str):
    """YOLO model for 'person' or 'face' from models/ultralytics, or None (OpenCV fallback)."""
    if kind in _DET:
        return _DET[kind]
    model = None
    try:
        from ultralytics import YOLO
        root = os.path.join(folder_paths.models_dir, "ultralytics")
        cands = []
        for sub in ("bbox", "segm", ""):
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                cands += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".pt") and kind in f.lower()]
        if cands:
            model = YOLO(cands[0])
    except Exception:  # noqa: BLE001 - fall back to OpenCV
        model = None
    _DET[kind] = model
    return model


def _detect(frames: list[np.ndarray]) -> list[dict]:
    """Per frame: person boxes (area fraction) and face count."""
    import cv2
    out = [{"persons": [], "faces": 0} for _ in frames]
    pm, fm = _yolo("person"), _yolo("face")
    if pm is not None:
        for i, r in enumerate(pm(frames, verbose=False, conf=0.35, classes=[0])):
            h, w = frames[i].shape[:2]
            for b in r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []:
                out[i]["persons"].append(float((b[2] - b[0]) * (b[3] - b[1]) / (w * h)))
    else:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        for i, fr in enumerate(frames):
            h, w = fr.shape[:2]
            rects, _ = hog.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), winStride=(8, 8))
            out[i]["persons"] = [float(rw * rh / (w * h)) for (_, _, rw, rh) in rects]
    if fm is not None:
        for i, r in enumerate(fm(frames, verbose=False, conf=0.4)):
            out[i]["faces"] = int(len(r.boxes)) if r.boxes is not None else 0
    else:
        casc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        for i, fr in enumerate(frames):
            out[i]["faces"] = int(len(casc.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), 1.1, 5)))
    return out


def shot_stats(path: str, analysis: dict, seg: dict, samples: int = 6) -> dict:
    """Sampled content statistics for one shot (cached)."""
    key = (path, os.path.getmtime(path), float(analysis["fps"]), seg["start"], seg["end"], samples)
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    k = max(1, min(samples, seg["end"] - seg["start"]))
    pick = np.linspace(seg["start"], seg["end"] - 1, k).round().astype(int)
    w = 640
    h = max(32, int(round(w * analysis["height"] / max(1, analysis["width"]))))
    frames = _read_frames(path, src[pick], (w, h))
    det = _detect(frames)
    raw = analysis.get("raw") or []
    motion = float(np.mean(raw[seg["start"] + 1:seg["end"]])) if seg["end"] - seg["start"] > 1 and raw else 0.0
    st = {
        "persons": int(max(len(d["persons"]) for d in det)),
        "person_area": round(float(max([max(d["persons"]) for d in det if d["persons"]] or [0.0])), 4),
        "person_frames": int(sum(1 for d in det if d["persons"])),
        "faces": int(max(d["faces"] for d in det)),
        "brightness": round(float(np.mean([f.mean() / 255.0 for f in frames])), 4),
        "motion": round(motion, 4), "sampled": int(k),
    }
    _STATS_CACHE[key] = st
    return st


def skip_reason(stats: dict, length: int, f: dict) -> str:
    """Why a shot should be skipped under these filters ('' = keep)."""
    if f.get("min_frames") and length < int(f["min_frames"]):
        return f"shorter than {int(f['min_frames'])} frames"
    if f.get("skip_dark") and stats["brightness"] < float(f.get("dark_level", 0.06)):
        return "dark / fade"
    if f.get("skip_static") and stats["motion"] < float(f.get("static_level", 0.004)):
        return "static"
    if f.get("person") and stats["persons"] == 0:
        return "no person"
    if f.get("person") and float(f.get("min_person_area") or 0) > 0 and stats["person_area"] < float(f["min_person_area"]):
        return f"person smaller than {float(f['min_person_area']) * 100:.0f}% of the frame"
    if int(f.get("max_persons") or 0) > 0 and stats["persons"] > int(f["max_persons"]):
        return f"more than {int(f['max_persons'])} people"
    if f.get("face") and stats["faces"] == 0:
        return "no face"
    return ""


def filters_active(f: dict) -> bool:
    return any(f.get(k) for k in ("person", "face", "skip_dark", "skip_static")) or \
        int(f.get("max_persons") or 0) > 0 or int(f.get("min_frames") or 0) > 0


def apply_filters(plan: dict, analysis: dict, path: str, segs: list[dict]) -> list[dict]:
    """Mark each shot with stats and an automatic skip reason; per-shot 'force' overrides it."""
    f = dict(DEFAULT_FILTERS); f.update(plan.get("filters") or {})
    meta = plan.get("segs") or []
    need = filters_active(f)
    for i, s in enumerate(segs):
        force = (meta[i] or {}).get("force", "auto") if i < len(meta) and isinstance(meta[i], dict) else "auto"
        s["force"] = force
        s["stats"] = shot_stats(path, analysis, s, int(f.get("samples") or 6)) if need else None
        s["skip_reason"] = skip_reason(s["stats"], s["end"] - s["start"], f) if need else ""
        if not s["skip_reason"] and s.get("cast_skip"):
            s["skip_reason"] = "no linked person"
        s["run"] = s["enabled"] and (force == "run" or (force != "skip" and not s["skip_reason"]))
    return segs


# ---------------------------------------------------------------------------- cast (people by face)

_FACE_APP: Any = None
_CAST_CACHE: dict[tuple, dict] = {}


def _face_app():
    """InsightFace detector + ArcFace recogniser (buffalo_l), or None when unavailable."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP or None
    try:
        import insightface
        root = os.path.join(folder_paths.models_dir, "insightface")
        # CPU only: light, no VRAM, and no CUDA/cuDNN loading (a missing cuDNN aborts the whole process)
        app = insightface.app.FaceAnalysis(name="buffalo_l", root=root, allowed_modules=["detection", "recognition"],
                                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        # small models: a few threads are faster than onnxruntime's default pool (one per core) and stay light
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = min(4, os.cpu_count() or 4)
        so.inter_op_num_threads = 1
        for model in app.models.values():
            model.session = ort.InferenceSession(model.model_file, so, providers=["CPUExecutionProvider"])
        _FACE_APP = app
    except Exception as exc:  # noqa: BLE001
        print(f"[BFS Shot Planner] face analysis unavailable: {exc!r}")
        _FACE_APP = False
    return _FACE_APP or None


def _cast_key(path: str, analysis: dict, step_s: float = 0.5, threshold: float = 0.42, min_share: float = 0.01) -> tuple:
    return (path, os.path.getmtime(path), float(analysis["fps"]), round(step_s, 3), round(threshold, 3), round(min_share, 4))


def cached_cast(path: str, analysis: dict) -> dict | None:
    """The cast of a video when it was already analysed with the default settings, else None (never computes)."""
    try:
        return _CAST_CACHE.get(_cast_key(path, analysis))
    except OSError:
        return None


def analyze_cast(path: str, analysis: dict, step_s: float = 0.5, threshold: float = 0.42,
                 min_share: float = 0.01, max_faces: int = 3) -> dict:
    """Detect faces every `step_s` seconds (CPU), keep the `max_faces` most probable ones per frame, group them
    into people by ArcFace similarity.

    Returns {"people": [{id, thumb, count, share}], "samples": [{f, faces: [{pid, area}]}]} where
    `f` is a timeline frame. Person ids are stable for the same video and settings.
    """
    import cv2
    key = _cast_key(path, analysis, step_s, threshold, min_share)
    if key in _CAST_CACHE:
        return _CAST_CACHE[key]
    app = _face_app()
    if app is None:
        raise RuntimeError("Face analysis needs the insightface package and the buffalo_l models in models/insightface.")
    fps = float(analysis["fps"])
    step = max(1, int(round(step_s * fps)))
    frames_idx = np.arange(0, analysis["n"], step)
    src = _timeline(analysis["n_src"], analysis["fps_src"], fps)
    w = 960
    h = max(32, int(round(w * analysis["height"] / max(1, analysis["width"]))))
    frames = _read_frames(path, src[frames_idx], (w, h), stage="Reading frames for faces")
    dets = []   # (sample index, embedding, area fraction, crop)
    from insightface.app.common import Face
    rec = app.models["recognition"]
    for si, fr in enumerate(frames):
        _status("Finding faces", si, len(frames))
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        boxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
        # only the most probable faces of the frame get recognised, so a crowd stays cheap and out of the cast
        cand = []
        for k in range(boxes.shape[0]):
            x0, y0, x1, y1 = [int(v) for v in boxes[k, :4]]
            area = max(0, x1 - x0) * max(0, y1 - y0) / float(w * h)
            if area >= 0.0015 and boxes[k, 4] >= 0.5:
                cand.append((float(boxes[k, 4]), k, area))
        for score, k, area in sorted(cand, reverse=True)[:max_faces]:
            f = Face(bbox=boxes[k, :4], kps=kpss[k] if kpss is not None else None, det_score=score)
            rec.get(bgr, f)
            x0, y0, x1, y1 = [int(v) for v in f.bbox]
            pad = int(0.25 * max(x1 - x0, y1 - y0))
            crop = fr[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
            dets.append((si, f.normed_embedding.astype(np.float32), area, crop))
    # greedy online clustering on cosine similarity, then merge close clusters
    cents, members = [], []
    for k, (_, e, _, _) in enumerate(dets):
        if cents:
            sims = np.array([c @ e / (np.linalg.norm(c) + 1e-8) for c in cents])
            j = int(sims.argmax())
            if sims[j] >= threshold:
                members[j].append(k); cents[j] = cents[j] + e
                continue
        cents.append(e.copy()); members.append([k])
    merged = True
    while merged:
        merged = False
        for a in range(len(cents)):
            for b in range(a + 1, len(cents)):
                ca, cb = cents[a] / np.linalg.norm(cents[a]), cents[b] / np.linalg.norm(cents[b])
                if ca @ cb >= threshold:
                    cents[a] = cents[a] + cents[b]; members[a] += members[b]
                    del cents[b]; del members[b]; merged = True
                    break
            if merged:
                break
    total = max(1, len(frames))
    groups = [m for m in members if len({dets[k][0] for k in m}) / total >= min_share]
    groups.sort(key=lambda m: -len({dets[k][0] for k in m}))
    people, owner = [], {}
    for pid, m in enumerate(groups):
        best = max(m, key=lambda k: dets[k][2])
        crop = cv2.resize(dets[best][3], (96, 96), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        seen = sorted({dets[k][0] for k in m})
        people.append({"id": pid, "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                       "count": len(seen), "share": round(len(seen) / total, 4),
                       "first": int(frames_idx[seen[0]]), "last": int(frames_idx[seen[-1]])})
        for k in m:
            owner[k] = pid
    samples = [{"f": int(f), "faces": []} for f in frames_idx]
    for k, (si, _, area, _) in enumerate(dets):
        if k in owner:
            samples[si]["faces"].append({"pid": owner[k], "area": round(float(area), 5)})
    out = {"people": people, "samples": samples, "step": step}
    _CAST_CACHE[key] = out
    return out


def shot_people(cast: dict, start: int, end: int) -> dict:
    """Who appears in [start, end): per person, sampled frames seen and summed face area; plus the main one."""
    score: dict[int, list[float]] = {}
    for smp in cast["samples"]:
        if start <= smp["f"] < end:
            for fc in smp["faces"]:
                v = score.setdefault(fc["pid"], [0, 0.0]); v[0] += 1; v[1] += fc["area"]
    main = max(score, key=lambda p: (score[p][0], score[p][1])) if score else -1
    return {"people": sorted(score), "main": main}


def person_change_points(cast: dict, start: int, end: int, min_run: int) -> list[int]:
    """Frames inside [start, end) where the main (largest) face switches to another person for at least min_run."""
    runs = []
    for smp in cast["samples"]:
        if start <= smp["f"] < end:
            main = max(smp["faces"], key=lambda fc: fc["area"])["pid"] if smp["faces"] else -1
            if main >= 0 and (not runs or runs[-1][1] != main):
                runs.append([smp["f"], main])
    pts = []
    for (f0, p0), (f1, p1) in zip(runs, runs[1:]):
        if f1 - (pts[-1] if pts else start) >= min_run and end - f1 >= min_run:
            pts.append(f1)
    return pts


# ---------------------------------------------------------------------------- shot masks (SAM 3) and crop / uncrop

SAM3_FILE = "sam3.1_multiplex_fp16.safetensors"
SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
_SAM3: dict[str, Any] = {}
_MASK_CACHE: dict[tuple, dict] = {}
# global mask settings (the planner's Mask card); a shot only says what to segment (text / points / key frame)
DEFAULT_MASK = {"text": "", "points": [], "key": 0, "threshold": 0.5, "max_objects": 4, "invert": False,
                "fill_holes": True, "temporal_expand": 2, "blockify": 0, "padding": 0.15, "expand": 16,
                "feather": 12, "paste": "mask"}
SHOT_MASK_KEYS = ("text", "points", "key")


def _sam3_path() -> str:
    """The official SAM 3.1 checkpoint in models/checkpoints, downloaded once when missing."""
    p = folder_paths.get_full_path("checkpoints", SAM3_FILE)
    if p:
        return p
    import urllib.request
    dst = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], SAM3_FILE)
    tmp = dst + ".part"
    print(f"[BFS Shot Planner] downloading {SAM3_FILE} (1.7 GB) from Comfy-Org/sam3.1 ...")
    with urllib.request.urlopen(SAM3_URL) as r, open(tmp, "wb") as f:
        total, done, step = int(r.headers.get("Content-Length") or 0), 0, 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total and done * 10 // total > step:
                step = done * 10 // total
                print(f"[BFS Shot Planner] {SAM3_FILE}: {step * 10}%")
    os.replace(tmp, dst)
    return dst


def _sam3():
    if "model" not in _SAM3:
        import comfy.sd
        out = comfy.sd.load_checkpoint_guess_config(_sam3_path(), output_vae=False, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        _SAM3["model"], _SAM3["clip"] = out[0], out[1]
    return _SAM3["model"], _SAM3["clip"]


def mask_spec(m: dict | None, cfg: dict | None = None) -> dict:
    """Global settings (cfg) with the shot's own target (text / points / key frame) on top."""
    spec = dict(DEFAULT_MASK)
    spec.update({k: v for k, v in (cfg or {}).items() if v is not None and k not in SHOT_MASK_KEYS})
    spec.update({k: v for k, v in (m or {}).items() if v is not None and k in SHOT_MASK_KEYS})
    return spec


def shape_mask(masks: torch.Tensor, spec: dict) -> torch.Tensor:
    """invert -> fill holes -> temporal expand -> blockify, on [N,H,W] 0/1 masks."""
    x = masks.float()
    if spec.get("invert"):
        x = 1 - x
    if spec.get("fill_holes"):
        try:
            from scipy.ndimage import binary_fill_holes
            x = torch.from_numpy(np.stack([binary_fill_holes(f > 0.5) for f in x.numpy()]).astype(np.float32))
        except ImportError:
            pass
    t = int(spec.get("temporal_expand") or 0)
    if t > 0 and x.shape[0] > 1:
        N, H, W = x.shape
        x = torch.nn.functional.max_pool1d(x.permute(1, 2, 0).reshape(-1, 1, N), 2 * t + 1, 1, t) \
            .reshape(H, W, N).permute(2, 0, 1)
    b = int(spec.get("blockify") or 0)
    if b > 1:
        H, W = x.shape[1:]
        cov = torch.nn.functional.avg_pool2d(x[:, None], b, b, ceil_mode=True)[:, 0]
        x = (cov >= 0.5).float().repeat_interleave(b, 1).repeat_interleave(b, 2)[:, :H, :W]
    return (x > 0.5).to(torch.uint8)


def _track(model, imgs: torch.Tensor, init: torch.Tensor | None, cond, spec: dict) -> torch.Tensor:
    from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack
    data = SAM3_VideoTrack.execute(images=imgs, model=model, initial_mask=init, conditioning=cond,
                                   detection_threshold=float(spec["threshold"]),
                                   max_objects=int(spec["max_objects"]), detect_interval=1).args[0]
    return SAM3_TrackToMask.execute(track_data=data, object_indices="").args[0].float().cpu()


def segment_frames(imgs: torch.Tensor, spec: dict) -> torch.Tensor:
    """[N,H,W,3] -> [N,H,W] masks with SAM 3: tracked from the points on the key frame, or from the text."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    model, clip = _sam3()
    text = str(spec.get("text") or "").strip()
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text)) if text else None
    pts = spec.get("points") or []
    N, H, W = imgs.shape[:3]
    if not pts:
        if cond is None:
            raise ValueError("the shot's mask needs a text prompt or points")
        return _track(model, imgs, None, cond, spec)
    key = max(0, min(N - 1, int(spec.get("key") or 0)))
    pos = [{"x": p["x"] * W, "y": p["y"] * H} for p in pts if p.get("label", 1)]
    neg = [{"x": p["x"] * W, "y": p["y"] * H} for p in pts if not p.get("label", 1)]
    first = SAM3_Detect.execute(model=model, image=imgs[key:key + 1], positive_coords=json.dumps(pos),
                                negative_coords=json.dumps(neg), threshold=float(spec["threshold"]),
                                refine_iterations=2).args[0][:1].float()
    fwd = _track(model, imgs[key:], first, None, spec)                     # key frame -> end
    if key == 0:
        return fwd
    back = _track(model, imgs[:key + 1].flip(0), first, None, spec).flip(0)  # key frame -> start
    return torch.cat([back[:-1], fwd], 0)


def shot_mask(path: str, analysis: dict, start: int, length: int, spec: dict) -> dict:
    """Masks of one shot (timeline frames start..start+length) at a working size, cached; plus the crop box."""
    key = (path, os.path.getmtime(path), float(analysis["fps"]), int(start), int(length),
           json.dumps({k: spec[k] for k in ("text", "points", "key", "threshold", "max_objects")}, sort_keys=True))
    if key not in _MASK_CACHE:
        src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
        idx = np.clip(np.arange(start, start + length), 0, analysis["n"] - 1)
        sw, sh = analysis["width"], analysis["height"]
        s = 640 / max(sw, sh)
        w, h = max(32, int(sw * s) // 2 * 2), max(32, int(sh * s) // 2 * 2)
        frames = _read_frames(path, src[idx], (w, h), stage="Reading the shot")
        imgs = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        _status(f"Segmenting {len(frames)} frames with SAM 3", force=True)
        _MASK_CACHE[key] = {"masks": (segment_frames(imgs, spec) > 0.5).to(torch.uint8), "size": (w, h)}
        _node_boundary()
    out = dict(_MASK_CACHE[key])
    out["masks"] = shape_mask(out["masks"], spec)
    out["box"] = crop_box(out["masks"], float(spec["padding"]))
    return out


def crop_box(masks: torch.Tensor, padding: float) -> list[float] | None:
    """One box for the whole shot (union of every frame's mask) plus padding, normalised [x0, y0, x1, y1]."""
    if masks.numel() == 0 or not bool(masks.any()):
        return None
    union = masks.amax(0)
    ys, xs = torch.nonzero(union, as_tuple=True)
    H, W = union.shape
    x0, x1, y0, y1 = xs.min().item() / W, (xs.max().item() + 1) / W, ys.min().item() / H, (ys.max().item() + 1) / H
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    return [max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)]


def grow_blur(m: torch.Tensor, grow: int, blur: int) -> torch.Tensor:
    """[N,H,W] in 0-1: dilate by `grow` px, then soften the edge by `blur` px."""
    x = m.float()[:, None]
    if grow > 0:
        x = torch.nn.functional.max_pool2d(x, 2 * grow + 1, 1, grow)
    for _ in range(2 if blur > 0 else 0):
        x = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (blur,) * 4, mode="replicate"), 2 * blur + 1, 1)
    return x[:, 0].clamp(0, 1)


def box_px(box: list[float], W: int, H: int, multiple: int = 2) -> tuple[int, int, int, int]:
    x0, y0 = int(box[0] * W), int(box[1] * H)
    x1, y1 = max(x0 + multiple, int(round(box[2] * W))), max(y0 + multiple, int(round(box[3] * H)))
    return x0, y0, min(W, x1), min(H, y1)


def crop_panel(img: torch.Tensor, shot: dict) -> torch.Tensor:
    """Cut a duet panel off a decoded canvas (the shot was conditioned with a duet panel)."""
    info = shot.get("panel")
    if not info or img.shape[1:3] == (shot.get("height"), shot.get("width")):
        return img
    h, w = info["h"] * 16, info["w"] * 16
    y0 = info["strip_h"] * 16 if info["position"] == "top" else 0
    x0 = info["strip_w"] * 16 if info["position"] == "left" else 0
    if img.shape[1] < y0 + h or img.shape[2] < x0 + w:
        return img
    return img[:, y0:y0 + h, x0:x0 + w]


def uncrop(result: torch.Tensor, shot: dict) -> torch.Tensor:
    """Paste a cropped shot's result back into its full frames (feathered by the mask or the box)."""
    c = shot["crop"]
    full = shot["full_frames"]
    W, H = full.shape[2], full.shape[1]
    x0, y0, x1, y1 = box_px(c["box"], W, H)
    n = min(result.shape[0], full.shape[0])
    res = torch.nn.functional.interpolate(result[:n, ..., :3].movedim(-1, 1).float(), size=(y1 - y0, x1 - x0),
                                          mode="bilinear", align_corners=False).movedim(1, -1).clamp(0, 1)
    out = full[:n].clone()
    if c.get("paste") == "box":
        a = torch.zeros(n, H, W)
        a[:, y0:y1, x0:x1] = 1
        alpha = grow_blur(a, 0, int(c.get("feather", 12)))
    else:
        m = c["mask"][:n].float()
        alpha = grow_blur(m, int(c.get("expand", 8)), int(c.get("feather", 12)))
    a = alpha[:, y0:y1, x0:x1, None]
    out[:, y0:y1, x0:x1] = res * a + out[:, y0:y1, x0:x1] * (1 - a)
    if result.shape[0] > n:   # frames past the planned length stay as generated, pasted the same way
        out = torch.cat([out, out[-1:].expand(result.shape[0] - n, -1, -1, -1)], 0)
    return out


def mask_preview(path: str, analysis: dict, start: int, length: int, spec: dict, count: int = 6) -> dict:
    """A few frames of the shot with the mask in red and the crop box, as data URLs."""
    import cv2
    r = shot_mask(path, analysis, start, length, spec)
    masks, (w, h), box = r["masks"], r["size"], r["box"]
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    picks = sorted(set(int(round(i)) for i in np.linspace(0, length - 1, min(count, length))))
    frames = _read_frames(path, src[np.clip(np.array(picks) + start, 0, analysis["n"] - 1)], (w, h))
    out = []
    for i, fr in zip(picks, frames):
        img = fr.copy()
        m = masks[min(i, masks.shape[0] - 1)].numpy().astype(bool)
        img[m] = (img[m] * 0.45 + np.array([255, 40, 60]) * 0.55).astype(np.uint8)
        if box:
            x0, y0, x1, y1 = box_px(box, w, h)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 210, 90), 2)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append({"f": start + i, "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})
    cover = float(masks.float().mean()) if masks.numel() else 0.0
    return {"box": box, "frames": out, "coverage": cover, "empty": box is None}


# ---------------------------------------------------------------------------- VLM suggestions (optional)

DEFAULT_VLM = {"enabled": False, "frames": 3, "max_tokens": 1024, "auto_segment": True, "auto_shot": True,
               "instruction": "", "describe_preset": "full body", "describe_custom": "",
               "write_prompt": False, "write_task": "character swap", "write_change": ""}
_VLM: dict[str, Any] = {}            # the VLM connected to a planner (kept for the panel's Analyse button)
_VLM_CACHE: dict[tuple, dict] = {}

VLM_QUESTION = """You see {n} frames (in time order) of ONE camera shot from a video that will be edited with an AI video model.
Answer with one JSON object only, no code fence:
{{"segment": "the main person or object a user would edit, as a short English noun phrase a segmentation model understands, describing what it looks like (e.g. 'woman in a black top', 'man in a grey suit', 'red car'); never on-screen text",
"shot": "one or two sentences for a video prompt: camera distance and angle, camera movement, where the subject is in the frame and what they do, in time order; describe actions and framing, not the person's face or clothes",
"people": <number of people visible>,
"recommend": "run" or "skip" (skip when the shot has nobody to edit, is a title card, a black frame or text only),
"reason": "a few words"}}{extra}"""


async def _off_loop(fn):
    """Run slow work in a thread so the server keeps answering (and the panel gets progress events)."""
    import asyncio
    _outside_prompt()
    return await asyncio.get_running_loop().run_in_executor(None, fn)


def _outside_prompt() -> None:
    """Models run from a panel button (no prompt running): ComfyUI's progress hook reads the server's last prompt
    and node ids, which do not exist until a first prompt has run."""
    try:
        from server import PromptServer
        srv = PromptServer.instance
        for attr in ("last_prompt_id", "last_node_id"):
            if not hasattr(srv, attr):
                setattr(srv, attr, None)
    except Exception:  # noqa: BLE001
        pass


def _node_boundary() -> None:
    """What ComfyUI does between two nodes: drop the per-thread CUDA malloc graph and prefetch queues. Needed when
    one node runs a model several times (a second text generation in the same node otherwise hits a device assert)."""
    try:
        import comfy.model_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
    except Exception:  # noqa: BLE001 - older ComfyUI without it
        pass


def vlm_generate(clip, prompt: str, images: torch.Tensor, max_tokens: int) -> str:
    """Text from a vision-language CLIP. Some models end the answer before writing anything for some wordings:
    retry with a reworded request, then with sampling, and clean leftovers of the chat template."""
    from comfy_extras.nodes_textgen import TextGenerate
    tries = [(prompt, {"sampling_mode": "off"}),
             ("Look at the picture(s) carefully. " + prompt + " Write the answer now.", {"sampling_mode": "off"}),
             (prompt, {"sampling_mode": "on", "temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.0,
                       "repetition_penalty": 1.05, "presence_penalty": 0.0, "seed": 1})]
    text = ""
    for q, mode in tries:
        text = TextGenerate.execute(clip=clip, prompt=q, max_length=int(max_tokens), sampling_mode=mode,
                                    image=images, mtp="off").args[0]
        _node_boundary()
        text = str(text or "").strip()
        for junk in ("assistant\n", "assistant:", "assistant"):
            if text.lower().startswith(junk):
                text = text[len(junk):].strip()
        if text:
            break
    return text


def vlm_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_VLM)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    return out


def _parse_json(text: str) -> dict:
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[a:b + 1]) if a >= 0 and b > a else {}
    except json.JSONDecodeError:
        return {}


def vlm_shot(clip, path: str, analysis: dict, start: int, end: int, cfg: dict) -> dict:
    """Ask the VLM about one shot (a few frames); cached per shot and settings."""
    key = (path, os.path.getmtime(path), int(start), int(end), id(clip),
           json.dumps({k: cfg[k] for k in ("frames", "max_tokens", "instruction")}, sort_keys=True))
    if key in _VLM_CACHE:
        return _VLM_CACHE[key]
    from comfy_extras.nodes_textgen import TextGenerate
    _status("Asking the VLM about the shot", force=True)
    n = max(1, min(8, int(cfg["frames"])))
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    picks = np.clip(np.linspace(start, end - 1, n).round().astype(int), 0, analysis["n"] - 1)
    s = 448 / max(analysis["width"], analysis["height"])
    size = (max(32, int(analysis["width"] * s) // 2 * 2), max(32, int(analysis["height"] * s) // 2 * 2))
    imgs = torch.from_numpy(np.stack(_read_frames(path, src[picks], size)).astype(np.float32) / 255.0)
    extra = ("\n" + cfg["instruction"].strip()) if str(cfg.get("instruction") or "").strip() else ""
    q = VLM_QUESTION.format(n=n, extra=extra)
    text = vlm_generate(clip, q, imgs, int(cfg["max_tokens"]))

    out = _parse_json(text)
    out = {"segment": str(out.get("segment") or "").strip(), "shot": str(out.get("shot") or "").strip(),
           "people": out.get("people"), "recommend": str(out.get("recommend") or "run").lower(),
           "reason": str(out.get("reason") or "").strip(), "raw": "" if out else text}
    _VLM_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------- reference descriptions (VLM)

DESCRIBE_PRESETS = {
    "full body": ("Describe the person in these reference pictures for a video-generation prompt: one paragraph of "
                  "3 to 5 sentences in English. Only what is visible: apparent gender and age, skin tone and texture, "
                  "face shape and distinctive facial features (eyes, eyebrows, nose, lips, facial hair, wrinkles, "
                  "freckles), hair colour, length, texture and style (hairline, parting, bald areas), body build and "
                  "proportions, then the clothing piece by piece with colours and materials, shoes and accessories. "
                  "Describe only physical traits and clothing: never the pose, gesture, action, expression, "
                  "camera, framing or background. Concrete words only: no negations, no names, no opinions."),
    "head / face": ("Describe the head and face of the person in these reference pictures for a video-generation "
                    "prompt: one paragraph of 3 to 4 sentences in English. Only what is visible: apparent gender and "
                    "age, skin tone and texture, head and face shape, eyes, eyebrows, nose, lips, jawline, ears, "
                    "facial hair, wrinkles, freckles or marks, glasses, and the hair: colour, length, texture, style, "
                    "hairline. Describe only physical traits: never the pose, head angle, gaze, expression, action, "
                    "camera or background. Concrete words only: no negations, no names, no opinions."),
    "face attributes": ("Describe this face in a short comma-separated attribute list, in exactly this style: \"Male, oval "
                        "face shape, average-sized head with strong jawline, light brown skin, dark eyes, black tousled "
                        "hair, silver hoop earring.\" Cover, in order: gender, head/face shape and proportions (e.g. "
                        "oval/round/square/heart-shaped, narrow/wide, jaw structure, whether the head reads as "
                        "small/average/large relative to the shoulders), skin tone, eye color, hair color and style, and any "
                        "distinctive features (facial hair, jewelry, makeup, glasses, etc.). Physical traits only, never "
                        "the pose, expression or background. Only output the description, nothing else."),
    "outfit": ("Describe only what the person in these reference pictures wears, piece by piece, for a "
               "video-generation prompt: garments, colours, materials, fit, shoes and accessories, in 2 or 3 "
               "sentences in English. Never the pose, action or background. No negations, no names."),
}
_DESCRIBE_CACHE: dict[tuple, str] = {}


def _image_key(img: torch.Tensor | None) -> str:
    if img is None:
        return ""
    x = torch.nn.functional.interpolate(img[:1, ..., :3].movedim(-1, 1).float(), size=(32, 32), mode="area")
    return hashlib.sha1((x * 255).round().to(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def describe_instruction(cfg: dict) -> str:
    preset = cfg.get("describe_preset", "full body")
    if preset == "custom":
        return str(cfg.get("describe_custom") or DESCRIBE_PRESETS["full body"]).strip()
    return DESCRIBE_PRESETS.get(preset, DESCRIBE_PRESETS["full body"])


def vlm_describe(clip, images: list, instruction: str, max_tokens: int = 320) -> str:
    """One description for a set of reference pictures (e.g. a close-up and a full-body photo of one person)."""
    images = [im for im in images if im is not None]
    if not images:
        return ""
    key = (tuple(_image_key(im) for im in images), instruction, id(clip), int(max_tokens))
    if key in _DESCRIBE_CACHE:
        return _DESCRIBE_CACHE[key]
    from comfy_extras.nodes_textgen import TextGenerate
    _status("Describing the references with the VLM", force=True)
    side = 448   # same size for the batch: letterbox every picture on grey
    batch = []
    for im in images:
        x = im[:1, ..., :3].movedim(-1, 1).float()
        s = side / max(x.shape[-2:])
        nh, nw = max(2, int(x.shape[-2] * s)), max(2, int(x.shape[-1] * s))
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, side, side), 0.5)
        canvas[:, :, (side - nh) // 2:(side - nh) // 2 + nh, (side - nw) // 2:(side - nw) // 2 + nw] = x
        batch.append(canvas)
    imgs = torch.cat(batch, 0).movedim(1, -1).clamp(0, 1)
    text = vlm_generate(clip, instruction, imgs, int(max_tokens))
    text = " ".join(str(text).replace("```", " ").split())
    _DESCRIBE_CACHE[key] = text
    return text


def ref_set_key(ref: str, ref2: str) -> str:
    """Key of a reference set in plan['ref_details'] (edited descriptions)."""
    return f"{ref or ''}|{ref2 or ''}"


# ---------------------------------------------------------------------------- prompt writer (VLM)

WRITER_TASKS = ["character swap", "style", "setting", "appearance", "lighting / weather", "custom"]

WRITER_INSTRUCTION = """You write the prompt for MiniMax H3, a video model, for a split-screen "duet": the KEPT FOOTAGE (the first {nf} images, one camera shot in time order) stays exactly as it is in one half of the frame, and the other half is generated moving in sync with it. {refs_txt}

Task: {task}. {change}

Rules (render-tested, follow them exactly):
1. Never describe the kept footage's performer, face, hair, clothes, props or room, not even to contrast them: whatever you name gets drawn, whatever you leave out is copied from the kept footage.
2. The kept footage has no tag: call it "the kept footage". Never write <Video 1>.
3. In the shot paragraph, restate the new look with two or three concrete words from the reference pictures (hair, outfit, face){face_rule}.
4. Describe what the camera records in the kept footage: distance, angle and movement, where the subject is, and what they do, one sentence per real action, in time order.
5. Write only what is seen. No negations (no "not", "no", "never", "without"), no quality words ("realistic", "high quality", "cinematic").
6. Keep the literal text {{layout}} exactly where it is in the template.

Output exactly these six sections in this order, nothing before or after:

subject_definitions:
{subject_line}

summary:
[reference generation] The target video is a split screen: the kept footage beside <the generated half>, which moves in sync with it.

retention_analysis:
<one line per subject: "... fully_preserved - ... are retained.">
The kept footage: fully_preserved - the panel is kept exactly.

detailed_description:
The target video is in <style and medium>. {{layout}}

[Shot 1] <camera, framing, the subject restated, the actions>

overall_soundscape:
<the sounds of the scene>

non_diegetic_music:
None."""

TASK_SUBJECT = {
    "character swap": '"<Subject 1> is the person whose appearance comes from <Picture 1>[ and <Picture 2>]: <face, hair, skin, outfit piece by piece, from the pictures>."',
    "style": '"The kept footage is the motion reference of the split screen." (and, when there are reference pictures, "<Picture 1> is the style reference.")',
    "setting": '"<Subject 1> is the place: <the requested place with two or three concrete details>." (from <Picture 1> when given)',
    "appearance": '"<Subject 1> is the performer of the kept footage, now <the requested look in concrete seen words>."',
    "lighting / weather": '"The kept footage is the motion reference of the split screen."',
    "custom": '"<Subject 1> is ..." for whatever the change brings in, from the pictures when given.',
}

TASK_HINTS = {
    "character swap": "Replace the performer with the person from the reference pictures; the place stays the same room.",
    "style": "Keep the same performance and camera, redrawn in the requested style.",
    "setting": "Keep the same performer and performance, moved to the requested place.",
    "appearance": "Keep the same performance, with the performer's look changed as requested.",
    "lighting / weather": "Keep the same performance and place, under the requested light or weather.",
    "custom": "Do what the change below asks, keeping the kept footage's performance and camera in sync.",
}


def _letterbox_batch(images: list, side: int = 448) -> torch.Tensor:
    batch = []
    for im in images:
        x = im[:1, ..., :3].movedim(-1, 1).float()
        s = side / max(x.shape[-2:])
        nh, nw = max(2, int(x.shape[-2] * s)), max(2, int(x.shape[-1] * s))
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, side, side), 0.5)
        canvas[:, :, (side - nh) // 2:(side - nh) // 2 + nh, (side - nw) // 2:(side - nw) // 2 + nw] = x
        batch.append(canvas)
    return torch.cat(batch, 0).movedim(1, -1).clamp(0, 1)


def _clean_written(text: str) -> str:
    t = text.replace("```", "").strip()
    i = t.find("subject_definitions:")
    if i > 0:
        t = t[i:]
    t = t.replace("<Video 1>", "the kept footage")
    if "{layout}" not in t and "detailed_description:" in t:
        a = t.index("detailed_description:") + len("detailed_description:")
        nl = t.find("\n[Shot", a)
        cut = nl if nl > 0 else len(t)
        t = t[:cut].rstrip() + " {layout}\n" + t[cut:]
    return t


def vlm_write_prompt(clip, path: str, analysis: dict, start: int, end: int, refs: list, task: str, change: str,
                     max_tokens: int = 1024, frames: int = 4) -> str:
    """A duet prompt for one shot, written by the VLM from frames of the shot and the reference pictures."""
    src = _timeline(analysis["n_src"], analysis["fps_src"], float(analysis["fps"]))
    n = max(1, min(8, int(frames)))
    picks = np.clip(np.linspace(start, end - 1, n).round().astype(int), 0, analysis["n"] - 1)
    fr = [torch.from_numpy(f.astype(np.float32) / 255.0)[None] for f in _read_frames(path, src[picks], None)]
    return write_duet_prompt(clip, fr, refs, task, change, max_tokens)


def write_duet_prompt(clip, frames: list, refs: list, task: str, change: str, max_tokens: int = 1024,
                      mode: str = "canvas") -> str:
    """A duet prompt from frames of the kept footage ([1,H,W,3] each) and the reference pictures. The VLM only fills
    a few fields (look, medium, camera and actions, sounds); the six REF2VA sections are assembled here, so a small
    model cannot break the format."""
    refs = [r for r in refs if r is not None]
    fr = [f for f in frames if f is not None]
    key = ("write2", tuple(_image_key(f) for f in fr), tuple(_image_key(r) for r in refs), task, change, id(clip),
           int(max_tokens), mode)
    if key in _DESCRIBE_CACHE:
        return _DESCRIBE_CACHE[key]
    _status("Writing the prompt with the VLM", force=True)
    swap = task == "character swap" and bool(refs)
    # two separate questions: mixing the clip's frames and the reference photos in one batch makes the VLM
    # describe the reference photo (studio, standing still) as if it were the video
    look = ""
    if swap:
        look = vlm_describe(clip, refs, "Describe the person in these reference pictures in ONE sentence of concrete seen "
                            "words: apparent gender and age, face, hair colour, length and style, skin, and the clothing "
                            "piece by piece with colours. Physical traits and clothing only: never the pose, expression, "
                            "camera or background.", max_tokens).strip().rstrip(".")
    q = (f"These {len(fr)} images are frames, in time order, of one camera shot of a video. "
         "Answer with one JSON object only, no code fence:\n{"
         + '"medium": "what the video is, e.g. handheld vertical smartphone footage under soft window light", '
         + '"shot": "the camera distance, angle and movement, then what the person does, in time order, one short '
           'sentence per real action; call them the person and never describe their face, hair or clothes", '
         + '"sounds": "the sounds of the scene in a few words"}')
    text = vlm_generate(clip, q, _letterbox_batch(fr), max_tokens)
    d = _parse_json(text)
    medium = str(d.get("medium") or "real camera footage").strip().rstrip(".")
    shot = str(d.get("shot") or "").strip()
    sounds = str(d.get("sounds") or "the sounds of the kept footage").strip().rstrip(".")
    for w in ("The person", "the person"):
        shot = shot.replace(w, "<Subject 1>" if swap else w)
    pics = " and ".join(f"<Picture {k + 1}>" for k in range(len(refs)))
    chg = change.strip().rstrip(".")
    split = mode != "shifted"
    lead = ("a split screen: the kept footage beside " if split else "")
    if swap:
        defs = f"<Subject 1> is the person whose appearance comes from {pics}" + (f": {look}." if look else ".")
        if chg:
            defs += f" {chg}."
        who = "<Subject 1>"
        summary = (f"[reference generation] The target video is {lead}<Subject 1>, who " if split else
                   "[reference generation] The target video shows <Subject 1>, who ") + "moves in sync with the kept footage, in the same place."
        keep = f"<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair and clothing from {pics} are retained."
        short = ", ".join(look.split(", ")[:3]) if look else ""
        restate = f" <Subject 1>, the face from <Picture 1>{', ' + short if short else ''}, performs every movement in sync with the kept footage."
        style = f"The target video is in a realistic style, as {medium}."
    else:
        what = {"style": f"redrawn as {chg or 'the requested style'}", "setting": f"moved to {chg or 'the requested place'}",
                "appearance": f"with the performer now {chg or 'changed'}", "lighting / weather": f"under {chg or 'the new light'}",
                }.get(task, chg or "changed as requested")
        defs = (f"<Subject 1> is the place: {chg}." if task == "setting" and chg else
                "The kept footage is the motion reference.") + (f" {pics} show the look to follow." if refs else "")
        summary = (f"[reference generation] The target video is {lead}the same performance, {what}." if split else
                   f"[reference generation] The target video shows the same performance as the kept footage, {what}.")
        keep = f"The performance and camera: fully_preserved - every movement and expression in sync with the kept footage."
        restate = f" The same performance, {what}, in sync with the kept footage."
        style = (f"The target video is in {chg}." if task == "style" and chg else f"The target video is in a realistic style, as {medium}.")
    out = "\n\n".join([
        "subject_definitions:\n" + defs,
        "summary:\n" + summary,
        "retention_analysis:\n" + keep + "\nThe kept footage: fully_preserved - the panel is kept exactly.",
        "detailed_description:\n" + style + " {layout}\n\n[Shot 1] " + (shot or "The same framing and camera as the kept footage.") + restate,
        "overall_soundscape:\n" + (sounds[0].upper() + sounds[1:] if sounds else "The sounds of the kept footage") + ".",
        "non_diegetic_music:\nNone.",
    ])
    _DESCRIBE_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------- continuity between shots

CHAIN_MODES = ("off", "reference", "first frame")
_LAST_RESULT: dict[str, Any] = {}   # auto loop: the last shot a render node produced (index, count, frames)


def pick_frame(frames: torch.Tensor, length: int, which: str) -> torch.Tensor:
    """One frame [1,H,W,3] of a shot's result: its first, middle or last real frame (not the overlap)."""
    n = max(1, min(int(length), frames.shape[0]))
    i = {"first": 0, "middle": n // 2}.get(which, n - 1)
    return frames[i:i + 1, ..., :3].float()


def remember_result(shot: dict, frames: torch.Tensor) -> None:
    _LAST_RESULT.clear()
    _LAST_RESULT.update(index=int(shot.get("index", 0)), count=int(shot.get("count", 0)), frames=frames.detach().cpu())


def chain_image(shot: dict) -> torch.Tensor | None:
    """The previous shot's result frame this shot continues from, if it asked for one and it exists."""
    if shot.get("chain", "off") == "off" or int(shot.get("index", 0)) == 0:
        return None
    if shot.get("chain_image") is not None:
        return shot["chain_image"]
    if _LAST_RESULT.get("index") == int(shot["index"]) - 1 and _LAST_RESULT.get("count") == int(shot.get("count", 0)):
        prev_len = shot.get("prev_length") or _LAST_RESULT["frames"].shape[0]
        return pick_frame(_LAST_RESULT["frames"], prev_len, shot.get("chain_frame", "first"))
    return None


def _fit_to(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x = img[..., :3].movedim(-1, 1).float()
    return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)


# ---------------------------------------------------------------------------- queue loop state

def run_id_for(plan_json: str, path: str) -> str:
    h = hashlib.sha1((plan_json + str(os.path.getmtime(path))).encode()).hexdigest()[:16]
    return h


def run_dir(run_id: str) -> str:
    d = os.path.join(folder_paths.get_temp_directory(), "bfs_shotloop", run_id)
    os.makedirs(d, exist_ok=True)
    return d


def run_state(run_id: str) -> dict:
    f = os.path.join(run_dir(run_id), "state.json")
    if os.path.isfile(f):
        with open(f) as fh:
            return json.load(fh)
    return {"done": [], "count": 0}


def save_state(run_id: str, st: dict) -> None:
    with open(os.path.join(run_dir(run_id), "state.json"), "w") as fh:
        json.dump(st, fh)


def _notify(event: str, data: dict) -> None:
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, data)
    except Exception:  # noqa: BLE001 - headless runs have no UI to notify
        pass


# ---------------------------------------------------------------------------- images

def _load_image(name: str) -> torch.Tensor | None:
    if not name:
        return None
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(_input_path(name))).convert("RGB")
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]


def _grey(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.full((1, h, w, 3), 0.5)


# ---------------------------------------------------------------------------- nodes

class BFSShotPlanner:
    """Split a video into shots that fit the model, with a reference and prompt per shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": json.dumps(DEFAULT_PLAN), "multiline": True,
                                    "tooltip": "The panel writes this. JSON with the video, split settings, "
                                               "boundaries and the per-shot reference and prompt."}),
            },
            "optional": {
                "ref_image": ("IMAGE", {"tooltip": "Default reference for every shot that has none of its own "
                                                   "(overrides the panel's global reference)."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Default second reference (e.g. a full-body photo)."}),
                "prompt": ("STRING", {"forceInput": True,
                                      "tooltip": "Default prompt for shots without their own "
                                                 "(overrides the panel's global prompt)."}),
                "vlm": ("CLIP", {"tooltip": "Optional vision-language model (CLIPLoader with qwen3vl_4b / qwen3vl_8b). "
                                            "It looks at every shot and suggests what to segment, a description of the "
                                            "shot (fills {shot} in the prompt) and whether to run it. Settings in the "
                                            "panel's VLM card."}),
            },
        }

    RETURN_TYPES = ("BFS_SHOT", "INT", "FLOAT", "INT", "INT", "AUDIO", "STRING", "BFS_SHOT_TIMELINE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("shots", "count", "fps", "width", "height", "audio", "summary", "timeline", "ref_image", "ref_image_2")
    OUTPUT_IS_LIST = (True, False, False, False, False, False, False, False, True, True)
    OUTPUT_TOOLTIPS = (
        "One item per shot. Every node that receives this list runs once per shot; connect it to "
        "BFS Shot Unpack or BFS Shot H3 Conditioning, sample, decode, then BFS Shot Join.",
        "Number of shots that will run.", "Timeline frame rate.", "Generation width.",
        "Generation height.", "The whole soundtrack, trimmed to the planned duration.",
        "Human-readable plan.",
        "Every shot in order, including the ones that do not run (disabled or filtered out). Connect it "
        "to BFS Shot Join so skipped shots are filled with the original video (or dropped).",
        "The references the shots use, without repeats: one image when every shot shares the same reference.",
        "The second references the shots use, without repeats.")
    FUNCTION = "plan_shots"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Split a long video into model-sized shots (at camera cuts, fixed length, or by hand), "
                   "give every shot its own reference image and prompt, and run the rest of the graph once "
                   "per shot. Join the decoded shots with BFS Shot Join.")

    @classmethod
    def IS_CHANGED(cls, plan, **kwargs):
        try:
            if _load_plan(plan).get("run") == "queue":
                return float("nan")   # the next shot depends on the loop state on disk
        except ValueError:
            pass
        return plan

    def plan_shots(self, plan, ref_image=None, ref_image_2=None, prompt=None, vlm=None):
        p = _load_plan(plan)
        if vlm is not None:
            _VLM["clip"] = vlm       # the panel's Analyse button uses it too
        vcfg = vlm_cfg(p.get("vlm_cfg"))
        use_vlm = vlm is not None and vcfg["enabled"]
        suggestions = []
        path = _input_path(p["video"])
        fps = float(p["fps"])
        a = analyze(path, fps)
        if p.get("cast") or p.get("cast_split"):
            analyze_cast(path, a)   # warms the cache so the plan uses the same people as the panel
        all_segs = apply_filters(p, a, path, resolve_plan(p, a, path))
        segs = [s for s in all_segs if s["run"]]
        if not segs:
            raise ValueError("BFS Shot Planner: no shot is left to run (all disabled or filtered out).")
        W, H = generation_size(a["width"], a["height"], float(p["megapixels"]), int(p["multiple"]))
        src = _timeline(a["n_src"], a["fps_src"], fps)
        audio = _read_audio(path) if p.get("audio_mode", "auto") != "silent" else None
        g_ref = ref_image if ref_image is not None else _load_image(p.get("global_ref", ""))
        g_ref2 = ref_image_2 if ref_image_2 is not None else _load_image(p.get("global_ref2", ""))
        g_prompt = prompt if prompt is not None else p.get("global_prompt", "")
        queue = p.get("run") == "queue"
        rid = run_id_for(plan, path) if queue else ""
        todo = list(range(len(segs)))
        if queue:
            st = run_state(rid)
            st["count"] = len(segs)
            save_state(rid, st)
            pending = [i for i in todo if i not in st["done"]]
            todo = pending[:1] if pending else [len(segs) - 1]
        cache_imgs: dict[str, torch.Tensor] = {}

        def ref_for(name, default):
            if not name:
                return default
            if name not in cache_imgs:
                cache_imgs[name] = _load_image(name)
            return cache_imgs[name]

        g_names = {"ref": "__socket__" if ref_image is not None else p.get("global_ref", ""),
                   "ref2": "__socket__" if ref_image_2 is not None else p.get("global_ref2", "")}

        def unique(field, default):
            seen, out = set(), []
            for s in segs:
                key = s[field] or g_names[field]          # the image this shot actually uses
                img = ref_for(s[field], default)
                if img is None or key in seen:
                    continue
                seen.add(key)
                out.append(img)
            return out or [_grey()]

        used_refs, used_refs2 = unique("ref", g_ref), unique("ref2", g_ref2)
        details = p.get("ref_details") or {}

        def fill_details(text, rname, r2name, rimg, r2img):
            """{details} = the description of this shot's references: edited in the panel, else from the VLM."""
            if "{details}" not in text:
                return text
            d = details.get(ref_set_key(rname, r2name), "")
            if not d and vlm is not None:
                d = vlm_describe(vlm, [rimg, r2img], describe_instruction(vcfg), int(vcfg["max_tokens"]))
            return text.replace("{details}", d)

        # the VLM answers for every shot first: switching between it and SAM 3 mid-generation breaks the VLM
        vlm_out = {i: vlm_shot(vlm, path, a, segs[i]["start"], segs[i]["end"], vcfg) for i in todo} if use_vlm else {}
        written = {}
        if vlm is not None and vcfg.get("write_prompt"):   # the VLM writes the prompt of every shot without its own
            for i in todo:
                if not segs[i]["prompt"]:
                    written[i] = vlm_write_prompt(vlm, path, a, segs[i]["start"], segs[i]["end"],
                                                  [ref_for(segs[i]["ref"], g_ref), ref_for(segs[i]["ref2"], g_ref2)],
                                                  vcfg["write_task"], vcfg["write_change"], int(vcfg["max_tokens"]))
        shots = []
        for i, s in enumerate(segs):
            if i not in todo:
                continue
            idx = np.arange(s["start"], s["start"] + s["gen_len"])
            idx = np.clip(idx, 0, a["n"] - 1)          # past the end: hold the last frame
            frames = _read_frames(path, src[idx], (W, H))
            ft = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
            ref = ref_for(s["ref"], g_ref)
            ref2 = ref_for(s["ref2"], g_ref2)
            prev_len = (segs[i - 1]["end"] - segs[i - 1]["start"]) if i > 0 else 0
            chain_img = None
            if queue and i > 0 and s["chain"] != "off":
                fp = os.path.join(run_dir(rid), f"shot_{i - 1:04d}.pt")
                if os.path.exists(fp):
                    prev = torch.load(fp)["frames"].float() / 255.0
                    chain_img = _fit_to(pick_frame(prev, prev_len, s["chain_frame"]), W, H)
            sug = vlm_out.get(i)
            if sug is not None:
                suggestions.append(dict(sug, start=s["start"], end=s["end"]))
                if vcfg["auto_segment"] and sug["segment"] and not (s["mask"].get("text") or s["mask"].get("points")):
                    s["mask"] = dict(s["mask"], text=sug["segment"])
            crop, full_frames = None, None
            if s.get("crop"):
                spec = mask_spec(s.get("mask"), p.get("mask_cfg"))
                r = shot_mask(path, a, s["start"], s["gen_len"], spec)
                if r["box"] is not None:
                    full_frames = ft
                    sx0, sy0, sx1, sy1 = box_px(r["box"], a["width"], a["height"])
                    cw, chh = generation_size(sx1 - sx0, sy1 - sy0, float(p["megapixels"]), int(p["multiple"]))
                    hi = _read_frames(path, src[idx], None)               # source resolution, then crop
                    ft = torch.from_numpy(np.stack([_fit(f[sy0:sy1, sx0:sx1], (cw, chh)) for f in hi])
                                          .astype(np.float32) / 255.0)
                    m = torch.nn.functional.interpolate(r["masks"][:, None].float(), size=(H, W), mode="nearest")[:, 0]
                    crop = {"box": r["box"], "mask": m, "paste": spec["paste"], "expand": int(spec["expand"]),
                            "feather": int(spec["feather"])}
                    bx0, by0, bx1, by1 = box_px(r["box"], r["size"][0], r["size"][1])
                    crop["crop_mask"] = torch.nn.functional.interpolate(
                        r["masks"][:, None, by0:by1, bx0:bx1].float(), size=(chh, cw), mode="nearest")[:, 0]
            shots.append({
                "index": i, "count": len(segs), "start": s["start"], "end": s["end"],
                "length": s["end"] - s["start"], "gen_length": s["gen_len"], "fps": fps,
                "cut_before": s["cut_before"], "width": ft.shape[2], "height": ft.shape[1], "frames": ft,
                "crop": crop, "full_frames": full_frames,
                "ref": ref, "ref2": ref2,
                "prompt": fill_details((s["prompt"] or written.get(i) or g_prompt or "").replace(
                    "{shot}", sug["shot"] if (sug and vcfg["auto_shot"]) else ""),
                    s["ref"] or g_names["ref"], s["ref2"] or g_names["ref2"], ref, ref2),
                "audio": _slice_audio(audio, s["start"] / fps, s["gen_len"] / fps),
                "run_id": rid, "queue": queue,
                "chain": s["chain"], "chain_frame": s["chain_frame"], "chain_image": chain_img,
                "prev_length": prev_len,
                "source": {"path": path, "frames": [int(x) for x in src[idx]]},
            })
        if suggestions:
            _notify("bfs-shotloop-vlm", {"video": p["video"], "segs": suggestions})
        total = sum(s["end"] - s["start"] for s in segs)
        # no (usable) soundtrack: a silent track of the right length, so Create Video never gets None
        full_audio = _slice_audio(audio, segs[0]["start"] / fps, total / fps) if audio else silence(total / fps)
        full_total = sum(s["end"] - s["start"] for s in all_segs)
        timeline = {"path": path, "fps": fps, "width": W, "height": H, "fill": p.get("skip_fill", "original"),
                    "audio": _slice_audio(audio, all_segs[0]["start"] / fps, full_total / fps) if audio else None,
                    "segs": [{"start": s["start"], "end": s["end"], "run": s["run"], "cut_before": s["cut_before"],
                              "run_index": segs.index(s) if s["run"] else -1} for s in all_segs],
                    "src": src.tolist(), "n": a["n"]}
        skipped = [(i, s.get("skip_reason") or ("disabled" if not s["enabled"] else "skip")) for i, s in enumerate(all_segs) if not s["run"]]
        lines = [f"{len(segs)} shots, {total} frames ({total / fps:.2f}s) at {fps:g} fps, {W}x{H}"
                 + (f" | queue loop: running shot {shots[0]['index'] + 1}/{len(segs)}" if queue else "")]
        for i, why in skipped:
            lines.append(f"skip shot {i + 1} (frames {all_segs[i]['start']}-{all_segs[i]['end'] - 1}): {why}")
        for i, g in sorted(vlm_out.items()):
            lines.append(f"VLM #{i + 1}: segment '{g['segment']}', {g['recommend']}"
                         f"{' (' + g['reason'] + ')' if g['reason'] else ''} | {g['shot']}")
        for s in shots:
            lines.append(f"#{s['index'] + 1}: frames {s['start']}-{s['end'] - 1} ({s['length']} -> generate "
                         f"{s['gen_length']}){' cut' if s['cut_before'] else ''}"
                         f"{' ref' if s['ref'] is not None else ''}{' ref2' if s['ref2'] is not None else ''}")
        return (shots, len(segs), fps, W, H, full_audio, "\n".join(lines), timeline, used_refs, used_refs2)


class BFSShotUnpack:
    """Open one shot into plain values for any workflow (runs once per shot)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot": ("BFS_SHOT",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "IMAGE", "AUDIO", "INT", "INT", "INT", "BFS_SHOT",
                    "IMAGE", "MASK")
    RETURN_NAMES = ("guide_frames", "ref_image", "ref_image_2", "prompt", "length", "first_frame",
                    "audio", "width", "height", "index", "shot", "previous_result", "mask")
    OUTPUT_TOOLTIPS = (
        "The shot's guide frames, already at a length the model accepts.",
        "This shot's reference (a grey placeholder if none was set).",
        "This shot's second reference (a grey placeholder if none was set).",
        "This shot's prompt.", "Frames to generate (the grid-valid length).",
        "First guide frame of the shot.", "Soundtrack of the shot (silence if the video has none).",
        "Generation width.", "Generation height.", "Shot index (0-based).",
        "The same shot, unchanged: connect it to BFS Shot Repack after editing the pieces.",
        "The previous shot's result frame this shot continues from (Continuity in the planner; queue loop), "
        "or a grey image.",
        "The shot's SAM 3 mask over the guide frames (in the crop when the shot is cropped), or all ones.")
    FUNCTION = "unpack"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "Split one shot into its guide frames, references, prompt and length."

    def unpack(self, shot):
        audio = shot["audio"] or {"waveform": torch.zeros(1, 1, int(44100 * shot["gen_length"] / shot["fps"])),
                                  "sample_rate": 44100}
        return (shot["frames"], shot["ref"] if shot["ref"] is not None else _grey(),
                shot["ref2"] if shot["ref2"] is not None else _grey(), shot["prompt"], shot["gen_length"],
                shot["frames"][:1], audio, shot["width"], shot["height"], shot["index"], shot,
                chain_image(shot) if chain_image(shot) is not None else _grey(),
                shot["crop"]["crop_mask"] if shot.get("crop") else torch.ones(shot["frames"].shape[:3]))


class BFSShotRepack:
    """Put edited pieces back into a shot (runs once per shot), e.g. a reference with its background removed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"shot": ("BFS_SHOT",)},
            "optional": {
                "guide_frames": ("IMAGE", {"tooltip": "Replacement guide frames. Resized to the shot's generation "
                                                      "size; padded or trimmed to its length."}),
                "ref_image": ("IMAGE", {"tooltip": "Replacement reference (e.g. background removed)."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Replacement second reference."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Replacement prompt."}),
            },
        }

    RETURN_TYPES = ("BFS_SHOT",)
    RETURN_NAMES = ("shot",)
    FUNCTION = "repack"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Rebuild a shot after editing its pieces (Unpack -> any processing -> Repack). Inputs left "
                   "unconnected keep the shot's original values; timing, cuts and audio are unchanged.")

    def repack(self, shot, guide_frames=None, ref_image=None, ref_image_2=None, prompt=None):
        out = dict(shot)
        if guide_frames is not None:
            fr = guide_frames
            H, W = shot["height"], shot["width"]
            if tuple(fr.shape[1:3]) != (H, W):
                arr = (fr.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                fr = torch.from_numpy(np.stack([_fit(a, (W, H)) for a in arr]).astype(np.float32) / 255.0)
            n = shot["gen_length"]
            if fr.shape[0] != n:
                print(f"[BFS Shot Repack] shot {shot['index'] + 1}: guide has {fr.shape[0]} frames, "
                      f"expected {n}; {'trimming' if fr.shape[0] > n else 'holding the last frame'}")
                fr = fr[:n] if fr.shape[0] > n else torch.cat([fr, fr[-1:].expand(n - fr.shape[0], -1, -1, -1)], 0)
            out["frames"] = fr
        if ref_image is not None:
            out["ref"] = ref_image[:1]
        if ref_image_2 is not None:
            out["ref2"] = ref_image_2[:1]
        if prompt is not None:
            out["prompt"] = prompt
        return (out,)


SETTING_MODES = ["off", "on (generation size)", "on (source size, slower)"]


def tv_static(img: torch.Tensor, mask: torch.Tensor, grow: float = 0.03, seed: int = 0) -> torch.Tensor:
    """[1,H,W,3] with the masked region (dilated by `grow` of the short side) covered in black-and-white TV static,
    TSC's 'person noised out' setting picture: the place is seen in full detail, the person is not."""
    H, W = img.shape[1:3]
    m = torch.nn.functional.interpolate(mask.float().reshape(1, 1, *mask.shape[-2:]), size=(H, W), mode="nearest")[0, 0]
    g = max(1, int(min(H, W) * grow))
    m = torch.nn.functional.max_pool2d(m[None, None], 2 * g + 1, stride=1, padding=g)[0, 0] > 0.5
    gen = torch.Generator().manual_seed(seed)
    grain = 2
    n = (torch.rand((H + grain - 1) // grain, (W + grain - 1) // grain, generator=gen) > 0.5).float()
    n = n.repeat_interleave(grain, 0).repeat_interleave(grain, 1)[:H, :W]
    out = img.clone()
    out[0][m] = n[m][:, None].expand(-1, 3).to(out)
    return out


def setting_picture(shot: dict, mode: str, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The middle frame of the shot with the person covered in TV static. Mask: the given one, else the shot's SAM 3
    crop mask, else SAM 3 'person' on that frame."""
    src = shot.get("source") or {}
    k = len(shot["frames"]) // 2
    if mode == SETTING_MODES[2] and src.get("path"):
        f = _read_frames(src["path"], np.array([src["frames"][min(k, len(src["frames"]) - 1)]]), None)[0]
        img = torch.from_numpy(f.astype(np.float32) / 255.0)[None]
        short = min(img.shape[1:3])
        if short > 2048:   # what H3's 'max' reference size keeps anyway
            s = 2048 / short
            img = torch.nn.functional.interpolate(img.movedim(-1, 1), scale_factor=s, mode="bilinear",
                                                  align_corners=False).movedim(1, -1)
    else:
        full = shot.get("full_frames")
        img = (full if full is not None else shot["frames"])[k:k + 1]
    if mask is not None:
        m = mask[min(k, mask.shape[0] - 1)] if mask.ndim == 3 else mask
    elif shot.get("crop") is not None:
        cm = shot["crop"]["mask"]
        m = cm[min(k, cm.shape[0] - 1)]
    else:
        spec = dict(DEFAULT_MASK, text="person", max_objects=8)
        small = torch.nn.functional.interpolate(img.movedim(-1, 1), size=_fit_size(img.shape[1:3], 640),
                                                mode="bilinear", align_corners=False).movedim(1, -1)
        m = segment_frames(small, spec)[0]
        _node_boundary()
    return tv_static(img, m)


def _fit_size(hw, side: int) -> tuple[int, int]:
    s = side / max(hw)
    return max(32, int(hw[0] * s) // 2 * 2), max(32, int(hw[1] * s) // 2 * 2)


def setting_line(k: int, swap: bool) -> str:
    who = "<Subject 1>" if swap else "the performer"
    return (f"<Picture {k}> shows the setting, the same place as the kept footage in full detail; the noise patch in it "
            f"is where {who} stands.")


def add_setting(text: str, k: int, swap: bool) -> str:
    """`{setting}` becomes <Picture k>; without it, a sentence goes at the end of subject_definitions."""
    if "{setting}" in text:
        return text.replace("{setting}", f"<Picture {k}>")
    line = setting_line(k, swap)
    if "subject_definitions:" in text:
        a = text.index("subject_definitions:")
        b = text.find("\n\n", a)
        b = len(text) if b < 0 else b
        return text[:b].rstrip() + " " + line + text[b:]
    return (text.rstrip() + "\n\n" + line).strip()


class BFSShotH3Conditioning:
    """Native MiniMax H3 conditioning for one shot: references, prompt and the shot as a guide."""

    GUIDE_MODES = ["aligned guide (Add Guide)", "native reference video (Video 1)", "both", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot": ("BFS_SHOT",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide_mode": (cls.GUIDE_MODES, {"default": cls.GUIDE_MODES[0], "tooltip":
                    "aligned guide: the shot's frames sit on the generated frames (MiniMax H3 Add Guide), "
                    "frame by frame. native reference video: the shot enters as <Video 1> like the reference "
                    "video input. both: the two together."}),
                "use_ref_2": ("BOOLEAN", {"default": True, "tooltip": "Also pass the second reference."}),
                "first_frame": (["none", "shot's first frame"], {"default": "none", "tooltip":
                    "Anchor the shot's own first frame as an extra aligned image guide."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "with_audio": ("BOOLEAN", {"default": False, "tooltip":
                    "Attach the shot's soundtrack to the guide / reference video (needs audio_vae)."}),
                "duet": (["off", "canvas", "shifted RoPE"], {"default": "off", "tooltip":
                    "Pin the shot's own clip in a side panel and generate in sync with it (training-free duet). "
                    "canvas: panel and video share one wide grid. shifted RoPE: the video keeps its own RoPE "
                    "positions and the panel sits past its edge (connect the model and use the model output). "
                    "BFS Shot Join cuts the panel off by itself. In the prompt the panel has no tag: call it "
                    "'the kept footage' by its side ('the LEFT half')."}),
                "model": ("MODEL", {"tooltip": "Needed for 'shifted RoPE': route the model through this node."}),
                "panel_position": (["left", "right", "top", "bottom"], {"default": "left"}),
                "panel_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.01,
                                         "tooltip": "Panel size against the video (1.0 = two equal halves)."}),
                "panel_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "0 pins the panel exactly; 0.05-0.2 loosens it for bigger changes."}),
                "rope_gap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0,
                                       "tooltip": "shifted RoPE only: empty RoPE steps (2x2 patches) between video and panel. Keep it small "
                                                  "against the video width (0-2 at low resolution): a gap close to the "
                                                  "video's width makes the model draw its own split screen."}),
                "task": (["planner prompt"] + WRITER_TASKS, {"default": "planner prompt", "tooltip":
                    "Optional prompt writer for the duet. 'planner prompt' (default) uses the shot's prompt from the "
                    "planner as it is. A task writes the prompt of every shot in the duet format instead: with a VLM "
                    "connected it looks at the shot and its references and writes it; without one, a template for the "
                    "task. Needs duet on (canvas or shifted RoPE). See the written text on the prompt output."}),
                "instruction": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "What changes, in seen words, for the task: 'a 1990s anime cel style', 'a sunny beach at sunset', "
                    "'an elderly woman with short grey hair'. Empty is fine for character swap (the person comes from "
                    "the references)."}),
                "vlm": ("CLIP", {"tooltip": "Optional VLM (CLIPLoader with a Qwen3-VL text encoder) that writes the "
                                            "prompt when a task is chosen. Without it the task's template is used."}),
                "setting_ref": (SETTING_MODES, {"default": "off", "tooltip":
                    "TSC's trick: one more reference picture, the shot's middle frame with the person covered in TV "
                    "static, so the model sees the place in full detail (the panel / guide is often small). It is "
                    "the last <Picture n>; a sentence about it is added to subject_definitions (or write {setting} "
                    "where you want its tag). Mask: setting_mask, else the shot's SAM 3 crop mask, else SAM 3 "
                    "'person' on that frame. 'source size' uses the video's own resolution (up to 2048 short edge): "
                    "sharper, slower."}),
                "setting_mask": ("MASK", {"tooltip": "Optional mask of the person to cover in the setting picture."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "MODEL", "STRING")
    RETURN_NAMES = ("positive", "latent", "model", "prompt")
    FUNCTION = "condition"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Build MiniMax H3 conditioning for one shot with the native nodes: Reference to Video "
                   "(prompt, references, length) plus the shot as an aligned guide and/or reference video.")

    def condition(self, shot, clip, vae, guide_mode, use_ref_2, first_frame, ref_image_size,
                  audio_vae=None, with_audio=False, duet="off", model=None, panel_position="left", panel_size=1.0,
                  panel_noise=0.0, rope_gap=0.0, task="planner prompt", instruction="", vlm=None,
                  setting_ref="off", setting_mask=None):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
        try:
            from .bfs_h3_side_panel import build_prompt, layout_text, make_info
        except ImportError:
            from bfs_h3_side_panel import build_prompt, layout_text, make_info
        refs = {}
        if shot["ref"] is not None:
            refs["ref_image_0"] = shot["ref"]
        if use_ref_2 and shot["ref2"] is not None:
            refs[f"ref_image_{len(refs)}"] = shot["ref2"]
        prev = chain_image(shot)
        if prev is not None and shot.get("chain") == "reference":
            refs[f"ref_image_{len(refs)}"] = prev   # one more <Picture n>, after the shot's own
        audio = shot["audio"] if (with_audio and audio_vae is not None) else None
        native = guide_mode in (self.GUIDE_MODES[1], self.GUIDE_MODES[2])
        aligned = guide_mode in (self.GUIDE_MODES[0], self.GUIDE_MODES[2])
        on_canvas = duet != "off"
        rope_mode = "shifted" if duet == "shifted RoPE" else "canvas"
        text = shot["prompt"] or ""
        if task and task != "planner prompt":
            if not on_canvas:
                raise ValueError("the prompt writer (task) writes duet prompts: set duet to canvas or shifted RoPE, "
                                 "or set task to 'planner prompt'")
            if vlm is not None:
                fr = shot["frames"]
                idx = sorted(set(np.linspace(0, fr.shape[0] - 1, min(4, fr.shape[0])).round().astype(int).tolist()))
                text = write_duet_prompt(vlm, [fr[i:i + 1] for i in idx], list(refs.values()), task, instruction,
                                         1024, rope_mode)
                _node_boundary()
            else:
                text = build_prompt(task if task != "custom" else "appearance", instruction, len(refs), 0, rope_mode)
        setting = setting_picture(shot, setting_ref, setting_mask) if setting_ref and setting_ref != "off" else None
        if setting is not None:
            if ref_image_size == "match":   # keep 'match' for the people, the setting keeps its own size
                for k, r in list(refs.items()):
                    sc = min(1.0, math.sqrt(shot["width"] * shot["height"] / (r.shape[1] * r.shape[2])))
                    if sc < 1.0:
                        refs[k] = torch.nn.functional.interpolate(
                            r[:1].movedim(-1, 1), size=(max(16, int(r.shape[1] * sc)), max(16, int(r.shape[2] * sc))),
                            mode="bilinear", align_corners=False).movedim(1, -1)
                if setting_ref == SETTING_MODES[1]:
                    sc = min(1.0, math.sqrt(shot["width"] * shot["height"] / (setting.shape[1] * setting.shape[2])))
                    if sc < 1.0:
                        setting = torch.nn.functional.interpolate(
                            setting.movedim(-1, 1), size=(int(setting.shape[1] * sc), int(setting.shape[2] * sc)),
                            mode="bilinear", align_corners=False).movedim(1, -1)
                ref_image_size = "max"
            refs[f"ref_image_{len(refs)}"] = setting
            text = add_setting(text, len(refs), task == "character swap" or "<Subject 1>" in text)
        if "{layout}" in text:
            fill = layout_text(make_info(shot["width"], shot["height"], panel_position, panel_size, 0), rope_mode) \
                if on_canvas else ""
            text = text.replace("{layout}", fill).replace("  ", " ")
        kwargs = dict(clip=clip, prompt=text, width=shot["width"], height=shot["height"],
                      length=shot["gen_length"], ref_image_size=ref_image_size, vae=vae,
                      audio_vae=audio_vae, ref_images=refs or None)
        if native:
            kwargs["ref_videos"] = {"ref_video_1": shot["frames"]}
            if audio is not None:
                kwargs["ref_video_audios"] = {"ref_video_audio_1": audio}
        positive, latent = MiniMaxH3ReferenceToVideo.execute(**kwargs).args[:2]
        if aligned and (not on_canvas or audio is not None):
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 audio_vae=audio_vae if audio is not None else None,
                                                 image=shot["frames"], audio=audio).args[0]
        if first_frame != "none":
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 image=shot["frames"][:1]).args[0]
        prev = chain_image(shot)
        if prev is not None and shot.get("chain") == "first frame":
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 image=prev).args[0]
        shot.pop("panel", None)
        if on_canvas:
            try:
                from .bfs_h3_side_panel import BFSH3SidePanel, patch_model_rope
            except ImportError:
                from bfs_h3_side_panel import BFSH3SidePanel, patch_model_rope
            # the guide goes straight onto the canvas (guides added above are moved onto it by the panel step)
            guide = shot["frames"] if aligned and audio is None else None
            positive, latent, info, _, _ = BFSH3SidePanel().apply(
                positive, latent, vae, shot["frames"], panel_position, panel_size, "contain", 0, "all frames",
                panel_noise, guide, 0)
            shot["panel"] = info        # BFS Shot Join crops the decoded canvas back to the video
            if duet == "shifted RoPE":
                if model is None:
                    raise ValueError("duet 'shifted RoPE' needs the model input (and its model output in the sampler)")
                model = patch_model_rope(model, info, rope_gap)
        return (positive, latent, model, text)


class BFSShotJoin:
    """Put the decoded shots back together in order, trimmed to their true lengths."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded frames of every shot (the list from VAE Decode)."}),
                "shots": ("BFS_SHOT", {"tooltip": "The shots list from BFS Shot Planner."}),
                "crossfade": ("INT", {"default": 4, "min": 0, "max": 24, "tooltip":
                    "Frames to blend across boundaries that are not camera cuts, using the overlap each "
                    "shot generated past its end. 0 = hard joins everywhere."}),
            },
            "optional": {"audio": ("AUDIO", {"tooltip": "Soundtrack to return with the video "
                                                       "(e.g. the planner's audio output)."}),
                         "timeline": ("BFS_SHOT_TIMELINE", {"tooltip": "The planner's timeline. With it, shots that "
                             "did not run are filled with the original video (or dropped, per the planner's setting) "
                             "and the soundtrack follows."}),
                         "comparison": ("BOOLEAN", {"default": False, "tooltip": "Also output a side-by-side video: "
                             "original shot | references | result, with the shot's info on top and its prompt below."}),
                         "label": ("STRING", {"default": "", "multiline": False, "tooltip": "Extra text for the "
                             "comparison's top bar, e.g. the model, LoRA, steps and seed."}),
                         "comparison_height": ("INT", {"default": 360, "min": 0, "max": 4096, "step": 16, "tooltip":
                             "Height of each column in the comparison (0 = full size). Smaller is much lighter "
                             "for long videos."})},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "IMAGE")
    RETURN_NAMES = ("images", "audio", "fps", "comparison")
    FUNCTION = "join"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "Concatenate the generated shots in order, trim each to its length and cross-fade soft joins."

    def join(self, images, shots, crossfade, audio=None, timeline=None, comparison=None, label=None,
             comparison_height=None):
        tl = timeline[0] if timeline else None
        want = bool(comparison[0]) if comparison else False
        self._comp_h = int(comparison_height[0]) if comparison_height else 360
        self._want = want
        lab = (label[0] if label else "") or ""
        self._parts = []   # (shot or None, frames in the output, original frames) per piece, for the comparison
        images = [crop_panel(img, sh) for img, sh in zip(images, shots)]
        images = [uncrop(img, sh) if sh.get("crop") and sh.get("full_frames") is not None else img
                  for img, sh in zip(images, shots)]
        if shots and shots[0].get("queue"):
            out = self._join_queue(images, shots, crossfade, audio, tl)
        else:
            out = self._join_all(images, shots, crossfade, audio, tl)
        if len(out) == 3 and not isinstance(out[0], torch.Tensor):   # queue loop, not the last shot
            return out + (out[0],)
        video = out[0]
        comp = comparison_video(video, self._parts, lab, self._comp_h) if want else video[:1]
        return tuple(out) + (comp,)

    def _join_queue(self, images, shots, crossfade, audio, tl=None):
        from comfy_execution.graph_utils import ExecutionBlocker
        shot = shots[0]
        rid = shot["run_id"]
        d = run_dir(rid)
        img = images[0]
        meta = {k: v for k, v in shot.items() if k in ("index", "count", "start", "end", "length", "gen_length",
                                                         "fps", "cut_before", "prompt", "ref", "ref2")}
        if getattr(self, "_want", False):   # originals only for the comparison: small and uint8
            src = shot.get("full_frames") if shot.get("full_frames") is not None else shot.get("frames")
            if src is not None:
                h = getattr(self, "_comp_h", 360) or src.shape[1]
                sc = min(1.0, h / src.shape[1])
                small = _scale_batch(src, max(16, int(src.shape[1] * sc)), max(16, int(src.shape[2] * sc)))
                meta["frames"] = (small * 255).round().to(torch.uint8)
        torch.save({"frames": (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu(), "shot": meta},
                   os.path.join(d, f"shot_{shot['index']:04d}.pt"))
        st = run_state(rid)
        if shot["index"] not in st["done"]:
            st["done"].append(shot["index"])
        st["count"] = shot["count"]
        save_state(rid, st)
        done = len(set(st["done"]))
        _notify("bfs-shotloop-progress", {"run_id": rid, "done": done, "count": shot["count"]})
        if done < shot["count"]:
            _notify("bfs-shotloop-next", {"run_id": rid, "done": done, "count": shot["count"]})
            blocker = ExecutionBlocker(None)
            return (blocker, blocker, blocker)
        stored = [torch.load(os.path.join(d, f"shot_{i:04d}.pt")) for i in range(shot["count"])]
        imgs = [x["frames"].float() / 255.0 for x in stored]
        meta = [x["shot"] for x in stored]
        return self._join_all(imgs, meta, crossfade, audio, tl)

    def _join_all(self, images, shots, crossfade, audio=None, tl=None):
        if tl is not None and any(not s["run"] for s in tl["segs"]):
            return self._join_timeline(images, shots, crossfade, audio, tl)
        xf = int(crossfade[0]) if crossfade else 0
        if len(images) != len(shots):
            raise ValueError(f"BFS Shot Join: got {len(images)} image batches for {len(shots)} shots. "
                             "Connect the decoded images of the same shot list.")
        order = sorted(range(len(shots)), key=lambda i: shots[i]["index"])
        H, W = images[order[0]].shape[1:3]
        out = []
        prev_tail = None  # frames the previous shot generated past its end
        for j in order:
            img = images[j]
            if img.shape[1:3] != (H, W):
                img = torch.nn.functional.interpolate(img.movedim(-1, 1), size=(H, W), mode="bilinear",
                                                      align_corners=False).movedim(1, -1)
            L = shots[j]["length"]
            body = img[:L]
            if body.shape[0] < L:   # shorter than planned: hold the last frame
                body = torch.cat([body, body[-1:].expand(L - body.shape[0], -1, -1, -1)], 0)
            if xf > 0 and prev_tail is not None and not shots[j]["cut_before"]:
                n = min(xf, prev_tail.shape[0], body.shape[0])
                if n > 0:
                    w = torch.linspace(0, 1, n + 2)[1:-1].view(-1, 1, 1, 1)
                    body = body.clone()
                    body[:n] = prev_tail[:n] * (1 - w) + body[:n] * w
            out.append(body)
            if hasattr(self, "_parts"):
                self._parts.append((shots[j], L, shots[j].get("full_frames") if shots[j].get("full_frames") is not None
                                    else shots[j].get("frames")))
            prev_tail = img[L:]
        video = torch.cat(out, 0)
        fps = float(shots[order[0]]["fps"])
        a = audio[0] if audio else None
        if not _usable(a):   # no soundtrack: a silent one, so Create Video always gets valid audio
            return (video, silence(video.shape[0] / fps), fps)
        if a is not None:
            n = int(round(video.shape[0] / fps * a["sample_rate"]))
            wf = a["waveform"][..., :n]
            if wf.shape[-1] < n:
                wf = torch.nn.functional.pad(wf, (0, n - wf.shape[-1]))
            a = {"waveform": wf, "sample_rate": a["sample_rate"]}
        return (video, a, fps)


def _join_timeline_impl(self, images, shots, crossfade, audio, tl):
    """Rebuild the whole timeline: generated shots where they ran, original video (or nothing) elsewhere."""
    xf = int(crossfade[0]) if crossfade else 0
    by_idx = {shots[j]["index"]: images[j] for j in range(len(shots))}
    meta = {s["index"]: s for s in shots}
    first = images[0]
    H, W = first.shape[1:3]
    fps = float(tl["fps"])
    src = np.asarray(tl["src"])
    out, kept = [], []
    prev_tail = None
    for seg in tl["segs"]:
        L = seg["end"] - seg["start"]
        if seg["run"]:
            img = by_idx[seg["run_index"]]
            if img.shape[1:3] != (H, W):
                img = torch.nn.functional.interpolate(img.movedim(-1, 1), size=(H, W), mode="bilinear",
                                                      align_corners=False).movedim(1, -1)
            body = img[:L]
            if body.shape[0] < L:
                body = torch.cat([body, body[-1:].expand(L - body.shape[0], -1, -1, -1)], 0)
            if xf > 0 and prev_tail is not None and not seg["cut_before"]:
                n = min(xf, prev_tail.shape[0], body.shape[0])
                if n > 0:
                    w = torch.linspace(0, 1, n + 2)[1:-1].view(-1, 1, 1, 1)
                    body = body.clone(); body[:n] = prev_tail[:n] * (1 - w) + body[:n] * w
            out.append(body); kept.append((seg["start"], L))
            if hasattr(self, "_parts"):
                sm = meta.get(seg["run_index"]) or {}
                self._parts.append((sm, L, sm.get("full_frames") if sm.get("full_frames") is not None else sm.get("frames")))
            prev_tail = img[L:]
        else:
            prev_tail = None
            if tl.get("fill", "original") == "drop":
                continue
            frames = _read_frames(tl["path"], src[seg["start"]:seg["end"]], (W, H))
            orig = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
            out.append(orig); kept.append((seg["start"], L))
            if hasattr(self, "_parts"):
                self._parts.append(({"skipped": True, "start": seg["start"], "end": seg["end"], "fps": fps,
                                     "cut_before": seg["cut_before"]}, L, orig))
    video = torch.cat(out, 0)
    a = (audio[0] if audio else None) or tl.get("audio")
    if not _usable(a):
        return (video, silence(video.shape[0] / fps), fps)
    if a is not None:
        sr = a["sample_rate"]; base = tl["segs"][0]["start"]
        if tl.get("fill", "original") == "drop":
            parts = [a["waveform"][..., int(round((st - base) / fps * sr)):int(round((st - base + L) / fps * sr))] for st, L in kept]
            wf = torch.cat(parts, -1)
        else:
            wf = a["waveform"]
        n = int(round(video.shape[0] / fps * sr))
        wf = wf[..., :n]
        if wf.shape[-1] < n:
            wf = torch.nn.functional.pad(wf, (0, n - wf.shape[-1]))
        a = {"waveform": wf, "sample_rate": sr}
    return (video, a, fps)


BFSShotJoin._join_timeline = _join_timeline_impl


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_bar(lines: list[str], width: int, size: int, max_lines: int, fg=(235, 235, 235), bg=(18, 18, 22)):
    """A dark bar with word-wrapped lines, as a float tensor [h, width, 3]."""
    from PIL import Image, ImageDraw
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped = []
    for line in lines:
        words, cur = line.split(), ""
        for w in words:
            t = (cur + " " + w).strip()
            if probe.textlength(t, font=font) <= width - 2 * size and cur:
                cur = t
            elif not cur:
                cur = w
            else:
                wrapped.append(cur); cur = w
        wrapped.append(cur)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][: max(0, len(wrapped[-1]) - 3)] + "..."
    lh = int(size * 1.3)
    img = Image.new("RGB", (width, lh * len(wrapped) + size), bg)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(wrapped):
        d.text((size, size // 2 + i * lh), t, fill=fg, font=font)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)


def _column_labels(columns: list[tuple[str, int]], size: int) -> torch.Tensor:
    """One row with each label centred over its column."""
    from PIL import Image, ImageDraw
    font = _font(size)
    total = sum(w for _, w in columns)
    img = Image.new("RGB", (total, int(size * 1.6)), (30, 30, 36))
    d = ImageDraw.Draw(img)
    x = 0
    for name, w in columns:
        tw = d.textlength(name, font=font)
        d.text((x + (w - tw) / 2, size * 0.25), name, fill=(255, 210, 90), font=font)
        x += w
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)


def _fit_box(img: torch.Tensor | None, w: int, h: int) -> torch.Tensor:
    """[1,H,W,3] or None -> [h,w,3], letterboxed on dark grey."""
    out = torch.full((h, w, 3), 0.1)
    if img is None:
        return out
    x = img[:1, ..., :3].movedim(-1, 1).float()
    s = min(w / x.shape[-1], h / x.shape[-2])
    nw, nh = max(1, int(x.shape[-1] * s)), max(1, int(x.shape[-2] * s))
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)[0].movedim(0, -1)
    out[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = x.clamp(0, 1)
    return out


def _scale_batch(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[N,H,W,C] -> [N,h,w,3] float on the CPU (area filter when shrinking)."""
    x = x[..., :3].float().cpu()
    if tuple(x.shape[1:3]) == (h, w):
        return x
    mode = "area" if x.shape[1] >= h else "bilinear"
    y = torch.nn.functional.interpolate(x.movedim(-1, 1), size=(h, w), mode=mode,
                                        **({} if mode == "area" else {"align_corners": False}))
    return y.movedim(1, -1).clamp(0, 1)


def comparison_video(video: torch.Tensor, parts: list, label: str = "", height: int = 480) -> torch.Tensor:
    """original | references | result for every output frame, with shot info on top and the prompt below.
    Built at `height` px per column (0 = full size) so long videos stay light."""
    vH, vW = video.shape[1:3]
    s = min(1.0, height / vH) if height and height > 0 else 1.0
    H, W = max(16, int(vH * s) // 2 * 2), max(16, int(vW * s) // 2 * 2)
    rw = max(32, W // 2 // 2 * 2)
    total_w = W * 2 + rw
    size = max(12, total_w // 70)
    count = sum(1 for sh, _, _ in parts if sh and not sh.get("skipped"))
    cols = _column_labels([("original", W), ("references", rw), ("result", W)], size)
    pieces, pos, k = [], 0, 0
    for shot, L, orig in parts:
        shot = shot or {}
        fps = float(shot.get("fps") or 24.0)
        skipped = bool(shot.get("skipped"))
        if not skipped:
            k += 1
        t0, t1 = shot.get("start", 0) / fps, shot.get("end", 0) / fps
        head = [f"{'skipped (original video)' if skipped else f'shot {k}/{count}'}   "
                f"{int(t0 // 60)}:{t0 % 60:05.2f} -> {int(t1 // 60)}:{t1 % 60:05.2f}   {L} frames"
                f"{' -> ' + str(shot.get('gen_length')) + ' generated' if shot.get('gen_length') else ''}"
                f"{'   cut' if shot.get('cut_before') else ''}"]
        if label:
            head.append(label)
        top = torch.cat([_text_bar(head, total_w, size, 3), cols], 0)
        prompt = " ".join(str(shot.get("prompt") or "").split())
        bottom = _text_bar([("prompt: " + prompt) if prompt else ("" if skipped else "prompt: (none)")], total_w,
                           max(10, int(size * 0.8)), 6)
        refs = [r for r in (shot.get("ref"), shot.get("ref2")) if r is not None]
        if refs:
            rh = H // len(refs)
            col = torch.cat([_fit_box(r, rw, rh) for r in refs] + ([torch.full((H - rh * len(refs), rw, 3), 0.1)]
                                                                    if H - rh * len(refs) else []), 0)
        else:
            col = _fit_box(None, rw, H)
        res = _scale_batch(video[pos:pos + L], H, W)
        if orig is not None and orig.shape[0]:
            idx = torch.clamp(torch.arange(L), max=orig.shape[0] - 1)
            o = orig[idx]
            o = _scale_batch(o.float() / 255.0 if o.dtype == torch.uint8 else o, H, W)
        else:
            o = torch.full((L, H, W, 3), 0.1)
        body = torch.cat([o, col[None].expand(L, -1, -1, -1), res], 2)
        pieces.append((top, body, bottom))
        pos += L
    if not pieces:
        return video[:1]
    # same height for every frame, and both sides on a multiple of 16: video encoders (x264 / yuv420p) need even sizes
    hmax = max(t.shape[0] + b.shape[1] + bt.shape[0] for t, b, bt in pieces)
    H16, W16 = -(-hmax // 16) * 16, -(-total_w // 16) * 16
    out = torch.full((sum(b.shape[0] for _, b, _ in pieces), H16, W16, 3), 18 / 255)
    i = 0
    for top, body, bottom in pieces:
        n = body.shape[0]
        y = 0
        out[i:i + n, y:y + top.shape[0], :total_w] = top; y += top.shape[0]
        out[i:i + n, y:y + body.shape[1], :total_w] = body; y += body.shape[1]
        out[i:i + n, y:y + bottom.shape[0], :total_w] = bottom
        i += n
    return out


NODE_CLASS_MAPPINGS = {
    "BFSShotPlanner": BFSShotPlanner,
    "BFSShotUnpack": BFSShotUnpack,
    "BFSShotRepack": BFSShotRepack,
    "BFSShotH3Conditioning": BFSShotH3Conditioning,
    "BFSShotJoin": BFSShotJoin,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSShotPlanner": "BFS Shot Planner",
    "BFSShotUnpack": "BFS Shot Unpack",
    "BFSShotRepack": "BFS Shot Repack",
    "BFSShotH3Conditioning": "BFS Shot H3 Conditioning",
    "BFSShotJoin": "BFS Shot Join",
}

# ---------------------------------------------------------------------------- http api

try:
    from aiohttp import web
    from server import PromptServer

    def _list_input(exts):
        root = folder_paths.get_input_directory()
        out = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/")
                    out.append(rel)
        return sorted(out, key=str.lower)

    @PromptServer.instance.routes.get("/bfs/shotloop/files")
    async def _bfs_shot_files(request):
        # images are only listed on request: an input folder can hold thousands of them
        images = _list_input(IMAGE_EXTS) if request.query.get("images") in ("1", "true") else []
        return web.json_response({"videos": _list_input(VIDEO_EXTS), "images": images})

    @PromptServer.instance.routes.get("/bfs/shotloop/analyze")
    async def _bfs_shot_analyze(request):
        name = request.query.get("video", "")
        fps = float(request.query.get("fps", "24") or 24)
        try:
            a = await _off_loop(lambda: analyze(_input_path(name), fps))
        except Exception as exc:  # noqa: BLE001 - the panel shows the reason
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(a)

    @PromptServer.instance.routes.post("/bfs/shotloop/plan")
    async def _bfs_shot_plan(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))

            def work():
                a = analyze(_input_path(p["video"]), float(p["fps"]))
                path = _input_path(p["video"])
                if p.get("cast") or p.get("cast_split"):
                    try:
                        analyze_cast(path, a)
                    except Exception:  # noqa: BLE001 - plan without people
                        pass
                segs = resolve_plan(p, a, path)
                cuts, used = find_cuts(path, a, p.get("detector", "adaptive"), float(p["sensitivity"]))
                return a, segs, cuts, used
            a, segs, cuts, used = await _off_loop(work)
            W, H = generation_size(a["width"], a["height"], float(p["megapixels"]), int(p["multiple"]))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": segs, "cuts": cuts, "width": W, "height": H, "detector": used,
                                  "max_len": snap_down(int(round(float(p["max_s"]) * float(p["fps"]))), p["grid"])})
    @PromptServer.instance.routes.post("/bfs/shotloop/filters")
    async def _bfs_shot_filters(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            f = dict(DEFAULT_FILTERS); f.update(p.get("filters") or {})
            path = _input_path(p["video"])

            def work():
                a = analyze(path, float(p["fps"]))
                segs = resolve_plan(p, a, path)
                for i, s in enumerate(segs):   # stats always, so the panel can show them before any filter is on
                    _status("Detecting people and faces", i, len(segs))
                    s["stats"] = shot_stats(path, a, s, int(f.get("samples") or 6))
                    s["skip_reason"] = skip_reason(s["stats"], s["end"] - s["start"], f)
                return segs
            segs = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": [{"start": s["start"], "end": s["end"], "stats": s["stats"],
                                            "skip_reason": s["skip_reason"]} for s in segs]})

    @PromptServer.instance.routes.get("/bfs/shotloop/frame")
    async def _bfs_shot_frame(request):
        import cv2
        q = request.rel_url.query
        try:
            path = _input_path(q["video"])
            a = analyze(path, float(q.get("fps", 24)))
            f = max(0, min(a["n"] - 1, int(q.get("f", 0))))
            src = _timeline(a["n_src"], a["fps_src"], float(a["fps"]))
            w = int(q.get("w", 960))
            h = max(16, int(round(w * a["height"] / max(1, a["width"]))))
            fr = _read_frames(path, src[[f]], (w, h))[0]
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.Response(body=buf.tobytes(), content_type="image/jpeg")

    @PromptServer.instance.routes.post("/bfs/shotloop/mask")
    async def _bfs_shot_mask(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            a = analyze(path, float(p["fps"]))
            segs = resolve_plan(p, a, path)
            seg = segs[int(body.get("index", 0))]
            spec = mask_spec(body.get("mask") or seg.get("mask"), p.get("mask_cfg"))
            import asyncio   # SAM 3 takes seconds: keep the server responsive
            _outside_prompt()
            out = await asyncio.get_running_loop().run_in_executor(
                None, lambda: mask_preview(path, a, seg["start"], seg["end"] - seg["start"], spec, int(body.get("count", 6))))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(out)

    @PromptServer.instance.routes.post("/bfs/shotloop/vlm")
    async def _bfs_shot_vlm(request):
        body = await request.json()
        try:
            clip = _VLM.get("clip")
            if clip is None:
                raise ValueError("The panel has no VLM yet. Connect the VLM (a CLIPLoader with a Qwen3-VL text encoder, "
                                 "the same one Generate Text uses) to the planner's vlm input and run the workflow once "
                                 "(Queue): ComfyUI only hands models to nodes when they run. After that the panel's "
                                 "VLM buttons use it.")
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            a = analyze(path, float(p["fps"]))
            segs = resolve_plan(p, a, path)
            want = body.get("indices")
            pick = [segs[i] for i in want] if want else segs
            cfg = vlm_cfg(p.get("vlm_cfg"))
            import asyncio
            _outside_prompt()
            out = await asyncio.get_running_loop().run_in_executor(
                None, lambda: [dict(vlm_shot(clip, path, a, s["start"], s["end"], cfg), start=s["start"], end=s["end"])
                               for s in pick])
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": out})

    @PromptServer.instance.routes.post("/bfs/shotloop/describe")
    async def _bfs_shot_describe(request):
        """Descriptions for reference sets: body {plan, sets: [[ref, ref2], ...]} -> {texts: {key: text}}."""
        body = await request.json()
        try:
            clip = _VLM.get("clip")
            if clip is None:
                raise ValueError("The panel has no VLM yet. Connect the VLM (a CLIPLoader with a Qwen3-VL text encoder, "
                                 "the same one Generate Text uses) to the planner's vlm input and run the workflow once "
                                 "(Queue): ComfyUI only hands models to nodes when they run. After that the panel's "
                                 "VLM buttons use it.")
            p = _load_plan(json.dumps(body.get("plan", {})))
            cfg = vlm_cfg(p.get("vlm_cfg"))
            sets = [tuple(x) for x in body.get("sets", []) if any(x)]

            def work():
                out = {}
                for r1, r2 in sets:
                    out[ref_set_key(r1, r2)] = vlm_describe(clip, [_load_image(r1) if r1 else None, _load_image(r2) if r2 else None],
                                                            describe_instruction(cfg), int(cfg["max_tokens"]))
                return out
            texts = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"texts": texts})

    @PromptServer.instance.routes.post("/bfs/shotloop/write")
    async def _bfs_shot_write(request):
        """The VLM writes one shot's duet prompt: body {plan, index} -> {prompt}."""
        body = await request.json()
        try:
            clip = _VLM.get("clip")
            if clip is None:
                raise ValueError("The panel has no VLM yet. Connect the VLM to the planner's vlm input and run the "
                                 "workflow once (Queue): ComfyUI only hands models to nodes when they run.")
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            cfg = vlm_cfg(p.get("vlm_cfg"))

            def work():
                a = analyze(path, float(p["fps"]))
                seg = resolve_plan(p, a, path)[int(body.get("index", 0))]
                r1 = _load_image(seg["ref"] or p.get("global_ref", "")) if (seg["ref"] or p.get("global_ref")) else None
                r2 = _load_image(seg["ref2"] or p.get("global_ref2", "")) if (seg["ref2"] or p.get("global_ref2")) else None
                return vlm_write_prompt(clip, path, a, seg["start"], seg["end"], [r1, r2], cfg["write_task"],
                                        cfg["write_change"], int(cfg["max_tokens"]))
            text = await _off_loop(work)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"prompt": text})

    @PromptServer.instance.routes.post("/bfs/shotloop/cast")
    async def _bfs_shot_cast(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            path = _input_path(p["video"])
            c = await _off_loop(lambda: analyze_cast(path, analyze(path, float(p["fps"]))))
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"people": c["people"], "step": c["step"], "samples": len(c["samples"])})

    @PromptServer.instance.routes.post("/bfs/shotloop/progress")
    async def _bfs_shot_progress(request):
        body = await request.json()
        try:
            plan_json = json.dumps(body.get("plan", {}))
            p = _load_plan(plan_json)
            rid = run_id_for(body.get("plan_raw") or plan_json, _input_path(p["video"]))
            st = run_state(rid)
            if body.get("reset"):
                import shutil
                shutil.rmtree(run_dir(rid), ignore_errors=True)
                st = {"done": [], "count": st.get("count", 0)}
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"run_id": rid, "done": len(set(st["done"])), "count": st.get("count", 0)})
except Exception as _exc:  # noqa: BLE001 - nodes still work without the panel
    print(f"[BFSNodes] Shot loop HTTP routes not registered: {_exc!r}")
