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


def _read_frames(path: str, src_indices: np.ndarray, size: tuple[int, int] | None) -> list[np.ndarray]:
    """Decode the requested source frames (sorted, may repeat) as RGB uint8, sequentially."""
    import cv2
    want = sorted(set(int(i) for i in src_indices))
    got: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    i, k = 0, 0
    last = None
    while k < len(want):
        ok = cap.grab()
        if not ok:
            break
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


def _read_audio(path: str, sample_rate: int = 44100) -> dict | None:
    """Decode the whole soundtrack as float32 [1, C, T] with ffmpeg; None when silent."""
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
    small = []
    for fr in _read_frames(path, src, None):
        g = cv2.resize(fr, (64, 36), interpolation=cv2.INTER_AREA)
        small.append(g)
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
    frames_for_thumbs = _read_frames(path, src[::k], (tw, thumb_h))
    for j, fr in enumerate(frames_for_thumbs):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        th.append({"f": int(j * k), "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})
    out = dict(info, fps=float(fps), n=n, thumbs=th, thumb_w=tw, thumb_h=thumb_h,
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
    segs = plan_segments(analysis["n"], cuts, mode, max_len, min_len,
                         int(plan.get("max_parts") or 0), max_total, manual)
    meta = plan.get("segs") or []
    for i, s in enumerate(segs):
        m = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
        s["enabled"] = bool(m.get("enabled", True))
        s["ref"] = m.get("ref") or ""
        s["ref2"] = m.get("ref2") or ""
        s["prompt"] = m.get("prompt") or ""
        s["cut_before"] = bool(s.get("cut_before")) or (s["start"] in cuts)
        s["gen_len"] = snap_up(s["end"] - s["start"], grid)
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
        s["run"] = s["enabled"] and (force == "run" or (force != "skip" and not s["skip_reason"]))
    return segs


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

    def plan_shots(self, plan, ref_image=None, ref_image_2=None, prompt=None):
        p = _load_plan(plan)
        path = _input_path(p["video"])
        fps = float(p["fps"])
        a = analyze(path, fps)
        all_segs = apply_filters(p, a, path, resolve_plan(p, a, path))
        segs = [s for s in all_segs if s["run"]]
        if not segs:
            raise ValueError("BFS Shot Planner: no shot is left to run (all disabled or filtered out).")
        W, H = generation_size(a["width"], a["height"], float(p["megapixels"]), int(p["multiple"]))
        src = _timeline(a["n_src"], a["fps_src"], fps)
        audio = _read_audio(path)
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
            shots.append({
                "index": i, "count": len(segs), "start": s["start"], "end": s["end"],
                "length": s["end"] - s["start"], "gen_length": s["gen_len"], "fps": fps,
                "cut_before": s["cut_before"], "width": W, "height": H, "frames": ft,
                "ref": ref, "ref2": ref2, "prompt": s["prompt"] or g_prompt or "",
                "audio": _slice_audio(audio, s["start"] / fps, s["gen_len"] / fps),
                "run_id": rid, "queue": queue,
            })
        total = sum(s["end"] - s["start"] for s in segs)
        full_audio = _slice_audio(audio, segs[0]["start"] / fps, total / fps) if audio else None
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "IMAGE", "AUDIO", "INT", "INT", "INT")
    RETURN_NAMES = ("guide_frames", "ref_image", "ref_image_2", "prompt", "length", "first_frame",
                    "audio", "width", "height", "index")
    OUTPUT_TOOLTIPS = (
        "The shot's guide frames, already at a length the model accepts.",
        "This shot's reference (a grey placeholder if none was set).",
        "This shot's second reference (a grey placeholder if none was set).",
        "This shot's prompt.", "Frames to generate (the grid-valid length).",
        "First guide frame of the shot.", "Soundtrack of the shot (silence if the video has none).",
        "Generation width.", "Generation height.", "Shot index (0-based).")
    FUNCTION = "unpack"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "Split one shot into its guide frames, references, prompt and length."

    def unpack(self, shot):
        audio = shot["audio"] or {"waveform": torch.zeros(1, 1, int(44100 * shot["gen_length"] / shot["fps"])),
                                  "sample_rate": 44100}
        return (shot["frames"], shot["ref"] if shot["ref"] is not None else _grey(),
                shot["ref2"] if shot["ref2"] is not None else _grey(), shot["prompt"], shot["gen_length"],
                shot["frames"][:1], audio, shot["width"], shot["height"], shot["index"])


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


class BFSShotH3Conditioning:
    """Native MiniMax H3 conditioning for one shot: references, prompt and the shot as a guide."""

    GUIDE_MODES = ["aligned guide (Add Guide)", "native reference video (<Video 1>)", "both", "none"]

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
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "condition"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Build MiniMax H3 conditioning for one shot with the native nodes: Reference to Video "
                   "(prompt, references, length) plus the shot as an aligned guide and/or reference video.")

    def condition(self, shot, clip, vae, guide_mode, use_ref_2, first_frame, ref_image_size,
                  audio_vae=None, with_audio=False):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
        refs = {}
        if shot["ref"] is not None:
            refs["ref_image_0"] = shot["ref"]
        if use_ref_2 and shot["ref2"] is not None:
            refs[f"ref_image_{len(refs)}"] = shot["ref2"]
        audio = shot["audio"] if (with_audio and audio_vae is not None) else None
        native = guide_mode in (self.GUIDE_MODES[1], self.GUIDE_MODES[2])
        aligned = guide_mode in (self.GUIDE_MODES[0], self.GUIDE_MODES[2])
        kwargs = dict(clip=clip, prompt=shot["prompt"], width=shot["width"], height=shot["height"],
                      length=shot["gen_length"], ref_image_size=ref_image_size, vae=vae,
                      audio_vae=audio_vae, ref_images=refs or None)
        if native:
            kwargs["ref_videos"] = {"ref_video_1": shot["frames"]}
            if audio is not None:
                kwargs["ref_video_audios"] = {"ref_video_audio_1": audio}
        positive, latent = MiniMaxH3ReferenceToVideo.execute(**kwargs).args[:2]
        if aligned:
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 audio_vae=audio_vae if audio is not None else None,
                                                 image=shot["frames"], audio=audio).args[0]
        if first_frame != "none":
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                                 image=shot["frames"][:1]).args[0]
        return (positive, latent)


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
                             "and the soundtrack follows."})},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "join"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "Concatenate the generated shots in order, trim each to its length and cross-fade soft joins."

    def join(self, images, shots, crossfade, audio=None, timeline=None):
        tl = timeline[0] if timeline else None
        if shots and shots[0].get("queue"):
            return self._join_queue(images, shots, crossfade, audio, tl)
        return self._join_all(images, shots, crossfade, audio, tl)

    def _join_queue(self, images, shots, crossfade, audio, tl=None):
        from comfy_execution.graph_utils import ExecutionBlocker
        shot = shots[0]
        rid = shot["run_id"]
        d = run_dir(rid)
        img = images[0]
        torch.save({"frames": (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu(),
                    "shot": {k: v for k, v in shot.items() if k in ("index", "count", "start", "end", "length",
                                                                   "gen_length", "fps", "cut_before")}},
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
            prev_tail = img[L:]
        video = torch.cat(out, 0)
        fps = float(shots[order[0]]["fps"])
        a = audio[0] if audio else None
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
            prev_tail = img[L:]
        else:
            prev_tail = None
            if tl.get("fill", "original") == "drop":
                continue
            frames = _read_frames(tl["path"], src[seg["start"]:seg["end"]], (W, H))
            out.append(torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)); kept.append((seg["start"], L))
    video = torch.cat(out, 0)
    a = (audio[0] if audio else None) or tl.get("audio")
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
            a = analyze(_input_path(name), fps)
        except Exception as exc:  # noqa: BLE001 - the panel shows the reason
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(a)

    @PromptServer.instance.routes.post("/bfs/shotloop/plan")
    async def _bfs_shot_plan(request):
        body = await request.json()
        try:
            p = _load_plan(json.dumps(body.get("plan", {})))
            a = analyze(_input_path(p["video"]), float(p["fps"]))
            path = _input_path(p["video"])
            segs = resolve_plan(p, a, path)
            cuts, used = find_cuts(path, a, p.get("detector", "adaptive"), float(p["sensitivity"]))
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
            a = analyze(path, float(p["fps"]))
            segs = resolve_plan(p, a, path)
            for s in segs:   # stats always, so the panel can show them before any filter is on
                s["stats"] = shot_stats(path, a, s, int(f.get("samples") or 6))
                s["skip_reason"] = skip_reason(s["stats"], s["end"] - s["start"], f)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"segs": [{"start": s["start"], "end": s["end"], "stats": s["stats"],
                                            "skip_reason": s["skip_reason"]} for s in segs]})

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
