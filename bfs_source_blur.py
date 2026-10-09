"""Optional blur of the source person in what the model SEES of the source (aligned guide, native reference video,
duet panel), keeping the mouth sharp: the model still reads the pose, the motion and the lips, but has no face to copy,
so the identity comes from the reference. JalenBrunson's trick for reliable ref2v swaps (Sapiens2 there); here the
masks come from SAM 3.1, already used by the planner. The kept area (inpaint) and the output are never blurred.
"""
from __future__ import annotations

import torch

SOURCE_BLUR = ["off", "face (keep mouth)", "head + hands (keep mouth)", "person (keep mouth)"]
_PARTS = {SOURCE_BLUR[1]: ("face",), SOURCE_BLUR[2]: ("head", "hand"), SOURCE_BLUR[3]: ("person",)}
_CACHE: dict = {}


def _sl():
    try:
        from . import bfs_shot_loop as SL
    except ImportError:
        import bfs_shot_loop as SL
    return SL


def _small(frames: torch.Tensor, side: int = 640) -> torch.Tensor:
    H, W = frames.shape[1:3]
    s = side / max(H, W)
    x = frames[..., :3].float()
    if s >= 1:
        return x
    return torch.nn.functional.interpolate(x.movedim(-1, 1), size=(max(32, int(H * s) // 2 * 2), max(32, int(W * s) // 2 * 2)),
                                           mode="bilinear", align_corners=False).movedim(1, -1)


def _grow(m: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return m
    return torch.nn.functional.max_pool2d(m[:, None], 2 * px + 1, stride=1, padding=px)[:, 0]


def blur_mask(frames: torch.Tensor, mode: str, log: str = "") -> torch.Tensor | None:
    """[N,h,w] float mask (working size) of what to blur: the parts for `mode` minus the (slightly grown) mouth."""
    parts = _PARTS.get(mode)
    if not parts:
        return None
    SL = _sl()
    small = _small(frames)
    key = (mode, tuple(small.shape), SL._image_key(frames[:1]), SL._image_key(frames[-1:]),
           SL._image_key(frames[frames.shape[0] // 2:frames.shape[0] // 2 + 1]))
    if key in _CACHE:
        return _CACHE[key]
    m = torch.zeros(small.shape[:3])
    for p in parts:
        m = torch.maximum(m, (SL.segment_frames(small, dict(SL.DEFAULT_MASK, text=p, max_objects=8, threshold=0.4)) > 0.5).float())
    mouth = (SL.segment_frames(small, dict(SL.DEFAULT_MASK, text="mouth", max_objects=8, threshold=0.3)) > 0.5).float()
    keep = _grow(mouth, max(1, int(min(small.shape[1:3]) * 0.012)))
    out = (_grow(m, max(1, int(min(small.shape[1:3]) * 0.015))) * (1 - keep)).clamp(0, 1)
    if log:
        print(f"{log}: source blur '{mode}': {float(m.mean()) * 100:.1f}% of the frame, mouth kept "
              f"({'found' if float(mouth.sum()) > 0 else 'not found: the whole part is blurred'})", flush=True)
    if len(_CACHE) > 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def blur_source(frames: torch.Tensor | None, mode: str, strength: float = 1.0, log: str = "") -> torch.Tensor | None:
    """The frames with the source person blurred per `mode` (mouth kept). `strength` 0-1 sets how coarse the blur is."""
    if frames is None or not mode or mode == "off" or strength <= 0:
        return frames
    m = blur_mask(frames, mode, log)
    if m is None or float(m.sum()) == 0:
        return frames
    x = frames[..., :3].float()
    N, H, W = x.shape[:3]
    # a heavy blur: down to ~1/12-1/40 of the size and back, then a soft pass; blocky detail never survives it
    f = max(4, int(min(H, W) * (0.025 + 0.06 * float(strength))))
    low = torch.nn.functional.interpolate(x.movedim(-1, 1), size=(max(2, H // f), max(2, W // f)), mode="area")
    blur = torch.nn.functional.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
    blur = torch.nn.functional.avg_pool2d(blur, 5, stride=1, padding=2).movedim(1, -1)
    mm = torch.nn.functional.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)
    mm = torch.nn.functional.avg_pool2d(mm, 9, stride=1, padding=4)[:, 0, ..., None].clamp(0, 1)
    if mm.shape[0] != N:   # one mask per frame; a shorter list holds its last one
        idx = [min(i, mm.shape[0] - 1) for i in range(N)]
        mm = mm[idx]
    out = x * (1 - mm) + blur * mm
    del low, blur, mm
    return out
