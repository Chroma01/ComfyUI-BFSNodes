"""Optional blur of the source person in what the model SEES of the source (aligned guide, native reference video,
duet panel): the model still reads the pose, the motion and (if kept sharp) the lips, but has no face to copy, so the
identity comes from the reference. JalenBrunson's trick for reliable ref2v swaps. The kept area (inpaint) and the
output are never blurred.

Modes are "<region> (<what stays sharp>)":
  region     face / head + hands / person: found by the segmenter, every person in the frame;
             masked area: the shot's own mask (the person being replaced, with its expand), no extra segmentation
  sharp      keep mouth: lips, teeth and tongue; keep lips + tongue: the teeth are blurred too (always Sapiens2);
             mouth too: nothing stays sharp
Segmenter: SAM 3.1 (text prompts, already used by the planner) or Sapiens2 (body-part classes, one pass, finer lips).
Effect: blur, or invert the colours (keeps the shapes and the expression, loses skin tone and colours).
"""
from __future__ import annotations

import torch

_REGIONS = ("face", "head + hands", "person", "masked area")
_SHARP = ("keep mouth", "keep lips + tongue", "mouth too")
SOURCE_BLUR = ["off"] + [f"{r} ({k})" for k in _SHARP for r in _REGIONS]
SEGMENTERS = ["SAM 3.1", "Sapiens2"]
EFFECTS = ["blur", "invert colours"]
_SAM_PARTS = {"face": ("face",), "head + hands": ("head", "hand"), "person": ("person",)}
_SAP_FACE = ("Face_Neck", "Eyeglass", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")
_SAP_PARTS = {"face": _SAP_FACE, "head + hands": _SAP_FACE + ("Hair", "Left_Hand", "Right_Hand")}
_SAP_SHARP = {"keep mouth": ("Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"),
              "keep lips + tongue": ("Lower_Lip", "Upper_Lip", "Tongue")}
_CACHE: dict = {}


def parse(mode: str) -> tuple[str, str] | None:
    """'face (keep mouth)' -> ('face', 'keep mouth'); None for off / unknown."""
    if not mode or "(" not in mode:
        return None
    region, sharp = mode.rstrip(")").split(" (", 1)
    return (region, sharp) if region in _REGIONS and sharp in _SHARP else None


def _sl():
    try:
        from . import bfs_shot_loop as SL
    except ImportError:
        import bfs_shot_loop as SL
    return SL


def _sap():
    try:
        from . import bfs_sapiens as SP
    except ImportError:
        import bfs_sapiens as SP
    return SP

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


def blur_mask(frames: torch.Tensor, mode: str, log: str = "", segmenter: str = "SAM 3.1",
              area: torch.Tensor | None = None) -> torch.Tensor | None:
    """[N,h,w] float mask (working size) of what to blur: the region for `mode` minus what stays sharp (grown a
    little). `area` [N,H,W]: the shot's own mask, for the 'masked area' region."""
    pr = parse(mode)
    if not pr:
        return None
    region, sharp = pr
    if region == "masked area" and area is None:
        if log:
            print(f"{log}: source blur 'masked area' needs a mask on the shot (SAM 3 text or points): blurring the "
                  "whole person instead", flush=True)
        region = "person"
    SL = _sl()
    small = _small(frames)
    h, w = small.shape[1:3]
    sap_region = region != "masked area" and segmenter == "Sapiens2"
    sap_sharp = sharp == "keep lips + tongue" or (sharp == "keep mouth" and segmenter == "Sapiens2")
    key = (region, sharp, segmenter, tuple(small.shape), SL._image_key(frames[:1]), SL._image_key(frames[-1:]),
           SL._image_key(frames[frames.shape[0] // 2:frames.shape[0] // 2 + 1]),
           None if area is None else (tuple(area.shape), float(area.float().sum())))
    if key in _CACHE:
        return _CACHE[key]
    cls = _sap().segment(small) if sap_region or sap_sharp else None
    if region == "masked area":
        m = torch.nn.functional.interpolate(area[:, None].float(), size=(h, w), mode="nearest")[:, 0].clamp(0, 1)
        m_grow = 0                                   # already grown by the mask's own expand
    elif sap_region:
        m = (cls > 0).float() if region == "person" else _sap().parts(cls, _SAP_PARTS[region])
        m_grow = max(1, int(min(h, w) * 0.01))
    else:
        m = torch.zeros(small.shape[:3])
        for p in _SAM_PARTS[region]:
            m = torch.maximum(m, (SL.segment_frames(small, dict(SL.DEFAULT_MASK, text=p, max_objects=8, threshold=0.4)) > 0.5).float())
        m_grow = max(1, int(min(h, w) * 0.015))
    if sharp == "mouth too":
        keep, found = torch.zeros_like(m), None
    elif sap_sharp:
        keep = _sap().parts(cls, _SAP_SHARP[sharp])
        keep = _grow(keep, max(1, int(min(h, w) * 0.005)))
        found = float(keep.sum()) > 0
    else:
        mouth = (SL.segment_frames(small, dict(SL.DEFAULT_MASK, text="mouth", max_objects=8, threshold=0.3)) > 0.5).float()
        keep = _grow(mouth, max(1, int(min(h, w) * 0.012)))
        found = float(mouth.sum()) > 0
    out = (_grow(m, m_grow) * (1 - keep)).clamp(0, 1)
    if log:
        kept = "nothing kept sharp" if found is None else \
            f"{sharp[5:]} kept ({'found' if found else 'not found: the whole region is blurred'})"
        used = "shot mask" if region == "masked area" else "Sapiens2" if sap_region else "SAM 3.1"
        print(f"{log}: source blur '{region}' ({used}{' + Sapiens2 lips' if sap_sharp and not sap_region else ''}): "
              f"{float(m.mean()) * 100:.1f}% of the frame, {kept}", flush=True)
    if len(_CACHE) > 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def _soft(m: torch.Tensor, H: int, W: int, N: int) -> torch.Tensor:
    """The working-size mask at the frame size with a soft edge, one per frame ([N,H,W,1])."""
    mm = torch.nn.functional.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)
    mm = torch.nn.functional.avg_pool2d(mm, 9, stride=1, padding=4)[:, 0, ..., None].clamp(0, 1)
    if mm.shape[0] != N:   # one mask per frame; a shorter list holds its last one
        mm = mm[[min(i, mm.shape[0] - 1) for i in range(N)]]
    return mm


def blur_source(frames: torch.Tensor | None, mode: str, strength: float = 1.0, log: str = "",
                segmenter: str = "SAM 3.1", effect: str = "blur", area: torch.Tensor | None = None) -> torch.Tensor | None:
    """The frames with the source person blurred (or colour-inverted) per `mode`. `strength` 0-1 sets how coarse the
    blur is; `area`: the shot's mask for the 'masked area' modes."""
    if frames is None or not mode or mode == "off" or strength <= 0:
        return frames
    m = blur_mask(frames, mode, log, segmenter, area)
    if m is None or float(m.sum()) == 0:
        return frames
    x = frames[..., :3].float()
    N, H, W = x.shape[:3]
    if effect == "invert colours":
        mm = _soft(m, H, W, N)
        return x * (1 - mm) + (1 - x) * mm
    # a heavy blur: down to ~1/12-1/40 of the size and back, then a soft pass; blocky detail never survives it
    f = max(4, int(min(H, W) * (0.025 + 0.06 * float(strength))))
    low = torch.nn.functional.interpolate(x.movedim(-1, 1), size=(max(2, H // f), max(2, W // f)), mode="area")
    blur = torch.nn.functional.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
    blur = torch.nn.functional.avg_pool2d(blur, 5, stride=1, padding=2).movedim(1, -1)
    mm = _soft(m, H, W, N)
    out = x * (1 - mm) + blur * mm
    del low, blur, mm
    return out
