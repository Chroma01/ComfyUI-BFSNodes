"""Sliding history for MiniMax H3: the end of the previous shot's result as clean frames placed just BEFORE the video
on H3's timeline, so a long continuous take is generated window by window with motion (and look) carried across.

Idea from Akatz's H3 Relay windowed edit and Ethanfel's ComfyUI-MiniMaxH3-Context-Loop (both GPL-3.0); this is an
independent implementation. A guide keyframe marked ``anchor = "history"`` is packed like any H3 guide, then its rows
are kept at the target's origin and everything else on the target timeline (the video, its audio, the other guides)
moves forward by the history's span. Text and reference rows are untouched.

The layout is patched in memory only (ComfyUI files are never edited), once, and only changes layouts that carry a
history keyframe; when ComfyUI's own H3 layout already understands history anchors the patch stands down.
"""
from __future__ import annotations

import logging

import torch

LOG = logging.getLogger("bfs.h3_history")
HISTORY_FRAMES = (17, 34, 51)   # whole H3 VAE groups: 1+4+4+4+4 frames per 5 latent steps
_PATCH = "_bfs_history_patch"


def history_span(latent_t: int) -> float:
    """Timeline span of `latent_t` video latent steps (FRAME_RESCALE per pixel frame)."""
    import comfy.ldm.minimax.model as mm
    return float(sum(mm._video_t_spans(int(latent_t))))


def relocate(layout, keyframes) -> bool:
    """Moves the target timeline past the history keyframes. Returns whether anything changed."""
    hist = [i for i, kf in enumerate(keyframes or ()) if kf.get("anchor") == "history"]
    if not hist:
        return False
    # keyframe -> its packed segments: every guide with a video latent owns the next "cond" segment, every guide
    # with audio the next "cond_audio" one, in keyframe order (the order PackedLayout packs them)
    cond = [s for s in layout.segments if s[2] == "cond"]
    cond_a = [s for s in layout.segments if s[2] == "cond_audio"]
    own, ci, ai = {}, 0, 0
    for i, kf in enumerate(keyframes):
        segs = []
        if kf.get("latent") is not None and ci < len(cond):
            segs.append(cond[ci]); ci += 1
        if kf.get("audio_latent") is not None and ai < len(cond_a):
            segs.append(cond_a[ai]); ai += 1
        own[i] = segs
    span = sum(history_span(keyframes[i]["latent"].shape[2]) for i in hist if keyframes[i].get("latent") is not None)
    if span <= 0:
        return False
    keep = torch.zeros(layout.position_ids.shape[0], dtype=torch.bool)
    for i in hist:
        for a, b, _ in own[i]:
            keep[a:b] = True
    vid = next(((a, b) for a, b, k in layout.segments if k == "video"), None)
    if vid is not None and keep.any():
        # another history implementation (e.g. H3 Relay's, or a future ComfyUI) already moved the target: skip
        if float(layout.position_ids[vid[0]:vid[1], 0].min()) - float(layout.position_ids[keep, 0].min()) >= span - 1e-6:
            return False
    move = torch.zeros_like(keep)
    for a, b, kind in layout.segments:
        if kind in ("video", "audio", "cond", "cond_audio"):
            move[a:b] = True
    move &= ~keep
    pos = layout.position_ids.clone()
    pos[move, 0] += span
    layout.position_ids = pos
    LOG.info("H3 history: %d rows before the video, target timeline moved by %.2f", int(keep.sum()), span)
    return True


def install() -> bool:
    """Wraps PackedLayout.__init__ once (process-local). False when ComfyUI is missing or already supports it."""
    try:
        import comfy.ldm.minimax.model as mm
    except ImportError:
        return False
    cls = mm.PackedLayout
    if getattr(cls.__init__, _PATCH, False):
        return True
    import inspect
    if "history" in inspect.getsource(cls):      # native history anchors: nothing to do
        return False
    orig = cls.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        orig(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=keyframes, refs=refs)
        relocate(self, keyframes)

    setattr(__init__, _PATCH, True)
    cls.__init__ = __init__
    return True


def add_history(positive, vae, frames: torch.Tensor):
    """Appends `frames` ([N,H,W,3] at the video's size, N = 17k: the end of the previous result) as a history keyframe.
    Returns (positive, frames used)."""
    import node_helpers
    n = (int(frames.shape[0]) // 17) * 17
    if n < 17:
        return positive, 0
    install()
    lat = vae.encode(frames[-n:, ..., :3].float())
    kfs = list(positive[0][1].get("minimax_keyframes", []))
    kfs.append({"resolved_frame_index": 0, "latent": lat, "anchor": "history"})
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": kfs}), n
