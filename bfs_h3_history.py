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
# H3's VAE encodes 17-frame clips into 5 tokens (1+4+4+4+4 frames) and drops the last 3 tokens of the whole video, so only
# 17k+5 frames encode without loss: 17 frames come back as 2 tokens = the FIRST 5 frames, leaving a 12-frame hole between
# the history and the shot it sits before
HISTORY_FRAMES = (5, 22, 39, 56)


def history_length(wanted: int, available: int) -> int:
    """The longest valid history (17k+5 frames) up to `wanted` and `available`; 0 when not even 5 fit. A wanted value
    off the grid (old plans saved 17/34/51) is read as the next valid one up (22/39/56)."""
    wanted, available = max(0, int(wanted)), max(0, int(available))
    if wanted < 5:
        return 0
    wanted = 5 + -(-(wanted - 5) // 17) * 17
    n = min(wanted, available)
    return 0 if n < 5 else 5 + ((n - 5) // 17) * 17
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
    """Appends the last 17k+5 frames of `frames` ([N,H,W,3]) from the previous shot as an H3 history keyframe.
    Returns (positive, frames used)."""
    import node_helpers
    n = history_length(frames.shape[0], frames.shape[0])
    if n < 5:
        return positive, 0
    install()
    # encoded from CPU frames; the latent comes back on ComfyUI's intermediate device like any H3 guide, and the VAE
    # is loaded / freed by ComfyUI's model management (nothing here keeps GPU memory)
    with torch.no_grad():
        lat = vae.encode(frames[-n:, ..., :3].float().cpu())
    kfs = list(positive[0][1].get("minimax_keyframes", []))
    kfs.append({"resolved_frame_index": 0, "latent": lat, "anchor": "history"})
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": kfs}), n


def pin_head(latent: dict, vae, frames: torch.Tensor):
    """Continuity 'head': the end of the previous shot's result (17k+5 frames, [N,H,W,3]) encoded over the FIRST latent
    steps of this shot's video, with the sampler's noise mask at 0 there, so those frames are kept exactly as they were
    (no new swap on them) and the rest of the shot is generated continuing from them. BFS Shot Join cuts them off again.
    Same masked-prefix idea as Ethanfel's Context Loop 'masked AV' continuation. Returns (latent, frames pinned)."""
    import comfy.nested_tensor
    n = history_length(frames.shape[0], frames.shape[0])
    samples = latent["samples"]
    nested = getattr(samples, "is_nested", False)
    video = samples.tensors[0] if nested else samples
    T, h, w = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
    if n < 5:
        return latent, 0
    px = frames[-n:, ..., :3].float().cpu()
    if tuple(px.shape[1:3]) != (h * 16, w * 16):
        px = torch.nn.functional.interpolate(px.movedim(-1, 1), size=(h * 16, w * 16), mode="bilinear",
                                             align_corners=False).movedim(1, -1)
    with torch.no_grad():
        enc = vae.encode(px).to(video)
    t = int(enc.shape[2])
    if t >= T:
        LOG.warning("H3 head: %d pinned latent steps leave nothing to generate in a %d-step video", t, T)
        return latent, 0
    video = video.clone()
    video[:, :, :t] = enc
    old = latent.get("noise_mask")
    vm = old.tensors[0] if getattr(old, "is_nested", False) else old
    vm = torch.ones((1, 1, T, h, w), dtype=torch.float32) if vm is None else vm.float().clone()
    vm = torch.broadcast_to(vm, (vm.shape[0], vm.shape[1], T, h, w)).clone()
    vm[:, :, :t] = 0.0
    out = dict(latent)
    if nested:
        audio = samples.tensors[1]
        am = old.tensors[1] if getattr(old, "is_nested", False) else torch.ones_like(audio)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((vm.to(video.device), am))
    else:
        out["samples"], out["noise_mask"] = video, vm.to(video.device)
    LOG.info("H3 head: first %d frames (%d latent steps) pinned from the previous shot", n, t)
    return out, n
