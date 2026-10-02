"""Virtual side panel for MiniMax H3: a held reference strip next to the video, cut off before decoding.

The latent grows by a strip (left, right, top or bottom). The strip holds the panel content (a
reference image, or a clip) through a noise mask, so H3 treats it like its own condition rows
(clean, at the cond timestep) while the video area is generated from noise. Everything stays in
one canvas, so the video attends to the panel at the same instant, the way a split-screen "duet"
works, and no training is needed. BFS H3 Side Panel Crop removes the strip from the latent before
VAE Decode, so the panel never reaches the output.

An aligned guide (the latent guide the body-swap LoRAs use) can be added by this node: the guide
frames are placed in the video area of the canvas and the panel fills the strip. Guides already
on the conditioning (Add Guide for MiniMax H3) are re-encoded onto the canvas too.
"""
from __future__ import annotations

import torch

try:
    import comfy.nested_tensor
    import comfy.utils
except ImportError:  # unit tests without ComfyUI
    comfy = None

GRAY = 0.5   # #808080
PATCH_PX = 32  # one DiT patch (2x2 latent cells of 16 px)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)   # H3 video latent: frames per latent step, k % 5

POSITIONS = ["top", "left", "right", "bottom"]
FITS = ["contain", "cover", "stretch"]
HOLDS = ["all frames", "first latent frame"]


def frame_count_of(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def snap32(x: float) -> int:
    return max(PATCH_PX, int(round(x / PATCH_PX)) * PATCH_PX)


def strip_size(width: int, height: int, position: str, size: float, gap_px: int) -> tuple[int, int, int, int]:
    """(strip_w, strip_h, content_w, content_h) in pixels; the strip includes the gap."""
    if position in ("left", "right"):
        cw = snap32(width * size)
        return cw + gap_px, height, cw, height
    ch = snap32(height * size)
    return width, ch + gap_px, width, ch


def _resize(img: torch.Tensor, w: int, h: int, fit: str) -> torch.Tensor:
    """[N,H,W,C] -> [N,h,w,3] placed on gray: contain (letterbox), cover (center crop) or stretch."""
    img = img[..., :3].float()
    x = img.movedim(-1, 1)
    sh, sw = x.shape[-2:]
    if fit == "stretch":
        return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True).movedim(1, -1).clamp(0, 1)
    s = (min if fit == "contain" else max)(w / sw, h / sh)
    nw, nh = max(1, round(sw * s)), max(1, round(sh * s))
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
    out = torch.full((x.shape[0], 3, h, w), GRAY)
    if fit == "contain":
        top, left = (h - nh) // 2, (w - nw) // 2
        out[:, :, top:top + nh, left:left + nw] = x
    else:
        top, left = (nh - h) // 2, (nw - w) // 2
        out = x[:, :, top:top + h, left:left + w]
    return out.movedim(1, -1)


def strip_frames(panel: torch.Tensor, frames: list[int], width: int, height: int, position: str,
                 size: float, gap_px: int, fit: str) -> torch.Tensor:
    """Strip pixels for the given video frame indices. A one-image panel is static; a clip is indexed
    by frame (held on its last frame when shorter)."""
    sw, sh, cw, ch = strip_size(width, height, position, size, gap_px)
    idx = [min(f, panel.shape[0] - 1) for f in frames]
    content = _resize(panel[idx], cw, ch, fit)
    out = torch.full((len(frames), sh, sw, 3), GRAY)
    if position == "left":
        out[:, :, :cw] = content          # gap next to the video
    elif position == "right":
        out[:, :, gap_px:] = content
    elif position == "top":
        out[:, :ch] = content
    else:
        out[:, gap_px:] = content
    return out


def compose(video: torch.Tensor, strip: torch.Tensor, position: str) -> torch.Tensor:
    """Place a strip next to video frames ([N,H,W,3] each, same N)."""
    if position == "left":
        return torch.cat([strip, video], dim=2)
    if position == "right":
        return torch.cat([video, strip], dim=2)
    if position == "top":
        return torch.cat([strip, video], dim=1)
    return torch.cat([video, strip], dim=1)


def join_latent(target: torch.Tensor, strip: torch.Tensor, position: str) -> torch.Tensor:
    """Same as compose for [B,C,T,h,w] latents."""
    if position == "left":
        return torch.cat([strip, target], dim=4)
    if position == "right":
        return torch.cat([target, strip], dim=4)
    if position == "top":
        return torch.cat([strip, target], dim=3)
    return torch.cat([target, strip], dim=3)


def make_info(width: int, height: int, position: str, size: float, gap_px: int) -> dict:
    """Layout of the canvas in latent cells: the video area (h, w) and the strip beside it."""
    sw, sh, _, _ = strip_size(width, height, position, size, gap_px)
    return {"position": position, "h": height // 16, "w": width // 16,
            "strip_h": sh // 16 if position in ("top", "bottom") else 0,
            "strip_w": sw // 16 if position in ("left", "right") else 0}


def target_box(info: dict, scale: int = 1) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of the video area, in latent cells (scale 1) or pixels (scale 16)."""
    h, w, sh, sw, pos = info["h"], info["w"], info["strip_h"], info["strip_w"], info["position"]
    y0 = sh if pos == "top" else 0
    x0 = sw if pos == "left" else 0
    return y0 * scale, (y0 + h) * scale, x0 * scale, (x0 + w) * scale


def panel_mask(info: dict, latent_t: int, hold: str, target_mask: torch.Tensor | None = None,
               panel_noise: float = 0.0) -> torch.Tensor:
    """[1,1,T,H',W'] denoise mask: panel_noise (0 = hard pin) on the strip, 1 generates the video area."""
    H = info["h"] + (info["strip_h"] if info["position"] in ("top", "bottom") else 0)
    W = info["w"] + (info["strip_w"] if info["position"] in ("left", "right") else 0)
    m = torch.ones(1, 1, latent_t, H, W)
    t_hold = latent_t if hold == "all frames" else 1
    y0, y1, x0, x1 = target_box(info)
    keep = torch.ones(H, W, dtype=torch.bool)
    keep[y0:y1, x0:x1] = False
    m[:, :, :t_hold, keep] = float(panel_noise)
    if target_mask is not None:
        tm = comfy.utils.reshape_mask(target_mask, (1, 1, latent_t, info["h"], info["w"])) if comfy else target_mask
        m[:, :, :, y0:y1, x0:x1] = tm[:1, :1]
    return m


def layout_text(info: dict) -> str:
    """How the kept region is named in a prompt (TSC's wording: 42-58% of the canvas is 'the LEFT half')."""
    pos = info["position"]
    side = {"left": "LEFT", "right": "RIGHT", "top": "TOP", "bottom": "BOTTOM"}[pos]
    other = {"left": "RIGHT", "right": "LEFT", "top": "BOTTOM", "bottom": "TOP"}[pos]
    horizontal = pos in ("left", "right")
    strip = info["strip_w"] if horizontal else info["strip_h"]
    total = strip + (info["w"] if horizontal else info["h"])
    share = strip / total
    word = "half" if 0.42 <= share <= 0.58 else None
    kept = f"the {side} half" if word else f"the {side} {round(share * 100)}% of the frame (a narrow strip)"
    gen = f"the {other} half" if word else f"the {other} {round((1 - share) * 100)}% of the frame"
    line = "vertical" if horizontal else "horizontal"
    text = (f"A split screen divided by a thin straight {line} line: {kept} is the kept footage; "
            f"{gen} is generated and moves in sync with it.")
    if (1 - share) > 1.3 * share:
        text += (f" The generated area is LARGER than the kept panel ({round((1 - share) * 100)}% of the frame against "
                 f"{round(share * 100)}%): restage the scene at that larger size, the same shots, framing proportions and "
                 "timing, a bigger picture, not a pixel-for-pixel mirror.")
    return text


ROPE_MODES = ["canvas", "shifted"]


def panel_frame_positions(info: dict, gap_patches: float = 0.0) -> torch.Tensor:
    """(h, w) RoPE coordinates of one canvas frame's 2x2-patch rows for the 'shifted' layout.

    The video area gets exactly the coordinates of a render without the panel (normalised to the video's
    own area), and the panel continues the same grid past the video's edge, `gap_patches` steps further out.
    """
    h, w = info["h"], info["w"]
    sqrt_a = (h * w) ** 0.5
    step = 64.0 / sqrt_a                               # one 2x2 patch, as in H3's area-normalised grid
    ys = (torch.arange(h // 2, dtype=torch.float64) * step + (1.0 - h / sqrt_a) / 2.0 * 32.0)
    xs = (torch.arange(w // 2, dtype=torch.float64) * step + (1.0 - w / sqrt_a) / 2.0 * 32.0)
    sh, sw, pos = info["strip_h"] // 2, info["strip_w"] // 2, info["position"]
    g = gap_patches * step
    if pos == "left":
        xs = torch.cat([xs[0] - g - step * torch.arange(sw, 0, -1, dtype=torch.float64), xs])
    elif pos == "right":
        xs = torch.cat([xs, xs[-1] + g + step * torch.arange(1, sw + 1, dtype=torch.float64)])
    elif pos == "top":
        ys = torch.cat([ys[0] - g - step * torch.arange(sh, 0, -1, dtype=torch.float64), ys])
    else:
        ys = torch.cat([ys, ys[-1] + g + step * torch.arange(1, sh + 1, dtype=torch.float64)])
    hh, ww = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def shift_layout(layout, info: dict, gap_patches: float):
    """Copy of an H3 PackedLayout with the canvas rows (target video and guides) on the shifted grid."""
    import copy
    frame = panel_frame_positions(info, gap_patches)
    out = copy.copy(layout)
    positions = layout.position_ids.clone()
    rows = frame.shape[0]
    for a, b, kind in layout.segments:
        if kind in ("video", "cond") and (b - a) % rows == 0:
            positions[a:b, 1:] = frame.repeat((b - a) // rows, 1)
    out.position_ids = positions
    return out


def patch_model_rope(model, info: dict, gap_patches: float):
    """Clone of the model whose H3 layout puts the panel on the shifted grid (see shift_layout)."""
    patched = model.clone()
    previous = patched.model_options.get("model_function_wrapper")
    cache = [None, None]

    def wrapper(model_function, args):
        conds = args.get("c", {})
        payload = conds.get("minimax_payload")
        native = (payload or {}).get("layout")
        if native is not None:
            if cache[0] is not native:
                cache[:] = [native, shift_layout(native, info, gap_patches)]
            args = dict(args, c=dict(conds, minimax_payload=dict(payload, layout=cache[1])))
        if previous is not None:
            return previous(model_function, args)
        return model_function(args["input"], args["timestep"], **args["c"])

    patched.set_model_unet_function_wrapper(wrapper)
    return patched


def _encode(vae, frames: torch.Tensor) -> torch.Tensor:
    return vae.encode(frames[..., :3])


def _decode(vae, latent: torch.Tensor) -> torch.Tensor:
    img = vae.decode(latent)
    return img.reshape((-1,) + tuple(img.shape[-3:]))


class BFSH3SidePanel:
    """Adds a held panel strip to an H3 AV latent; optional aligned guide on the same canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "MiniMax H3 AV latent at the output size (from Reference to Video / Image to Video / Empty AV Latent)."}),
                "vae": ("VAE",),
                "panel": ("IMAGE", {"tooltip": "What the panel shows: a reference image (static) or a clip (one image per frame)."}),
                "position": (POSITIONS, {"default": "left", "tooltip": "Side of the video the strip is added to (TSC's duet pins the source clip on the left)."}),
                "size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.01,
                                   "tooltip": "Strip size as a fraction of the video's height (top/bottom) or width (left/right), snapped to 32 px."}),
                "fit": (FITS, {"default": "contain", "tooltip": "contain keeps the whole panel on gray; cover fills the strip and crops; stretch distorts."}),
                "gap": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": "Gray separator between panel and video, in 32 px patches (held like the panel)."}),
                "panel_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "0 pins the panel exactly. 0.05-0.15 lets the model loosen it a little when the result copies too much of it (TSC's SOURCE NOISE)."}),
                "hold": (HOLDS, {"default": "all frames", "tooltip": "all frames: the panel is held for the whole clip. first latent frame: only the start is held, the rest of the strip is generated (and cropped)."}),
            },
            "optional": {
                "guide": ("IMAGE", {"tooltip": "Aligned latent guide (e.g. the source video), placed in the video area. 5, 22, 39... (17k+5) frames, or one image."}),
                "guide_frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "BFS_H3_PANEL", "IMAGE", "STRING")
    RETURN_NAMES = ("positive", "latent", "panel_info", "canvas_preview", "layout_text")
    FUNCTION = "apply"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("Training-free virtual panel: the panel is held in a strip next to the video (like a split-screen "
                   "duet) and BFS H3 Side Panel Crop removes it before decoding. Works with or without an aligned guide.")

    def apply(self, positive, latent, vae, panel, position, size, fit, gap, hold, panel_noise=0.0, guide=None, guide_frame_idx=0):
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or samples.tensors[0].ndim != 5:
            raise ValueError("BFS H3 Side Panel expects a MiniMax H3 AV latent")
        video, audio = samples.tensors[0], samples.tensors[1]
        T, h, w = video.shape[2], video.shape[3], video.shape[4]
        W, H = w * 16, h * 16
        if W % PATCH_PX or H % PATCH_PX:
            raise ValueError(f"the video size {W}x{H} must be a multiple of 32")
        F = frame_count_of(T)
        gap_px = gap * PATCH_PX
        info = make_info(W, H, position, size, gap_px)

        # held strip: encode the strip alone (the video area is generated, so it needs no pixels)
        strip_px = strip_frames(panel, list(range(F)), W, H, position, size, gap_px, fit)
        strip_lat = _encode(vae, strip_px).to(video)
        if strip_lat.shape[2] != T:
            raise ValueError(f"panel latent has {strip_lat.shape[2]} frames, the video {T}")
        canvas = join_latent(video, strip_lat, position)

        old_mask = latent.get("noise_mask")
        target_mask = old_mask.tensors[0] if getattr(old_mask, "is_nested", False) else old_mask
        vmask = panel_mask(info, T, hold, target_mask, panel_noise)
        amask = (old_mask.tensors[1] if getattr(old_mask, "is_nested", False) else torch.ones_like(audio))
        out_latent = dict(latent)
        out_latent["samples"] = comfy.nested_tensor.NestedTensor((canvas, audio))
        out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask.to(canvas.device), amask))

        # guides share the canvas grid: re-encode the existing ones onto it, then add the new one
        def to_canvas(frames_px, start):
            strip = strip_frames(panel, list(range(start, start + frames_px.shape[0])), W, H, position, size, gap_px, fit)
            return _encode(vae, compose(frames_px[..., :3].float().cpu(), strip, position))

        new_pos = []
        for cond, opts in positive:
            opts = dict(opts)
            kfs = []
            for kf in opts.get("minimax_keyframes", []):
                kf = dict(kf)
                if kf.get("latent") is not None and kf["latent"].shape[-1] == w and kf["latent"].shape[-2] == h:
                    kf["latent"] = to_canvas(_decode(vae, kf["latent"]).cpu(), kf["resolved_frame_index"])
                kfs.append(kf)
            if kfs:
                opts["minimax_keyframes"] = kfs
            new_pos.append([cond, opts])

        preview = compose(torch.full((1, H, W, 3), GRAY), strip_px[:1], position)
        if guide is not None:
            n = guide.shape[0]
            n = 1 if n < 5 else n - ((n - 5) % 17)
            start = guide_frame_idx if guide_frame_idx >= 0 else F + guide_frame_idx
            if start < 0 or start + n > F:
                raise ValueError(f"a {n} frame guide at frame {guide_frame_idx} does not fit in the video's {F} frames")
            g = _resize(guide[:n], W, H, "cover")
            kf = {"resolved_frame_index": start, "latent": to_canvas(g, start)}
            preview = compose(g[:1], strip_px[start:start + 1], position)
            for c in new_pos:
                c[1]["minimax_keyframes"] = list(c[1].get("minimax_keyframes", [])) + [kf]
        return (new_pos, out_latent, info, preview, layout_text(info))


class BFSH3SidePanelCrop:
    """Cuts the panel strip off (latent before decode, or decoded images)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"panel_info": ("BFS_H3_PANEL",)},
                "optional": {"latent": ("LATENT",), "images": ("IMAGE",)}}

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "images")
    FUNCTION = "crop"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = "Removes the side panel: crop the sampled latent before VAE Decode (cheaper), or crop decoded images."

    def crop(self, panel_info, latent=None, images=None):
        out_latent, out_images = None, None
        if latent is not None:
            y0, y1, x0, x1 = target_box(panel_info)
            s = latent["samples"]
            if getattr(s, "is_nested", False):
                v, a = s.tensors[0], s.tensors[1]
                s = comfy.nested_tensor.NestedTensor((v[:, :, :, y0:y1, x0:x1].contiguous(), a))
            else:
                s = s[:, :, :, y0:y1, x0:x1].contiguous()
            out_latent = {k: v for k, v in latent.items() if k != "noise_mask"}
            out_latent["samples"] = s
        if images is not None:
            y0, y1, x0, x1 = target_box(panel_info, 16)
            out_images = images[:, y0:y1, x0:x1]
        return (out_latent, out_images)


def h3_render(model, clip, vae, audio_vae, prompt, refs, width, height, length, steps, sampler_name, scheduler,
              seed, panel=None, guide=None, guide_frame_idx=0, position="left", size=1.0, fit="contain", gap=0,
              panel_noise=0.0, hold="all frames", ref_image_size="match", decode_canvas=False,
              rope_mode="canvas", rope_gap=0.0):
    """Reference to Video -> optional pinned panel -> optional aligned guide -> sample -> crop -> decode.
    Without a panel this is plain H3 ref2va with an aligned guide. `{layout}` in the prompt becomes the
    layout sentence. Returns (images, audio, canvas, layout_text, cropped latent)."""
    import comfy.samplers
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced

    text = layout_text(make_info(width, height, position, size, gap * PATCH_PX)) if panel is not None else ""
    prompt = prompt.replace("{layout}", text)
    positive, latent = MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=prompt, width=width, height=height, length=length, ref_image_size=ref_image_size,
        vae=vae, audio_vae=audio_vae, ref_images=refs or None).args[:2]
    info, preview = None, None
    if panel is not None:
        positive, latent, info, preview, _ = BFSH3SidePanel().apply(
            positive, latent, vae, panel, position, size, fit, gap, hold, panel_noise, guide, guide_frame_idx)
    elif guide is not None:
        positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=guide_frame_idx,
                                             vae=vae, image=guide).args[0]

    if info is not None and rope_mode == "shifted":
        model = patch_model_rope(model, info, rope_gap)
    guider = Guider_Basic(model)
    guider.set_conds(positive)
    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
    out = SamplerCustomAdvanced.execute(Noise_RandomNoise(seed), guider, comfy.samplers.sampler_object(sampler_name),
                                        sigmas, latent).args[0]
    cropped = BFSH3SidePanelCrop().crop(info, latent=out)[0] if info is not None else \
        {k: v for k, v in out.items() if k != "noise_mask"}
    video_lat, audio_lat = cropped["samples"].unbind()
    images = _decode(vae, video_lat)
    if info is not None and decode_canvas:
        canvas = _decode(vae, out["samples"].unbind()[0])
    else:
        canvas = preview if preview is not None else images[:1]
    audio = None
    if audio_vae is not None:
        from comfy_extras.nodes_audio import vae_decode_audio
        audio = vae_decode_audio(audio_vae, {"samples": audio_lat})
    return images, audio, canvas, text, cropped


def _sampler_lists():
    try:
        import comfy.samplers
        return comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES
    except ImportError:
        return ["euler"], ["beta"]


class BFSH3Duet:
    """Everything in one node: native references + prompt, the pinned panel, an optional aligned guide,
    sampling (euler / beta / CFG 1 like TSC's duet), decode and crop."""

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _sampler_lists()
        panel_req = BFSH3SidePanel.INPUT_TYPES()["required"]
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True,
                                      "tooltip": "REF2VA prompt. Name the panel by its place (e.g. 'the LEFT half is the kept footage', see layout_text) and never give it a tag; <Picture n> are the ref images."}),
                "width": ("INT", {"default": 576, "min": 32, "max": 4096, "step": 32, "tooltip": "Generated video width (the output)."}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32}),
                "length": ("INT", {"default": 0, "min": 0, "max": 3600, "tooltip": "Frames at 24 fps, snapped to 17k+5. 0 = the guide's or panel clip's length."}),
                "position": panel_req["position"], "size": panel_req["size"], "fit": panel_req["fit"],
                "gap": panel_req["gap"], "panel_noise": panel_req["panel_noise"], "hold": panel_req["hold"],
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "beta"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "decode_canvas": ("BOOLEAN", {"default": False, "tooltip": "Also decode the whole canvas, to check the sync (one more VAE decode)."}),
            },
            "optional": {
                "panel": ("IMAGE", {"tooltip": "Pinned in the strip: the source clip (TSC duet) or a reference image. "
                                               "Leave it empty for a plain guided render."}),
                "audio_vae": ("VAE",),
                "ref_image_1": ("IMAGE", {"tooltip": "<Picture 1>"}),
                "ref_image_2": ("IMAGE", {"tooltip": "<Picture 2>"}),
                "ref_image_3": ("IMAGE", {"tooltip": "<Picture 3>"}),
                "guide": ("IMAGE", {"tooltip": "Optional aligned latent guide in the video area (what the body-swap LoRAs use)."}),
                "guide_frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
                "rope_mode": (ROPE_MODES, {"default": "canvas", "tooltip": "canvas: the panel and the video share one wide grid (TSC). shifted: the video keeps the RoPE positions of a render without the panel and the panel sits past its edge, rope_gap patches further out."}),
                "rope_gap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0, "tooltip": "shifted only: empty RoPE steps (2x2 patches) between the video and the panel."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "STRING", "LATENT")
    RETURN_NAMES = ("images", "audio", "canvas", "layout_text", "latent")
    FUNCTION = "run"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("MiniMax H3 duet in one node: the panel (source clip or reference) is pinned beside the video, "
                   "the video is generated in sync with it, and only the video comes out. Optional aligned guide.")

    def run(self, model, clip, vae, prompt, width, height, length, position, size, fit, gap, panel_noise, hold,
            ref_image_size, steps, sampler_name, scheduler, seed, decode_canvas, panel=None, audio_vae=None,
            ref_image_1=None, ref_image_2=None, ref_image_3=None, guide=None, guide_frame_idx=0,
            rope_mode="canvas", rope_gap=0.0):
        if panel is None and guide is None:
            raise ValueError("BFS H3 Duet needs a panel, a guide, or both")
        if length <= 0:
            src = guide if guide is not None else panel
            length = src.shape[0] if src.shape[0] >= 5 else 124
        refs = {f"ref_image_{i}": r for i, r in enumerate((ref_image_1, ref_image_2, ref_image_3), 1) if r is not None}
        return h3_render(model, clip, vae, audio_vae, prompt, refs, width, height, length, steps, sampler_name,
                         scheduler, seed, panel, guide, guide_frame_idx, position, size, fit, gap, panel_noise, hold,
                         ref_image_size, decode_canvas, rope_mode, rope_gap)


class BFSShotH3Duet:
    """One shot of the shot loop, rendered with MiniMax H3: the shot pinned in a panel (duet), the shot as an
    aligned guide, or both. Planner -> this -> BFS Shot Join."""

    MODES = ["duet (pin the shot, no LoRA needed)", "guide (aligned, for body-swap LoRAs)", "duet + guide"]

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _sampler_lists()
        panel_req = BFSH3SidePanel.INPUT_TYPES()["required"]
        return {
            "required": {
                "shot": ("BFS_SHOT",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "mode": (cls.MODES, {"default": cls.MODES[0], "tooltip":
                    "duet: the shot's own clip is pinned beside the video and copied in sync (TSC's latent pin; "
                    "write the prompt for a split screen, {layout} inserts the sentence). guide: the shot sits on "
                    "the generated frames as a latent guide (use with a body-swap LoRA). duet + guide: both."}),
                "use_ref_2": ("BOOLEAN", {"default": True}),
                "position": panel_req["position"], "size": panel_req["size"], "fit": (FITS, {"default": "cover"}),
                "gap": panel_req["gap"], "panel_noise": panel_req["panel_noise"],
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "beta"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "decode_canvas": ("BOOLEAN", {"default": False}),
            },
            "optional": {"audio_vae": ("VAE",),
                "rope_mode": (ROPE_MODES, {"default": "canvas", "tooltip": "canvas: the panel and the video share one wide grid (TSC). shifted: the video keeps the RoPE positions of a render without the panel and the panel sits past its edge, rope_gap patches further out."}),
                "rope_gap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0, "tooltip": "shifted only: empty RoPE steps (2x2 patches) between the video and the panel."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "audio", "canvas", "layout_text")
    FUNCTION = "render"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Renders one shot with MiniMax H3 (duet, aligned guide, or both) and returns its frames for "
                   "BFS Shot Join. Runs once per shot of the Planner's list.")

    def render(self, shot, model, clip, vae, mode, use_ref_2, position, size, fit, gap, panel_noise,
               ref_image_size, steps, sampler_name, scheduler, seed, decode_canvas, audio_vae=None,
               rope_mode="canvas", rope_gap=0.0):
        refs = {}
        if shot.get("ref") is not None:
            refs["ref_image_1"] = shot["ref"]
        if use_ref_2 and shot.get("ref2") is not None:
            refs[f"ref_image_{len(refs) + 1}"] = shot["ref2"]
        duet, guided = mode != self.MODES[1], mode != self.MODES[0]
        return h3_render(model, clip, vae, audio_vae, shot["prompt"], refs, shot["width"], shot["height"],
                         shot["gen_length"], steps, sampler_name, scheduler, seed + int(shot.get("index", 0)),
                         panel=shot["frames"] if duet else None, guide=shot["frames"] if guided else None,
                         position=position, size=size, fit=fit, gap=gap, panel_noise=panel_noise,
                         ref_image_size=ref_image_size, decode_canvas=decode_canvas,
                         rope_mode=rope_mode, rope_gap=rope_gap)[:4]


NODE_CLASS_MAPPINGS = {
    "BFSH3Duet": BFSH3Duet,
    "BFSShotH3Duet": BFSShotH3Duet,
    "BFSH3SidePanel": BFSH3SidePanel,
    "BFSH3SidePanelCrop": BFSH3SidePanelCrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSH3Duet": "BFS H3 Duet (pinned panel, all in one)",
    "BFSShotH3Duet": "BFS Shot H3 Duet (render one shot)",
    "BFSH3SidePanel": "BFS H3 Side Panel (virtual reference panel)",
    "BFSH3SidePanelCrop": "BFS H3 Side Panel Crop",
}
