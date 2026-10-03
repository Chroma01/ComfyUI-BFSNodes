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

PANEL_PROMPT_HINT = (
    "How to refer to the side panel in the prompt: the panel has NO tag (it is not <Picture n> or <Video n>; "
    "the text encoder never sees it). Name it by its place: 'the LEFT half is the kept footage' (write {layout} "
    "to insert that sentence for the current size and side). Describe only the generated part: never describe "
    "the panel's performer, clothes or room, even to contrast them; what you leave undescribed is copied from "
    "the panel. Restate the new identity in every shot ('her face from <Picture 1>' plus two or three face, hair "
    "or outfit words). Give exact times for cuts ('[Shot 2] At 00:03.708, both halves cut together to ...').")

DUET_PROMPT_TEMPLATE = """subject_definitions:
<Subject 1> is the woman whose appearance comes from <Picture 1>: {skin, face shape, eyes, brows, nose, lips}, {hair colour, length and cut}, wearing {the outfit, piece by piece}.

summary:
[reference generation] The target video is a split screen: the kept footage beside <Subject 1>, who moves in sync with it, in the same room.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - {her key look words} are retained.
The kept footage: fully_preserved - the panel is kept exactly.

detailed_description:
The target video is in a realistic style, as {the medium, e.g. handheld vertical smartphone footage under soft daylight}. {layout}

[Shot 1] {Camera distance, angle and movement.} <Subject 1>, her face from <Picture 1> with {two face words}, {two hair or outfit words}, {what she does, one sentence per real action}.

overall_soundscape:
{The sounds of the scene and when they happen.}

non_diegetic_music:
None."""


TASKS = ["character swap", "style", "setting", "appearance", "lighting / weather", "custom prompt"]

CREDIT = ("Credits: the initial idea for this H3 node came from TSC's latent-pin duet (the source pinned beside the "
          "video with a noise mask). BFS had already used the same principle on LTX (a green side panel holding the "
          "reference) and implemented the virtual sidecar approach (reference tokens placed beside the frame in RoPE). "
          "New here: the shifted RoPE layout (the video keeps its own grid and the panel sits past its edge, with an "
          "optional gap), dynamic references, task prompts and the shot-loop node.")

PROMPT_EXAMPLE = """Example (character swap, source clip pinned on the left, one face picture and one full-body picture):

subject_definitions:
<Subject 1> is the woman whose appearance comes from <Picture 1> and <Picture 2>: fair skin, a narrow oval face, grey-green eyes, light brown hair in a low ponytail, wearing a white long-sleeved top under a navy denim apron.

summary:
[reference generation] The target video is a split screen: the kept footage beside <Subject 1>, who moves in sync with it, in the same bedroom.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - her face, ponytail, white top and navy apron are retained.
The kept footage: fully_preserved - the panel is kept exactly.

detailed_description:
The target video is in a realistic style, as handheld vertical smartphone footage under soft daylight. {layout}

[Shot 1] A medium close-up at chest height, the phone steady at eye level. <Subject 1>, her face from <Picture 1> with grey-green eyes and soft pink lips, her light brown ponytail and navy apron, talks to the camera with small nods.

[Shot 2] At 00:03.708, both halves cut together to a closer framing. <Subject 1>, her face from <Picture 1>, the white sleeves and apron straps visible, raises one open hand beside her mouth and smiles.

overall_soundscape:
Her voice speaking to the camera in a quiet bedroom.

non_diegetic_music:
None."""


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


def layout_text(info: dict, rope_mode: str = "canvas") -> str:
    """How the kept region is named in a prompt (TSC's wording: 42-58% of the canvas is 'the LEFT half').
    shifted RoPE: the video sits on its own grid, so a split-screen sentence would make the model split the video
    itself; the sentence then only says that the whole frame follows the kept footage."""
    if rope_mode == "shifted":
        return "The whole frame shows the generated video, moving in sync with the kept footage frame by frame."
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


def latent_mask(mask: torch.Tensor, latent_t: int, h: int, w: int, grow: int = 1) -> torch.Tensor:
    """Pixel mask frames [F,H,W] (1 = regenerate) -> [1,1,T,h,w] on H3's latent grid: a latent cell regenerates when
    any of its pixels or frames does (frames per latent step: 1, 4, 4, 4, 4...)."""
    m = mask.float()
    if m.ndim == 4:
        m = m[..., 0]
    m = torch.nn.functional.adaptive_max_pool2d(m[:, None], (h, w))[:, 0]
    if grow > 0:
        m = torch.nn.functional.max_pool2d(m[:, None], 2 * grow + 1, 1, grow)[:, 0]
    out, f = [], 0
    for k in range(latent_t):
        n = FRAME_PER_TOKEN[k % 5]
        seg = m[min(f, m.shape[0] - 1):min(f + n, m.shape[0])] if f < m.shape[0] else m[-1:]
        out.append(seg.amax(0)); f += n
    return (torch.stack(out)[None, None] > 0.5).float()


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
                "gap": ("INT", {"default": 0, "min": 0, "max": 8, "advanced": True, "tooltip": "Gray separator between panel and video, in 32 px patches (held like the panel)."}),
                "panel_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "0 pins the panel exactly. 0.05-0.15 lets the model loosen it a little when the result copies too much of it (TSC's SOURCE NOISE)."}),
                "hold": (HOLDS, {"default": "all frames", "advanced": True, "tooltip": "all frames: the panel is held for the whole clip. first latent frame: only the start is held, the rest of the strip is generated (and cropped)."}),
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
    DESCRIPTION = (CREDIT + "\n\nTraining-free virtual panel: the panel is held in a strip next to the video (like a split-screen "
                   "duet) and BFS H3 Side Panel Crop removes it before decoding. Works with or without an aligned guide.\n\n"
                   + PANEL_PROMPT_HINT)

    def apply(self, positive, latent, vae, panel, position, size, fit, gap, hold, panel_noise=0.0, guide=None, guide_frame_idx=0,
              keep_video=None, keep_mask=None):
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
        old_mask = latent.get("noise_mask")
        target_mask = old_mask.tensors[0] if getattr(old_mask, "is_nested", False) else old_mask
        if keep_mask is not None:
            # inpainting inside the duet: the video area starts from keep_video and only the masked region is regenerated
            src = keep_video if keep_video is not None else panel
            idx = [min(i, src.shape[0] - 1) for i in range(F)]
            video = _encode(vae, _resize(src[idx], W, H, "cover")).to(video)
            target_mask = latent_mask(keep_mask, T, h, w)
        canvas = join_latent(video, strip_lat, position)
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


def build_prompt(task: str, instruction: str, n_pictures: int, n_videos: int, rope_mode: str = "canvas") -> str:
    """A six-section REF2VA prompt for a duet task. `{layout}` stays in it and is filled at render time.
    shifted: no split-screen wording (it makes the model split the video itself)."""
    inst = instruction.strip().rstrip(".")
    pics = [f"<Picture {i}>" for i in range(1, n_pictures + 1)]
    pics_txt = " and ".join(pics) if len(pics) <= 2 else ", ".join(pics[:-1]) + " and " + pics[-1]
    seen = f" as shown in {pics_txt}" if pics else ""
    sync = "every movement, gesture and expression in sync with the kept footage"
    if task == "character swap":
        look = f"whose appearance comes from {pics_txt}" if pics else "the new person"
        defs = f"<Subject 1> is the person {look}" + (f": {inst}." if inst else ".")
        summary = ("[reference generation] The target video is a split screen: the kept footage beside <Subject 1>, "
                   "who moves in sync with it, in the same place.")
        keep = (f"<Subject 1> (appears in [Shot 1]): fully_preserved - the face, hair and outfit"
                f"{' from ' + pics[0] if pics else ''} are retained.")
        face = f", the face from {pics[0]}" if pics else ""
        shot = (f"[Shot 1] The same framing and camera movement as the kept footage. <Subject 1>{face}"
                f"{', ' + inst if inst else ''}, performs {sync}.")
        style = "The target video keeps the medium and the light of the kept footage."
    else:
        what = {"style": f"restyled as {inst or 'the style'}{seen}",
                "setting": f"moved to {inst or 'the place'}{seen}",
                "appearance": f"with the performer now {inst or 'changed'}{seen}",
                "lighting / weather": f"under {inst or 'the new light'}{seen}"}[task]
        defs = (f"<Subject 1> is {inst or 'the reference'}{seen}." if pics else "")
        summary = (f"[reference generation] The target video is a split screen: the kept footage beside the same "
                   f"performance, {what}.")
        keep = f"The performance and camera: fully_preserved - {sync}."
        shot = f"[Shot 1] The same framing, people and camera movement as the kept footage, {what}, {sync}."
        style = (f"The target video is in {inst}{'' if 'style' in inst.lower() else ' style'}." if task == "style" and inst
                 else "The target video keeps the medium of the kept footage.")
    sections = [
        "subject_definitions:\n" + (defs or "The kept footage is the motion reference of the split screen."),
        "summary:\n" + summary,
        "retention_analysis:\n" + keep + "\nThe kept footage: fully_preserved - the panel is kept exactly.",
        "detailed_description:\n" + style + " {layout}\n\n" + shot,
        "overall_soundscape:\nThe sounds of the kept footage, in sync.",
        "non_diegetic_music:\nNone.",
    ]
    text = "\n\n".join(sections)
    if rope_mode == "shifted":
        text = (text.replace("The target video is a split screen: the kept footage beside ", "The target video shows ")
                .replace("who moves in sync with it", "who moves in sync with the kept footage")
                .replace("the motion reference of the split screen", "the motion reference"))
    return text


def h3_render(model, clip, vae, audio_vae, prompt, refs, width, height, length, steps, sampler_name, scheduler,
              seed, panel=None, guide=None, guide_frame_idx=0, position="left", size=1.0, fit="contain", gap=0,
              panel_noise=0.0, hold="all frames", ref_image_size="match", decode_canvas=False,
              rope_mode="canvas", rope_gap=0.0, first_frame=None):
    """Reference to Video -> optional pinned panel -> optional aligned guide -> sample -> crop -> decode.

    `refs` holds the Reference to Video inputs ({"ref_images": {...}, "ref_videos": {...}, ...}). Without a
    panel this is plain H3 ref2va with an aligned guide. `{layout}` in the prompt becomes the layout sentence.
    Returns (images, audio, canvas, layout_text, cropped latent, prompt used)."""
    import comfy.samplers
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced

    text = layout_text(make_info(width, height, position, size, gap * PATCH_PX), rope_mode) if panel is not None else ""
    prompt = prompt.replace("{layout}", text)
    refs = {k: v for k, v in (refs or {}).items() if v}
    positive, latent = MiniMaxH3ReferenceToVideo.execute(
        clip=clip, prompt=prompt, width=width, height=height, length=length, ref_image_size=ref_image_size,
        vae=vae, audio_vae=audio_vae, **refs).args[:2]
    if first_frame is not None:   # anchored at frame 0; the panel step moves it onto the canvas
        positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae,
                                             image=first_frame[:1]).args[0]
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
    if audio is None:   # no audio VAE: a silent track of the video's length, so Create Video gets valid audio
        audio = {"waveform": torch.zeros(1, 2, max(1, int(round(images.shape[0] / 24.0 * 44100)))), "sample_rate": 44100}
    return images, audio, canvas, text, cropped, prompt


def _sampler_lists():
    try:
        import comfy.samplers
        return comfy.samplers.SAMPLER_NAMES, comfy.samplers.SCHEDULER_NAMES
    except ImportError:
        return ["euler"], ["beta"]


def _resolve_prompt(prompt: str, task: str, instruction: str, n_pictures: int, n_videos: int) -> str:
    if prompt and prompt.strip():
        return prompt
    if task == "custom prompt":
        raise ValueError("task 'custom prompt' needs a prompt")
    return build_prompt(task, instruction, n_pictures, n_videos)


try:
    from comfy_api.latest import io as _io
    PANEL_INFO_T = _io.Custom("BFS_H3_PANEL")
except ImportError:  # unit tests without ComfyUI
    _io = None

if _io is not None:
    class BFSH3Duet(_io.ComfyNode):
        """Everything in one node: references (as many as the official node takes), the pinned panel, an optional
        aligned guide, a task prompt, sampling, decode and crop."""

        @classmethod
        def define_schema(cls):
            samplers, schedulers = _sampler_lists()
            return _io.Schema(
                node_id="BFSH3Duet",
                display_name="BFS H3 Duet (pinned panel, all in one)",
                category="BFS/MiniMax H3",
                description=("MiniMax H3 duet in one node: a panel (the source clip or a picture) is pinned beside the "
                             "video, the video is generated in sync with it, and only the video comes out. For any "
                             "edit that keeps the source's motion and timing: character swap, style, setting, "
                             "appearance, light. Optional aligned guide.\n\n" + PANEL_PROMPT_HINT + "\n\n" + CREDIT),
                inputs=[
                    _io.Model.Input("model"),
                    _io.Clip.Input("clip"),
                    _io.Vae.Input("vae"),
                    _io.Vae.Input("audio_vae", optional=True),
                    _io.Image.Input("panel", optional=True, tooltip="Pinned beside the video: the source clip (copied in sync) or a picture."),
                    _io.Image.Input("guide", optional=True, tooltip="Optional aligned latent guide in the video area (for body-swap LoRAs)."),
                    _io.Combo.Input("task", options=TASKS, default=TASKS[0], tooltip=
                        "Writes a DRAFT prompt when the prompt box is empty: character swap (the person from the "
                        "pictures), style, setting, appearance or lighting / weather (the instruction says what). The "
                        "draft cannot see the video, so it is generic: a prompt written for the clip (see the example "
                        "in the prompt tooltip, and the 'prompt' output for the draft) gives much better results. A "
                        "text in the prompt box always wins."),
                    _io.String.Input("instruction", default="", multiline=False, tooltip=
                        "What changes, in a few seen words: 'a 1990s anime cel style', 'a beach at sunset', "
                        "'an elderly woman with grey hair', 'night lit by pink and blue neon', or the new person's look."),
                    _io.String.Input("prompt", default="", multiline=True, tooltip=
                        "Your own REF2VA prompt (overrides the task). <Picture n> / <Video n> / <Audio n> are the "
                        "references in order. " + PANEL_PROMPT_HINT + "\n\n" + PROMPT_EXAMPLE),
                    _io.Int.Input("width", default=448, min=32, max=4096, step=32, tooltip="Generated video width (the output)."),
                    _io.Int.Input("height", default=800, min=32, max=4096, step=32),
                    _io.Int.Input("length", default=0, min=0, max=3600, tooltip="Frames at 24 fps, snapped to 17k+5. 0 = the guide's or panel clip's length."),
                    _io.Combo.Input("position", options=POSITIONS, default="left"),
                    _io.Float.Input("size", default=1.0, min=0.1, max=1.5, step=0.01, tooltip="Panel size against the video (1.0 = two equal halves)."),
                    _io.Combo.Input("fit", advanced=True, options=FITS, default="contain", tooltip="contain keeps the whole clip (smaller, with grey around) so nothing is cropped; cover fills the panel and crops; stretch distorts."),
                    _io.Int.Input("gap", advanced=True, default=0, min=0, max=8, tooltip="Grey separator in 32 px patches (pixels, held)."),
                    _io.Float.Input("panel_noise", default=0.0, min=0.0, max=1.0, step=0.01, tooltip=
                        "0 copies the panel exactly (swaps, light). 0.1-0.2 gives room for big changes (style, creatures)."),
                    _io.Combo.Input("hold", advanced=True, options=HOLDS, default=HOLDS[0]),
                    _io.Combo.Input("rope_mode", advanced=True, options=ROPE_MODES, default="canvas", tooltip=
                        "canvas: panel and video share one wide grid (TSC). shifted (BFS): the video keeps the RoPE "
                        "positions of a render without the panel and the panel sits past its edge."),
                    _io.Float.Input("rope_gap", advanced=True, default=0.0, min=0.0, max=256.0, step=1.0, tooltip="shifted only: empty RoPE steps (2x2 patches) between video and panel. Keep it small against the video width (0-2 at low resolution): a large gap makes the model draw its own split screen."),
                    _io.Combo.Input("ref_image_size", advanced=True, options=["match", "max"], default="match"),
                    _io.Int.Input("steps", default=20, min=1, max=200),
                    _io.Combo.Input("sampler_name", options=samplers, default="euler"),
                    _io.Combo.Input("scheduler", options=schedulers, default="beta"),
                    _io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=True),
                    _io.Boolean.Input("decode_canvas", advanced=True, default=False, tooltip="Also decode the whole canvas, to check the sync."),
                    _io.Int.Input("guide_frame_idx", advanced=True, default=0, min=-9999, max=9999, optional=True),
                    _io.Autogrow.Input("ref_images", optional=True, template=_io.Autogrow.TemplatePrefix(
                        input=_io.Image.Input("ref_image", tooltip="<Picture n>, in order"), prefix="ref_image_", min=0, max=9)),
                    _io.Autogrow.Input("ref_videos", optional=True, template=_io.Autogrow.TemplatePrefix(
                        input=_io.Image.Input("ref_video", tooltip="<Video n> reference (24 fps)"), prefix="ref_video_", min=0, max=3)),
                    _io.Autogrow.Input("ref_video_audios", optional=True, template=_io.Autogrow.TemplatePrefix(
                        input=_io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"), prefix="ref_video_audio_", min=0, max=3)),
                    _io.Autogrow.Input("ref_audios", optional=True, template=_io.Autogrow.TemplatePrefix(
                        input=_io.Audio.Input("ref_audio", tooltip="<Audio n> reference"), prefix="ref_audio_", min=0, max=3)),
                ],
                outputs=[_io.Image.Output(display_name="images"), _io.Audio.Output(display_name="audio"),
                         _io.Image.Output(display_name="canvas"), _io.String.Output(display_name="layout_text"),
                         _io.Latent.Output(display_name="latent"), _io.String.Output(display_name="prompt")],
            )

        @classmethod
        def execute(cls, model, clip, vae, task, instruction, prompt, width, height, length, position, size, fit, gap,
                    panel_noise, hold, rope_mode, rope_gap, ref_image_size, steps, sampler_name, scheduler, seed,
                    decode_canvas, audio_vae=None, panel=None, guide=None, guide_frame_idx=0, ref_images=None,
                    ref_videos=None, ref_video_audios=None, ref_audios=None):
            if panel is None and guide is None:
                raise ValueError("BFS H3 Duet needs a panel, a guide, or both")
            if length <= 0:
                src = guide if guide is not None else panel
                length = src.shape[0] if src.shape[0] >= 5 else 124
            ref_images = {k: v for k, v in (ref_images or {}).items() if v is not None}
            ref_videos = {k: v for k, v in (ref_videos or {}).items() if v is not None}
            text = _resolve_prompt(prompt, task, instruction, len(ref_images), len(ref_videos))
            refs = {"ref_images": ref_images, "ref_videos": ref_videos,
                    "ref_video_audios": {k: v for k, v in (ref_video_audios or {}).items() if v is not None},
                    "ref_audios": {k: v for k, v in (ref_audios or {}).items() if v is not None}}
            return _io.NodeOutput(*h3_render(model, clip, vae, audio_vae, text, refs, width, height, length, steps,
                                             sampler_name, scheduler, seed, panel, guide, guide_frame_idx, position,
                                             size, fit, gap, panel_noise, hold, ref_image_size, decode_canvas,
                                             rope_mode, rope_gap))


if _io is not None:
    class BFSH3DuetConditioning(_io.ComfyNode):
        """Duet conditioning only (no sampling): references + prompt (written by a VLM from the task when connected)
        + the pinned panel. Goes into BasicGuider / SamplerCustomAdvanced; BFS H3 Side Panel Crop before VAE Decode."""

        @classmethod
        def define_schema(cls):
            return _io.Schema(
                node_id="BFSH3DuetConditioning",
                display_name="BFS H3 Duet Conditioning (prompt writer)",
                category="BFS/MiniMax H3",
                description=("MiniMax H3 duet conditioning for your own sampler. The panel (the source clip) is pinned beside "
                             "the video. The prompt: yours if you write one; otherwise a connected VLM writes it in the duet "
                             "format from the task and instruction, looking at the clip and the references; otherwise a "
                             "draft. Outputs positive + latent (with the panel's noise mask) + model (RoPE shift applied in "
                             "shifted mode) + panel_info for BFS H3 Side Panel Crop.\n\n" + PANEL_PROMPT_HINT + "\n\n" + CREDIT),
                inputs=[
                    _io.Clip.Input("clip", tooltip="The MiniMax H3 text encoder (Qwen3-VL 32B)."),
                    _io.Vae.Input("vae"),
                    _io.Image.Input("panel", tooltip="The clip pinned beside the video (copied in sync), or a picture."),
                    _io.Combo.Input("task", options=["character swap", "style", "setting", "appearance", "lighting / weather", "custom"],
                                    default="character swap", tooltip="What to do. With a VLM connected it writes the prompt for this task."),
                    _io.String.Input("instruction", default="", multiline=True, tooltip=
                        "What changes, in seen words: 'a 1990s anime cel style', 'a sunny beach at sunset', 'an elderly woman with "
                        "short grey hair', or for custom anything you want done. Empty is fine for character swap (the person "
                        "comes from the pictures)."),
                    _io.String.Input("prompt", default="", multiline=True, tooltip=
                        "Your own prompt (wins over the VLM and the draft). " + PANEL_PROMPT_HINT),
                    _io.Int.Input("width", default=448, min=32, max=4096, step=32),
                    _io.Int.Input("height", default=800, min=32, max=4096, step=32),
                    _io.Int.Input("length", default=0, min=0, max=3600, tooltip="Frames (17k+5). 0 = the panel clip's length."),
                    _io.Combo.Input("position", options=POSITIONS, default="left"),
                    _io.Float.Input("size", default=1.0, min=0.1, max=1.5, step=0.01),
                    _io.Float.Input("panel_noise", default=0.0, min=0.0, max=1.0, step=0.01,
                                    tooltip="0 pins the panel exactly; 0.1-0.2 for big changes."),
                    _io.Combo.Input("rope_mode", options=ROPE_MODES, default="canvas",
                                    tooltip="canvas (one wide grid) or shifted (connect the model and use the model output)."),
                    _io.Combo.Input("fit", options=FITS, default="contain", advanced=True),
                    _io.Int.Input("gap", default=0, min=0, max=8, advanced=True),
                    _io.Combo.Input("hold", options=HOLDS, default=HOLDS[0], advanced=True),
                    _io.Float.Input("rope_gap", default=0.0, min=0.0, max=256.0, step=1.0, advanced=True),
                    _io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True),
                    _io.Int.Input("vlm_max_tokens", default=1024, min=128, max=4096, advanced=True),
                    _io.Clip.Input("vlm", optional=True, tooltip="Optional VLM (CLIPLoader with a Qwen3-VL text encoder) that writes the prompt."),
                    _io.Model.Input("model", optional=True, tooltip="Needed for rope_mode = shifted."),
                    _io.Vae.Input("audio_vae", optional=True),
                    _io.Image.Input("guide", optional=True, tooltip="Optional aligned latent guide in the video area."),
                    _io.Mask.Input("keep_mask", optional=True, tooltip=
                        "Inpainting inside the duet: 1 = regenerate (e.g. the person, dilated), 0 = keep. The video area "
                        "starts from keep_video (or the panel clip) and only the masked region is generated, so the "
                        "background and its light stay pixel-exact."),
                    _io.Image.Input("keep_video", optional=True, tooltip="The video kept outside the mask (default: the panel clip)."),
                    _io.Autogrow.Input("ref_images", optional=True, template=_io.Autogrow.TemplatePrefix(
                        input=_io.Image.Input("ref_image", tooltip="<Picture n>, in order"), prefix="ref_image_", min=0, max=9)),
                ],
                outputs=[_io.Conditioning.Output(display_name="positive"), _io.Latent.Output(display_name="latent"),
                         _io.Model.Output(display_name="model"), PANEL_INFO_T.Output(display_name="panel_info"),
                         _io.String.Output(display_name="prompt"), _io.Image.Output(display_name="canvas_preview")],
            )

        @classmethod
        def execute(cls, clip, vae, panel, task, instruction, prompt, width, height, length, position, size, panel_noise,
                    rope_mode, fit, gap, hold, rope_gap, ref_image_size, vlm_max_tokens, vlm=None, model=None,
                    audio_vae=None, guide=None, ref_images=None, keep_mask=None, keep_video=None):
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
            refs = [v for v in (ref_images or {}).values() if v is not None]
            if length <= 0:
                length = panel.shape[0] if panel.shape[0] >= 5 else 124
            text = prompt if prompt and prompt.strip() else ""
            if not text and vlm is not None:
                try:
                    from .bfs_shot_loop import write_duet_prompt
                except ImportError:
                    write_duet_prompt = None
                if write_duet_prompt:
                    idx = sorted(set(int(round(x)) for x in torch.linspace(0, panel.shape[0] - 1, min(4, panel.shape[0])).tolist()))
                    text = write_duet_prompt(vlm, [panel[i:i + 1] for i in idx], refs, task, instruction, int(vlm_max_tokens),
                                             rope_mode)
            if not text:
                text = build_prompt(task if task != "custom" else "appearance", instruction, len(refs), 0, rope_mode)
            info = make_info(width, height, position, size, gap * PATCH_PX)
            text = text.replace("{layout}", layout_text(info, rope_mode))
            positive, latent = MiniMaxH3ReferenceToVideo.execute(
                clip=clip, prompt=text, width=width, height=height, length=length, ref_image_size=ref_image_size, vae=vae,
                audio_vae=audio_vae, ref_images={f"ref_image_{i}": r for i, r in enumerate(refs)} or None).args[:2]
            positive, latent, info, preview, _ = BFSH3SidePanel().apply(
                positive, latent, vae, panel, position, size, fit, gap, hold, panel_noise, guide, 0,
                keep_video=keep_video, keep_mask=keep_mask)
            if rope_mode == "shifted":
                if model is None:
                    raise ValueError("rope_mode 'shifted' needs the model input (use the node's model output in the sampler)")
                model = patch_model_rope(model, info, rope_gap)
            return _io.NodeOutput(positive, latent, model, info, text, preview)


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
                    "duet: the shot's own clip is pinned beside the video and copied in sync. guide: the shot sits on "
                    "the generated frames as a latent guide (use with a body-swap LoRA). duet + guide: both."}),
                "task": (TASKS, {"default": TASKS[0], "tooltip": "Writes the prompt when the shot has none (no "
                                  "per-shot or global prompt in the Planner)."}),
                "instruction": ("STRING", {"default": "", "tooltip": "What changes, in a few seen words (see BFS H3 Duet)."}),
                "use_ref_2": ("BOOLEAN", {"default": True}),
                "position": panel_req["position"], "size": panel_req["size"],
                "fit": (FITS, {"default": "contain", "advanced": True, "tooltip": "contain keeps the whole clip (smaller, with grey around) so nothing is cropped"}),
                "gap": panel_req["gap"], "panel_noise": panel_req["panel_noise"],
                "ref_image_size": (["match", "max"], {"default": "match", "advanced": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "beta"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "decode_canvas": ("BOOLEAN", {"default": False, "advanced": True}),
            },
            "optional": {"audio_vae": ("VAE",),
                         "rope_mode": (ROPE_MODES, {"default": "canvas", "advanced": True}),
                         "rope_gap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0, "advanced": True})},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "canvas", "layout_text", "prompt")
    FUNCTION = "render"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Renders one shot with MiniMax H3 (duet, aligned guide, or both) and returns its frames for "
                   "BFS Shot Join. Runs once per shot of the Planner's list. The prompt comes from the Planner (per "
                   "shot or global), or from the task when there is none.\n\nDuet modes: " + PANEL_PROMPT_HINT
                   + "\n\n" + CREDIT)

    def render(self, shot, model, clip, vae, mode, task, instruction, use_ref_2, position, size, fit, gap,
               panel_noise, ref_image_size, steps, sampler_name, scheduler, seed, decode_canvas, audio_vae=None,
               rope_mode="canvas", rope_gap=0.0):
        imgs = {}
        if shot.get("ref") is not None:
            imgs["ref_image_1"] = shot["ref"]
        if use_ref_2 and shot.get("ref2") is not None:
            imgs[f"ref_image_{len(imgs) + 1}"] = shot["ref2"]
        duet, guided = mode != self.MODES[1], mode != self.MODES[0]
        try:
            from .bfs_shot_loop import chain_image, remember_result
        except ImportError:
            chain_image = remember_result = None
        prev = chain_image(shot) if chain_image else None
        if prev is not None and shot.get("chain") == "reference":
            imgs[f"ref_image_{len(imgs) + 1}"] = prev
        text = _resolve_prompt(shot.get("prompt", ""), task, instruction, len(imgs), 0)
        out = h3_render(model, clip, vae, audio_vae, text, {"ref_images": imgs}, shot["width"], shot["height"],
                        shot["gen_length"], steps, sampler_name, scheduler, seed + int(shot.get("index", 0)),
                        panel=shot["frames"] if duet else None, guide=shot["frames"] if guided else None,
                        position=position, size=size, fit=fit, gap=gap, panel_noise=panel_noise,
                        ref_image_size=ref_image_size, decode_canvas=decode_canvas,
                        rope_mode=rope_mode, rope_gap=rope_gap,
                        first_frame=prev if prev is not None and shot.get("chain") == "first frame" else None)
        if remember_result:
            remember_result(shot, out[0])   # the next shot can continue from it (auto loop)
        return out[:4] + (out[5],)


NODE_CLASS_MAPPINGS = {
    "BFSShotH3Duet": BFSShotH3Duet,
    "BFSH3SidePanel": BFSH3SidePanel,
    "BFSH3SidePanelCrop": BFSH3SidePanelCrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSShotH3Duet": "BFS Shot H3 Duet (render one shot)",
    "BFSH3SidePanel": "BFS H3 Side Panel (virtual reference panel)",
    "BFSH3SidePanelCrop": "BFS H3 Side Panel Crop",
}
if _io is not None:
    NODE_CLASS_MAPPINGS["BFSH3Duet"] = BFSH3Duet
    NODE_DISPLAY_NAME_MAPPINGS["BFSH3Duet"] = "BFS H3 Duet (pinned panel, all in one)"
    NODE_CLASS_MAPPINGS["BFSH3DuetConditioning"] = BFSH3DuetConditioning
    NODE_DISPLAY_NAME_MAPPINGS["BFSH3DuetConditioning"] = "BFS H3 Duet Conditioning (prompt writer)"
