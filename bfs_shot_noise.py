"""BFS Shot Noise: the sampler's starting noise, with optional structure. Every option is off at 0, and with all of
them off the output is exactly ComfyUI's RandomNoise for the same seed.

What can be shaped (the video part of the latent only; the audio noise is always plain):
  seed per shot      same for every shot / increment / random per shot / overrides (as BFS Shot Seed)
  shared_base        the COARSE pattern of the noise (block averages on a fixed grid) comes from a base seed instead
                     of the shot's seed; the fine detail stays the shot's own. The coarse pattern decides layout and
                     who the subject becomes in the first steps, so shots of one video (or re-rolls of one clip)
                     agree on it. The grid is defined in blocks, not in latent cells, so the same base seed gives the
                     same coarse pattern at any resolution, and along the timeline it is one long tape: a shot reads
                     the part at its own position, whatever its length.
  distribution       gaussian (default) / laplacian / student-t / uniform, all at unit variance
  spectrum_tilt      1/f^alpha spectrum: more (or less) energy in the large blobs ("fractal" noise)
  pyramid            coarser layers of noise added on top
  static_share       a part of the noise that is the same in every frame (temporally correlated noise)
  guide_structure    outside the mask, a little of the guide video's own latent is mixed into the noise, so the
                     background starts closer to the guide; inside the mask the noise stays free (no pull toward
                     the original person)
  ref_tint           inside the mask, the noise leans toward the reference picture's average latent (its colours)

All mixes keep the noise at unit variance (a*signal + sqrt(1-a^2)*noise, block means replaced by block means of the same
variance). The model was trained on plain noise: small values, and compare with 0."""
import hashlib
import math

import torch

import comfy.nested_tensor
import comfy.sample

try:
    from .bfs_shot_loop import shot_seed, BFSShotSeed
    from .bfs_h3_side_panel import latent_mask
except ImportError:                                   # run as a script (tests)
    from bfs_shot_loop import shot_seed, BFSShotSeed
    from bfs_h3_side_panel import latent_mask


_pick_seed = shot_seed            # BFSShotNoise.make has an input of the same name


def _gen(*key) -> torch.Generator:
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256("/".join(str(k) for k in key).encode()).digest()[:8], "big") >> 1)


def _edges(n: int, blocks: int) -> torch.Tensor:
    """Block id of each of n cells for `blocks` near-equal blocks (blocks is capped at n)."""
    blocks = max(1, min(int(blocks), n))
    return torch.clamp((torch.arange(n) * blocks) // n, max=blocks - 1)


def replace_coarse(x: torch.Tensor, base_seed: int, amount: float, grid: int = 6, time_block: int = 2, t0: int = 0) -> torch.Tensor:
    """x [C,T,h,w] unit noise. The mean of every (time block, y block, x block) cell is replaced (amount 1) or mixed
    with a mean drawn from `base_seed`; the result is unit noise again. Blocks: `grid` along the short side, the long
    side in proportion; `time_block` latent frames, counted from timeline position t0."""
    if amount <= 0:
        return x
    C, T, h, w = x.shape
    hb, wb = (grid, max(1, round(grid * w / h))) if h <= w else (max(1, round(grid * h / w)), grid)
    yid, xid = _edges(h, hb), _edges(w, wb)
    hb, wb = int(yid.max()) + 1, int(xid.max()) + 1
    tid_abs = (t0 + torch.arange(T)) // max(1, int(time_block))
    tids = torch.unique(tid_abs)
    tloc = torch.searchsorted(tids, tid_abs)
    nb = len(tids) * hb * wb
    ids = ((tloc[:, None, None] * hb + yid[None, :, None]) * wb + xid[None, None, :]).reshape(-1)
    flat = x.reshape(C, -1).double()
    cnt = torch.zeros(nb, dtype=torch.float64).index_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
    mean = torch.zeros(C, nb, dtype=torch.float64).index_add_(1, ids, flat) / cnt
    z = torch.stack([torch.randn(C, hb, wb, generator=_gen("coarse", base_seed, int(t), hb, wb), dtype=torch.float64) for t in tids], 1).reshape(C, nb)
    a = float(min(1.0, amount))
    new = math.sqrt(1.0 - a * a) * mean + a * z / cnt.sqrt()
    return (flat + (new - mean)[:, ids]).reshape(C, T, h, w).to(x.dtype)


DISTRIBUTIONS = ["gaussian", "laplacian", "student-t", "uniform"]


def draw(shape, kind: str, seed: int) -> torch.Tensor:
    """Unit-variance noise of another distribution (heavier or lighter tails than the gaussian the model was trained on)."""
    g = _gen("dist", kind, seed)
    if kind == "laplacian":
        u = torch.rand(shape, generator=g, dtype=torch.float64) - 0.5
        return (-torch.sign(u) * torch.log1p(-2 * u.abs().clamp(max=0.5 - 1e-12)) / math.sqrt(2.0)).float()
    if kind == "student-t":                                   # 5 degrees of freedom
        n = torch.randn(shape, generator=g, dtype=torch.float64)
        chi = torch.randn((5,) + tuple(shape), generator=g, dtype=torch.float64).pow(2).sum(0)
        return (n / torch.sqrt(chi / 5.0) * math.sqrt(3.0 / 5.0)).float()
    if kind == "uniform":
        return ((torch.rand(shape, generator=g, dtype=torch.float64) - 0.5) * math.sqrt(12.0)).float()
    return torch.randn(shape, generator=g)


def tilt_spectrum(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """x [C,T,h,w]: power spectrum multiplied by 1/f^alpha over time and space (alpha > 0 = more energy in the large
    blobs, 'fractal' noise; alpha < 0 = more in the fine grain), then back to zero mean and unit variance per channel."""
    if alpha == 0:
        return x
    C, T, h, w = x.shape
    f = torch.sqrt(torch.fft.fftfreq(T)[:, None, None] ** 2 + torch.fft.fftfreq(h)[None, :, None] ** 2 + torch.fft.rfftfreq(w)[None, None, :] ** 2)
    f[0, 0, 0] = f[f > 0].min()
    y = torch.fft.irfftn(torch.fft.rfftn(x.double(), dim=(1, 2, 3)) * f.pow(-alpha / 2.0), s=(T, h, w), dim=(1, 2, 3))
    y = y - y.mean(dim=(1, 2, 3), keepdim=True)
    return (y / y.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)).to(x.dtype)


def add_pyramid(x: torch.Tensor, amount: float, seed: int, levels: int = 4) -> torch.Tensor:
    """x [C,T,h,w] plus coarser copies of fresh noise (each level half the size, weight amount^level), unit variance."""
    if amount <= 0:
        return x
    C, T, h, w = x.shape
    y = x.double()
    for i in range(1, levels + 1):
        hh, ww = max(1, h >> i), max(1, w >> i)
        if hh == 1 and ww == 1:
            break
        n = torch.randn(C, T, hh, ww, generator=_gen("pyramid", seed, i), dtype=torch.float64)
        y = y + (amount ** i) * torch.nn.functional.interpolate(n, size=(h, w), mode="bilinear", align_corners=False)
    y = y - y.mean(dim=(1, 2, 3), keepdim=True)
    return (y / y.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)).to(x.dtype)


def mix(x: torch.Tensor, signal: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.clamp(0, 1)
    return torch.sqrt(1.0 - weight ** 2) * x + weight * signal


class ShotNoise:
    def __init__(self, seed, base_seed, opts, guide=None, tint=None, mask=None, t0=0):
        self.seed, self.base_seed, self.o, self.guide, self.tint, self.mask, self.t0 = int(seed), int(base_seed), opts, guide, tint, mask, int(t0)

    def generate_noise(self, input_latent):
        samples = input_latent["samples"]
        inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        noise = comfy.sample.prepare_noise(samples, self.seed, inds)            # exactly RandomNoise
        o = self.o
        if not any(o[k] > 0 for k in ("shared_base", "static_share", "guide_structure", "ref_tint", "pyramid")) \
                and o["spectrum_tilt"] == 0 and o["distribution"] == "gaussian":
            return noise
        nested = getattr(noise, "is_nested", False)
        parts = list(noise.unbind()) if nested else [noise]
        v = parts[0]
        if v.ndim != 5:
            return noise
        out = []
        for b in range(v.shape[0]):
            x = v[b].float()
            C, T, h, w = x.shape
            if o["distribution"] != "gaussian":
                x = draw((C, T, h, w), o["distribution"], self.seed + b)
            x = add_pyramid(tilt_spectrum(x, o["spectrum_tilt"]), o["pyramid"], self.seed + b)
            if o["static_share"] > 0:
                s = torch.randn(C, 1, h, w, generator=_gen("static", self.base_seed, h, w))
                x = mix(x, s.expand_as(x), torch.tensor(float(o["static_share"])))
            x = replace_coarse(x, self.base_seed, o["shared_base"], o["base_grid"], o["time_block"], self.t0)
            m = None
            if self.mask is not None:
                m = latent_mask(self.mask, T, h, w, grow=1)[0].to(x)               # [1,T,h,w], 1 = the person
            if o["guide_structure"] > 0 and self.guide is not None:
                g = self.guide[min(b, self.guide.shape[0] - 1)].float().cpu()
                if tuple(g.shape[1:]) != (T, h, w):
                    g = torch.nn.functional.interpolate(g[None], size=(T, h, w), mode="trilinear", align_corners=False)[0]
                g = (g - g.mean(dim=(1, 2, 3), keepdim=True)) / g.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                x = mix(x, g, float(o["guide_structure"]) * ((1.0 - m) if m is not None else torch.ones(1, T, h, w)))
            if o["ref_tint"] > 0 and self.tint is not None:
                x = mix(x, self.tint.view(C, 1, 1, 1).expand_as(x), float(o["ref_tint"]) * (m if m is not None else torch.ones(1, T, h, w)))
            out.append(x.to(v.dtype))
        parts[0] = torch.stack(out)
        if nested:
            return comfy.nested_tensor.NestedTensor(parts)
        return parts[0]


class BFSShotNoise:
    """Starting noise for the sampler with optional structure (see the node description)."""

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda tip, mx=1.0: ("FLOAT", {"default": 0.0, "min": 0.0, "max": mx, "step": 0.01, "tooltip": tip})
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The seed (the base seed when a shot is connected)."}),
                "shot_seed": (BFSShotSeed.MODES, {"default": BFSShotSeed.MODES[0], "tooltip":
                    "Only with a shot connected: the seed of each shot (same / counting up / unrelated but repeatable)."}),
                "shared_base": f("0 = off (plain noise). The coarse pattern of the noise comes from base_seed instead of "
                                 "the shot's seed: every shot, any length or size, agrees on it; the fine detail stays "
                                 "per shot. 1 = fully shared. Try 0.5 - 1."),
                "static_share": f("0 = off. A part of the noise that is identical in every frame (temporally "
                                  "correlated noise). Try 0.1 - 0.3; too much freezes the motion.", 0.9),
                "guide_structure": f("0 = off. Needs guide_frames + vae. Outside the mask the guide's latent is mixed "
                                     "into the noise, so the background starts closer to the guide (without a mask: "
                                     "everywhere, which also pulls toward the original person). Try 0.05 - 0.2.", 0.6),
                "distribution": (DISTRIBUTIONS, {"default": "gaussian", "tooltip":
                    "gaussian = what the model was trained on. laplacian / student-t: more extreme values (stronger, "
                    "sparser kicks). uniform: none. All at unit variance. Like AdvancedNoise's types, but for H3's "
                    "video + audio latent (the audio noise always stays gaussian)."}),
                "spectrum_tilt": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip":
                    "0 = off (white noise). > 0: more energy in the large blobs (fractal / pink noise, AdvancedNoise's "
                    "'fractal' alpha): the coarse layout, where the swap is decided, gets a harder push. < 0: more "
                    "fine grain. Try 0.2 - 0.6; large values burn colours."}),
                "pyramid": f("0 = off. Adds coarser layers of noise (each half the size, weight = this value per "
                             "level): another way to put more energy in the large scales. Try 0.3 - 0.6.", 0.95),
                "ref_tint": f("0 = off. Needs ref_image + vae. Inside the mask (everywhere without one) the noise "
                              "leans toward the reference's average latent, i.e. its colours. Try 0.05 - 0.15.", 0.6),
            },
            "optional": {
                "shot": ("BFS_SHOT", {"tooltip": "From BFS Shot Planner / Unpack: per-shot seeds, and the shared base is "
                                                 "read at the shot's position on the timeline."}),
                "base_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip":
                    "Seed of the shared coarse pattern and of static_share. -1 = the seed above. Put here the seed of "
                    "a result that worked to keep its coarse noise while the detail changes with the seed."}),
                "overrides": ("STRING", {"default": "", "tooltip": "Seeds for single shots: 'shot:seed, shot:seed' (shots count from 1)."}),
                "base_grid": ("INT", {"default": 6, "min": 1, "max": 32, "advanced": True, "tooltip":
                    "Blocks of the coarse pattern along the short side (the long side in proportion). Fewer = only the "
                    "broadest layout is shared."}),
                "time_block": ("INT", {"default": 2, "min": 1, "max": 32, "advanced": True, "tooltip": "Latent frames per block of the coarse pattern."}),
                "vae": ("VAE", {"tooltip": "The video VAE, for guide_structure and ref_tint."}),
                "guide_frames": ("IMAGE", {"tooltip": "The guide video frames (guide_structure)."}),
                "ref_image": ("IMAGE", {"tooltip": "The reference picture (ref_tint)."}),
                "mask": ("MASK", {"tooltip": "The person, one mask per frame (white = the person): guide_structure stays "
                                             "outside it, ref_tint inside it."}),
            },
        }

    RETURN_TYPES = ("NOISE", "INT")
    RETURN_NAMES = ("noise", "seed")
    OUTPUT_TOOLTIPS = ("For SamplerCustomAdvanced.", "The seed this shot used.")
    FUNCTION = "make"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Optional structured starting noise. With every amount at 0 it is exactly RandomNoise. shared_base: "
                   "the coarse noise pattern is the same for every shot / size / length (from base_seed), the detail "
                   "per shot. static_share: a part identical in all frames. guide_structure: the guide's latent mixed "
                   "in outside the mask. ref_tint: lean toward the reference's colours inside the mask. Experimental: "
                   "the model was trained on plain noise, so use small amounts and compare with 0.")

    def make(self, seed, shot_seed, shared_base, static_share, guide_structure, ref_tint, distribution="gaussian", spectrum_tilt=0.0,
             pyramid=0.0, shot=None, base_seed=-1, overrides="",
             base_grid=6, time_block=2, vae=None, guide_frames=None, ref_image=None, mask=None):
        index = int(shot.get("index", 0)) if shot else 0
        s = _pick_seed(int(seed), index, shot_seed, overrides) if shot else int(seed)
        base = int(seed) if int(base_seed) < 0 else int(base_seed)
        guide = tint = None
        if guide_structure > 0:
            if vae is None or guide_frames is None:
                raise ValueError("BFS Shot Noise: guide_structure needs vae and guide_frames")
            guide = vae.encode(guide_frames[..., :3])
            guide = guide.tensors[0] if getattr(guide, "is_nested", False) else guide
        if ref_tint > 0:
            if vae is None or ref_image is None:
                raise ValueError("BFS Shot Noise: ref_tint needs vae and ref_image")
            r = vae.encode(ref_image[:1, ..., :3])
            r = (r.tensors[0] if getattr(r, "is_nested", False) else r).float().cpu()
            tint = (r.mean(dim=(0, 2, 3, 4)) / r.std(dim=(0, 2, 3, 4)).clamp_min(1e-6)).clamp(-2, 2)
        t0 = int(round(int(shot.get("start", 0)) / 4)) if shot else 0          # timeline position in latent frames
        opts = {"shared_base": float(shared_base), "static_share": float(static_share), "guide_structure": float(guide_structure),
                "ref_tint": float(ref_tint), "base_grid": int(base_grid), "time_block": int(time_block),
                "distribution": distribution, "spectrum_tilt": float(spectrum_tilt), "pyramid": float(pyramid)}
        return (ShotNoise(s, base, opts, guide, tint, mask, t0), s)


def sigmas_chart(sigmas: torch.Tensor, other: torch.Tensor | None = None, w: int = 640, h: int = 360):
    """(image [1,h,w,3], text): the noise level at the start of every step, with the zones where the steps are spent."""
    import numpy as np
    from PIL import Image, ImageDraw
    s = [float(v) for v in sigmas.flatten().tolist()]
    top = max(1e-6, s[0])
    img = Image.new("RGB", (w, h), (24, 24, 28)); d = ImageDraw.Draw(img)
    L, R, T, B = 52, w - 16, 34, h - 40
    y = lambda v: B - (v / top) * (B - T)
    zones = ((0.85, 1.01, (60, 40, 40), "layout / identity"), (0.5, 0.85, (48, 46, 36), "shapes"), (-0.01, 0.5, (36, 44, 52), "detail"))
    for lo, hi, col, name in zones:
        d.rectangle([L, y(min(hi, 1.0) * top), R, y(max(lo, 0.0) * top)], fill=col)
        d.text((R - 6 - 6 * len(name), y(min(hi, 1.0) * top) + 3), name, fill=(150, 150, 150))
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        d.line([L, y(v * top), R, y(v * top)], fill=(70, 70, 76)); d.text((8, y(v * top) - 6), f"{v * top:.2f}", fill=(170, 170, 170))

    def line(vals, col, labels):
        n = len(vals) - 1
        pts = [(L + (R - L) * i / max(1, n), y(v)) for i, v in enumerate(vals)]
        d.line(pts, fill=col, width=2)
        for i, (px, py) in enumerate(pts):
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)
            if labels and i < n:
                d.text((px - 10, py - 16 if i % 2 == 0 else py + 6), f"{vals[i]:.2f}", fill=(235, 235, 235))
                d.text((px - 3, B + 8), str(i + 1), fill=(170, 170, 170))

    if other is not None:
        line([float(v) for v in other.flatten().tolist()], (120, 120, 200), False)
    line(s, (255, 190, 70), True)
    steps = s[:-1]
    cnt = [sum(lo * top < v <= hi * top or (hi > 1 and v >= lo * top) for v in steps) for lo, hi, _, _ in zones]
    d.text((L, 10), f"{len(steps)} steps: {cnt[0]} on layout / identity (> {0.85 * top:.2f}), {cnt[1]} on shapes, {cnt[2]} on detail (< {0.5 * top:.2f})"
                    f"   last jump {steps[-1]:.2f} -> {s[-1]:.2f}", fill=(235, 235, 235))
    d.text((R - 24, B + 8), "step", fill=(170, 170, 170))
    text = (f"{len(steps)} steps | layout/identity {cnt[0]}, shapes {cnt[1]}, detail {cnt[2]} | sigmas: " + ", ".join(f"{v:.3f}" for v in s))
    return torch.from_numpy(np.asarray(img).astype("float32") / 255.0)[None], text


class BFSSigmasPreview:
    """Draws a sigma schedule: where the sampler's steps are spent (pass-through)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("SIGMAS", {"tooltip": "The schedule that goes to the sampler (after the scheduler and the shift)."})},
                "optional": {"compare": ("SIGMAS", {"tooltip": "Another schedule drawn behind it, e.g. the same steps with another shift."})}}

    RETURN_TYPES = ("SIGMAS", "IMAGE", "STRING")
    RETURN_NAMES = ("sigmas", "chart", "text")
    OUTPUT_TOOLTIPS = ("The same sigmas, untouched.", "The chart: connect a Preview Image.", "The values and the step count per zone.")
    FUNCTION = "show"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Shows a sigma schedule: the noise level at the start of each step and how many steps fall on layout / "
                   "identity (high noise), shapes and detail (low noise). With H3's video shift of 12 most steps sit at "
                   "the top; lowering the shift moves steps toward detail. Changes nothing: sigmas pass through.")

    def show(self, sigmas, compare=None):
        chart, text = sigmas_chart(sigmas, compare)
        return (sigmas, chart, text)


NODE_CLASS_MAPPINGS = {"BFSShotNoise": BFSShotNoise, "BFSSigmasPreview": BFSSigmasPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSShotNoise": "BFS Shot Noise", "BFSSigmasPreview": "BFS Sigmas Preview"}
