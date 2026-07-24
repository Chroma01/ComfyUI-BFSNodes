"""Color Mask -- recolors a plain white/binary mask (e.g. straight out of SAM2) into the
colored-silhouette or flat-color-marker convention scail2v2 was trained on.

Two styles, matching the two different mask roles in the trained recipe:
  - "silhouette": recolor white->color, keep the full shape. Use for the GUIDE mask slot
    (LTX Multiple Controls' mask_video) -- its shape is what tells the model WHERE/WHAT POSE
    to place the identity, so the shape must be preserved.
  - "flat_marker": recolor AND collapse the whole mask down to a small solid-color dot at its
    centroid, discarding the shape entirely. Use for the IDENTITY mask slot (identity_mask_
    image) -- a full body-shaped silhouette there re-introduces a competing pose signal (this
    is the exact bug flatten_ref_masks.py fixes on the training-data side: the reference's OWN
    held pose bled into the output through a shaped identity mask). The color is the only
    thing that matters there (it's what teaches "this color <-> this identity" correspondence
    for multi-character replacement), not the shape.
"""
import numpy as np
import torch

_COLOR_PRESETS = {
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "custom": None,
}


class LTXColorMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("IMAGE", {"tooltip": "Plain white/binary mask (e.g. from SAM2), one or more frames. "
                                "Any non-black pixel counts as 'inside' the mask, regardless of exact shade."}),
            "color": (list(_COLOR_PRESETS.keys()), {"default": "blue",
                      "tooltip": "Preset color to recolor the mask to (matches scail2v2's per-person palette: "
                                 "blue=person 0, red=person 1, x-sorted left-to-right). Pick 'custom' to use the "
                                 "custom_r/g/b inputs instead."}),
            "style": (["silhouette", "flat_marker"], {"default": "silhouette",
                       "tooltip": "silhouette: recolor, keep the full shape -- use for the GUIDE mask slot "
                                  "(mask_video). flat_marker: recolor AND collapse to a small dot at the mask's "
                                  "centroid -- use for the IDENTITY mask slot (identity_mask_image); a shaped "
                                  "mask there re-injects a competing pose signal (the ref's own held pose)."}),
            "marker_radius_frac": ("FLOAT", {"default": 0.06, "min": 0.01, "max": 0.3, "step": 0.01,
                             "tooltip": "flat_marker only: dot radius as a fraction of min(H, W). Matches the "
                                        "training-side default (flatten_ref_masks.py)."}),
        }, "optional": {
            "custom_r": ("INT", {"default": 255, "min": 0, "max": 255}),
            "custom_g": ("INT", {"default": 0, "min": 0, "max": 255}),
            "custom_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            "background_black": ("BOOLEAN", {"default": True,
                             "tooltip": "On: output background is black (matches guide_mask's white-bg-colored-"
                                        "shape convention actually being colored-on-black internally once VAE-"
                                        "encoded -- for identity_mask specifically, training used a black bg). "
                                        "Off: keep the mask's original background color where nothing was "
                                        "recolored (rare; only if your source mask isn't black/white)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("colored_mask",)
    FUNCTION = "apply"
    CATEGORY = "LTX/identity"
    DESCRIPTION = ("Recolors a plain white/binary SAM2-style mask into scail2v2's trained convention. "
                   "'silhouette' for the guide mask slot, 'flat_marker' (small color dot, no shape) for the "
                   "identity mask slot -- see each input's tooltip for why the two differ.")

    def apply(self, mask, color, style, marker_radius_frac,
              custom_r=255, custom_g=0, custom_b=0, background_black=True):
        if color == "custom":
            rgb = (int(custom_r), int(custom_g), int(custom_b))
        else:
            rgb = _COLOR_PRESETS[color]

        arr = (mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # [N,H,W,C]
        n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
        color_arr = np.array(rgb, dtype=np.uint8)
        out = np.zeros((n, h, w, 3), dtype=np.uint8)

        for i in range(n):
            frame = arr[i]
            inside = frame[..., :3].sum(axis=-1) > 10  # non-black = inside the mask
            if not inside.any():
                continue
            if style == "silhouette":
                out[i][inside] = color_arr
                if not background_black:
                    out[i][~inside] = frame[..., :3][~inside]
            else:  # flat_marker
                ys, xs = np.where(inside)
                cy, cx = int(ys.mean()), int(xs.mean())
                radius = max(2, int(marker_radius_frac * min(h, w)))
                yy, xx = np.ogrid[:h, :w]
                dot = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
                out[i][dot] = color_arr

        out_t = torch.from_numpy(out.astype(np.float32) / 255.0)
        return (out_t,)


NODE_CLASS_MAPPINGS = {"LTXColorMask": LTXColorMask}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXColorMask": "LTX Color Mask"}
