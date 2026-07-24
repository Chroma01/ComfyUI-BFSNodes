"""LTX Multiple Controls — single-node version of chaining several LTXIdentityOverlapConditioning
(LTX Identity Transfer) calls for a guide (motion/structure) + mask (replacement region) +
identity (appearance) recipe, e.g. character-replace-in-driving-video training.

WHY THIS NODE EXISTS: LTXIdentityOverlapConditioning stores its reference specs in
model.model_options["transformer_options"]["_id_ref_specs"], and each call OVERWRITES that key
with its own specs list rather than merging with whatever the upstream node already put there.
Chaining N of those nodes therefore only keeps the LAST node's reference active at inference —
the earlier ones (e.g. the guide) silently have ZERO effect despite the graph looking correct.
This node builds every slot's ref_specs in ONE apply() call so they all end up in the same list.

Each slot (guide/mask/identity/identity_mask) is fully independent: its own source_id,
phase_scale, layout, ref_resize_mode -- no slot is hardcoded to a fixed layout/phase, only
given sane defaults that match the trained recipe (guide/mask: source_id=0 => no RoPE phase
tag, same as source_phase=false at train time; identity: source_id=2 => tagged, same as
source_phase=true). All four slots are optional -- leave any input unconnected to skip that
slot entirely.

identity_mask is the scail2v2-style color-pointer condition (a small flat color dot/blob, NOT
a body silhouette -- see identity_mask_image's tooltip for why body-shaped masks re-introduce
a competing pose signal). Defaults to inheriting identity's own source_id/phase_scale (they're
trained as one group); override identity_mask_source_id to split them into separate groups if
your checkpoint used a different grouping (e.g. scail2v2's 3-phase variant: guide=1, mask=2,
identity+identity_mask=3, all source_phase as noted above).
"""
import logging

import torch

from .ltx_identity_overlap import (
    _anchored_crop_resize,
    _draw_crop_overlay,
    _find_ltxv,
    _install_patches,
    _letterbox_resize,
)
from . import ltx_identity_overlap as _ido

log = logging.getLogger("LTXMultipleControls")

_LAYOUT_CHOICES = ["overlap", "st_drc", "strata"]
_RESIZE_CHOICES = ["match_target", "match_target_letterbox", "native_resolution"]


def _encode_ref(vae, latent, img, ref_resize_mode, crop_anchor, w_sf, h_sf, downscale_factor=1):
    """Resize (per ref_resize_mode/crop_anchor) + VAE-encode ONE reference IMAGE batch
    ([N,H,W,C], typically a whole video's frames). Mirrors LTXIdentityOverlapConditioning's
    own _encode_one so behavior is byte-identical per slot.

    downscale_factor>1: encode at target_size/downscale_factor instead of full target size
    (mirrors the trainer's reference_downscale_factor -- fewer tokens/cheaper VRAM for a
    condition that doesn't need full detail, e.g. guide/mask). Snapped to a multiple of
    w_sf/h_sf (must stay divisible by the VAE's own spatial downscale, same constraint as
    training's "must be divisible by 32" check)."""
    import comfy.utils

    if ref_resize_mode == "native_resolution":
        _, src_h, src_w, _ = img.shape
        tgt_w = max(w_sf, round(src_w / w_sf) * w_sf)
        tgt_h = max(h_sf, round(src_h / h_sf) * h_sf)
    else:
        _, _, _, lat_h, lat_w = latent["samples"].shape
        tgt_w, tgt_h = lat_w * w_sf, lat_h * h_sf
    if downscale_factor != 1:
        tgt_w = max(w_sf, round(tgt_w / downscale_factor / w_sf) * w_sf)
        tgt_h = max(h_sf, round(tgt_h / downscale_factor / h_sf) * h_sf)
    _, src_h0, src_w0, _ = img.shape
    crop_box = (0, 0, src_w0, src_h0)
    if ref_resize_mode == "match_target_letterbox":
        ref_px = _letterbox_resize(img, tgt_w, tgt_h)[:, :, :, :3]
    elif ref_resize_mode == "match_target" and crop_anchor != "center":
        ref_px, crop_box = _anchored_crop_resize(img, tgt_w, tgt_h, anchor=crop_anchor)
        ref_px = ref_px[:, :, :, :3]
    else:
        ref_px = comfy.utils.common_upscale(img.movedim(-1, 1), tgt_w, tgt_h, "bilinear", "center").movedim(1, -1)[:, :, :, :3]
        if ref_resize_mode == "match_target":
            _, crop_box = _anchored_crop_resize(img[:1], tgt_w, tgt_h, anchor="center")
    ref_lat = vae.encode(ref_px)
    overlay = _draw_crop_overlay(img[:1], crop_box)
    return ref_lat, ref_px[:1].clone(), overlay, crop_box, src_w0, src_h0


class LTXMultipleControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "latent": ("LATENT",),
        }, "optional": {
            "guide_video": ("IMAGE", {"tooltip": "Motion/structure driving video frames (IMAGE batch, e.g. from "
                             "GetVideoComponents). Leave unconnected to skip the guide slot."}),
            "guide_source_id": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 1.0,
                             "tooltip": "scail2v2 3-phase recipe default: 1. 0 = no RoPE phase tag at all "
                                        "(older 2-group recipes); either way this must stay source_phase=false "
                                        "(distinct id from identity) for the guide's positions to line up with "
                                        "the target frame-by-frame."}),
            "guide_phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            "guide_layout": (_LAYOUT_CHOICES, {"default": "overlap"}),
            "guide_ref_resize_mode": (_RESIZE_CHOICES, {"default": "match_target"}),
            "guide_downscale_factor": ("INT", {"default": 1, "min": 1, "max": 8,
                             "tooltip": "Encode at target_size/N instead of full res (cheaper VRAM). Must match "
                                        "whatever the checkpoint was trained with -- 1 for the no-downscale "
                                        "recipes, 2 for the earlier downscaled-guide recipes. Wrong value = "
                                        "positions the model never learned, not just a quality hit."}),

            "mask_video": ("IMAGE", {"tooltip": "Per-frame replacement-region mask, same frame count/alignment "
                             "as guide_video -- scail2v2 expects a COLORED silhouette (see the Color Mask node), "
                             "not plain white/binary. Leave unconnected to skip."}),
            "mask_source_id": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0,
                             "tooltip": "scail2v2 3-phase recipe default: 2 (own group, separate from guide's 1 "
                                        "and identity's 3 -- older 2-group recipes shared this with guide at 0)."}),
            "mask_phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            "mask_layout": (_LAYOUT_CHOICES, {"default": "overlap"}),
            "mask_ref_resize_mode": (_RESIZE_CHOICES, {"default": "match_target"}),
            "mask_downscale_factor": ("INT", {"default": 1, "min": 1, "max": 8,
                             "tooltip": "Same as guide_downscale_factor -- match the checkpoint's training recipe."}),

            "identity_image": ("IMAGE", {"tooltip": "Appearance reference (face/character), no positional "
                             "correspondence with the target needed. Leave unconnected to skip."}),
            "identity_source_id": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 8.0, "step": 1.0,
                             "tooltip": "scail2v2 3-phase recipe default: 3. Nonzero = tagged with its own RoPE "
                                        "phase (matches source_phase=true)."}),
            "identity_phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            "identity_layout": (_LAYOUT_CHOICES, {"default": "overlap"}),
            "identity_ref_resize_mode": (_RESIZE_CHOICES, {"default": "native_resolution"}),
            "identity_downscale_factor": ("INT", {"default": 1, "min": 1, "max": 8,
                             "tooltip": "Same as guide_downscale_factor -- match the checkpoint's training recipe."}),

            "identity_mask_image": ("IMAGE", {"tooltip": "scail2-style color-pointer marker for the identity "
                             "slot (small flat color dot/blob, NOT a body silhouette -- a body-shaped mask here "
                             "would re-inject a competing pose signal, exactly the bug the flat-marker design "
                             "fixes on the training side). Same source_id/phase as identity_image by default "
                             "(they're trained as one group) -- override identity_mask_source_id if your "
                             "checkpoint used a different grouping. Leave unconnected to skip."}),
            "identity_mask_source_id": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 8.0, "step": 1.0,
                             "tooltip": "-1 = inherit identity_source_id/identity_phase_scale (matches training, "
                             "where ref+ref_mask share one group). Set >=0 to give it its own separate phase."}),
            "identity_mask_phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            "identity_mask_layout": (_LAYOUT_CHOICES, {"default": "overlap"}),
            "identity_mask_ref_resize_mode": (_RESIZE_CHOICES, {"default": "native_resolution"}),
            "identity_mask_downscale_factor": ("INT", {"default": -1, "min": -1, "max": 8,
                             "tooltip": "-1 = inherit identity_downscale_factor. Same meaning otherwise as "
                                        "guide_downscale_factor -- match the checkpoint's training recipe."}),

            "crop_anchor": (["center", "top", "bottom", "left", "right"], {"default": "center",
                             "tooltip": "Shared by all slots using ref_resize_mode=match_target with a mismatched "
                                        "aspect ratio -- which part of the source survives the crop."}),
            "reference_guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1,
                             "tooltip": "ST-DRC-style reference-CFG applied to the WHOLE combined reference set "
                                        "(all active slots together). 1.0 = off."}),
            "debug_log": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "debug")
    FUNCTION = "apply"
    CATEGORY = "LTX/identity"
    DESCRIPTION = ("Guide + mask + identity + identity_mask reference conditioning in ONE node call, so all "
                   "active slots' RoPE reference specs actually combine (chaining LTXIdentityOverlapConditioning "
                   "nodes does NOT -- each call overwrites the previous one's specs). Every slot is optional and "
                   "its own source_id/phase_scale/layout/ref_resize_mode -- nothing is hardcoded, only defaulted "
                   "to match the trained recipe (guide/mask: source_id=0 i.e. no phase; identity: source_id=2; "
                   "identity_mask: inherits identity's grouping by default, or scail2v2's 3-phase recipe: "
                   "guide=1, mask=2, identity+identity_mask=3).")

    def apply(self, model, positive, negative, vae, latent,
              guide_video=None, guide_source_id=0.0, guide_phase_scale=1.0,
              guide_layout="overlap", guide_ref_resize_mode="match_target", guide_downscale_factor=1,
              mask_video=None, mask_source_id=0.0, mask_phase_scale=1.0,
              mask_layout="overlap", mask_ref_resize_mode="match_target", mask_downscale_factor=1,
              identity_image=None, identity_source_id=2.0, identity_phase_scale=1.0,
              identity_layout="overlap", identity_ref_resize_mode="native_resolution", identity_downscale_factor=1,
              identity_mask_image=None, identity_mask_source_id=-1.0, identity_mask_phase_scale=1.0,
              identity_mask_layout="overlap", identity_mask_ref_resize_mode="native_resolution",
              identity_mask_downscale_factor=-1,
              crop_anchor="center", reference_guidance_scale=1.0, debug_log=False):
        import comfy.samplers

        _ido._DEBUG_ENABLED = bool(debug_log)
        m = model.clone()
        ltxv = _find_ltxv(m)
        _, w_sf, h_sf = vae.downscale_index_formula

        # -1 = inherit the identity slot's own grouping (matches training: ref+ref_mask share
        # one source_id/phase_scale by default).
        if identity_mask_source_id < 0:
            identity_mask_source_id = identity_source_id
            identity_mask_phase_scale = identity_phase_scale
        if identity_mask_downscale_factor < 0:
            identity_mask_downscale_factor = identity_downscale_factor

        slots = [
            ("guide", guide_video, guide_source_id, guide_phase_scale, guide_layout, guide_ref_resize_mode, guide_downscale_factor),
            ("mask", mask_video, mask_source_id, mask_phase_scale, mask_layout, mask_ref_resize_mode, mask_downscale_factor),
            ("identity", identity_image, identity_source_id, identity_phase_scale, identity_layout, identity_ref_resize_mode, identity_downscale_factor),
            ("identity_mask", identity_mask_image, identity_mask_source_id, identity_mask_phase_scale,
             identity_mask_layout, identity_mask_ref_resize_mode, identity_mask_downscale_factor),
        ]

        ref_specs = []
        summary = []
        for name, img, source_id, phase_scale, layout, resize_mode, downscale_factor in slots:
            if img is None:
                continue
            ref_lat, _px, _overlay, crop_box, src_w0, src_h0 = _encode_ref(
                vae, latent, img, resize_mode, crop_anchor, w_sf, h_sf, downscale_factor=downscale_factor)
            seg_value = float(source_id) * float(phase_scale)
            ref_specs.append({"latent": ref_lat, "seg_value": seg_value, "layout": layout,
                               "strata_slot": len(ref_specs), "downscale_factor": downscale_factor})
            dsf_note = f", downscale={downscale_factor}x" if downscale_factor != 1 else ""
            summary.append(f"{name}: {img.shape[0]}f {src_w0}x{src_h0}px -> {layout}, seg={seg_value:g}, mode={resize_mode}{dsf_note}")

        if not ref_specs:
            log.warning("LTXMultipleControls: no slots connected -- passing through unchanged.")
            return (m, positive, negative, latent, "LTX Multiple Controls: no active slots (nothing connected).")

        _install_patches(ltxv)
        ltxv._id_rope_theta = 10000.0
        m.model_options = dict(m.model_options)
        to = dict(m.model_options.get("transformer_options", {}))
        to["_id_ref_specs"] = ref_specs
        m.model_options["transformer_options"] = to

        if reference_guidance_scale != 1.0:
            noref_to = dict(to)
            noref_to.pop("_id_ref_specs", None)
            ref_scale = float(reference_guidance_scale)

            def _ref_cfg_function(args):
                cond = args["cond"]
                uncond = args["uncond"]
                cond_scale = args["cond_scale"]
                denoised = uncond + (cond - uncond) * cond_scale
                noref_model_options = dict(args["model_options"])
                noref_model_options["transformer_options"] = noref_to
                (noref_pred,) = comfy.samplers.calc_cond_batch(
                    args["model"], [args["input_cond"]], args["input"], args["timestep"], noref_model_options,
                )
                noref_denoised = args["input"] - noref_pred
                denoised = denoised + (ref_scale - 1.0) * (cond - noref_denoised)
                return denoised

            m.set_model_sampler_cfg_function(_ref_cfg_function, disable_cfg1_optimization=True)

        dbg = (
            "=== LTX Multiple Controls ===\n"
            f"{len(ref_specs)} active slot(s):\n  " + "\n  ".join(summary) + "\n"
            f"reference-CFG: {'off' if reference_guidance_scale == 1.0 else f'ON, scale={reference_guidance_scale}'}\n"
            "Set LTX_IDOVERLAP_DEBUG=1 for per-step shape logs. Connect negative + CFG 3-5, no LightX2V."
        )
        log.info("\n" + dbg)
        return (m, positive, negative, latent, dbg)


NODE_CLASS_MAPPINGS = {"LTXMultipleControls": LTXMultipleControls}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXMultipleControls": "LTX Multiple Controls"}
