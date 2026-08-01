"""LTX-2.3 Multishot — prompt builder + ordered reference batch for the shotplan strata LoRA.

The multishot LoRA (`multishot_strata_r128_v1`, trained on Pensioner/shotplan keyframes) takes
ONE text prompt for the whole clip, in the ShotPlan caption format:

    <global preamble: setting, palette, lighting, camera grammar, mood>
    Shot 1: <shot size> <camera move> <what happens>
    Shot 2: ...

plus ONE reference keyframe per shot, stacked as strata slots. This node assembles both from a
per-shot UI so you don't hand-write the format.

WIRING (no dedicated conditioning node is needed -- LTX Identity Transfer already does the right
thing for this checkpoint):

    references (IMAGE batch) -> LTX Identity Transfer  with layout="strata", source_id=2,
                                                            phase_scale=1.0
    prompt (STRING)          -> your LTX text encode   -> positive

LTX Identity Transfer builds each ref's spec as
``seg_value = (source_id + i) * phase_scale`` and ``strata_slot = i`` over the image batch, so
source_id=2 over 4 images reproduces the trained convention exactly: source_id 2/3/4/5 on strata
slots 0/1/2/3. The batch ORDER IS THE SHOT ORDER -- the LoRA was trained with
`shuffle_reference_slots: false`, so slot N means shot N and nothing else.

Shots are COMPACTED, not positional: if you fill image_1 and image_3 and leave image_2 empty,
you get "Shot 1"/"Shot 2" on strata slots 0/1. Slot 0 must always be the opening shot -- the LoRA
never saw a gap in the middle of the slot sequence.

Unfilled shots are padded with a flat mid-gray frame, matching the `blank.jpg` (RGB 128)
placeholder the dataset builder used for samples with fewer than 4 shots -- the model was trained
to read that as "this slot carries no information".

The dropdown vocabularies are the terms actually present in the 17,525 shot headers of
`train_meta_16fps.json`, ordered by frequency (close-up 32%, medium shot 26%, medium close-up 20%
...; static 65%, pan 16%, tracking 9.5% ...). Terms outside these lists are out of distribution
for the LoRA.

Note the text encoder truncates at 1024 tokens and real ShotPlan captions already run long at 3-4
shots -- keep each action line to a sentence or two, or the last shot's description gets cut.
"""
import logging

import torch

log = logging.getLogger("LTXMultishotPrompt")

MAX_SHOTS = 4

# Vocabularies EXTRACTED from the 12,470 `Shot N:` headers of Pensioner/shotplan
# train_meta_16fps.json -- the header is the span between "Shot N:" and the start of the action
# sentence, and these lists cover 98% (size) / 97% (move) of them. Percentages are of all headers.
#
# Do NOT count these terms over the whole caption: the action text says things like "A close-up
# of the young man", which inflates close-up ~7x if the window isn't cut at the action boundary.
SHOT_SIZES = [
    "Medium shot",        # 37.4%
    "Medium close-up",    # 27.3%
    "Wide shot",          #  9.0%
    "Extreme close-up",   #  8.0%
    "Medium long shot",   #  7.4%
    "Close-up",           #  4.5%
    "Long shot",          #  1.7%
    "Medium wide shot",   #  1.1%
    "Full shot",          #  0.2%
    "(auto)",
]
# Modifiers COMBINE with a shot size ("over-the-shoulder close-up static"); they are not
# alternatives to one. Emitted before the size, matching the most frequent header forms.
SHOT_MODIFIERS = [
    "(none)",
    "Over-the-shoulder",  # 4.8%
    "Low angle",          # 3.1%
    "High angle",         # 2.9%
    "POV",                # 0.2%
    "Overhead",           # 0.1%
]
CAMERA_MOVES = [
    "Static",                   # 55.1%
    "Tracking",                 # 10.1%
    "Pan",                      #  6.6%
    "Static / locked-off",      #  5.1%
    "Pan right",                #  4.2%
    "Pan left",                 #  3.6%
    "Handheld",                 #  3.3%
    "Slow push-in",             #  2.7%
    "Tilt down",                #  1.4%
    "Dolly-in",                 #  1.0%
    "Static with slight pan",   #  0.8%
    "Tracking / follow",        #  0.8%
    "Tilt up",                  #  0.7%
    "(auto)",
]


def _gray_like(ref: torch.Tensor) -> torch.Tensor:
    """A single mid-gray frame shaped like `ref` ([1,H,W,C]) -- the dataset's blank.jpg is a
    flat RGB 128, i.e. 0.5 in ComfyUI's 0-1 float IMAGE convention."""
    return torch.full((1, ref.shape[1], ref.shape[2], ref.shape[3]), 0.5,
                      dtype=ref.dtype, device=ref.device)


def _resize_to(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinear-resize an IMAGE [N,H,W,C] to h x w so every slot can be stacked into one batch."""
    if img.shape[1] == h and img.shape[2] == w:
        return img
    import comfy.utils
    return comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)


class LTXMultishotPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "preamble": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": "Describes the WHOLE sequence -- setting, colour palette, lighting, "
                           "overall camera grammar, mood. This is the paragraph that precedes "
                           "'Shot 1:' in the training captions. Leave empty to emit only the "
                           "shot blocks."}),
        }
        optional = {}
        for i in range(1, MAX_SHOTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": f"Reference keyframe for shot {i} (strata slot {i - 1}, source_id "
                           f"{i + 1}). Leave unconnected to end the sequence -- the slot is "
                           f"padded with the trained mid-gray blank."})
            optional[f"shot_modifier_{i}"] = (SHOT_MODIFIERS, {"default": "(none)",
                "tooltip": "Combines WITH the shot size (e.g. 'Over-the-shoulder' + 'Close-up' "
                           "-> 'Over-the-shoulder close-up'). Not a size on its own."})
            optional[f"shot_size_{i}"] = (SHOT_SIZES, {"default": "Medium shot"})
            optional[f"camera_move_{i}"] = (CAMERA_MOVES, {"default": "Static"})
            optional[f"action_{i}"] = ("STRING", {
                "multiline": True, "default": "",
                "tooltip": f"What happens in shot {i} -- subject, wardrobe, action, framing "
                           f"details. One or two sentences; the encoder caps the whole prompt "
                           f"at 1024 tokens."})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("prompt", "references", "num_shots")
    FUNCTION = "build"
    CATEGORY = "BFS/video"
    DESCRIPTION = ("Builds a ShotPlan-format multishot prompt plus the ordered, gray-padded "
                   "reference batch for the multishot strata LoRA. Feed `references` into LTX "
                   "Identity Transfer with layout='strata', source_id=2, phase_scale=1.0, and "
                   "`prompt` into your LTX text encode.")

    def build(self, preamble, **kw):
        shots, images = [], []
        for i in range(1, MAX_SHOTS + 1):
            img = kw.get(f"image_{i}")
            action = (kw.get(f"action_{i}") or "").strip()
            if img is None and not action:
                continue
            mod = kw.get(f"shot_modifier_{i}", "(none)")
            size = kw.get(f"shot_size_{i}", "(auto)")
            move = kw.get(f"camera_move_{i}", "(auto)")
            # Dataset form: "<modifier> <size> <move>", e.g. "Over-the-shoulder close-up static".
            # The size is lowercased after a modifier so the header reads as one phrase.
            if mod and mod != "(none)" and size != "(auto)":
                size = f"{mod} {size[0].lower()}{size[1:]}"
            elif mod and mod != "(none)":
                size = mod
            header = " ".join(p for p in (size, move) if p and p != "(auto)")
            body = " ".join(p for p in (header, action) if p)
            shots.append(f"Shot {len(shots) + 1}: {body}".rstrip())
            images.append(img)

        if not shots:
            raise ValueError(
                "LTX Multishot Prompt: connect at least one image or write at least one action.")

        parts = ([preamble.strip()] if preamble.strip() else []) + shots
        prompt = " ".join(parts)

        # Every slot must share one resolution to stack into a single IMAGE batch; the first
        # connected reference sets it (that is the shot the model opens on).
        first = next((im for im in images if im is not None), None)
        if first is None:
            raise ValueError(
                "LTX Multishot Prompt: at least one image_N must be connected -- the LoRA is "
                "conditioned on keyframes, an action-only prompt has nothing to anchor to.")
        h, w = first.shape[1], first.shape[2]

        batch = []
        for im in images:
            batch.append(_gray_like(first) if im is None else _resize_to(im[:1], h, w))
        # Pad the unused slots: the LoRA always sees 4 strata slots, blank == "no information".
        while len(batch) < MAX_SHOTS:
            batch.append(_gray_like(first))

        references = torch.cat(batch, dim=0)
        log.info("[LTXMultishotPrompt] %d shot(s), refs %s, prompt %d chars",
                 len(shots), tuple(references.shape), len(prompt))
        return (prompt, references, len(shots))


NODE_CLASS_MAPPINGS = {"LTXMultishotPrompt": LTXMultishotPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXMultishotPrompt": "LTX Multishot Prompt + Refs"}
