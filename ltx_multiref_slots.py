"""LTX multi-subject references with learned slot tags.

Three nodes:

* **LTX Reference Images** — collects up to six reference images into one ordered list, so the
  slot order is visible in one place and references of different sizes travel together.
* **LTX Multi Reference Slots** — up to six reference images, each becoming its own reference
  block. Every block gets a distinct ``source_id`` (rotary phase) and, when the checkpoint
  carries one, the trained per-slot embedding added in feature space.
* **LTX Reference Tags** — builds the ``<Image 1> is the ...`` prompt prefix that binds each tag
  to its slot.

The prompt tags are the part people get wrong. Nothing in the model ties ``<Image 1>`` to slot 1:
that correspondence exists only because the training captions used it consistently. Sampling with
a different convention than training used simply does not bind, however well the slots are
separated — hence the second node, so the prompt is built the same way every time.

The slot embedding lives in the LoRA's own safetensors under
``diffusion_model.reference_slot_embedding.*`` with its hyperparameters in the file metadata, so
it is read straight from the LoRA file: ComfyUI's LoRA loader only applies LoRA weights and drops
everything else, which would silently sample the checkpoint without a signal it was trained on.
"""

import math

import torch

MAX_REFERENCES = 6
_PREFIX = "diffusion_model.reference_slot_embedding."


class _SlotEmbedding(torch.nn.Module):
    """Fourier features of the slot index through a small MLP. Mirrors the trainer's module —
    same parameter names and shapes, so a checkpoint loads without remapping."""

    def __init__(self, token_dim=128, num_frequencies=16, hidden_dim=256):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1 + 2 * num_frequencies, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, slot_index):
        index = torch.tensor([float(slot_index)], dtype=torch.float32, device=self.frequencies.device)
        scaled = index * self.frequencies
        features = torch.cat([index, torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.net(features.to(self.net[0].weight.dtype))


def _load_slot_embedding(lora_name):
    """Rebuild the slot embedding from a LoRA file, or None when it carries none."""
    import folder_paths
    from safetensors import safe_open

    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise ValueError(f"LoRA not found: {lora_name}")

    with safe_open(path, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith(_PREFIX)]
        if not keys:
            return None
        metadata = f.metadata() or {}
        state = {k[len(_PREFIX):]: f.get_tensor(k).float() for k in keys}

    # Prefer the file's own metadata; fall back to the shapes when it is absent.
    hidden_dim = int(metadata.get("reference_slot_embedding_hidden_dim", state["net.0.weight"].shape[0]))
    num_frequencies = int(metadata.get("reference_slot_embedding_num_frequencies", 0)) or (
        (state["net.0.weight"].shape[1] - 1) // 2
    )
    token_dim = int(metadata.get("reference_slot_embedding_dim", state["net.2.weight"].shape[0]))

    module = _SlotEmbedding(token_dim=token_dim, num_frequencies=num_frequencies, hidden_dim=hidden_dim)
    module.load_state_dict(state)
    module.eval()
    return module


class LTXReferenceImages:
    """Collect reference images into an ordered list. Images only — nothing else wired in.

    A plain IMAGE batch cannot carry references of different sizes (torch stacking needs one
    shape), and subject references routinely differ: a portrait crop next to a landscape product
    shot. Emitting a list keeps each one at its own size, and keeps the slot order explicit and
    visible on one node instead of spread across the conditioning node's inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                f"image_{i}": (
                    "IMAGE",
                    {"tooltip": f"Reference for slot {i} — the subject your prompt calls <Image {i}>. "
                                f"Leave unconnected to skip; connected slots renumber in order, so "
                                f"wiring 1 and 3 makes them <Image 1> and <Image 2>."},
                )
                for i in range(1, MAX_REFERENCES + 1)
            },
        }

    RETURN_TYPES = ("LTX_REFS", "INT")
    RETURN_NAMES = ("references", "count")
    FUNCTION = "collect"
    CATEGORY = "BFS/LTX"
    DESCRIPTION = (
        "Collects up to six reference images, in slot order, for LTX Multi Reference Slots. "
        "Images of different sizes are fine — each keeps its own."
    )

    def collect(self, **images):
        refs = [
            images[f"image_{i}"]
            for i in range(1, MAX_REFERENCES + 1)
            if images.get(f"image_{i}") is not None
        ]
        return (refs, len(refs))


class LTXReferenceTags:
    """Build the prompt prefix that binds each tag to its reference slot."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"subject_{i}": (
                "STRING",
                {"default": "", "multiline": False,
                 "tooltip": f"What reference image {i} is, as a short noun phrase — 'the woman', "
                            f"'the red sports car'. Becomes '<Image {i}> is <this>.'. Leave empty to skip."},
            )
            for i in range(1, MAX_REFERENCES + 1)
        }
        optional["scene"] = (
            "STRING",
            {"default": "", "multiline": True,
             "tooltip": "The scene description, appended after the declarations. Refer back to the "
                        "subjects by tag here (\"<Image 1> hands <Image 2> a cup\") — a tag that only "
                        "ever appears in the declaration has to carry both jobs at once, which is a "
                        "harder association for the model to use."},
        )
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "BFS/LTX"
    DESCRIPTION = (
        "Assembles '<Image 1> is the woman. <Image 2> is the man. <scene>'. Order matters: tag N "
        "must describe the image wired into reference_N on the slots node, and must match the "
        "convention the checkpoint was trained with. Describe the subjects DISTINCTLY — two "
        "references both called 'the woman' carry no information about which is which."
    )

    def build(self, scene="", **subjects):
        parts = []
        for i in range(1, MAX_REFERENCES + 1):
            text = (subjects.get(f"subject_{i}") or "").strip().rstrip(".")
            if text:
                parts.append(f"<Image {i}> is {text}.")
        scene = (scene or "").strip()
        if scene:
            parts.append(scene)
        return (" ".join(parts),)


class LTXMultiReferenceSlots:
    """N reference images, each on its own slot: distinct rotary phase plus the trained tag."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        optional = {}
        for i in range(1, MAX_REFERENCES + 1):
            optional[f"reference_{i}"] = (
                "IMAGE",
                {"tooltip": f"Reference image for slot {i} — the subject your prompt calls "
                            f"<Image {i}>. Leave unconnected to skip; connected slots are numbered "
                            f"in order, so wiring 1 and 3 gives them source ids 1 and 2."},
            )
        optional["references"] = (
            "LTX_REFS",
            {"tooltip": "Output of LTX Reference Images. Use this OR the individual reference_N "
                        "slots below — the list wins if both are connected."},
        )
        optional["slot_embedding_lora"] = (
            ["none"] + folder_paths.get_filename_list("loras"),
            {"default": "none",
             "tooltip": "The LoRA file to read the trained slot embedding from. Point this at the "
                        "SAME LoRA you load for sampling. ComfyUI's LoRA loader applies only LoRA "
                        "weights and drops the slot embedding, so without this the references are "
                        "untagged and the checkpoint is sampled without a signal it was trained on. "
                        "'none' = phase separation only (older checkpoints)."},
        )
        optional["start_source_id"] = (
            "INT",
            {"default": 1, "min": 0, "max": 8,
             "tooltip": "Source id of the first connected reference; the rest count up. The target "
                        "is always 0. Start at 1 unless the checkpoint was trained otherwise."},
        )
        optional["phase_scale"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1})
        optional["slot_strength"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
             "tooltip": "Scales the learned slot tag. The tag and the adapter were trained "
                        "together at strength 1.0, so raising the LoRA on the loader without "
                        "raising this leaves them out of proportion — the adapter shouts while "
                        "the tag that tells the references apart stays quiet. Match this to the "
                        "LoRA strength you set on the loader (an undertrained LoRA often wants "
                        "1.2-1.5, and the tag wants the same)."},
        )
        optional["layout"] = (
            ["overlap", "st_drc", "strata"],
            {"default": "overlap",
             "tooltip": "Where the references sit in the RoPE grid. Must match training."},
        )
        optional["ref_resize_mode"] = (
            ["match_target", "match_target_letterbox", "native_resolution"],
            {"default": "match_target_letterbox",
             "tooltip": "Letterbox is the safe default for subject references: it keeps the whole "
                        "image, where match_target centre-crops and can cut a subject in half when "
                        "the aspect ratios differ."},
        )
        optional["reference_temporal_offset_latents"] = ("INT", {"default": 0, "min": -8, "max": 8})
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "info")
    FUNCTION = "apply"
    CATEGORY = "BFS/LTX"
    DESCRIPTION = (
        "Attaches up to six reference images as separate reference blocks, each with its own "
        "source id and, when the LoRA carries one, its trained slot embedding. Pair with LTX "
        "Reference Tags so the prompt names the slots the way training did."
    )

    def apply(self, model, positive, negative, vae, latent, references=None,
              slot_embedding_lora="none", start_source_id=1, phase_scale=1.0, slot_strength=1.0,
              layout="overlap", ref_resize_mode="match_target_letterbox",
              reference_temporal_offset_latents=0, **slots):
        from .ltx_identity_overlap import _find_ltxv, _install_patches
        from .ltx_multiple_controls import _encode_ref

        images = list(references) if references else [
            slots[f"reference_{i}"]
            for i in range(1, MAX_REFERENCES + 1)
            if slots.get(f"reference_{i}") is not None
        ]
        if not images:
            return (model, positive, negative, latent, "no references connected — pass-through")

        slot_module = None
        if slot_embedding_lora != "none":
            slot_module = _load_slot_embedding(slot_embedding_lora)
            if slot_module is None:
                raise ValueError(
                    f"{slot_embedding_lora} carries no slot embedding "
                    f"(no '{_PREFIX}*' keys). Set slot_embedding_lora to 'none' if this "
                    "checkpoint was trained with phase separation only."
                )

        m = model.clone()
        ltxv = _find_ltxv(m)
        w_sf, h_sf = ltxv.vae_scale_factors[2], ltxv.vae_scale_factors[1]

        ref_specs = []
        lines = []
        for index, img in enumerate(images):
            source_id = int(start_source_id) + index
            ref_lat, _, _, _, _, _ = _encode_ref(
                vae, latent, img, ref_resize_mode, "center", w_sf, h_sf
            )
            spec = {
                "latent": ref_lat,
                "seg_value": float(source_id) * float(phase_scale),
                "layout": layout,
                "strata_slot": index,
                "temporal_offset_latents": int(reference_temporal_offset_latents),
            }
            if slot_module is not None:
                with torch.no_grad():
                    spec["slot_vector"] = slot_module(source_id).detach() * float(slot_strength)
            ref_specs.append(spec)
            lines.append(f"<Image {index + 1}> -> source_id {source_id}")

        _install_patches(ltxv)
        ltxv._id_rope_theta = 10000.0
        m.model_options = dict(m.model_options)
        transformer_options = dict(m.model_options.get("transformer_options", {}))
        transformer_options["_id_ref_specs"] = ref_specs
        m.model_options["transformer_options"] = transformer_options

        tag = (f"trained slot embedding x{slot_strength:g}" if slot_module is not None
               else "phase only (no slot embedding)")
        info = f"{len(images)} references, {layout}, {tag}\n" + "\n".join(lines)
        return (m, positive, negative, latent, info)


NODE_CLASS_MAPPINGS = {
    "LTXReferenceImages": LTXReferenceImages,
    "LTXMultiReferenceSlots": LTXMultiReferenceSlots,
    "LTXReferenceTags": LTXReferenceTags,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXReferenceImages": "LTX Reference Images",
    "LTXMultiReferenceSlots": "LTX Multi Reference Slots",
    "LTXReferenceTags": "LTX Reference Tags",
}
