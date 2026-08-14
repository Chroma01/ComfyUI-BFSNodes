# ComfyUI-BFSNodes

This repository includes general video composition helpers and the **LTXV Edit Anything** nodes required by Edit Anything LoRAs trained for LTX-Video 2.3 style `video_to_video_ref_adaln` workflows. The first public reference LoRA is expected to be `edit_anything_reference_v0.1_r128_12000.safetensors`.

## Nodes

### LTXV Edit Anything (Apply)

A unified conditioning node for LTXV Edit Anything LoRAs. It combines three conditioning paths from one LoRA checkpoint:

- **IC-LoRA sequence conditioning**: appends a reference image and optional guide video frames as clean conditioning frames.
- **Role embedding**: injects the learned reference-role token embedding before `patchify_proj`, when the LoRA contains role embedding weights.
- **AdaLN reference conditioning**: pools the encoded reference image and injects a global appearance/style condition into the LTXV timestep path, when the LoRA contains the AdaLN projector weights.

Use this node when a LoRA was trained with the Edit Anything / `video_to_video_ref_adaln` strategy and requires reference-image conditioning beyond a standard LoRA load.

### LTXV Apply Neutral Mask

Replaces masked-out regions in an image or video batch with a clean solid background (`white`, `neutral_gray`, or `black`). This is useful for preparing reference images or guide frames before passing them to Edit Anything conditioning.

### LTXV Resize Reference By Mask

Resizes a reference object onto a clean canvas using a mask bounding box as the target object size. This helps match the scale expected by the guide/video workflow while keeping the reference image clean.

### Video Composition Nodes

The repository also includes the existing BFS video composition utilities:

- **Reserved Region Frame Composer**
- **Frame Ranged Face Loader**
- **Face Sequence Batch**

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-BFSNodes.git
```

Install the Python dependencies if your ComfyUI environment does not already provide them:

```bash
cd ComfyUI-BFSNodes
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## LoRA Placement

Put your Edit Anything LoRA checkpoint in ComfyUI's LoRA folder, for example:

```text
ComfyUI/models/loras/
```

Subfolders are supported by ComfyUI, so organized paths such as this are fine:

```text
ComfyUI/models/loras/ltx-2/2.3/edit_anything/edit_anything_reference_v0.1_r128_12000.safetensors
```

## Basic Workflow

The expected graph order is:

```text
LTXV model loader
  -> standard ComfyUI Load LoRA
  -> LTXV Edit Anything (Apply)
  -> KSampler / sampler node
```

Connect the node inputs as follows:

- `model`: the model output after the standard LoRA loader.
- `positive` / `negative`: your conditioning.
- `vae`: the LTXV VAE.
- `latent`: the target video latent.
- `ref_image`: the appearance/reference image.
- `lora_name`: the same Edit Anything LoRA checkpoint.
- `guide_frames` optional: frames used for motion or structure guidance.

The node returns the patched model, updated positive/negative conditioning, updated latent, and a preview of the resized reference image.

## Recommended Starting Values

Start with:

```text
guide_strength: 1.0
ref_strength: 1.0
role_strength: 1.0
adaln_scale: 1.0
enable_adaln: true
enable_role_embedding: true
resize_mode: pad_to_fit
```

If the model ignores the reference image, try increasing `role_strength`. If the output copies color/texture too strongly, reduce `adaln_scale`.

## Compatibility Notes

These nodes target ComfyUI's LTXV/LTX-Video model implementation. The main apply node monkey-patches LTXV internals at runtime so the reference-role embedding and AdaLN conditioning match the training-time conditioning path.

A plain LoRA loader alone is not enough for Edit Anything LoRAs that include role embedding and/or AdaLN reference conditioning. Use the standard LoRA loader first, then pass the result through **LTXV Edit Anything (Apply)**.

## Reference temporal offset

Nodes that inject clean appearance references expose a reference-only temporal offset in VAE
latent steps. The default is `0`, matching the standard frame-zero placement used by existing
LoRAs and preserving their expected behavior. Negative values are an experimental way to reduce
the overlap frame-0 artifact where the reference composition briefly appears at the start of the
generated video.

- `0`: standard frame-zero placement (default)
- `-1`: one latent step before frame zero (`-8` pixel frames with the standard LTX VAE)
- `-2`: two latent steps before frame zero (stronger separation, possibly weaker identity)

The offset is applied only to appearance/reference inputs. Guide videos, masks, identity masks,
and target video positions remain unchanged. In **LTX Multiple Controls**, the field is named
`identity_temporal_offset_latents` and affects only `identity_image`.

## Requirements

- ComfyUI
- PyTorch
- safetensors
- An LTXV/LTX-Video model and compatible VAE
- An Edit Anything LoRA trained for this conditioning strategy

## License

This project follows the repository license. See [LICENSE](LICENSE).
