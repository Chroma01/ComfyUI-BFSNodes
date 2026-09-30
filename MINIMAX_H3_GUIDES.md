# MiniMax-H3 downscaled latent guides

BFSNodes 1.46.0 adds two nodes for H3 upscale LoRAs trained in
[alisson-anjos/ai-toolkit](https://github.com/alisson-anjos/ai-toolkit/tree/minimax-h3-latent-guides)
with `target_crop_v2` / `target_grid_stride_v1` geometry:

- **MiniMax-H3 Downscaled Latent Guide (BFS)**: preprocess and encode an image
  or video guide at a lower resolution, and return a model with compatible
  spatial conditioning coordinates.
- **MiniMax-H3 Guide Target — Image / Video (BFS)**: make the joint target
  latent at the intended output dimensions, with one frame for an image or
  `17*n+5` frames for video. It does not resize the target canvas implicitly.

The native Add Guide node enlarges guides to the output canvas. This node
keeps the smaller encoded guide and selects every factor-th origin from the
**target patch grid**. Guide time stays aligned to the target. It modifies a
cloned model through its UNet wrapper; no global PackedLayout replacement is
installed, and other models/conditionings continue to use their native paths.

## Install / update

```bash
cd ComfyUI/custom_nodes/ComfyUI-BFSNodes
git pull --ff-only
```

Restart ComfyUI. Use a ComfyUI build with native H3 support; the layout tests
use the exact implementation in commit
[`8cfe5e1`](https://github.com/Comfy-Org/ComfyUI/blob/8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1/comfy_extras/nodes_minimax_h3.py#L166).
Incompatible H3 layout/VAE APIs produce explicit errors.

## Image workflow

1. Load the **H3 Ref2V** model and the trained upscale LoRA with your normal
   LoRA loader. Load the official complete H3 video VAE with its encoder and
   the H3 Qwen3-VL text encoder (`CLIPLoader` type `minimax`).
2. Encode the caption without passing the guide image into the VLM. Start
   with the training prompt:
   `h3upscale, increase resolution and restore fine details while preserving the original content and visual style.`
3. Create a **Guide Target** with width `1024`, height `768`, frames `1`.
4. Connect the model after LoRA loading, positive conditioning, target latent,
   VAE, and source image to **Downscaled Latent Guide**. Set
   `downscale_factor=4`, `frame_idx=0`. Select the same `lora_name` to validate
   training metadata; that selector does **not** load the LoRA a second time.
5. Connect **both returned MODEL and positive CONDITIONING** to the guider /
   sampler. Use the returned model for sigma scheduling too. The sampler's
   `latent_image` stays the target latent from step 3, not a guide latent.
6. Decode with the H3 VAE. The third output is the **actual reduced pixel
   guide preview**; at this canvas it is `256 x 192`.

An [API-format image workflow template](examples/minimax_h3_upscale_image_api.json)
is included. Replace `YOUR_H3_UPSCALE_LORA.safetensors`,
`YOUR_INPUT_IMAGE.png`, and model filenames with files installed in your
ComfyUI. The template uses seed 42, Euler, 28 steps, full denoise, and BasicGuider.
It can be submitted as the `prompt` field of a ComfyUI `/prompt` API request.
It is an API graph, not a saved frontend canvas workflow.

## Video workflow

Create a target with `frames=73` (approximately 3 seconds at 24 fps) and supply
an IMAGE batch of **exactly 73 source video frames** resampled to 24 fps, in
chronological order. The video VAE produces 22 latent-time slices. Other valid
lengths include 5, 22, 39, 56, 90, 107 and 124; a one-frame batch is an image.
The node rejects invalid lengths instead of silently truncating or stretching
motion. It does not perform video decoding or FPS conversion itself.

Keep the same target canvas / guide factor as training. Increasing target
resolution or frame count increases target activation memory even though guide
spatial tokens are reduced. Start with a known-fitting canvas on the target GPU.
To save video, connect decoded frames to your video encoder/combiner at 24 fps.

An optional `audio` soundtrack plus `audio_vae` can add aligned guide audio.
Supply audio starting at the guide clip's first frame. It is cropped to the
clip and available target duration, and is never sent to the text encoder.
Omit it for visual-only inference. Native reference conditioning and native
same-resolution guides can be chained alongside this node; their row positions
are retained.

## Geometry and encoding

- Each target pixel axis must be a multiple of `32 * downscale_factor`.
- The source is fitted to the target canvas with the fork's centered bicubic
  crop, then downscaled with Lanczos before VAE encoding. This also applies
  when the supplied source is already small: inspect `guide_preview`.
- H3 VAE spatial compression is 16. At output `1024 x 768`, factor 4 yields
  guide pixels `256 x 192`, guide latents `16 x 12`, and target latents `64 x 48`.
- The posterior is sampled with an isolated CPU generator (seed 42) and
  rounded through FP16 **before** latent normalization, matching the fork's
  image and video sampling paths. The native VAE mean-only encoding is not used.
- An image has one guide latent-time slice; 73 video frames have 22. Video
  time coordinates use H3's `(1,4,4,4,4)` temporal pattern, not a uniform stride.
- The corrected packed layout keeps target/reference/audio coordinates and
  reindexes condition positions, frozen/update masks and segment boundaries.
  Downscaled guides receive the target spatial grid at the training stride;
  coordinates from a separately normalized small grid are not multiplied.

## Validation and limits

CPU tests cover the audited native layout, factors 1/2/4, exact crop/resize,
posterior sampling, mixed/native guides, references, audio row preservation,
73-frame counts, metadata checks, clone isolation and wrapper composition.
Golden position fixtures come from the actual AI Toolkit packing implementation.
Run `python -m unittest discover -s tests -v`.

A full pretrained ComfyUI GPU sampling run has not been validated as part of
this change. Matching preprocessing and coordinates establishes the conditioning
contract; it does not establish the trained LoRA's visual quality. Compare
LoRA strength 0 and 1 on held-out inputs before evaluating upscale fidelity.

## Experimental reference layouts and source phase (1.47.0)

Optional inputs on Downscaled Guide expose `guide_rope_layout` (overlap/sidecar),
`reference_rope_layout` (native/overlap/sidecar), `reference_source_phase`,
`reference_phase_scale`, `reference_sidecar_margin`, and `source_id`. Defaults
preserve 1.46.0 behavior. H3 sidecar margins use normalized RoPE units, not pixels.
Select your LoRA to check these values against `minimax_h3_reference_rope` metadata.
Legacy checkpoints imply overlap/native/phase-off. Changing only inference geometry
on a legacy LoRA does not establish that the model can use it.

**MiniMax-H3 Identity Reference + RoPE (BFS)** adds a separate single-image latent
reference without passing it to the VLM. Supply the same target latent and full H3
VAE. Chain both model and conditioning through Guide and Identity Reference, then
connect both final outputs to the guider/sampler (and model to the scheduler).
Use guide `source_id=1`, identity `source_id=2` for training control channels 1 and 2.
The identity reference keeps its aspect and is resized down only to the trainer's
/32 reference-image bucket; the guide factor does not resize the portrait.
The reference preview shows the actual pixels used.

This node requires a checkpoint trained with a matching identity reference role
and `control_latent_only: true` for caption-only conditioning. The LoRA selector
checks that mode too. It does not load LoRA weights. Adding a portrait to a LoRA
trained only with a low-resolution guide is not a validated identity-preservation
recipe. Keep caption/source conventions consistent during training.

Source phase adds the phase `source_id * phase_scale * 10000**(-d/48)` to H3's
rotary angles. Text and targets remain source zero. The scoped ModelPatcher object
patch composes this phase only when this node's marked conditioning is active;
unmarked conditioning and other models keep their original rotary path.

For a first experiment use overlap plus phase and compare to the phase-off baseline.
Sidecar and phase have CPU geometry/rotary parity tests, including mixed image,
video and audio, but no full pretrained H3 quality validation yet. They distinguish
sources; they do not guarantee the same face. See the
[training configuration guide](https://github.com/alisson-anjos/ai-toolkit/blob/minimax-h3-latent-guides/H3_REFERENCE_ROPE.md).
