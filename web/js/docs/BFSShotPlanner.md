# BFS Shot Planner / Shot Loop

Turn a long video into shots a video model can actually follow, give every shot its own
reference image and prompt, run the rest of the workflow once per shot, and join the results
back into one video with the original timing and soundtrack.

## Why

A video model only follows a guide reliably inside the clip length it was trained on. An H3
body-swap LoRA trained on 107-frame clips keeps the scene, framing and motion at 4.5 s and
loses them at 10 s, where it starts summarising the clip instead of following it frame by
frame. Splitting the source into model-sized shots, at camera cuts when there are any, keeps
every shot inside that range.

## Nodes

| node | what it does |
|---|---|
| **BFS Shot Planner** | Pick a video, split it on a timeline, set a reference and prompt per shot. Outputs the shots as a ComfyUI **list**. |
| **BFS Shot Unpack** | Opens one shot into plain values: guide frames, reference, second reference, prompt, length, first frame, audio, size, index. Use it with any model. |
| **BFS Shot Repack** | Puts edited pieces back into a shot: Unpack's `shot` output goes to Repack's `shot`, the edited piece (e.g. the reference with its background removed) to its input. Unconnected inputs keep their values; timing, cuts and audio are unchanged. |
| **BFS Shot H3 Conditioning** | Ready-made MiniMax H3 conditioning for one shot, built with the native nodes (Reference to Video + Add Guide). Optional *duet*: the shot's clip pinned in a side panel (canvas or shifted RoPE; route the model through the node for the shift); the join cuts the panel off. Optional prompt writer: pick a *task* (character swap, style, setting, appearance, lighting / weather, custom) and an *instruction*; with a VLM on its *vlm* input it writes each shot's duet prompt from the shot and its references, otherwise a template for the task (*planner prompt*, the default, keeps the planner's prompt). The written text comes out on *prompt*. Optional *setting_ref* (TSC's tip): one more reference, the shot's middle frame with the person covered in TV static, so the model sees the place in full detail; it is the last `<Picture n>` (write `{setting}` for its tag, otherwise a sentence is added to subject_definitions). |
| **BFS Shot Join** | Concatenates the decoded shots in order, trims each to its true length, cross-fades soft joins and returns the soundtrack. With *comparison* on it also returns a side-by-side video (original shot \| references \| result) with the shot's info and your *label* on top and its prompt below, ready for Create Video. |
| **BFS Shot H3 Duet** | Renders one shot with H3: the shot's clip pinned beside the video (duet, no LoRA), the shot as an aligned guide (for body-swap LoRAs), or both. |

## How the loop works

ComfyUI runs any node that receives a list once per item and pairs several lists by index. The
planner's `shots` output is a list, so everything downstream of it (conditioning, sampler,
decode) runs once per shot, and shot *i* always gets guide *i*, reference *i* and prompt *i*.
Single values such as the model and VAE are shared by every run. The join takes the whole list
back (it is an `INPUT_IS_LIST` node) and puts the video together.

Two run modes:

- **Auto loop (one run)**: every shot in one queue run. Simple; holds all decoded shots in memory.
- **Queue loop (one shot per run)**: each run generates the next pending shot and stores it on
  disk. The join blocks every node after it until the last shot, then outputs the full video,
  so a Save Video node only fires once. With *Auto-queue the next shot* on, the panel re-queues
  the workflow after each shot. *Reset loop* starts over. Changing the plan starts a new run.

## Guide frames and lengths

Each shot is generated at a length the model accepts (frame grid: H3 `17n+5`, LTX/Wan `8n+1`,
Wan `4n+1`, or any). When a shot is shorter than that length, the extra guide frames come from
the video that follows it, so the guide never repeats or stretches. The join cuts each result
back to the shot's own frames, which keeps the timing identical to the source. Those extra
frames are also a real overlap, which the join uses to cross-fade boundaries that are not
camera cuts.

## Splitting

- **Camera cuts** (default): cuts from **PySceneDetect** (`adaptive` or `content`; install
  `scenedetect`), or the built-in detector when the package is missing. Shots shorter than the
  minimum merge into a neighbour when the merge still fits; shots longer than the maximum split
  into equal parts.
- **Fixed length**: equal parts no longer than the maximum.
- **By hand**: drag the white handles on the timeline to move a boundary, double-click the
  shot bar to split, *Merge with next* to join, or select a shot and press **Delete** to remove the cut at its
  start (it merges into the previous shot). Your edits are what runs.

*Max seconds / shot* is converted to frames and snapped down to the grid (4.5 s at 24 fps =
107 frames for H3). *Max shots* and *Max total seconds* cap a long source.

## Audio

The source card shows whether the video has a usable audio track (rate, channels) or why not. *Use the
video's audio* or *silent track*. Without a usable track (none, unreadable, or silent chosen) the planner's
`audio`, BFS Shot Join's `audio` and the H3 Duet nodes return a **silent track of the right length** instead
of nothing, so Create Video / Save Video always work. Audio is read with ffmpeg, or with PyAV when the ffmpeg
binary is missing.

## Preview

*Play shot* plays only the selected shot, from its first to its last frame, and stops (or loops with
*loop shot*); every shot card has its own ▶. *Play all* plays the whole plan and jumps over shots
that will not run. The panel shows the shot's start, end and current time (and frame), a yellow
playhead follows on the timeline, and clicking the timeline seeks. The source video streams from
the input folder, so only what you play is loaded.

## Filters

Skip shots that should not run, decided per shot from a few sampled frames:

| filter | skips a shot when |
|---|---|
| Needs a person | no person is detected |
| Min person size | the largest person covers less than that fraction of the frame (wide shots) |
| Max people | more people than that (crowds) |
| Needs a face | no face is detected |
| Skip dark / fades | the mean brightness is below the level |
| Skip static | there is almost no motion (title cards, freeze frames) |
| Min frames | the shot is shorter than that |

People and faces use YOLO from `models/ultralytics` (`person_yolov8m-seg.pt`, `face_yolov8m.pt`, the
Impact Pack files) with OpenCV fallbacks. *Analyse people & faces* shows the numbers on every shot
card. Each shot can override the filters (always run / always skip).

Skipped shots do not run. Connect the planner's `timeline` output to BFS Shot Join and they are
filled with the original video, keeping the full duration and soundtrack, or removed (with the
matching audio) when *Skipped shots in the output* is set to remove them.

## Cast (people by face)

*Find people* samples the video every 0.5 s, detects faces and groups them into people by face
identity (InsightFace `buffalo_l` in `models/insightface`, on the CPU with a few threads so it stays
light and never touches the GPU; the result is cached per video). Each
person card shows the face, screen share and first/last appearance (click the face to seek).
Give a person a reference (and a second one) by clicking the slot, or pick one from the recent
references; *ignore* leaves that person out. Every shot card shows who is in it, with the main
person (most screen time) highlighted and linked people outlined in green.

| option | effect |
|---|---|
| Shots use their main person's reference | a shot without its own reference takes its main person's (shown with a dashed green outline) |
| Split where the main person changes | adds a boundary where the largest face switches to another person for at least a second, then re-splits |
| Only run shots with a linked person | shots where no linked, non-ignored person appears are skipped ("no linked person") |

A reference set on the shot itself always wins; the global reference is the last fallback.

## References and prompts

Every shot can have its own reference, second reference and prompt; shots without them use the
global ones. Click an empty reference slot to upload an image straight from your computer; a filled
slot has **✕** to remove it and, on a shot, **→ all** to use it for every shot. The last ten
references you picked are shown under the shot editor for one-click reuse (click = reference,
shift+click = second reference). Connected `ref_image`, `ref_image_2` and `prompt` inputs override
the panel's global values, so an edited first frame or a prompt from another node can drive the
defaults.

The planner also outputs `ref_image` and `ref_image_2`: the references the shots actually use,
without repeats, so a plan where every shot shares one reference returns a single image.

## Mask & crop (SAM 3)

Generate only part of the frame. In a shot's editor, type what to segment in English (`person in white`,
`woman with red hair`, `red car`; up to 32 tokens, commas for several things) or click **🎯 Points** to pick
positive / negative points on any frame of the shot, then **👁 Preview mask** (red = mask, yellow = crop box).
Tick **✂ Crop to mask** and the shot is cropped to one box around the mask (the union over all its frames, so
the crop does not shake), at the source resolution, and sent to the model at the generation size; **BFS Shot
Join** pastes the result back into the full frame, feathered by the mask (or the whole box). **BFS Shot
Unpack** also outputs the shot's mask (`mask`), e.g. for an inpainting model.

The **Mask & crop** card holds the global settings, with defaults that work as they are: fill holes on,
temporal expand 2 frames (less flicker), expand 16 px, feather 12 px, padding 15 %, paste by mask, blockify off
(16 aligns the mask to H3's latent grid), threshold 0.5, 4 objects. *Show masks on shots* overlays the preview
on the shot cards. SAM 3 uses the official `sam3.1_multiplex_fp16.safetensors` in `models/checkpoints`,
downloaded from Comfy-Org/sam3.1 the first time.

## VLM suggestions

Connect a vision-language model to the planner's `vlm` input (CLIPLoader with `qwen3vl_4b` or `qwen3vl_8b`).
It looks at a few frames of every shot and suggests what to segment, a description of the shot (camera,
framing, action) and whether to run it. In the **VLM** card: *Use the VLM when the workflow runs* applies the
suggestions at run time (mask text for shots without one; `{shot}` in a prompt becomes that shot's
description), *Analyse shots* asks from the panel (after one run with the VLM connected), and each shot's
editor shows its suggestion with buttons to apply it. The summary output lists them too.

## Describe references ({details})

The VLM can describe the references, for any task (not tied to swaps). In the **VLM** card pick an instruction
preset: *full body* (face, hair, skin, age, build, clothing piece by piece), *head / face*, *face attributes* (a
short comma-separated list), *outfit*, or *custom* (your own instruction). **📝 Describe refs** writes one
description per reference set: the global references, every shot's and every cast person's. They are saved in
the plan and editable. Write **`{details}`** anywhere in a prompt and every shot gets the description of its own
references; a set without a description is described by the VLM at run time.

The VLM buttons in the panel work after the workflow has run once with the VLM connected (ComfyUI only hands
models to nodes when they run). Tested with Qwen3-VL 8B; some smaller or modified models return empty answers
for some wordings: the planner retries with a reworded request and with sampling, but if a preset stays empty,
use the 8B or another preset.

## Settings reference

Every setting in the panel has a hover tooltip (ⓘ). The main ones:

| setting | what it does |
|---|---|
| Mode | Camera cuts (shots start at cuts, long ones split evenly) or fixed length |
| Detector / Sensitivity | PySceneDetect adaptive or content, or the built-in one; higher sensitivity finds more cuts |
| Frame grid | frame counts the model accepts (H3 17n+5, LTX/Wan 8n+1, Wan 4n+1) |
| Max / Min seconds per shot | longest shot sent to the model; shorter shots merge into a neighbour |
| Timeline fps | the frame rate the planner works on (generation and audio use it) |
| Max shots / Max total seconds | test on the first shots or seconds only |
| Megapixels / Size multiple | generation size at the source's aspect ratio, snapped to the multiple |
| Run | auto loop (all shots in one run) or queue loop (one shot per run, stored on disk) |
| Skipped shots in the output | keep the original video there or remove those shots |
| Mask: padding / expand / feather / temporal expand / blockify | crop context, paste-back growth and softness, flicker hold, square blocks |
| Mask: threshold / max objects / paste back | SAM 3 text matching, objects tracked, paste only the mask or the whole box |
| VLM: frames per shot / max tokens | how much the VLM sees and how long it may answer |

## Continuity between shots

Each shot (after the first) can continue from the **previous shot's generated result**: in the shot editor,
*Continuity* = *previous shot as reference* adds a frame of it as one more `<Picture n>` after the shot's own
references (good across camera cuts, keeps the person consistent), and *previous shot as first frame* anchors it
at frame 0 of the shot (for continuous action without a cut). Pick which frame: the previous result's first,
middle or last. *Continuity → all* applies the setting to every shot after the first.

It needs the previous result to exist: it works in the **queue loop** (the planner reads the stored result) and in
the **auto loop with BFS Shot H3 Duet** (it renders shot by shot and keeps the last result). BFS Shot Unpack also
outputs it as `previous_result`. This is different from H3 Conditioning's *first_frame*, which anchors the shot's
own frame from the source video.

## MiniMax H3 with an aligned guide (example)

```
BFS Shot Planner ── shots ──► BFS Shot H3 Conditioning (clip, vae, guide_mode = aligned guide)
                                 └─ positive, latent ─► CFGGuider / sampler ─► VAE Decode ─┐
BFS Shot Join ◄── images ─────────────────────────────────────────────────────────────────┘
      └─► Create Video (fps, audio) ─► Save Video
```

For other models, use **BFS Shot Unpack** and wire `guide_frames`, `ref_image`, `prompt` and
`length` into that model's own guide and reference nodes; the list mapping works the same way.
