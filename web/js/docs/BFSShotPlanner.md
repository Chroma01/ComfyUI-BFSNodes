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

## The panel

The planner's panel has five tabs, in the order you work. The timeline stays on top of every tab.

| tab | what is there |
|---|---|
| 🎞 **Video** | pick or upload the video, audio, how it is split into shots (cuts or fixed length), generation size |
| 🎬 **Shots** | one card per shot (scroll sideways, **← →** move between shots), the preview player and the selected shot's editor: reference & prompt, who is replaced (mask, generation mode, `{target}`), continuity, **Copy to other shots** |
| 👥 **People & masks** | the cast found by face (a reference per person) and the global mask settings |
| 📝 **Prompts & refs** | the global reference and prompt, the VLM and the reference descriptions (`{details}`) |
| ▶ **Run** | **Checks** (what to fix before running, with a button to each shot), auto / queue loop, filters, test limits |

Shot cards show badges for what each shot has: 🎯 mask, 🎭 / ✂ generation mode, 🧑 `{target}`, ⛓ continuity and ⚠ problems.
The header's ⚠ count opens the checks.

**Copy to other shots** (in the shot editor) copies the chosen parts of the selected shot to all shots, or to the
shots after it: references, prompt, mask text, mask points (each shot gets them on its frame at the same relative
position), generation mode, `{target}`, continuity. It replaces the old "→ all" buttons.

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
  shot bar to split, *⇥ Merge* to join, or select a shot and press **Delete** to remove the cut at its
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
slot has **✕** to remove it; **Copy to other shots** gives a shot's references to every shot. The last ten
references you picked are shown under the shot editor for one-click reuse (click = reference,
shift+click = second reference). Connected `ref_image`, `ref_image_2` and `prompt` inputs override
the panel's global values, so an edited first frame or a prompt from another node can drive the
defaults.

The planner also outputs `ref_image` and `ref_image_2`: the references the shots actually use,
without repeats, so a plan where every shot shares one reference returns a single image.

## Mask & crop (SAM 3)

Generate only part of the frame. In a shot's editor (*Who is replaced*), type what to segment in English
(`person in white`, `woman with red hair`, `red car`; up to 32 tokens, commas for several things) or click
**🎯 Points** to pick positive / negative points on any frame of the shot, then **👁 Preview** (red = mask,
yellow = crop box). Then pick how the shot is generated:

| mode | what is generated | crop / uncrop |
|---|---|---|
| **Full frame** | the whole frame; the mask only feeds `{target}` and the setting picture | no |
| **🧩 Frame + paste** | the whole frame (so the model follows the guide's pose freely and the new person may be bigger); BFS Shot Join then pastes only the person onto the original frame: the shot's mask plus the new subject's outline (SAM 3 on the result with the shot's *new subject* text, default `person`; `paste_new_outline`) | no crop; paste in the Join |
| **🎭 Mask only** | only the mask, on the whole frame; everything else stays exactly as it was | no |
| **✂ Crop** | a box around the mask (more pixels for a small person); BFS Shot Join pastes it back | yes |
| **✂🎭 Crop + mask** | only the mask, inside the crop: the most detail with the background kept | yes |

*Frame + paste* is the safest for a different body (e.g. a woman in profile replaced by a man facing the camera):
*Mask only* binds the new person to the old outline, *Frame + paste* does not. The two mask modes need **BFS Shot H3 Conditioning → inpaint = per shot (planner)** (the default; *only the mask*
forces it on every shot, *off* never). The crop is one box around the mask (the union over all its frames, so it
does not shake), at the source resolution, sent to the model at the generation size; **BFS Shot Join** pastes the
result back, feathered by the mask (or the whole box). **BFS Shot Unpack** also outputs the shot's mask (`mask`),
e.g. for an inpainting model. In **BFS Shot Join**'s comparison video, cropped shots show their mask (red) and crop
box (yellow) over the original column (`comparison_mask`, on by default).

**Mask opacity** (mask modes, per shot, slider under the modes): the value of the white inside the generation mask.
1 regenerates the masked area completely. A grey mask (e.g. 0.85) keeps part of the original there: H3 puts those
rows at *opacity* × the noise level, so they start from the original partly visible. 0.8-0.9 keeps pose, outline and lighting while still swapping; too low copies the original
person. With `guide_mode = aligned guide` the model also sees the whole original shot as its guide, so it follows
the motion either way. (Not applied with the duet panel.)

**Mask guide** (experimental, *Mask only* shots, BFS Shot H3 Conditioning → `mask_guide`): the shot with only the
masked region visible, the rest grey, so the model looks at the subject's pose and outline on its own while the mask
limits where it generates. *+ extra guide* adds it as a second aligned guide next to the normal one (the model sees
both); *instead of the full guide* replaces the normal one; *+ reference video* gives it as a native reference video
(`<Video n>`, after the shot's own one when guide_mode uses it; write `{mask_video}` in the prompt where its tag goes, otherwise a sentence is added). `mask_guide_look` sets what it shows of the subject: *grey blurred* (default: volume, light and head direction, no colours or face, so the model follows the motion without copying the old subject), *colour* (as it is; can make the model copy the old subject), *silhouette* (flat shape), *edges* (outlines) or *pose (people)* (the skeleton of the people in the mask, OpenPose colours; needs ultralytics and a YOLO pose model in `models/ultralytics`, e.g. `pose/yolov8m-pose.pt`). `mask_ref_size` makes that reference video smaller (default 1/2, ~1/4 of the tokens): it only has to show the pose and outline. Compare with
*off*: the LoRAs were trained with one full guide, and it can also pull the old subject's look.

### Masks from a video (rotoscoping) instead of SAM 3

A mask made elsewhere (After Effects Roto Brush, DaVinci Resolve Magic Mask, another ComfyUI workflow…) can replace
SAM 3. It is a black and white video, **white = the subject**, and it works with every mode (Mask only, Crop,
Crop + mask), `{target}`, the setting picture and the comparison overlay. Three places, the most specific wins:

| where | covers | wins over |
|---|---|---|
| shot editor → **🎞 mask video** | only that shot (its first frame = the shot's first frame) | everything |
| the shot's own SAM 3 text / points | that shot | the two below |
| the planner's optional **`mask`** input (MASK or IMAGE batch, e.g. Load Video + Convert Image to Mask) | the whole video, one mask per source frame | the panel's mask video |
| People & masks → **🎞 Mask video (whole video)** | the whole video, matched by time | SAM 3 for shots without a mask |

Any size and frame rate: it is matched to the shot by time and resized. The console says which mask each shot used.

**Stitch finishing** (BFS Shot Join, cropped shots only; adapted from Neko (Nekodificador)'s *NKD Inpaint Stitch*, MIT, built on AbleJones's workflow and nodes):

| Option | What it does |
|---|---|
| `edge_hardness` (0-1) | hardens the soft edge of the paste; raise it when a faint ghost of the original person shows around the new one |
| `match_colors` (0-1) | corrects the colour / brightness drift of the generated patch (Reinhard in LAB), with statistics over the whole shot so it does not flicker |
| `match_region` | *around the subject (swap)*: measured on a ring of background around the mask, so a new person keeps their own colours. *inside the subject*: measured inside the mask, for retouching the same content |
| `seamless_edges` | Poisson blend (OpenCV) for stubborn seams; slower |

**Which mode.** A generation mask alone keeps the background exact but generates the person at the frame's own size,
so a small person gets few pixels; a crop alone gives the person more resolution, but the whole box is regenerated
and pasted back (that is what the finishing above is for). *Mask only* is the simplest when the person fills a good
part of the frame; *Crop + mask* is best for small people. In the mask modes the latent starts from the shot's own
frames and its SAM 3 mask (grown by *expand* plus one latent cell) becomes the H3 generation mask. Raise *expand*
when the new person is bigger than the old one (longer hair, wider body): the generated area cannot go past the
grown mask; their shadow or reflection outside the mask stays from the original.

Credit to **Neko (Nekodificador)** and **AbleJones**, whose workflow and nodes the crop + generation mask follows.
ComfyUI builds without native H3 generation masks need a per-row mask patch on the model
(e.g. ComfyUI-MiniMaxH3-PerRowMasking).

The **People & masks** tab holds the global mask settings, with defaults that work as they are: fill holes on,
temporal expand 2 frames (less flicker), expand 16 px, feather 12 px, padding 15 %, paste by mask, blockify off
(16 aligns the mask to H3's latent grid), threshold 0.5, 4 objects. *Show masks on shots* overlays the preview
on the shot cards. SAM 3 uses the official `sam3.1_multiplex_fp16.safetensors` in `models/checkpoints`,
downloaded from Comfy-Org/sam3.1 the first time.

## VLM suggestions

Connect a vision-language model to the planner's `vlm` input (CLIPLoader with `qwen3vl_4b` or `qwen3vl_8b`).
It looks at a few frames of every shot and suggests what to segment, a description of the shot (camera,
framing, action) and whether to run it. In the **Prompts & refs** tab: *Use the VLM when the workflow runs* applies the
suggestions at run time (mask text for shots without one; `{shot}` in a prompt becomes that shot's
description), *Analyse shots* asks from the panel (after one run with the VLM connected), and each shot's
editor shows its suggestion with buttons to apply it. The summary output lists them too.

## Describe references ({details})

The VLM can describe the references, for any task (not tied to swaps). In the **Prompts & refs** tab pick an instruction
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
middle or last. **Copy to other shots** with *continuity* ticked applies it to every shot after the first.

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

## Choosing who is replaced (`{target}`), step by step

With several people on screen, a prompt that says "replace the person" leaves the model to guess. `{target}`
puts a short description of the right person into the prompt of every shot ("the young woman with dark hair in a
pink top"), so the model knows WHICH one to swap. The selection is made with SAM 3, and the description by the
VLM, so it works for anyone or anything you can click on.

### What you need

- The planner's **`vlm`** input connected to a Qwen3-VL (CLIPLoader, type *stable_diffusion*, `qwen3vl_8b` recommended).
- The workflow run **once** with it connected. ComfyUI only hands models to nodes when they run; after that the
  panel buttons can use it.
- Without a VLM it still works: you get the cut-out and type the description yourself.

### 1. Put `{target}` in the prompt

Write `{target}` wherever the prompt names the person being replaced: in the global prompt (all shots) or in a
shot's own prompt. For example, the swap caption the H3 LoRAs were trained with:

```
... only the face, body and clothing of {target} are replaced.  summary: [video generation + reference] Replace
{target} in the guide video with <Subject 1>, keeping the scene, camera and the complete motion and facial
performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of {target}
and moves with the same timing, body pose, gestures, ...
```

`{target}` can be combined with `{details}` (the description of the references) and `{shot}` (the VLM's
description of the shot).

### 2. Select the person

Open a shot and click **🎯 Points…**. The modal shows one frame of the shot; the slider below picks the frame.

- **Click on the person's body** (torso), not only the face. A click on the face selects the head, and the
  description then misses the clothes.
- **Right-click** (or shift+click) puts an *exclude* point, e.g. on a second person standing close.
- **👁 Segment** shows what SAM 3 picked (red). Add or remove points until only that person is red.
- Instead of points you can just type what to segment (`woman in pink`, `man with glasses`, `dog`): no clicking is
  needed, SAM 3 finds and tracks it through the shot. Points are for when the text is ambiguous (two similar people).
  Points win when both are set. **Copy to other shots** with *mask text* (and *mode*) ticked gives the text (and
  the generation mode) to every shot.

Then save:

| Button | What it does |
|---|---|
| **Save (this shot)** | keeps the selection on this shot only |
| **Save → all shots** | applies the same points to every shot, each on its frame at the same relative position as the frame you clicked on (e.g. clicked in the middle of this shot → every shot uses its middle frame) |

**Save → all shots** is for a person who stays in the same place across shots (a sequential video, a fixed camera).

### 3. Describe the person

Click **🧑 Describe target** in the shot's editor:
- SAM 3 cuts the person out of the selected frame, and a thumbnail of the cut-out appears next to the field.
- The VLM writes a short phrase into the **`{target}`** field.
- The field stays editable: correct it or write your own. Keep it short and visual (who, hair, main clothes with
  colours) and start it with "the".
- **Copy to other shots** with *{target}* ticked copies the description to every shot. Use it when the same person is replaced throughout;
  otherwise describe shot by shot.

### 4. Check and fix

- **👁 Preview** on a few shots shows the selection on several frames of each shot.
- In a shot where the selection picked the wrong thing, **✕ Clear** removes its points, text and mode. Select
  again in that shot only (**Save (this shot)**) and describe again.
- **Copy to other shots** with *mask points* ticked copies that shot's points to every shot without opening the modal.

### What happens when the workflow runs

Every shot's prompt gets its own `{target}` text. A shot without a description gets "the person". The selection is
only used to make the description: it does not crop or mask anything unless the shot's mode is *Mask only*, *Crop*
or *Crop + mask*.

### Examples

- **Two people, swap only one:** in the first shot, click on the woman's torso, put an exclude point on the man,
  then **Save → all shots** and **🧑 Describe target** → "the young woman with long black hair in a white blouse".
  **Copy to other shots** with *{target}* ticked.
- **The person changes between shots:** describe shot by shot, e.g. shot 1 "the man in the grey suit", shot 2
  "the woman in the red dress". Each shot keeps its own text.
- **Animals:** it works the same way, e.g. "the small black dog in front".

### Troubleshooting

| Problem | Fix |
|---|---|
| "No VLM yet" | connect the `vlm` input and run the workflow once, or type the description |
| "Select the person first" | the shot has no points or mask text: use **🎯 Points…** |
| "SAM 3 found nothing" | click again on the person (on the torso), or try another frame with the slider |
| Only the face or the hair is described | the click was on the head: click the body |
| The other person is described | add an exclude point (right-click) on the other person, **👁 Segment**, save, describe again |
| The swap still hits the wrong person | make the description more specific (position: "on the left", clothing colours) |

