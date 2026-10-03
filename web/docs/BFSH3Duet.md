# BFS H3 Duet / Side Panel

Training-free split-screen ("duet") generation for MiniMax H3.

**Credits:** the initial idea for this H3 node came from TSC's latent-pin duet. BFS had already used the
same principle on LTX (a green side panel holding the reference) and implemented the virtual sidecar
approach (reference tokens placed beside the frame in RoPE). New here: the **shifted RoPE layout** (the
video keeps the RoPE positions of a render without the panel and the panel sits past its edge, with an
optional gap), dynamic references, task prompts and the shot-loop node.
A panel is pinned next to the video in the same canvas with a noise mask. H3 keeps the panel
exactly (it treats it like its own condition rows) and generates the video next to it, in sync.
The panel is cut off before decoding, so it never reaches the output.

## Nodes

| node | what it does |
|---|---|
| **BFS H3 Duet** | Everything in one node: native references + prompt, the pinned panel, an optional latent guide, sampling (euler / beta / CFG 1), decode and crop. Leave `panel` empty for a plain guided render. Outputs the generated video, its audio, the whole canvas (to check the sync) and the layout sentence for the prompt. |
| **BFS Shot H3 Conditioning** (duet option) | The shot loop's H3 conditioning can pin the shot's own clip as a duet panel too: *duet* = canvas or shifted RoPE (route the model through the node for the shift). BFS Shot Join cuts the panel off by itself, so the rest of the workflow (sampler, decode) stays as it is. |
| **BFS Shot H3 Duet** | One shot of the shot loop (Planner -> this -> BFS Shot Join): *duet* pins the shot's own clip beside the video (no LoRA needed), *guide* puts the shot on the generated frames as a latent guide (for body-swap LoRAs), *duet + guide* does both. Write `{layout}` in the prompt to insert the split-screen sentence. |
| **BFS H3 Side Panel** | Only the canvas: takes the conditioning and AV latent from Reference to Video, adds the pinned strip (and the guide), and returns them for your own sampler. Guides already added with Add Guide for MiniMax H3 are moved onto the canvas. |
| **BFS H3 Side Panel Crop** | Removes the strip from the sampled latent (before VAE Decode) or from decoded images. |

## What to pin

- **The source clip** (TSC's duet): the new character copies its motion, camera and cuts frame for
  frame. Give the new character with `<Picture 1>` (and the place with `<Picture 2>`). The pinned clip
  has no tag: the prompt calls it "the kept footage" (`layout_text` gives the sentence). See TSC's
  prompting guide: never describe the source performer, restate the identity in every shot.
- **A reference image**: a static picture of the person next to the video.

## Tasks

Use it for any edit that keeps the source's motion, camera and timing. With an empty prompt box the node
writes the prompt from **task** + **instruction** (the `prompt` output shows what it sent):

| task | instruction example | panel_noise |
|---|---|---|
| character swap | (empty, or the new person's look) + the person's pictures as references | 0 |
| style | `a 1990s anime cel style` | 0.1-0.2 |
| setting | `a sunny beach at sunset, waves rolling behind her` | 0 |
| appearance | `an elderly woman with short grey hair` | 0.1 |
| lighting / weather | `night, lit by pink and blue neon signs` | 0 |

It does not change the motion or the camera: the panel makes the video copy them in sync.

The task prompt is a **draft**: it cannot see the video, so it is generic. In tests, a prompt written for the
clip (the example below) kept the room, framing and sync; the drafts got the change only partly (the style
barely changed, the setting and light lost the framing, the swap took the reference's backdrop). Start from
the `prompt` output and describe the clip.

References work like the official Reference to Video node: up to 9 images (`<Picture n>`), 3 videos
(`<Video n>`, each with its soundtrack) and 3 audios (`<Audio n>`), new slots appear as you connect them.

## Prompt

The node's prompt box starts with a template. The side panel has **no tag**: it is not a `<Picture n>`
or `<Video n>`, and the text encoder never sees it. Name it by its place ("the LEFT half is the kept
footage"); `{layout}` inserts that sentence for the current side and size. Describe only the generated
part: never the panel's performer, clothes or room, even to contrast them, because whatever you leave
undescribed is copied from the panel. Restate the new identity in every shot ("her face from
`<Picture 1>`" plus two or three face, hair or outfit words) and give exact times for cuts.

Example (character swap, source clip pinned on the left, a face picture and a full-body picture):

```
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
None.
```

## Settings

| setting | effect |
|---|---|
| position / size | side of the strip and its size relative to the video (1.0 = two equal halves) |
| fit | contain (default: the whole clip, smaller, with grey around, nothing cropped), cover (fill and crop), stretch |
| gap | grey separator between panel and video, in 32 px patches |
| panel_noise | 0 pins the panel exactly; 0.05-0.15 loosens it when the result copies too much of it |
| hold | pin the panel for the whole clip, or only its first latent frame |
| guide | optional aligned latent guide in the video area (what the body-swap LoRAs use) |
| rope_mode | *canvas*: panel and video share one wide grid (TSC). *shifted*: the video keeps the RoPE positions of a render without the panel, and the panel sits past its edge |
| rope_gap | *shifted* only: empty RoPE steps (2x2 patches) between video and panel; the panel moves away in position without any pixels in between. Keep it small against the video width: at 320 px wide (10 patches) a gap of 8 made the model draw its own split screen; 0 works everywhere |

## Notes from tests

- Source clip pinned, no LoRA, no guide (TSC's setup): the new person follows the clip's gestures,
  timing and cuts, in the same room, and on-screen captions disappear.
- With a body-swap LoRA and a latent guide, the guide alone kept the room better than guide + panel:
  the panel pulled the background toward the reference picture's backdrop. The shifted RoPE (gap 0 or 8)
  did not change that, so it comes from the pinned panel itself, not from the canvas geometry.
- Duet with the shifted RoPE, gap 0 and gap 8: same quality and sync as the canvas layout. Use the panel without a
  guide, or the guide without a panel, until a LoRA is trained for both.
