# BFS H3 Duet / Side Panel

Training-free split-screen ("duet") generation for MiniMax H3, following TSC's latent-pin workflow.
A panel is pinned next to the video in the same canvas with a noise mask. H3 keeps the panel
exactly (it treats it like its own condition rows) and generates the video next to it, in sync.
The panel is cut off before decoding, so it never reaches the output.

## Nodes

| node | what it does |
|---|---|
| **BFS H3 Duet** | Everything in one node: native references + prompt, the pinned panel, an optional latent guide, sampling (euler / beta / CFG 1), decode and crop. Leave `panel` empty for a plain guided render. Outputs the generated video, its audio, the whole canvas (to check the sync) and the layout sentence for the prompt. |
| **BFS Shot H3 Duet** | One shot of the shot loop (Planner -> this -> BFS Shot Join): *duet* pins the shot's own clip beside the video (no LoRA needed), *guide* puts the shot on the generated frames as a latent guide (for body-swap LoRAs), *duet + guide* does both. Write `{layout}` in the prompt to insert the split-screen sentence. |
| **BFS H3 Side Panel** | Only the canvas: takes the conditioning and AV latent from Reference to Video, adds the pinned strip (and the guide), and returns them for your own sampler. Guides already added with Add Guide for MiniMax H3 are moved onto the canvas. |
| **BFS H3 Side Panel Crop** | Removes the strip from the sampled latent (before VAE Decode) or from decoded images. |

## What to pin

- **The source clip** (TSC's duet): the new character copies its motion, camera and cuts frame for
  frame. Give the new character with `<Picture 1>` (and the place with `<Picture 2>`). The pinned clip
  has no tag: the prompt calls it "the kept footage" (`layout_text` gives the sentence). See TSC's
  prompting guide: never describe the source performer, restate the identity in every shot.
- **A reference image**: a static picture of the person next to the video.

## Settings

| setting | effect |
|---|---|
| position / size | side of the strip and its size relative to the video (1.0 = two equal halves) |
| fit | contain (whole panel on grey), cover (fill and crop), stretch |
| gap | grey separator between panel and video, in 32 px patches |
| panel_noise | 0 pins the panel exactly; 0.05-0.15 loosens it when the result copies too much of it |
| hold | pin the panel for the whole clip, or only its first latent frame |
| guide | optional aligned latent guide in the video area (what the body-swap LoRAs use) |
| rope_mode | *canvas*: panel and video share one wide grid (TSC). *shifted*: the video keeps the RoPE positions of a render without the panel, and the panel sits past its edge |
| rope_gap | *shifted* only: empty RoPE steps (2x2 patches) between video and panel; the panel moves away in position without any pixels in between |

## Notes from tests

- Source clip pinned, no LoRA, no guide (TSC's setup): the new person follows the clip's gestures,
  timing and cuts, in the same room, and on-screen captions disappear.
- With a body-swap LoRA and a latent guide, the guide alone kept the room better than guide + panel:
  the panel pulled the background toward the reference picture's backdrop. The shifted RoPE (gap 0 or 8)
  did not change that, so it comes from the pinned panel itself, not from the canvas geometry.
- Duet with the shifted RoPE, gap 0 and gap 8: same quality and sync as the canvas layout. Use the panel without a
  guide, or the guide without a panel, until a LoRA is trained for both.
