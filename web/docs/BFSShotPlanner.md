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
| **BFS Shot H3 Conditioning** | Ready-made MiniMax H3 conditioning for one shot, built with the native nodes (Reference to Video + Add Guide). |
| **BFS Shot Join** | Concatenates the decoded shots in order, trims each to its true length, cross-fades soft joins and returns the soundtrack. |

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
  shot bar to split, *Merge with next* to join. Your edits are what runs.

*Max seconds / shot* is converted to frames and snapped down to the grid (4.5 s at 24 fps =
107 frames for H3). *Max shots* and *Max total seconds* cap a long source.

## References and prompts

Every shot can have its own reference, second reference and prompt; shots without them use the
global ones. Connected `ref_image`, `ref_image_2` and `prompt` inputs override the panel's global
values, so an edited first frame or a prompt from another node can drive the defaults.

## MiniMax H3 with an aligned guide (example)

```
BFS Shot Planner ── shots ──► BFS Shot H3 Conditioning (clip, vae, guide_mode = aligned guide)
                                 └─ positive, latent ─► CFGGuider / sampler ─► VAE Decode ─┐
BFS Shot Join ◄── images ─────────────────────────────────────────────────────────────────┘
      └─► Create Video (fps, audio) ─► Save Video
```

For other models, use **BFS Shot Unpack** and wire `guide_frames`, `ref_image`, `prompt` and
`length` into that model's own guide and reference nodes; the list mapping works the same way.
