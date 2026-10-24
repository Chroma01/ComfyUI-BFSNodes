# Click-to-Picture SAM 3.1 bindings and token weights (experimental)

Requires BFSNodes **1.115.29**, a ComfyUI version with the native SAM3 nodes,
and the ordinary H3 models used in the existing native workflow.

## Workflows

- [Token / word editor only](examples/h3_tara_token_weight_editor.comfy.json) ([API](examples/h3_tara_token_weight_editor.api.json)).
- [Click selection + SAM 3.1 + token editor](examples/h3_tara_cast_sam31.comfy.json) ([API](examples/h3_tara_cast_sam31.api.json)).
- [Same selection with native shot loop](examples/h3_tara_cast_sam31_loop.comfy.json) ([API](examples/h3_tara_cast_sam31_loop.api.json)).

Update BFSNodes, restart ComfyUI and refresh the browser. This prototype has CPU
logic/mathematical checks, **not** an end-to-end browser/GPU video validation.
The current training process was not modified or interrupted.

## Select a performer and bind Picture X

1. Load the guide with VHS `force_rate=24`, `select_every_nth=1`. Connect consecutive Picture 1..N images to Select Cast. Its `references` output
   supplies the exact same bundle to Describe References through `reference_set`;
   there is no need to wire each image twice.
2. Leave the selector's `setup_only` on and queue once. Downstream generation is
   blocked while its player and reference thumbnails load.
3. Choose a Picture thumbnail, pause/scrub the guide and click its source
   performer. A numbered point is shown on that key frame. Ctrl/Cmd-click adds
   positive points; Shift-click excludes a region. Each Picture uses one key
   frame; marking it on another frame replaces that Picture's key frame.
   Native video controls occupy the bottom strip, so choose points above them.
4. Mark only the reference Pictures used for performer bindings. A background or
   style Picture does not need a performer point. Marks are stored in the
   workflow JSON. A changed video requires an explicit reset/reselection;
   existing marks are not silently erased.
5. Turn `setup_only` off and queue. **BFS H3 SAM 3.1 Cast Tracking** calls native
   `SAM3_Detect`, `SAM3_VideoTrack` and `SAM3_TrackToMask`. It tracks forward and
   backward from each chosen key frame, in chunks of at most 150 frames, retaining
   a separate mask per Picture. Its player shows colored tracked regions and
   the report lists empty frames. These colors are preview only.
6. **Apply Cast Reference Bindings** connects those masks to the corresponding
   Picture attention controls after TARA. Start at `reference_strength=1.25` and
   `other_reference_strength=1`. It applies a regional attention adjustment;
   it is not inpainting or pixel/background locking.

The checkpoint loader expects
`ComfyUI/models/checkpoints/sam3.1_multiplex_fp16.safetensors` from the
[official Comfy-Org checkpoint](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors).
The workflow includes the loader and model-download metadata. No SAM 3D or
KJ point-editor dependency is required for this selector.

In the loop variant, selection/tracking is performed on the **whole loaded
24fps video before the loop**. Apply Cast Bindings slices each track to the
current segment, then repeats/trims its tail to the prepared H3 length. Camera
cuts still need to be entered in Loop Start. A single point anchor does not
solve identity tracking through arbitrary cuts or occlusion: inspect the mask
preview and empty-frame report. Correct failed selections before relying on them.
There is no automatic closed-loop mask repair in this version.

## Inspect and weight words/tokens

`Native Reference Set to Video positive → Token / Word Weight Editor → Guider positive`

`TARA (and optional cast binding) MODEL → Token / Word Weight Editor → Guider MODEL`

Also connect its MODEL output to the scheduler. The example uses CFG 1 and
connects the same positive to its guider's negative socket, as before.

1. Enable `inspect_only` if you want to inspect tokens without generating, then queue. This encodes conditioning and displays
   tokens but blocks sampling. In the combined workflow, finish cast setup first.
2. Click token chips; Ctrl/Cmd-click selects several. Double-click selects matching
   tokenizer word/fragments, including repeated occurrences. Use **+0.1**, **−0.1**,
   **Reset selected** or **Reset overrides**. The JSON widget stores the forces and prompt
   fingerprint in the workflow. The supported force range is **0.05..3**; start
   with modest changes such as **0.8 / 1 / 1.2**, using the same seed.
3. Turn `inspect_only` off and queue to render. Editing chips does not change an
   already running generation. If the prompt changes, reset weights rather than
   apply old indices to new words. Up to 16 disjoint weighted ranges are supported
   in attention mode; contiguous tokens with the same force share a range.

The panel shows **real tokenizer IDs/labels linked to original prompt rows**,
plus the original embedding L2 norm. A norm is not attention, saliency or word
importance. This does not invert an arbitrary embedding to recover words.
Native H3's Qwen tokenizer disables SD-style inline weights. The BFS native
encoder now adds an optional prompt trace after encoding; visual row tags are
checked before tracing/weighting. For another native workflow, supply its exact
original prompt and H3 CLIP to the editor as a fallback; the caller is responsible
for supplying the actual prompt used in that conditioning. A generic CLIP or a
conditioning with no safe H3 mapping is rejected rather than guessed.

### Two different experiments

- **attention logits:** leaves the conditioning tensor unchanged. For target
  video queries, adds `log(force)` to attention logits of the selected text keys.
  This multiplies their **unnormalized** attention weight before shared softmax;
  it does not multiply final probabilities by the same factor. Requires TARA
  upstream and the editor MODEL/positive outputs to be used together. Visual
  Qwen rows and native Picture/video latent keys receive no direct token edit.
- **embedding magnitude:** multiplies the selected text embedding vectors by
  their force after encoding; all other rows and metadata stay unchanged. No
  empty-prompt subtraction or global renormalization is applied. This is not
  claimed to reproduce the SD1.5 weighting algorithm.

Contextual encoders mix meaning across tokens. Reducing one token does not
remove its meaning from all other embeddings; strengthening a token can affect
other attributes. Neither mode guarantees improved identity, sharpness or a
locked background. No learned model weights or reference pixels are changed.

Both token-logit weighting and active cast attention gains use **Flash Partitions**
for this prototype, bypassing fused Kitchen one-pass. Remove Model Sparse
Attention when using nonneutral attention controls. The normal TARA backend is
preserved when no attention gain is active; embedding-only mode adds no attention
policy of its own. Additional partition calls can reduce speed.

The loop example defaults token weights to neutral and `inspect_only=false`.
Manual token indices are tied to each generated shot prompt. Nonneutral index
overrides are not automatically remapped across different loop prompts. From
1.115.30, exact `weighted_phrases` directives can be rematched per shot (see below). Regional Picture/mask bindings do
support the loop's full-video-mask slicing described above.

## Validation boundaries

Seven CPU checks cover selection validation, **mocked** SAM call ordering and
chunk/frame coverage, loop-mask slicing/padding, tokenizer trace provenance,
embedding/metadata preservation, attention policy cloning, example graph cycles
and links, and exact composition of text/Picture/temporal logits against a dense
softmax oracle. Native SAM GPU execution, browser interactions and rendered
quality/speed still require local validation. No claim of cross-platform GPU
validation follows from these checks.

## Phrase weights in the prompt (1.115.30)

The same editor now supports phrases without selecting individual token chips.

**Option A — inline in the final execution prompt:**

```text
motion_transfer: Replace the performer with (the goblin from <Picture 1>:1.3). Preserve (the original background and its details:1.2).
```

Keep `parse_prompt_weights=true` on **BFS H3 Native Reference Set to Video**.
It removes explicit `(phrase:number)` syntax **before encoding**, then attaches
clean-text character spans to the token trace. The encoder itself does not apply
these forces: use its positive output through **Token / Word Weight Editor**.
Its `effective_prompt` output is the cleaned text that was actually encoded.
Write annotations in the final execution prompt, not only in an earlier user
request that Generate Text may rewrite/remove.

**Option B — directly in the editor's `weighted_phrases` field:**

```text
(golden fur:1.3)
(original background:1.2)
```

These are exact, case-sensitive phrases from the **already encoded** prompt.
The editor maps them using actual tokenizer offsets; it needs no new CLIP pass.
Standalone word edges do not match substrings inside larger words (`man` does
not select `woman`). Every exact occurrence is weighted. A missing phrase fails
explicitly instead of guessing. You may also paste a full annotated prompt into
this field if its cleaned text exactly matches the encoded prompt.

Both forms weight every overlapping subword token of the phrase. A token that
crosses a phrase boundary is affected as a whole. Ordinary `(parentheses)` remain
literal; only an explicit numeric suffix on a phrase is a weight. Timestamp
parentheses such as `(00:03.333)` and numeric-only ratios stay literal. Escape `\(` and `\)` for a
literal parenthesized expression that looks like a weight. Explicit nested or
overlapping groups multiply their forces, which must stay within **0.05..3**.
Inline and editor-field phrase weights also multiply if both select a token.

The panel shows these phrase defaults. Manual per-token adjustments override the
phrase-derived value; a manual value of 1 can neutralize an inline boost.
**Reset selected / Reset overrides** restores phrase defaults, rather than
removing the annotations. Clear the annotations/`weighted_phrases` to remove
those defaults. Exact tokenizer character offsets are required for phrase
mapping; unsupported tokenizers fail rather than apply weights to guessed rows.

These forces use the editor's selected **attention logits** or **embedding
magnitude** mode. They are not an extra simultaneous weighting stage. In a shot
loop, phrase directives are matched afresh against each shot's encoded prompt;
a missing phrase stops that experiment. Old manual token-index overrides still
require resetting when a shot prompt changes.

Validation added CPU checks for nested/literal/escaped syntax, Unicode offsets,
repeated phrases and word boundaries, visual-row preservation, manual neutral
overrides, and cleaning before the mocked native encode call. GPU quality and
browser interaction remain pending local validation.

From 1.115.30, new token nodes/workflows default `inspect_only=false` so sampling
continues normally. Old saved workflows retain their explicit value; switch it
off manually or load the updated file. Inspect mode now prints an explicit
message and the examples include a native PreviewAny status output, even if the
custom panel did not load. Repeated ~0.1s runs with no sampler can be cached
inspection blockers, not a failed GPU compile.

For older slow Qwen2 tokenizers, the mapper can construct an offset-capable local
companion from the loaded tokenizer's vocabulary/merges and added tokens. It
never downloads a replacement tokenizer and only accepts offsets when token IDs
match the actual H3 tokenizer exactly. A CPU check also verified matching against
the real local ComfyUI Qwen vocabulary; full GPU/video quality remains untested.

## One workflow with optional SAM and loop (1.115.31)

[Load the optional workflow](examples/h3_tara_optional_sam_loop.comfy.json)
([API](examples/h3_tara_optional_sam_loop.api.json)). The **START HERE** options
node has `use_sam31` and `use_shot_loop`, both OFF by default:

- Both OFF: the whole loaded clip is generated in one pass; no marks or SAM checkpoint required.
- SAM ON: enable selection/tracking and regional Picture bindings. Queue to view
  the guide, choose a Picture and mark its performer; turn `setup_only` off and
  queue again. Selection status now appears in a native PreviewAny and the log.
- Loop ON: split into configured chunks/manual camera cuts, apply continuity
  between same-take segments, and join the results. With loop OFF, chunk/cut
  settings are ignored and there is only one segment. Very long single-pass
  videos still need suitable VRAM; native H3's frame-count limit still applies.
- Both ON: track the full loaded video once and slice the masks for each segment.

The optional SAM loader checks its checkpoint only when enabled; disabled SAM
branches are lazy and do not load model weights. No VAE is requested for SAM.
The H3 video VAE remains a separate loader. Original Picture order and VLM prompt
stages remain in the graph; the token editor is neutral and not inspect-only by default.

Native H3 input settings are now stored/restored by name in the frontend helper,
including recovery of old widget arrays that assigned a boolean to the
`ref_image_size` combo. Its only valid values remain `match` and `max`.
CPU checks cover disabled/lazy branches, single-pass bypass, links, and named
widget restoration; browser/GPU validation is pending.
