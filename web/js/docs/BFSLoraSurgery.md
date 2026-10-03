# BFS LoRA Surgery

Read any LoRA's layers and blocks, then scale or drop them and apply the result live.
No retraining, no files written.

## Why

When a LoRA learns a defect (smoothed skin, identity drifting when the subject is far from
the camera, a pose it refuses to copy) that defect usually lives in one module family and a
range of blocks, not in the whole adapter. Scaling or dropping that group and regenerating
under a fixed seed tells you where it lives, in minutes.

In the cases tested so far the culprit was the MLP **input** projection, in a subset of
blocks. The MLP output projection barely mattered and attention was innocent. Treat that as
a starting point rather than a rule: which group and which block range are right differs per
LoRA, which is why the panel measures instead of assuming.

**Names depend on the base model**, so the panel lists whatever the file you loaded actually
contains:

| base | family | MLP input | MLP output | blocks |
|---|---|---|---|---|
| Qwen-Image-2.1 | `transformer_blocks` | `img_mlp.gate_up` | `img_mlp.out` | 32 |
| Krea 2 | `blocks`, plus `txtfusion.*` | `mlp.gate`, `mlp.up` | `mlp.down` | 28 |
| MiniMax-H3 | `blocks`, plus `token_refiner.blocks` | `mlp.fc1` | `mlp.fc2` | 50 |

Other bases will show other names again (`w1`/`w3`, `fc1`, `proj_in`). Read the list, do not
assume a name.

## How it works

The node discovers the structure from the key names, so it is not tied to any model. It
handles the diffusers convention (`lora_A` / `lora_B`) and the kohya one (`lora_down` /
`lora_up` plus `alpha`). Families like `transformer_blocks`, `double_blocks`, `token_refiner`
or `txtfusion` are detected automatically, along with the block indices and module types in
each.

The panel lists every family with its module types, sorted by how large the update is
(`‖ΔW‖`), and that bar is a quick read on where training actually invested.

Scaling multiplies only the up/B factor, since `(sB)A = s(BA)`. Exact, and negative scales
work.

## Using it

1. Pick a LoRA in `lora_name`. The panel reads it and lists the families.
2. Click a type's `→` to select it, then click blocks to select them (shift-click for a range).
3. Set a scale and press **add rule**, or **add as regex** if you want the same selection
   written as a pattern you can hand-edit.
4. Queue the prompt. The edited LoRA is applied to the model in memory.

**Order matters.** Later rules override earlier ones for modules they both match, so put
boosts first and drops last. Use ↑/↓ to reorder.

`rules` is plain JSON and is saved with the workflow:

```json
[
  {"enabled": true, "match": {"type": "attn.*"}, "scale": 1.15},
  {"enabled": true, "match": {"type": "img_mlp.gate_up", "blocks": "8-15"}, "scale": 0},
  {"enabled": true, "match": {"blocks": "16-23", "type": "*"}, "scale": 0}
]
```

A match is either structural (`family`, `type`, `blocks`) or a `regex` tested against the
module's full path. `type` accepts a trailing `*`. `blocks` accepts `8-15`, `0,4,7` or
`8-15,24-31`.

## Saving the result

Two ways, depending on whether you are exploring or building a pipeline.

**From the panel.** Once the rules look right, type a name (or leave it blank) and press
`save as file`. It writes straight into `models/loras/surgery` and reports the result inline.
Nothing else to wire.

**From the graph.** `BFS LoRA Surgery` has a `rules` output. Connect it to the `rules` input
of **BFS LoRA Surgery (save)** and the file is produced as part of the run, with no JSON
copied by hand. Both nodes read the same `lora_name`, so point them at the same file.

Either way the rules are applied identically, since both call the same code.

Leave `filename` empty and the name is built from the rules, so the file says what was done
to it:

```
mylora__attn-x1.15__gate_up-b8_15-off__all-b16_23-off.safetensors
```

Names are capped and sanitized, so a long source name or a 30-rule recipe still produces a
valid filename: when it would exceed the limit, both halves are trimmed and a short hash of
the full recipe is appended, so two different recipes never collapse onto the same name.

The file also carries its own history. Alongside the original metadata (which is preserved
untouched), it writes:

* `bfs_surgery_summary`, one line any metadata viewer will show:
  `136/192 modules kept: dropped img_mlp.gate_up in blocks 8-15; all modules in blocks 16-23,
  scaled attn.* x1.15 (from mylora.safetensors)`
* `bfs_surgery`, the full record as JSON: tool and timestamp, source filename **and its
  sha256** so provenance survives a rename, the readable recipe, the exact rule list, and the
  module counts (total, kept, dropped, scaled)
* `ss_output_name` updated to the new name

That means a file you share months from now can still explain what was done to it, and can be
traced back to the checkpoint it came from even if someone renamed either one.

`subfolder` defaults to `surgery` to keep these out of your main list, `save_dtype` can
downcast, and `overwrite` is off by default, so a repeat run gets `_1`, `_2` rather than
destroying the previous one.

## A warning

A variant can win every metric you are looking at and still have destroyed what you trained.
Removing the MLP path entirely gave the sharpest skin in one of the cases here, and made the
LoRA stop copying expression from the source image. Always check a deliberately hard case
(a strong expression, an unusual angle) with your eyes, not only the numbers.

Pruning after training is also not the same as training with those layers excluded. Excluding
them during training may simply fail to converge, because the layer is where the defect lodges,
not where it comes from. That is usually the dataset.
