# BFS LoRA Surgery

Read any LoRA's layers and blocks, then scale or drop them and apply the result live.
No retraining, no files written.

## Why

When a LoRA learns a defect (smoothed skin, identity drifting when the subject is far from
the camera, a pose it refuses to copy) that defect usually lives in one module family and a
range of blocks, not in the whole adapter. Scaling or dropping that group and regenerating
under a fixed seed tells you where it lives, in minutes.

In two real cases, the culprit was the MLP **input** projection (`gate_up`, `w1`+`w3`,
`mlp.gate`/`mlp.up`, depending on the architecture), in a subset of blocks. The MLP output
projection barely mattered, and attention was innocent.

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

**BFS LoRA Surgery (save)** applies the same rules and writes a real `.safetensors` into
`models/loras`, for when a recipe is settled and you want it usable anywhere.

Leave `filename` empty and the name is built from the rules, so the file says what was done
to it:

```
mylora__attn-x1.15__gate_up-b8_15-off__all-b16_23-off.safetensors
```

The recipe also goes into the safetensors metadata under `bfs_surgery` (source file, the
rule list, modules kept and dropped), alongside the original metadata. That survives being
shared, so months later the file can still explain itself.

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
