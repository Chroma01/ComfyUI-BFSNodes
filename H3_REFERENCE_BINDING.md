# H3 Reference Attention Binding (experimental, 1.115.28)

A separate **BFS H3 Reference Attention Binding (Experimental)** node controls
how strongly generated video tokens attend to each native Picture. It is an
inference experiment, not a trained identity tracker or a proven quality fix.

[Load the example workflow](examples/h3_tara_reference_binding.comfy.json)
([API](examples/h3_tara_reference_binding.api.json)). Update BFSNodes, restart
ComfyUI and refresh the browser before loading it.

## Connect

`Model + LoRA → TARA → Reference Attention Binding → CFGGuider`

Also use the resulting model for the scheduler. Keep pictures/video conditioning
on the existing Native Reference Set to Video path. Nothing new connects to the
positive-conditioning socket. Multiple binding nodes may be chained after TARA.
The existing loop workflow is a separate example in [Native Shot Loop](H3_NATIVE_SHOT_LOOP.md).

## Controls

- `enabled`: off returns the original model without adding a reference policy.
- `picture_index`: existing native Picture number; video numbers are independent.
- `reference_strength`: gain for that Picture. **1 is neutral**. Start with **1.25**;
  compare 1, 1.25 and 1.5 with the same seed, prompt and all other settings.
- `other_reference_strength`: gain for other Pictures inside the same region.
  Start at **1**. A value such as **0.75** reduces their contribution there. This
  also affects background or continuity Pictures, so do not lower it indiscriminately.
- `target_mask` (optional): white selects generated-video regions where the gains
  apply. Supply one static mask or a tracked mask for every prepared shot frame
  at 24fps. It is a mask of the output/guide position, **not of the reference
  photograph**. No mask means the gain applies to the whole output video.

For two subjects, chain two nodes: Picture 1 with subject A's tracked mask;
Picture 2 with subject B's tracked mask. Their masks must follow them across
position changes and match output framing. This node does not segment or track.
Overlapping masks combine their log gains. It does not infer subject roles from
the prompt; the prompt's explicit identity assignments remain necessary.

A whole long-video mask cannot be connected directly to a loop's short-shot
binding without slicing/padding it to that prepared shot. The newer **Apply Cast Reference Bindings** node (1.115.29) supplies automatic
loop-mask slicing in its [SAM 3.1 workflow](H3_CAST_AND_TOKEN_CONTROLS.md). The raw
Reference Attention Binding node still expects a mask already matched to its shot.

## Exact rule and scope

For a target query inside mask weight `m` and a key belonging to Picture `i`,
the attention logit receives `m * log(gain_i)`. Thus unnormalized weight is
multiplied by `gain_i ** m`, before the shared softmax normalization. This is
**not** multiplication of the reference pixels, V tensors, decoded images, or
a promise that the final attention probability rises by that percentage.

Only target-video queries to native image keys are modified. Target-to-guide
keys keep TARA's temporal penalty; other keys and non-target queries retain their
original rules. Picture indices remain unchanged. Soft masks are area-downsampled
to target patch resolution and averaged over H3 latent temporal spans. The mask
must have either one frame or the prepared `17n+5` shot length, including padding.

The prototype composes reference gains with TARA through exact Flash Partitions
and LSE merging. **While nonzero gains are active, Kitchen one-pass is bypassed.**
Remove Model Sparse Attention for this experiment; sparse routing is not supported.
No new binary, pip install or compile is needed for the reference-control code,
but CUDA FP16/BF16 native FlashAttention with LSE is required. Partition overhead
may make this slower than the fused TARA-only Kitchen path.

Turning the node off, or setting both gains to 1, adds no reference policy.
This does not disable TARA itself. The log announces `[BFS H3 reference binding]
ACTIVE` and `using Flash Partitions` when the new policy is used.

## Evidence so far

CPU checks compare the composed output against an explicit dense softmax oracle,
with and without temporal penalties, including fractional masks and unchanged
non-target queries. Additional checks cover temporal mask reduction, invalid
Picture/mask rejection, exact disabled/neutral bypass, model cloning, policy cache
refresh and the original TARA partition policy. Six checks passed. No GPU video
quality, speed, identity consistency, Windows execution or browser validation
has been performed for this prototype. GPU training was left untouched.
