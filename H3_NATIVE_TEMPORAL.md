# Experimental H3 native temporal alignment

Matches ai-toolkit branch [`feat/h3-native-temporal-motion`](https://github.com/alisson-anjos/ai-toolkit/tree/feat/h3-native-temporal-motion), implementation `native_temporal_v1` (commit e0ecf3a). This mode retains a native video reference, aligns its temporal RoPE coordinates with the target, and softly penalizes target queries attending to distant guide times. It preserves native image-reference coordinates and native spatial geometry. It does not add learned weights or load a LoRA.

Existing Add Guide motion-transfer LoRAs were trained under a different conditioning contract, but the step-10500 example below shows useful transfer to this native temporal mode. This is an experimental observation, not a guarantee for every LoRA. CPU math/routing tests alone do not validate video quality.

## Video comparison: an Add Guide-trained LoRA with native temporal alignment

[**Play the synchronized side-by-side video**](https://github.com/alisson-anjos/ComfyUI-BFSNodes/raw/refs/heads/feat/h3-native-temporal/docs/media/h3_native_temporal_comparison.mp4)

All six columns play together in one 15-second video, starting at the beginning of each supplied export. From left to right:

1. Original guide.
2. Step-10500 motion-transfer LoRA + native temporal alignment, with guide scale **1.0**.
3. The same step-10500 LoRA + native temporal alignment, with guide scale **0.5**.
4. The same step-10500 LoRA + guide latents / Add Guide.
5. The same step-10500 LoRA + native reference, without the temporal patch.
6. Base model + native reference, without a LoRA or temporal patch.

The temporal columns are labeled **NEW**. The footer includes the character and setting reference thumbnails extracted from the supplied exports. The user confirmed that all supplied runs used the **same seed**. At guide scale 0.5, character appearance remains consistent in the reviewed samples, but some source poses are reproduced less closely than at 1.0 (for example, the raised-arm gesture around 3.7 seconds). Other generation settings were not fully captured in the exports.

**Apparently, a LoRA trained with guide latents can also work with this new native temporal method.** In this example, the step-10500 Add Guide-trained LoRA follows the source poses and camera framing with the temporal patch, while maintaining the reference dog's white T-shirt and jeans through the sampled beginning, middle and end. Its Add Guide result also follows the performance. Native reference without the temporal patch preserves the character more successfully with this LoRA than without it, but reproduces the choreography less closely. The base native-reference result loses the reference T-shirt.

A plausible explanation is that the LoRA's learned appearance and motion-transfer behavior can be reused through shared model weights, while the runtime patch supplies temporal correspondence through native reference attention. This example does not isolate that mechanism experimentally or establish universal compatibility. The temporal mode still uses native video-reference latents; it does **not** inject Add Guide latents.

**The reported identity drift concerns the separate, fresh native-temporal LoRA at step 750, not the step-10500 LoRA shown here.** The 750 checkpoint is not included in this comparison. Comparing those checkpoints would also compare different training histories and training durations.

The requested background replacement is partial: source architecture and layout remain mixed with the new street reference. The seed is confirmed equal by the user, but these supplied exports do not include complete prompt and configuration metadata, so this is a qualitative demonstration rather than a controlled benchmark. Pixel difference and PSNR against the guide cannot rank replacement quality when character and background changes are intentional. The 30 fps comparison export does not establish the model's generation frame rate.

## Standalone native workflow

1. Load the native MiniMax H3 base and the native-temporal LoRA.
2. Pass the guide as `<Video 1>` to **MiniMax H3 Reference to Video**, alongside any number of identity reference pictures supported by that native node. Picture and video numbering are independent.
3. Place **BFS MiniMax H3 Native Temporal Alignment (Experimental)** on the MODEL path, after LoRA loading. Route its MODEL output to your guider/sampler. Keep Reference to Video's positive and AV latent outputs connected normally.
4. Select `motion_video_index = 1` (or the actual `<Video N>` supplying motion).
5. Use the same 24 fps frame window for guide and output. H3 may trim frame counts to its supported cadence; use a matching valid count such as 107.

Do not add Add Guide, first-frame guides, VSA, Model Sparse Attention, or unknown attention/block replacements. The official Model Attention Backend node is supported from 1.115.13; see the Kitchen section below. The node raises a clear error for incompatible layouts and temporal lengths instead of silently stretching motion. Source video audio is supported by native packing but is not temporally biased by this patch. For CFG paths, every conditioning branch evaluated by the patched model must retain the native motion reference; an empty negative branch without references cannot provide its timeline. Training preview uses CFG 1.

## Planner

In **BFS Shot H3 Conditioning** (Advanced or Simple), choose:

```text
guide_mode = native temporal alignment (experimental)
```

Connect MODEL from your LoRA loader to the conditioning node, then connect the conditioning node's MODEL output to the guider. Advanced keeps its existing output order. Simple appends MODEL after positive/latent/prompt; old links keep their indices. Reload older nodes or recreate them if the new socket does not appear.

The shot is encoded as native `<Video 1>`, with identity pictures unchanged. The patch is applied separately to the returned model for each shot, and the current shot's packed temporal grid is used at sampling. Use 24 fps, first_frame none, mask_guide off and continuity off. Full-frame or ordinary native generation masks may still be used; masks do not create a temporal guide. Colored source overlays are retained if enabled, but color/mesh datasets have not yet been validated with this fresh experiment.

The Planner option uses the training defaults. To tune the policy manually, instead choose the ordinary `native reference video (Video 1)` conditioning mode and place the standalone node on the model path. Never install both patches on the same model path.

## Exact policy

Only target-video query to selected-motion-video key edges receive:

```text
score -= min(strength * (max(abs(t_target - t_guide) - window_seconds, 0) / falloff_seconds)^2, max_penalty)
```

Defaults: strength 2.0, window 0.20 s, falloff 0.25 s, maximum penalty 6.0. Set strength 0 to test temporal RoPE alignment alone. Bypass the node to test fully native baseline. Text, image references, target self-attention, reference audio, and token-refiner attention retain their normal behavior.

As of 1.115.11 the default backend is **flash partitions (no compile)**: it uses native CUDA FlashAttention with log-sum-exp outputs, positive partition merging and bounded frame chunks. No dense S×S bias tensor or torch.compile is used in this backend. CUDA FP16/BF16 and PyTorch's native FlashAttention-with-LSE primitive are required. The original compiled FlexAttention remains selectable for comparison. Start with model compilation/malloc graph optimizations disabled until validated on your local ComfyUI build.

Runtime confirmation:

```text
[BFS H3 native temporal] native_temporal_v1: <Video 1>, ... motion rows; temporal RoPE aligned; soft window=0.2s strength=2.0 cap=6.0; images remain native; Add Guide OFF
```

This log confirms the patch ran, not that visual motion or identity quality is correct.

## Validation

Tests compare an audited native ComfyUI layout with positions exported by the actual training implementation, including image references and reference audio. They verify original conditioning/layout immutability, temporal-length rejection, attention score math/output shapes, token-refiner fallback, per-model routing and incompatible patches. Planner tests verify native-video conditioning and returned MODEL routing. The user has since reported completed RTX 5090 model evaluations; the timings and limitations are documented in the illustrated explanation. Full visual-quality validation remains incomplete. The active training process is not restarted or modified by this source change.


## Performance changes in 1.115.11

See the [illustrated Flash Partitions explanation](docs/H3_FLASH_PARTITIONS.md) for attention scope, derivation, log-sum-exp merging, measurements and limitations.

The temporal rule and RoPE policy are unchanged. For motion keys, the modified softmax weight is decomposed exactly as `exp(-cap) + (exp(-penalty) - exp(-cap))`. The first term is ordinary attention over all motion keys. The positive correction is zero for distant times and can be computed with short per-frame batches. Non-motion keys keep weight one. All partitions are recombined with their log-sum-exp normalizers. This is not a hard temporal window or sparse approximation. BF16 partition merging has normal floating-point rounding differences; the small CUDA comparison against SDPA measured relative output error 0.00452.

A synthetic long-sequence benchmark on RTX PRO 6000 Blackwell / PyTorch 2.12.0+cu130, 107126 tokens, 8 heads, head dimension 128 measured approximately 0.628 s for original FlexAttention, 0.422 s for Flash partitions and 0.327 s for unmodified native FlashAttention. Training was running concurrently, so these are indicative isolated-kernel timings, not whole-model inference results or RTX 5090 speed guarantees. Small sequences have additional partition-launch overhead and can favor FlexAttention. No full-model GPU generation is claimed here.

Standalone node: `attention_backend` selects the implementation. Planner conditioning: `native_temporal_backend` selects it. Neither setting loads a LoRA or changes its weights. Defaults select Flash partitions; choose `flex (original)` for an A/B comparison.

Planner `native_guide_scale` defaults to 1.0 for compatibility. Try 0.5 on VRAM-constrained GPUs: it reduces only the motion-reference spatial dimensions (about four times fewer guide tokens), keeping all timestamps/frames, output dimensions and identity images unchanged. This changes the visual detail supplied by the guide and must be evaluated for quality. In standalone workflows resize only the motion video before Reference to Video to get the same effect.

The console logs the first attention's input shape and completion time and the first two full model evaluation times. If attention is fast but a full model evaluation remains slow, investigate model loading/offload and the remaining blocks rather than assuming compilation. These timings synchronize CUDA at those diagnostic points only. The active training code and its backend are unchanged.

## Comfy Kitchen and Model Attention Backend (1.115.13)

The official **Model Attention Backend** node is now recognized automatically. Select **comfy kitchen attention** there and route its MODEL through native temporal conditioning to the guider. Its official PyTorch selection is also accepted. Unrelated attention overrides and Model Sparse Attention remain incompatible. The runtime MODEL wrapper recognizes a supported dense selection applied before or after the temporal patch; keep only one temporal patch on the path.

With Kitchen selected, the console reports `backend=comfy kitchen (experimental)`. You can also explicitly select this backend in the standalone node or Planner conditioning. Without an upstream Kitchen selection, the existing default remains Flash Partitions. If ComfyUI itself falls back from unavailable Kitchen to PyTorch, the temporal patch does not claim Kitchen was selected. A supported selection from Model Attention Backend takes precedence over the BFS backend selector.

This backend calls `comfy_kitchen.int8_attention(attn_mask=...)` directly. Each target latent frame's spatial queries share one broadcast key bias. Non-target queries use ordinary Kitchen attention, while target-to-guide scores receive the same bounded temporal penalty as the original rule. Native temporal RoPE alignment remains active. It uses no Add Guide injection, no partition log-sum-exp, and no full sequence-by-sequence bias allocation: bias storage is `latent_frames * packed_rows`. It retains full-key attention, not sparse attention. Multiple frame calls can repeat Kitchen quantization work, so compatibility does not imply acceleration.

Initial isolated tests on an RTX PRO 6000 Blackwell with PyTorch 2.12.0+cu130 and comfy-kitchen 0.2.35:

- CPU tests compare the frame-wise policy against the dense mathematical oracle, including zero strength/cap and unchanged non-target queries. Routing tests cover automatic detection of the official dense node.
- A real CUDA BF16 test at `[1, 2, 1248, 128]` measured relative output norm error **0.01506** against the FP32 oracle, with finite outputs. INT8 quantization is approximate; this is not a visual quality measurement.
- At `[1, 2, 42836, 128]`, 22 latent frames and 7,062 guide rows, one warmed evaluation measured **0.1043 s Kitchen** versus **0.01739 s Flash Partitions**. This initial Kitchen path was slower. The test used two heads rather than H3's full 56, with concurrent training; these are diagnostics, not end-to-end or RTX 5090 timings. No speed claim or default-backend change follows from this result.

The active training backend is unchanged. Full ComfyUI video output and local performance still require validation. This implementation does not enable sol-attn, SLA or VSA alongside temporal alignment.
