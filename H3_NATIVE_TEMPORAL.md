# Experimental H3 native temporal alignment

Matches ai-toolkit branch [`feat/h3-native-temporal-motion`](https://github.com/alisson-anjos/ai-toolkit/tree/feat/h3-native-temporal-motion), implementation `native_temporal_v1` (commit e0ecf3a). This mode retains a native video reference, aligns its temporal RoPE coordinates with the target, and softly penalizes target queries attending to distant guide times. It preserves native image-reference coordinates and native spatial geometry. It does not add learned weights or load a LoRA.

Use the matching native-temporal LoRA when available. Existing Add Guide motion-transfer LoRAs were trained under a different conditioning contract. Quality is experimental; CPU math/routing tests are not a full video validation.

## Standalone native workflow

1. Load the native MiniMax H3 base and the native-temporal LoRA.
2. Pass the guide as `<Video 1>` to **MiniMax H3 Reference to Video**, alongside any number of identity reference pictures supported by that native node. Picture and video numbering are independent.
3. Place **BFS MiniMax H3 Native Temporal Alignment (Experimental)** on the MODEL path, after LoRA loading. Route its MODEL output to your guider/sampler. Keep Reference to Video's positive and AV latent outputs connected normally.
4. Select `motion_video_index = 1` (or the actual `<Video N>` supplying motion).
5. Use the same 24 fps frame window for guide and output. H3 may trim frame counts to its supported cadence; use a matching valid count such as 107.

Do not add Add Guide, first-frame guides, VSA, or other attention/block replacements. The node raises a clear error for incompatible layouts and temporal lengths instead of silently stretching motion. Source video audio is supported by native packing but is not temporally biased by this patch. For CFG paths, every conditioning branch evaluated by the patched model must retain the native motion reference; an empty negative branch without references cannot provide its timeline. Training preview uses CFG 1.

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

CUDA uses PyTorch compiled FlexAttention without allocating an S×S bias matrix. Requires a PyTorch build with FlexAttention and a compatible native ComfyUI H3 PackedLayout/attention override API. The tiny dense CPU implementation is a test oracle only and refuses long sequences. Start with model compilation/malloc graph optimizations disabled until validated on your local ComfyUI build.

Runtime confirmation:

```text
[BFS H3 native temporal] native_temporal_v1: <Video 1>, ... motion rows; temporal RoPE aligned; soft window=0.2s strength=2.0 cap=6.0; images remain native; Add Guide OFF
```

This log confirms the patch ran, not that visual motion or identity quality is correct.

## Validation

Tests compare an audited native ComfyUI layout with positions exported by the actual training implementation, including image references and reference audio. They verify original conditioning/layout immutability, temporal-length rejection, attention score math/output shapes, token-refiner fallback, per-model routing and incompatible patches. Planner tests verify native-video conditioning and returned MODEL routing. Full ComfyUI GPU generation remains pending local validation; the active training process is not restarted or modified by this source change.
