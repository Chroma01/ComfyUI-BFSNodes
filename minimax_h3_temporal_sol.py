"""Experimental temporal policy through native Comfy Kitchen sol-attn.

The public kernel accepts equal-length Q/K/V and a key-only bias. Therefore
this compatibility path runs one unbiased pass and one full-sequence pass per
latent target frame, retaining only that frame's query rows. It is deliberately
not advertised as a fast one-pass temporal sparse kernel.
"""
import inspect
import torch


def sol_function():
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise RuntimeError('Temporal sol-attn requires Comfy Kitchen.') from exc
    fn = getattr(ck, 'sol_attn', None)
    if fn is None or 'key_bias' not in inspect.signature(fn).parameters:
        raise RuntimeError('Update comfy-kitchen: temporal sol-attn requires sol_attn(key_bias=...).')
    return fn


def attention(q, k, v, plan, ctx, *, tau, token_aug=0, sink_q=(0, 0), kernel=None):
    if q.shape[0] != 1 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError('Temporal sol-attn requires one packed self-attention sequence with head dimension 128.')
    if plan.source_stop > plan.target_start:
        raise ValueError('Temporal sol-attn requires native motion references before the target video.')
    fn = sol_function() if kernel is None else kernel
    # Sol's key bias is applied only in the exact branch: all conditioning keys,
    # including the selected motion video, must be forced exact regardless of
    # the ordinary sparse node's sink_conditioning setting.
    sink_blocks = [0, (plan.target_start+63)//64]
    common = dict(tau=float(tau), sink_blocks=sink_blocks, topk_ratio=0., tail=True, token_aug=int(token_aug))
    qs, ks, vs = [x.transpose(1, 2) for x in (q, k, v)]
    if kernel is None and q.dtype == torch.float32:
        qs, ks, vs = [x.to(torch.bfloat16) for x in (qs, ks, vs)]
    out = fn(qs, ks, vs, sink_q=list(sink_q), **common).transpose(1, 2).clone()
    if ctx.strength == 0 or ctx.cap == 0:
        return out.to(q.dtype)
    cache_key = (q.device, q.shape[2], float(ctx.strength), float(ctx.window), float(ctx.falloff), float(ctx.cap))
    if getattr(plan, '_sol_bias_key', None) != cache_key:
        ft = ctx.times[0, plan.target_start:plan.target_stop:plan.target_frame_rows]
        st = ctx.times[0, plan.source_start:plan.source_stop]
        penalty = (ctx.strength*(torch.relu((ft[:, None]-st[None, :]).abs()-ctx.window)/ctx.falloff).square()).clamp(max=ctx.cap)
        bias = torch.zeros((plan.frames, q.shape[2]), device=q.device, dtype=torch.float32)
        bias[:, plan.source_start:plan.source_stop] = -penalty
        plan._sol_bias, plan._sol_bias_key = bias, cache_key
    for frame in range(plan.frames):
        lo = plan.target_start+frame*plan.target_frame_rows
        hi = lo+plan.target_frame_rows
        # Outputs for other frames are discarded: their temporal bias differs.
        result = fn(qs, ks, vs, key_bias=plan._sol_bias[frame], sink_q=[0, 0], **common)
        out[:, :, lo:hi] = result[:, lo:hi].transpose(1, 2)
    return out.to(q.dtype)
