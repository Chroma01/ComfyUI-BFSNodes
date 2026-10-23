"""Prefer native one-pass Kitchen kernels, with a frame-wise fallback.

The fallback avoids partition LSE.

Each target frame shares a key bias across its spatial queries. The compact
bias is O(frames * sequence), never an expanded sequence-by-sequence mask.
INT8 quantization changes numerical results; GPU performance needs measurement.
"""
import inspect
import torch
try:
    from . import minimax_h3_temporal_kernel as native_kernel
except ImportError:
    import minimax_h3_temporal_kernel as native_kernel


def kitchen_function():
    try:
        import comfy_kitchen
    except ImportError as exc:
        raise RuntimeError('Comfy Kitchen temporal attention requires comfy-kitchen with masked INT8 attention.') from exc
    fn = getattr(comfy_kitchen, 'int8_attention', None)
    if fn is None or 'attn_mask' not in inspect.signature(fn).parameters:
        raise RuntimeError('Update comfy-kitchen: this temporal backend requires int8_attention(attn_mask=...).')
    return fn


def attention(q, k, v, plan, ctx, kernel=None):
    if q.shape[0] != 1 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('Kitchen temporal attention requires one packed self-attention sequence')
    if kernel is None:
        native = native_kernel.run('dense',q,k,v,plan,ctx)
        if native is not None:
            return native
    kernel = kitchen_function() if kernel is None else kernel
    if ctx.strength == 0 or ctx.cap == 0:
        return kernel(q, k, v)
    sa, sb, ta, tb = plan.source_start, plan.source_stop, plan.target_start, plan.target_stop
    key = (q.device, q.shape[2], float(ctx.strength), float(ctx.window), float(ctx.falloff), float(ctx.cap))
    if getattr(plan, '_kitchen_key', None) != key:
        frame_times = ctx.times[0, ta:tb:plan.target_frame_rows]
        source_times = ctx.times[0, sa:sb]
        dt = (frame_times[:, None] - source_times[None, :]).abs()
        penalty = (ctx.strength * (torch.relu(dt-ctx.window)/ctx.falloff).square()).clamp(max=ctx.cap)
        biases = torch.zeros((plan.frames, q.shape[2]), device=q.device, dtype=torch.float32)
        biases[:, sa:sb] = -penalty
        plan._kitchen_biases, plan._kitchen_key = biases, key
    out = torch.empty_like(q)
    for lo, hi in ((0, ta), (tb, q.shape[2])):
        if hi > lo:
            out[:, :, lo:hi] = kernel(q[:, :, lo:hi], k, v)
    for frame in range(plan.frames):
        lo = ta + frame * plan.target_frame_rows
        hi = lo + plan.target_frame_rows
        # Broadcast only: Kitchen expands this key bias with zero query stride.
        mask = plan._kitchen_biases[frame].view(1, 1, 1, -1)
        out[:, :, lo:hi] = kernel(q[:, :, lo:hi], k, v, attn_mask=mask)
    return out
