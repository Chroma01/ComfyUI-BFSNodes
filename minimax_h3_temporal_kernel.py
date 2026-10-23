"""Optional native Comfy Kitchen extension; lazy import, no environment mutation."""
import importlib


def api_for(q):
    if q.device.type != 'cuda':
        return None
    try:
        api = importlib.import_module('comfy_kitchen.temporal_attention')
    except ImportError:
        return None
    if getattr(api, 'VERSION', None) != 'h3_native_temporal_kernel_v1':
        raise RuntimeError('Unsupported Comfy Kitchen temporal extension version; update the temporal overlay.')
    return api if api.is_available(q.device) else None


def policy_for(api, plan, ctx):
    key=(api.VERSION,ctx.strength,ctx.window,ctx.falloff,ctx.cap)
    if getattr(plan,'_native_temporal_policy_key',None)!=key:
        times = getattr(plan, 'frame_times', ())
        if not times:
            times = tuple(ctx.times[0, plan.target_start:plan.target_stop:plan.target_frame_rows].detach().cpu().tolist())
        plan._native_temporal_policy = api.TemporalPolicy(
            plan.source_start, plan.source_stop, plan.target_start, plan.target_stop,
            plan.source_frame_rows, plan.target_frame_rows, tuple(times),
            ctx.strength, ctx.window, ctx.falloff, ctx.cap)
        plan._native_temporal_policy_key=key
    return plan._native_temporal_policy


def run(kind, q, k, v, plan, ctx, **kwargs):
    api = api_for(q)
    if api is None:
        return None
    policy = policy_for(api, plan, ctx)
    if not getattr(plan, '_native_'+kind+'_logged', False):
        print(f'[BFS H3 native temporal] Kitchen ONE-PASS {kind} kernel ACTIVE; temporal bias fused; no per-frame attention loop.', flush=True)
        setattr(plan, '_native_'+kind+'_logged', True)
    dtype=q.dtype
    if str(dtype)=='torch.float32':
        import torch
        q,k,v=[t.to(torch.bfloat16) for t in (q,k,v)]
    fn = api.int8_attention if kind == 'dense' else api.sol_attn
    return fn(q,k,v,policy,**kwargs).to(dtype)
