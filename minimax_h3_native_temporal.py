"""Experimental native H3 temporal inference, matching ai-toolkit native_temporal_v1.

Per-model wrappers only: no global ComfyUI monkey patches or extra weights.
"""
import copy
import math
import torch
try:
    from . import minimax_h3_native_temporal_core as core
except ImportError:
    import minimax_h3_native_temporal_core as core

MARKER = 'bfs_h3_native_temporal_v1'


def temporal_layout(native, refs, keyframes, video_index=1):
    if keyframes:
        raise ValueError('Native temporal mode cannot be combined with Add Guide, first-frame guides or history. Remove those guides.')
    required = ('position_ids', 'segments', 'seq_len', 'signature')
    if any(not hasattr(native, x) for x in required):
        raise RuntimeError('Unsupported H3 PackedLayout API; update ComfyUI.')
    result = copy.copy(native)
    result.position_ids = native.position_ids.clone()
    source = torch.zeros(native.seq_len, dtype=torch.bool)
    target = torch.zeros_like(source)
    ta, tb, _ = next(s for s in native.segments if s[2] == 'video')
    target[ta:tb] = True
    # Native packing is text, each reference's optional audio then video, targets.
    cursor = next(s[1] for s in native.segments if s[2] == 'text')
    videos = 0
    for ref in refs:
        kind = ref['kind']
        if kind == 'image':
            cursor += (ref['latent_h']//2)*(ref['latent_w']//2)
        elif kind == 'audio':
            cursor += 2*ref['ref_audio_t']
        elif kind in ('video', 'video_audio'):
            videos += 1
            cursor += 2*ref.get('ref_audio_t', 0)
            n = ref['latent_t']*(ref['latent_h']//2)*(ref['latent_w']//2)
            if videos == video_index:
                source[cursor:cursor+n] = True
                result.position_ids[cursor:cursor+n, 0] += native.position_ids[ta, 0] - native.position_ids[cursor, 0]
            cursor += n
        else:
            raise ValueError('Unknown native H3 reference kind: '+str(kind))
    if not source.any():
        raise ValueError(f'Native temporal mode requires <Video {video_index}> as a native video reference. Connect Reference to Video positive conditioning.')
    st = torch.unique_consecutive(result.position_ids[source, 0])
    tt = torch.unique_consecutive(result.position_ids[target, 0])
    if st.shape != tt.shape or not torch.allclose(st, tt, atol=1e-9, rtol=0):
        raise ValueError('Native temporal guide and output must have identical temporal grids. Use the same frame window, 24 fps and matching H3 frame count (e.g. 107). No automatic motion stretching is applied.')
    return result, source, target


def attention_override(ctx, previous=None):
    def override(original, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        # Token refiner and other attention paths retain their original backend.
        n = q.shape[-2] if skip_reshape else q.shape[1]
        if n != ctx.times.shape[1] or k.shape[-2 if skip_reshape else 1] != n:
            if previous:
                return previous(original, q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape, **kwargs)
            return original(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape, **kwargs)
        if mask is not None:
            raise ValueError('Native temporal attention does not support another attention mask override.')
        if not skip_reshape:
            q, k, v = [x.view(x.shape[0], x.shape[1], heads, -1).transpose(1, 2) for x in (q,k,v)]
        out = core.attention(q, k, v, ctx)
        return out if skip_output_reshape else out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[1]*out.shape[3])
    return override


def patch_temporal_model(model, video_index=1, strength=2.0, window=0.2, falloff=0.25, cap=6.0):
    cfg = core.options(dict(native_temporal_alignment=True, native_temporal_strength=strength,
                            native_temporal_window_seconds=window, native_temporal_falloff_seconds=falloff,
                            native_temporal_max_penalty=cap))
    if isinstance(video_index, bool) or not isinstance(video_index, int) or video_index < 1:
        raise ValueError('motion_video_index must be a positive integer')
    if model is None:
        raise ValueError('Native temporal mode needs the MODEL input. Route this node MODEL output to the sampler/guider.')
    if model.model_options.get(MARKER):
        raise ValueError('Native temporal alignment is already installed. Use only one native temporal node per model path.')
    net = model.get_model_object('diffusion_model')
    if net.__class__.__name__ != 'MiniMaxH3Model':
        raise ValueError('Native temporal alignment requires a native ComfyUI MiniMaxH3 model.')
    opts = model.model_options.get('transformer_options', {})
    if opts.get('optimized_attention_override') or opts.get('patches_replace', {}).get('dit'):
        raise ValueError('Remove VSA and other attention/block replacement patches before native temporal alignment.')
    patched = model.clone()
    previous = patched.model_options.get('model_function_wrapper')
    cache = [None, None]
    logged = [False]

    def wrapper(model_function, args):
        conds = args.get('c', {})
        payload = conds.get('minimax_payload') or {}
        native = payload.get('layout')
        if native is None:
            raise RuntimeError('Native H3 layout missing. Update ComfyUI and use native AV latent conditioning.')
        if cache[0] is not native:
            cache[:] = [native, temporal_layout(native, payload.get('refs', []), payload.get('keyframes'), video_index)]
        layout, source, target = cache[1]
        device = args['input'][0].device if isinstance(args['input'], (list, tuple)) else args['input'].device
        to = dict(conds.get('transformer_options', {}))
        if to.get('optimized_attention_override') or to.get('patches_replace', {}).get('dit'):
            raise ValueError('Native temporal alignment conflicts with another attention/block replacement patch.')
        times = layout.position_ids[:, 0].float()[None].to(device)/40.0
        ctx = core.TemporalContext(times, source[None].to(device), target[None].to(device),
                                  torch.ones_like(source[None], device=device), strength, window, falloff, cap,
                                  torch.tensor([strength,window,falloff,cap],device=device,dtype=torch.float32))
        to['optimized_attention_override'] = attention_override(ctx)
        copied = dict(args, c=dict(conds, minimax_payload=dict(payload, layout=layout), transformer_options=to))
        if not logged[0]:
            print(f'[BFS H3 native temporal] {core.VERSION}: <Video {video_index}>, {int(source.sum())} motion rows; temporal RoPE aligned; soft window={window}s strength={strength} cap={cap}; images remain native; Add Guide OFF', flush=True)
            logged[0] = True
        if previous:
            return previous(model_function, copied)
        return model_function(copied['input'], copied['timestep'], **copied['c'])

    patched.set_model_unet_function_wrapper(wrapper)
    patched.model_options[MARKER] = dict(version=core.VERSION, video_index=video_index, **cfg)
    return patched


class BFSMiniMaxH3NativeTemporal:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',),
            'motion_video_index': ('INT', {'default':1, 'min':1, 'max':100,
                'tooltip':'Native <Video N> supplying motion. Picture numbering is independent.'}),
            'strength': ('FLOAT', {'default':2.0,'min':0.0,'max':20.0,'step':0.1}),
            'window_seconds': ('FLOAT', {'default':0.2,'min':0.0,'max':10.0,'step':0.01}),
            'falloff_seconds': ('FLOAT', {'default':0.25,'min':0.01,'max':10.0,'step':0.01}),
            'max_penalty': ('FLOAT', {'default':6.0,'min':0.0,'max':30.0,'step':0.1})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'BFS/MiniMax H3'
    DESCRIPTION = ('Experimental native_temporal_v1 inference for the matching motion-transfer LoRA. '
        'Keep the motion video in MiniMax H3 Reference to Video positive conditioning; route this MODEL '
        'to the guider. No Add Guide or VSA. Guide/output must share a 24fps temporal grid. '
        'Does not encode video or load the LoRA. Defaults match training.')

    def apply(self, model, motion_video_index=1, strength=2.0, window_seconds=0.2, falloff_seconds=0.25, max_penalty=6.0):
        return (patch_temporal_model(model,motion_video_index,strength,window_seconds,falloff_seconds,max_penalty),)

NODE_CLASS_MAPPINGS = {'BFSMiniMaxH3NativeTemporal': BFSMiniMaxH3NativeTemporal}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSMiniMaxH3NativeTemporal': 'BFS MiniMax H3 Native Temporal Alignment (Experimental)'}
