"""Apply H3 reference layouts to a cloned native Comfy layout."""
import torch
try:
    from .minimax_h3_reference_rope import shift_reference, add_source_phase, validate_options
except ImportError:
    from minimax_h3_reference_rope import shift_reference, add_source_phase, validate_options

MARKER='bfs_h3_aligned_guide'


def ref_span(ref):
    if ref['kind']=='image': return 1.0
    audio=float(ref.get('ref_audio_t',0))
    if ref['kind']=='audio': return audio
    vt=ref['latent_t']
    return max(audio,sum((1,4,4,4,4)[i%5]*5/3 for i in range(vt)))


def apply_rope_layout(layout, keyframes, refs):
    # Remove only experimental refs from the native cumulative target clock.
    spans=[ref_span(r) for r in refs]
    removed=sum(span for span,r in zip(spans,refs) if r.get(MARKER,{}).get('rope_layout','native')!='native')
    positions=layout.position_ids.clone()
    source=torch.zeros(layout.seq_len,dtype=torch.float32)
    target_rows=next((slice(a,b) for a,b,k in layout.segments if k=='video'))
    for a,b,kind in layout.segments:
        if kind in ('video','audio','cond','cond_audio'):positions[a:b,0]-=removed
    target=positions[target_rows]
    guide_segments=iter((a,b,k) for a,b,k in layout.segments if k in ('cond','cond_audio'))
    def apply_ranges(ranges, opts):
        if opts.get('rope_layout','overlap')=='sidecar' and ranges:
            indices=torch.cat([torch.arange(a,b) for a,b,_ in ranges])
            positions[indices]=shift_reference(positions[indices],target,'sidecar',opts.get('sidecar_margin',0.0))
        if opts.get('source_phase',False):
            for a,b,_ in ranges:source[a:b]=opts['source_id']*opts.get('phase_scale',1.0)
    for kf in keyframes:
        ranges=[]
        if kf.get('latent') is not None:ranges.append(next(guide_segments))
        if kf.get('audio_latent') is not None:ranges.append(next(guide_segments))
        opts=kf.get(MARKER,{})
        apply_ranges(ranges, opts)
    ref_segments=iter((a,b,k) for a,b,k in layout.segments if k in ('ref_img','ref_audio'))
    cursor=float(layout.signature[0])
    native_cursor=cursor
    target_origin=float(target[0,0])
    for ref,span in zip(refs,spans):
        ranges=[]
        if ref.get('ref_audio_t',0)>0:ranges.append(next(ref_segments))
        if ref['kind'] in ('image','video','video_audio'):ranges.append(next(ref_segments))
        opts=ref.get(MARKER,{})
        name=opts.get('rope_layout','native')
        origin=cursor if name=='native' else target_origin
        for a,b,kind in ranges:
            positions[a:b,0]+=origin-native_cursor
        apply_ranges(ranges, opts)
        native_cursor+=span
        if name=='native':cursor+=span
    layout.position_ids=positions
    layout.bfs_source_phase_values=source if torch.count_nonzero(source) else None
    return layout
