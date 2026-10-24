"""Experimental target-to-native-picture logit gains, with optional tracked masks.

Gains multiply unnormalized attention weights via log(gain), not image pixels,
V magnitude or the final decoded image. No trained weights are introduced.
"""
import math

import torch
import torch.nn.functional as F

OPTION = 'bfs_h3_reference_binding'


def image_spans(layout, refs):
    cursor=next(s[1] for s in layout.segments if s[2]=='text')
    result=[]
    for ref in refs:
        kind=ref['kind']
        if kind=='image':
            rows=(ref['latent_h']//2)*(ref['latent_w']//2)
            result.append((cursor,cursor+rows));cursor+=rows
        elif kind=='audio':cursor+=2*ref['ref_audio_t']
        elif kind in ('video','video_audio'):
            cursor+=2*ref.get('ref_audio_t',0)
            cursor+=ref['latent_t']*(ref['latent_h']//2)*(ref['latent_w']//2)
        else:raise ValueError('Unknown native reference kind: '+str(kind))
    return result


def target_mask(mask,layout,plan):
    total=plan.target_stop-plan.target_start
    if mask is None:return torch.ones(total)
    mask=mask.detach().float().cpu()
    if mask.ndim!=3 or not mask.numel() or not torch.isfinite(mask).all():
        raise ValueError('Reference binding mask must be a finite nonempty [frames,height,width] MASK')
    positions=layout.position_ids[plan.target_start:plan.target_stop]
    h=int(torch.unique(positions[:,1]).numel());w=int(torch.unique(positions[:,2]).numel())
    if h*w!=plan.target_frame_rows:
        raise ValueError('Unsupported H3 target spatial layout for reference binding')
    spans=[(1,4,4,4,4)[i%5] for i in range(plan.frames)]
    if len(mask) not in (1,sum(spans)):
        raise ValueError(f'Reference binding mask needs one static frame or {sum(spans)} frames matching the prepared 24fps shot; got {len(mask)}')
    resized=F.interpolate(mask[:,None].clamp(0,1),size=(h,w),mode='area')[:,0]
    if len(mask)==1:return resized.expand(plan.frames,-1,-1).reshape(-1).contiguous()
    cursor=0;frames=[]
    for span in spans:
        frames.append(resized[cursor:cursor+span].mean(0));cursor+=span
    return torch.stack(frames).reshape(-1)


def build_biases(layout,refs,plan,controls,device):
    spans=image_spans(layout,refs)
    gains=[torch.zeros(plan.target_stop-plan.target_start) for _ in spans]
    for control in controls:
        picture=control['picture']
        if not 1<=picture<=len(spans):
            raise ValueError(f'Reference binding Picture {picture} is not attached; only {len(spans)} native pictures found')
        mask=target_mask(control['mask'],layout,plan)
        for index in range(len(spans)):
            gain=control['strength'] if index==picture-1 else control['other_strength']
            gains[index]+=mask*math.log(gain)
    return [(a,b,g.to(device=device)) for (a,b),g in zip(spans,gains) if bool(g.any())]


class BFSMiniMaxH3ReferenceBinding:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model':('MODEL',),'enabled':('BOOLEAN',{'default':True}),
            'picture_index':('INT',{'default':1,'min':1,'max':9}),
            'reference_strength':('FLOAT',{'default':1.25,'min':.05,'max':4.,'step':.05,
                'tooltip':'Multiplier of unnormalized attention weights to this Picture inside the mask. 1 = neutral. Not a quality percentage.'}),
            'other_reference_strength':('FLOAT',{'default':1.,'min':.05,'max':4.,'step':.05,
                'tooltip':'Multiplier for other Pictures inside the same mask. Keep 1 initially; try 0.75 to reduce identity mixing. Includes environment/continuity pictures too.'})},
            'optional':{'target_mask':('MASK',{'tooltip':'One static mask or one mask per prepared shot frame at 24fps. White = affected. Must track the target subject; this node performs no tracking.'})}}
    RETURN_TYPES=('MODEL',); FUNCTION='apply'; CATEGORY='BFS/MiniMax H3'
    DESCRIPTION=('Experimental reference-attention control. Connect AFTER TARA and send this MODEL to the guider. '
        'Uses exact Flash Partitions instead of Kitchen/Sol kernels while active. No mask = global target influence. '
        'Masks refer to output/guide positions, not the reference image. Does not change prompt bindings or guarantee identity.')

    def apply(self,model,enabled=True,picture_index=1,reference_strength=1.25,other_reference_strength=1.,target_mask=None):
        if not enabled:return (model,)
        if not model.model_options.get('bfs_h3_native_temporal_v1'):
            raise ValueError('Connect Reference Attention Binding AFTER the BFS H3 TARA node, then route this MODEL to the guider')
        if isinstance(picture_index,bool) or not isinstance(picture_index,int) or not 1<=picture_index<=9:
            raise ValueError('Picture index must be an integer between 1 and 9')
        for gain in (reference_strength,other_reference_strength):
            if not math.isfinite(gain) or not .05<=gain<=4:
                raise ValueError('Reference attention multipliers must be finite and between 0.05 and 4')
        if reference_strength==1 and other_reference_strength==1:return (model,)
        if target_mask is not None:
            if target_mask.ndim!=3 or not target_mask.numel() or not torch.isfinite(target_mask).all():
                raise ValueError('target_mask must be finite nonempty [frames,height,width]')
            target_mask=target_mask.detach().float().cpu().clone()
        result=model.clone()
        options=dict(result.model_options.get('transformer_options',{}))
        options[OPTION]=list(options.get(OPTION,[]))+[dict(picture=picture_index,
            strength=float(reference_strength),other_strength=float(other_reference_strength),mask=target_mask)]
        result.model_options['transformer_options']=options
        return (result,)


NODE_CLASS_MAPPINGS={'BFSMiniMaxH3ReferenceBinding':BFSMiniMaxH3ReferenceBinding}
NODE_DISPLAY_NAME_MAPPINGS={'BFSMiniMaxH3ReferenceBinding':'BFS H3 Reference Attention Binding (Experimental)'}
