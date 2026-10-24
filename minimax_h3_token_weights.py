"""Inspectable H3 prompt-token provenance and opt-in weighting.

Words are recovered from the actual tokenizer and original prompt, never guessed
by inverting contextual embeddings. Visual rows are excluded by native H3 tags.
"""
import hashlib
import json
import math
import re

import torch

TRACE='bfs_h3_prompt_trace'
OPTION='bfs_h3_token_attention'


def attach_trace(conditioning,clip,prompt):
    tokens=clip.tokenize(prompt)
    batches=tokens.get('qwen3vl_32b')
    if not batches or len(batches)!=1:
        raise ValueError('Prompt tracing requires the native H3 Qwen3-VL tokenizer')
    ids=[int(t[0]) for t in batches[0]]
    decoder=clip.tokenizer.qwen3vl_32b
    offsets=None
    backend=getattr(decoder,'tokenizer',None)
    if backend is not None:
        try:
            encoded=backend(prompt,add_special_tokens=False,return_offsets_mapping=True)
            if list(encoded['input_ids'])==ids:offsets=encoded['offset_mapping']
        except (TypeError,ValueError,NotImplementedError):
            pass
    words=list(re.finditer(r'\S+',prompt))
    records=[]
    for i,token in enumerate(ids):
        label=decoder.decode([token],skip_special_tokens=False)
        word=label
        if offsets:
            a,b=offsets[i]
            match=next((m for m in words if m.start()<b and m.end()>a),None)
            if match:word=match.group()
        records.append(dict(index=i,token_id=token,label=label,word=word))
    result=[]
    for tensor,meta in conditioning:
        tags=meta.get('minimax_token_tags')
        if tags is None or tensor.ndim!=3 or tensor.shape[0]!=1:
            raise ValueError('Native H3 token tags and a single conditioning sequence are required for safe tracing')
        n=tensor.shape[1];tags=tags.reshape(-1)
        if not ids or len(ids)>n or len(tags)!=n or not bool(tags[n-len(ids):].eq(1).all()):
            raise ValueError('Prompt suffix does not match the native H3 text/vision token layout')
        trace=dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt=prompt,
                   sequence_length=n,start=n-len(ids),tokens=records)
        result.append([tensor,dict(meta,**{TRACE:trace})])
    return result


def parse_weights(text,trace):
    try:data=json.loads(text)
    except (ValueError,TypeError) as exc:raise ValueError('Token weights must be JSON produced by the editor') from exc
    if not isinstance(data,dict):raise ValueError('Token weights must be a JSON object')
    if data.get('prompt_sha256') and data['prompt_sha256']!=trace['prompt_sha256']:
        raise ValueError('Prompt changed: reset token weights in the editor before applying them')
    if not isinstance(data.get('weights',{}),dict):raise ValueError('weights must map token indices to forces')
    result={}
    for key,value in data.get('weights',{}).items():
        index=int(key);gain=float(value)
        if not 0<=index<len(trace['tokens']) or not math.isfinite(gain) or not .05<=gain<=3:
            raise ValueError('Token indices must exist and forces must be finite between 0.05 and 3')
        if gain!=1:result[index]=gain
    return result


def attention_biases(layout,plan,control,device):
    text_stop=next(b for a,b,kind in layout.segments if kind=='text')
    if control['sequence_length']!=text_stop:
        raise ValueError('Token attention weights belong to a different conditioning length; connect this editor MODEL and positive together')
    return [(start,stop,torch.full((plan.target_stop-plan.target_start,),math.log(gain),device=device))
            for start,stop,gain in control['ranges']]


def contiguous_ranges(weights,start):
    result=[]
    for index,gain in sorted(weights.items()):
        row=start+index
        if result and result[-1][1]==row and result[-1][2]==gain:
            a,b,g=result[-1];result[-1]=(a,row+1,g)
        else:result.append((row,row+1,gain))
    if len(result)>16:
        raise ValueError('This prototype supports up to 16 disjoint weighted token ranges; select fewer words or a contiguous phrase')
    return result


class BFSH3TokenWeightEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'conditioning':('CONDITIONING',),'model':('MODEL',),
            'mode':(['attention logits (experimental)','embedding magnitude (experimental)'],{}),
            'inspect_only':('BOOLEAN',{'default':True,'tooltip':'On: show tokens and stop before sampling. Adjust tokens in the panel; switch off and queue again to render.'}),
            'weights_json':('STRING',{'default':'{}','multiline':True})},
            'optional':{'clip':('CLIP',),'original_prompt':('STRING',{'forceInput':True})}}
    RETURN_TYPES=('CONDITIONING','MODEL','STRING')
    RETURN_NAMES=('positive','model','report')
    FUNCTION='edit';CATEGORY='BFS/MiniMax H3'
    DESCRIPTION=('Experimental word/subword controls. Requires traced BFS native H3 conditioning, or the exact original prompt and H3 CLIP. '
        'Does not decode embeddings back into words. Attention mode requires TARA upstream and uses Flash Partitions. '
        'No direct edits to visual rows, reference latents or pooled embeddings. Changes take effect on next queue.')

    def edit(self,conditioning,model,mode='attention logits (experimental)',inspect_only=True,weights_json='{}',clip=None,original_prompt=None):
        if not conditioning or TRACE not in conditioning[0][1]:
            if clip is None or original_prompt is None:
                raise ValueError('Connect traced BFS Native Reference Set to Video conditioning, or provide H3 CLIP and the exact original prompt. Embeddings alone cannot be decoded into original words.')
            conditioning=attach_trace(conditioning,clip,original_prompt)
        if len(conditioning)!=1:
            raise ValueError('Token editor currently supports one H3 conditioning sequence, not scheduled conditioning')
        tensor,meta=conditioning[0];trace=meta[TRACE]
        if tensor.shape[1]!=trace['sequence_length']:
            raise ValueError('Conditioning length no longer matches token trace')
        error=None
        try:weights=parse_weights(weights_json,trace)
        except ValueError as exc:weights={};error=str(exc)
        rows=[trace['start']+i for i in range(len(trace['tokens']))]
        tags=meta.get('minimax_token_tags')
        if tags is None or not bool(tags.reshape(-1)[rows].eq(1).all()):
            raise ValueError('Token trace overlaps visual or invalid rows; refusing to weight those embeddings')
        norms=tensor[0,rows].detach().float().norm(dim=-1).cpu().tolist()
        records=[dict(r,norm=norms[i],gain=weights.get(i,1.)) for i,r in enumerate(trace['tokens'])]
        ui={'bfs_h3_tokens':[dict(tokens=records,prompt_sha256=trace['prompt_sha256'],mode=mode,
              message=error or ('Inspect only: sampling blocked. Adjust weights, switch inspect_only off, queue again.' if inspect_only else 'Weights applied for this queued execution.'))]}
        if error or inspect_only:
            from comfy_execution.graph_utils import ExecutionBlocker
            stop=ExecutionBlocker(None)
            return {'ui':ui,'result':(stop,stop,error or 'Inspect only')}
        if mode=='embedding magnitude (experimental)':
            if weights:
                edited=tensor.clone()
                for index,gain in weights.items():edited[:,trace['start']+index,:]*=gain
                conditioning=[[edited,dict(meta)]]
        elif mode=='attention logits (experimental)':
            if weights:
                if not model.model_options.get('bfs_h3_native_temporal_v1'):
                    raise ValueError('Token attention mode needs TARA upstream; route this MODEL output to the guider and use its positive output together')
                spans=contiguous_ranges(weights,trace['start'])
                model=model.clone();opts=dict(model.model_options.get('transformer_options',{}))
                if OPTION in opts:
                    raise ValueError('Use only one Token Weight Editor per MODEL path')
                opts[OPTION]=dict(sequence_length=trace['sequence_length'],ranges=spans)
                model.model_options['transformer_options']=opts
        else:raise ValueError('Unknown token weight mode')
        report=json.dumps(dict(mode=mode,weighted_tokens=len(weights),forces=weights),indent=2)
        return {'ui':ui,'result':(conditioning,model,report)}


NODE_CLASS_MAPPINGS={'BFSH3TokenWeightEditor':BFSH3TokenWeightEditor}
NODE_DISPLAY_NAME_MAPPINGS={'BFSH3TokenWeightEditor':'BFS H3 Token / Word Weight Editor (Experimental)'}
