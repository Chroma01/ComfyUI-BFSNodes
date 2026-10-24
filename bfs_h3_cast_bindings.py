"""Click-to-Picture selections and native SAM 3.1 masks, independent of Planner."""
import hashlib
import json
import math
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from . import bfs_video_io as video_io
    from .bfs_h3_native_workflow import collect_references
    from .minimax_h3_reference_binding import BFSMiniMaxH3ReferenceBinding
except ImportError:
    import bfs_video_io as video_io
    from bfs_h3_native_workflow import collect_references
    from minimax_h3_reference_binding import BFSMiniMaxH3ReferenceBinding

CATEGORY='BFS/MiniMax H3/Cast Bindings'
COLORS=((.2,.9,.65),(.7,.4,1.),(1.,.6,.2),(.2,.7,1.),(1.,.3,.5))


def preview_video(images,prefix,overlay=None):
    import folder_paths
    folder=Path(folder_paths.get_temp_directory());name=f'{prefix}_{uuid.uuid4().hex}.mp4'
    h,w=images.shape[1:3];scale=min(1.,640/max(h,w));h=max(2,int(h*scale)//2*2);w=max(2,int(w*scale)//2*2)
    writer=video_io.Mp4Writer(str(folder/name),w,h,24.,crf=23,who='BFS Cast selector')
    digest=hashlib.sha256()
    try:
        for start in range(0,len(images),16):
            chunk=images[start:start+16,...,:3].detach().float().cpu()
            chunk=F.interpolate(chunk.movedim(-1,1),size=(h,w),mode='bilinear',align_corners=False).movedim(1,-1)
            raw=(chunk.clamp(0,1)*255).round().byte().numpy();digest.update(raw.tobytes())
            if overlay:
                for i,entry in enumerate(overlay):
                    mask=F.interpolate(entry['mask'][start:start+len(chunk),None].float(),size=(h,w),mode='bilinear',align_corners=False).movedim(1,-1)
                    color=torch.tensor(COLORS[i%len(COLORS)])
                    chunk=chunk*(1-mask*.45)+color*(mask*.45)
            writer.write(chunk.contiguous())
        writer.close()
    except Exception:
        writer.abort();raise
    return dict(filename=name,subfolder='',type='temp',frames=len(images),fps=24.,width=w,height=h,
                source=hashlib.sha256(f'{tuple(images.shape)}:{digest.hexdigest()}'.encode()).hexdigest())


def parse_selection(text,count,frames):
    try:data=json.loads(text)
    except (ValueError,TypeError) as exc:raise ValueError('Cast selection must be JSON from the selector') from exc
    if not isinstance(data,dict):raise ValueError('Cast selection must be an object')
    if not isinstance(data.get('bindings',[]),list):raise ValueError('bindings must be a list')
    seen=set()
    for entry in data.get('bindings',[]):
        if not isinstance(entry,dict):raise ValueError('Each binding must be an object')
        picture=entry.get('picture');frame=entry.get('frame')
        if isinstance(picture,bool) or not isinstance(picture,int) or not 1<=picture<=count or picture in seen:
            raise ValueError('Each selected Picture must be attached and occur only once')
        if isinstance(frame,bool) or not isinstance(frame,int) or not 0<=frame<frames:
            raise ValueError('Cast key frame is outside the loaded video')
        if not entry.get('positive'):raise ValueError('Place at least one positive point for each Picture')
        for point in entry.get('positive',[])+entry.get('negative',[]):
            if any(not isinstance(point.get(k),(int,float)) or not math.isfinite(point[k]) or not 0<=point[k]<=1 for k in ('x','y')):
                raise ValueError('Cast points must have normalized x/y coordinates between 0 and 1')
        seen.add(picture)
    return data


class BFSH3SelectCastBindings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'request':('STRING',{'multiline':True}),
            'selection_json':('STRING',{'default':'{}','multiline':True}),
            'setup_only':('BOOLEAN',{'default':True,'tooltip':'Queue to load video, pause and click subjects. Turn off and queue again to track and generate.'})},
            'optional':{f'ref_image_{i}':('IMAGE',) for i in range(1,10)}}
    RETURN_TYPES=('IMAGE','BFS_H3_CAST_SELECTION','STRING','BFS_H3_REFERENCE_SET');RETURN_NAMES=('images','cast_selection','request','references')
    FUNCTION='select';CATEGORY=CATEGORY
    DESCRIPTION='At 24 fps, choose Picture X in the player and click its source performer. Ctrl/Cmd-click adds points; Shift-click excludes. One key frame per Picture. No marks are burned into the guide.'

    def select(self,images,request,selection_json='{}',setup_only=True,**kwargs):
        refs=collect_references(kwargs);video=preview_video(images,'bfs_cast_source')
        import folder_paths
        from PIL import Image
        thumbs=[]
        for i,image in enumerate(refs,1):
            frame=image[0,...,:3].detach().float().cpu().clamp(0,1)
            h,w=frame.shape[:2];scale=min(1.,160/max(h,w))
            frame=F.interpolate(frame.movedim(-1,0)[None],size=(max(1,round(h*scale)),max(1,round(w*scale))),mode='bilinear',align_corners=False)[0].movedim(0,-1)
            name=f'bfs_cast_picture_{uuid.uuid4().hex}.png';Image.fromarray((frame*255).round().byte().numpy()).save(Path(folder_paths.get_temp_directory())/name)
            thumbs.append(dict(filename=name,type='temp',subfolder='',picture=i))
        problem=None
        try:selection=parse_selection(selection_json,len(refs),len(images))
        except ValueError as exc:selection={};problem=str(exc)
        if selection.get('bindings') and selection.get('source')!=video['source']:
            problem='Video changed: reset the marks and select the subjects again'
        if not selection.get('bindings'):problem=problem or 'Choose Picture X and click its subject in the video'
        ui={'bfs_cast_selector':[dict(video=video,pictures=thumbs,message=problem or 'Bindings ready. Turn setup_only off and queue to generate.') ]}
        if setup_only or problem:
            from comfy_execution.graph_utils import ExecutionBlocker
            stop=ExecutionBlocker(None);return {'ui':ui,'result':(stop,stop,stop,refs)}
        notes=[]
        for e in selection['bindings']:
            p=e['positive'][0]
            notes.append(f"<Picture {e['picture']}> supplies the appearance of the source subject selected at global loaded-video frame {e['frame']} ({e['frame']/24:.3f}s), normalized position x={p['x']:.3f}, y={p['y']:.3f}. This selection identifies the performer, not a color or a rendered annotation.")
        return {'ui':ui,'result':(images,selection,request+'\nExplicit user source bindings: '+' '.join(notes),refs)}


def track_sequence(images,model,mask,threshold):
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack,SAM3_TrackToMask
    pieces=[];start=0
    while start<len(images):
        end=min(len(images),start+150)
        data=SAM3_VideoTrack.execute(images=images[start:end],model=model,initial_mask=mask,
            detection_threshold=threshold,max_objects=1,detect_interval=1).args[0]
        tracked=SAM3_TrackToMask.execute(track_data=data,object_indices='0').args[0].float().cpu()
        pieces.append(tracked if start==0 else tracked[1:]);mask=tracked[-1:]
        if end==len(images):break
        start=end-1
    return torch.cat(pieces)


class BFSH3SAM31CastMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'model':('MODEL',),'cast_selection':('BFS_H3_CAST_SELECTION',),
            'threshold':('FLOAT',{'default':.5,'min':.05,'max':1.,'step':.05}),
            'max_tracking_side':('INT',{'default':640,'min':128,'max':1536,'step':32})}}
    RETURN_TYPES=('BFS_H3_CAST_MASKS','STRING');RETURN_NAMES=('cast_masks','report')
    FUNCTION='track';CATEGORY=CATEGORY
    DESCRIPTION='Native ComfyUI SAM 3.1: segment each Picture selection on its own key frame, track forward and backward, and retain a separate mask per Picture. Load sam3.1_multiplex_fp16.safetensors with CheckpointLoaderSimple. No text-only identity guessing.'

    def track(self,images,model,cast_selection,threshold=.5,max_tracking_side=640):
        from comfy_extras.nodes_sam3 import SAM3_Detect
        h,w=images.shape[1:3];scale=min(1.,max_tracking_side/max(h,w));size=(max(2,int(h*scale)//2*2),max(2,int(w*scale)//2*2))
        work=F.interpolate(images[...,:3].detach().float().cpu().movedim(-1,1),size=size,mode='bilinear',align_corners=False).movedim(1,-1)
        result=[];stats=[]
        for entry in cast_selection['bindings']:
            frame=entry['frame'];key=work[frame:frame+1]
            coords=lambda pts:json.dumps([dict(x=p['x']*size[1],y=p['y']*size[0]) for p in pts])
            initial=SAM3_Detect.execute(model=model,image=key,positive_coords=coords(entry['positive']),negative_coords=coords(entry.get('negative',[])),threshold=threshold,refine_iterations=2).args[0][:1].float().cpu()
            if not bool(initial.any()):raise ValueError(f"SAM 3.1 found an empty initial mask for Picture {entry['picture']}; adjust its selection points")
            forward=track_sequence(work[frame:],model,initial,threshold)
            if frame:
                backward=track_sequence(work[:frame+1].flip(0),model,initial,threshold).flip(0)
                mask=torch.cat((backward[:-1],forward))
            else:mask=forward
            result.append(dict(picture=entry['picture'],mask=mask))
            coverage=mask.mean((1,2));stats.append(dict(picture=entry['picture'],empty_frames=int((coverage==0).sum()),mean_coverage=float(coverage.mean())))
        overlay=preview_video(images,'bfs_cast_tracking',result)
        return {'ui':{'bfs_cast_preview':[dict(video=overlay,stats=stats)]},
                'result':(dict(bindings=result,source_frames=len(images)),json.dumps(stats,indent=2))}


class BFSH3ApplyCastBindings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'cast_masks':('BFS_H3_CAST_MASKS',),
            'prepared_length':('INT',{'default':107,'min':5,'step':17}),
            'reference_strength':('FLOAT',{'default':1.25,'min':.05,'max':4.,'step':.05}),
            'other_reference_strength':('FLOAT',{'default':1.,'min':.05,'max':4.,'step':.05}),
            'enabled':('BOOLEAN',{'default':True})},'optional':{'loop_state':('BFS_H3_NATIVE_LOOP',)}}
    RETURN_TYPES=('MODEL',);FUNCTION='apply';CATEGORY=CATEGORY
    DESCRIPTION='Applies every selected Picture/mask binding after TARA. Optionally slices full-video tracks to the current loop segment and repeats/trims its tail to the prepared H3 length.'

    def apply(self,model,cast_masks,prepared_length=107,reference_strength=1.25,other_reference_strength=1.,enabled=True,loop_state=None):
        if not enabled:return (model,)
        if loop_state is not None and cast_masks['source_frames']!=loop_state['source_frames']:
            raise ValueError('Full-video masks must use the same loaded 24fps video as Loop Start')
        for entry in cast_masks['bindings']:
            mask=entry['mask']
            if loop_state is not None:
                shot=loop_state['ranges'][loop_state['index']];mask=mask[shot['start']:shot['end']]
            mask=mask[:prepared_length]
            if len(mask)<prepared_length:mask=torch.cat((mask,mask[-1:].expand(prepared_length-len(mask),-1,-1)))
            model=BFSMiniMaxH3ReferenceBinding().apply(model,picture_index=entry['picture'],reference_strength=reference_strength,
                other_reference_strength=other_reference_strength,target_mask=mask)[0]
        return (model,)


CLASSES=(BFSH3SelectCastBindings,BFSH3SAM31CastMasks,BFSH3ApplyCastBindings)
NODE_CLASS_MAPPINGS={c.__name__:c for c in CLASSES}
NODE_DISPLAY_NAME_MAPPINGS=dict(zip(NODE_CLASS_MAPPINGS,('BFS H3 Select Cast → Picture (Click Video)','BFS H3 SAM 3.1 Cast Tracking','BFS H3 Apply Cast Reference Bindings')))
