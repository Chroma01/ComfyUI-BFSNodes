"""Sequential native H3 workflow expansion, independent of Shot Planner.

The loop body is expanded after each decoded segment. Models stay outside it.
Decoded segments are cached on disk; no mutable global session state is used.
"""
import json
import math
import uuid
from pathlib import Path

import torch

CATEGORY = 'BFS/MiniMax H3/Native Shot Loop'


def ranges(count, chunk_frames, cuts):
    if count < 1 or chunk_frames < 5:
        raise ValueError('Load a nonempty video and choose at least 5 frames per segment')
    try:
        cut_list = sorted(set(int(x.strip()) for x in cuts.split(',') if x.strip()))
    except ValueError as exc:
        raise ValueError('Camera cuts must be comma-separated source-frame indices') from exc
    if any(x <= 0 or x >= count for x in cut_list):
        raise ValueError('Camera cuts must be inside the loaded video frame range')
    result = []
    for a, b in zip([0] + cut_list, cut_list + [count]):
        for start in range(a, b, chunk_frames):
            result.append(dict(start=start, end=min(b, start+chunk_frames),
                               camera_cut=start in cut_list))
    return result


class BFSH3NativeShotLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',),
            'source_fps': ('FLOAT', {'default':24., 'min':.01}),
            'chunk_frames': ('INT', {'default':107, 'min':5, 'max':3600}),
            'camera_cuts': ('STRING', {'default':'', 'tooltip':'Comma-separated SOURCE frame indices, relative to the loaded clip. Empty = one continuous take split by chunk_frames. No automatic cut detection.'})},
            'optional': {'loop_state': ('BFS_H3_NATIVE_LOOP',)}}
    RETURN_TYPES = ('IMAGE', 'BFS_H3_NATIVE_LOOP', 'FLOAT', 'INT', 'INT')
    RETURN_NAMES = ('shot_images', 'loop_state', 'source_fps', 'shot_index', 'shot_count')
    FUNCTION = 'start'; CATEGORY = CATEGORY

    def start(self, images, source_fps=24., chunk_frames=107, camera_cuts='', loop_state=None):
        if not math.isfinite(source_fps) or abs(source_fps-24.)>1e-6:
            raise ValueError('Native H3 shot loop requires a 24 fps loaded video. Set VHS Load Video force_rate=24 before this node to keep boundaries and audio aligned.')
        if loop_state is None:
            import folder_paths
            folder = Path(folder_paths.get_temp_directory()) / 'bfs_h3_native_loop' / uuid.uuid4().hex
            folder.mkdir(parents=True)
            loop_state = dict(index=0, ranges=ranges(len(images),chunk_frames,camera_cuts),
                              folder=str(folder), segments=[], fps=24.,
                              source_frames=len(images), source_fps=source_fps)
        state = dict(loop_state)
        if len(images) != state['source_frames'] or source_fps != state['source_fps']:
            raise ValueError('Source video changed during the native shot loop')
        shot = state['ranges'][state['index']]
        print(f"[BFS H3 native loop] segment {state['index']+1}/{len(state['ranges'])}: source frames {shot['start']}..{shot['end']-1}",flush=True)
        frames=images[shot['start']:shot['end']]
        if len(frames)<5:
            frames=torch.cat((frames,frames[-1:].expand(5-len(frames),-1,-1,-1)))
        return frames,state,source_fps,state['index']+1,len(state['ranges'])


class BFSH3NativeShotContinuity:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'references': ('BFS_H3_REFERENCE_SET',),
            'prompt': ('STRING', {'forceInput':True}), 'loop_state': ('BFS_H3_NATIVE_LOOP',),
            'enabled': ('BOOLEAN', {'default':True})}}
    RETURN_TYPES = ('BFS_H3_REFERENCE_SET', 'STRING', 'STRING')
    RETURN_NAMES = ('references', 'prompt', 'continuity_status')
    FUNCTION = 'prepare'; CATEGORY = CATEGORY
    DESCRIPTION = 'Experimental native still-image continuity, not first-frame latent locking. Appends the previous generated last frame after all user pictures, only within a continuous take. Original Picture numbers stay fixed.'

    def prepare(self, references, prompt, loop_state, enabled=True):
        state=loop_state
        shot=state['ranges'][state['index']]
        if not enabled or not state['segments'] or shot['camera_cut']:
            reason='disabled' if not enabled else 'camera cut' if shot['camera_cut'] else 'first segment'
            return references,prompt,'Continuity OFF: '+reason
        if len(references)>=9:
            raise ValueError('Native continuity needs one free Picture slot: use at most 8 user reference images, or disable continuity')
        previous=torch.load(state['segments'][-1]['path'],map_location='cpu',weights_only=True)
        anchor=previous[-1:].clone()
        index=len(references)+1
        text=(f' Continuity anchor: <Picture {index}> is the last generated frame of the preceding segment of this same take. '
              'Use it only for consistent scene appearance, character identity and wardrobe at the boundary. '
              'The original assigned identity and environment pictures and user edits remain authoritative. '
              'Continue the current <Video 1> performance and timing; do not freeze or repeat the anchor pose, '
              'copy its earlier action, add another performer or introduce a transition.')
        return tuple(references)+(anchor,),prompt+text,f'Continuity ON: previous final frame is Picture {index} (native image reference, not a latent head guide)'


class BFSH3NativeShotLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'loop_state': ('BFS_H3_NATIVE_LOOP',)},
                'hidden': {'dynprompt':'DYNPROMPT','unique_id':'UNIQUE_ID'}}
    RETURN_TYPES=('IMAGE','FLOAT','STRING')
    RETURN_NAMES=('joined_images','fps','manifest')
    FUNCTION='finish'; CATEGORY=CATEGORY
    DESCRIPTION='Wire trimmed decoded shot frames here. Expands only the start-to-end workflow body until all segments are generated, then joins at 24 fps. Connect audio separately from the original video loader to Create Video.'

    def finish(self,images,loop_state,dynprompt=None,unique_id=None):
        state=dict(loop_state); index=state['index']; shot=state['ranges'][index]
        expected=shot['end']-shot['start']
        # Prepare Native Video pads tiny tail segments to H3's minimum grid.
        if len(images)<expected:
            raise ValueError(f'Segment {index+1} has {len(images)} decoded frames but needs {expected}; use pad last frame and remove video/frame caps')
        path=Path(state['folder'])/f'{index:05d}.pt'
        frames=images[:expected].detach().cpu().contiguous()
        tmp=path.with_suffix('.tmp');torch.save(frames,tmp);tmp.replace(path)
        segments=list(state['segments'])+[dict(path=str(path),frames=expected)]
        state.update(segments=segments,index=index+1)
        (path.parent/'manifest.json').write_text(json.dumps(state,indent=2))
        if state['index'] < len(state['ranges']):
            if dynprompt is None:
                raise ValueError('Native shot loop requires ComfyUI dynamic graph execution')
            return self.expand(dynprompt,str(unique_id),state)
        parts=[torch.load(x['path'],map_location='cpu',weights_only=True) for x in segments]
        shapes={tuple(x.shape[1:]) for x in parts}
        if len(shapes)!=1:
            raise ValueError('All segments must decode at the same resolution')
        return torch.cat(parts),24.,str(path.parent/'manifest.json')

    @staticmethod
    def expand(dynprompt,end,state):
        # Same start/end graph-expansion pattern used by BFS Shot Join, adapted
        # from Ethanfel's MiniMax H3 Context Loop / SxCP loop nodes (GPL-3.0).
        from comfy_execution.graph_utils import GraphBuilder,is_link
        feeds={};seen=set()
        def visit(nid):
            if nid in seen:return
            seen.add(nid)
            for value in dynprompt.get_node(nid).get('inputs',{}).values():
                if is_link(value):
                    parent=str(value[0]);feeds.setdefault(parent,[]).append(nid);visit(parent)
        visit(end)
        starts=[n for n in seen if dynprompt.get_node(n)['class_type']=='BFSH3NativeShotLoopStart']
        if len(starts)!=1:
            raise ValueError('Connect exactly one Native Shot Loop Start upstream of this Loop End')
        start=starts[0];body={start};pending=[start]
        while pending:
            for child in feeds.get(pending.pop(),[]):
                if child not in body:body.add(child);pending.append(child)
        graph=GraphBuilder();clones={n:graph.node(dynprompt.get_node(n)['class_type'], 'Next' if n==end else n) for n in body}
        for nid,node in clones.items():
            node.set_override_display_id(dynprompt.get_display_node_id(nid))
            for key,value in dynprompt.get_node(nid).get('inputs',{}).items():
                node.set_input(key,clones[str(value[0])].out(value[1]) if is_link(value) and str(value[0]) in body else value)
        clones[start].set_input('loop_state',state)
        return {'result':tuple(clones[end].out(i) for i in range(3)),'expand':graph.finalize()}


CLASSES=(BFSH3NativeShotLoopStart,BFSH3NativeShotContinuity,BFSH3NativeShotLoopEnd)
NODE_CLASS_MAPPINGS={c.__name__:c for c in CLASSES}
NODE_DISPLAY_NAME_MAPPINGS=dict(zip(NODE_CLASS_MAPPINGS,(
 'BFS H3 Native Shot Loop Start','BFS H3 Native Shot Continuity (Experimental)','BFS H3 Native Shot Loop End / Join')))
