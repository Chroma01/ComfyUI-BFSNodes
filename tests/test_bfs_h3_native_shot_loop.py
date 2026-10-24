import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_shot_loop_test',ROOT/'bfs_h3_native_shot_loop.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_ranges_respect_camera_cuts_and_cover_every_frame():
    spans=m.ranges(220,107,'110')
    assert [(s['start'],s['end'],s['camera_cut']) for s in spans]==[(0,107,False),(107,110,False),(110,217,True),(217,220,False)]
    assert sum(s['end']-s['start'] for s in spans)==220
    with pytest.raises(ValueError):m.ranges(220,107,'220')


def test_tail_padding_and_continuity_do_not_renumber_references(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'folder_paths',SimpleNamespace(get_temp_directory=lambda:str(tmp_path)))
    video=torch.zeros(8,8,8,3)
    start=m.BFSH3NativeShotLoopStart()
    _,state,*_=start.start(video,chunk_frames=5)
    previous=torch.ones(5,8,8,3);p=tmp_path/'previous.pt';torch.save(previous,p)
    state.update(index=1,segments=[{'path':str(p),'frames':5}])
    tail,state,*_=start.start(video,chunk_frames=5,loop_state=state)
    assert len(tail)==5
    refs=(torch.zeros(1,4,8,3),)
    output,prompt,status=m.BFSH3NativeShotContinuity().prepare(refs,'motion_transfer: test',state)
    assert output[0] is refs[0] and torch.equal(output[1],previous[-1:])
    assert '<Picture 2>' in prompt and 'ON' in status
    state['ranges'][1]['camera_cut']=True
    output,prompt,status=m.BFSH3NativeShotContinuity().prepare(refs,'original',state)
    assert output is refs and prompt=='original' and 'camera cut' in status
    state['ranges'][1]['camera_cut']=False
    with pytest.raises(ValueError,match='free Picture slot'):
        m.BFSH3NativeShotContinuity().prepare(refs*9,'original',state)
    # Join trims the padded tail and preserves exactly the original 8 frames.
    joined,fps,_=m.BFSH3NativeShotLoopEnd().finish(tail,state)
    assert len(joined)==8 and fps==24.


def test_graph_expansion_clones_only_loop_body_and_carries_state():
    # Use the actual ComfyUI graph builder when available; no GPU/model execution.
    comfy=Path('/data/comfyui')
    if not (comfy/'comfy_execution/graph_utils.py').exists():pytest.skip('ComfyUI graph utility not available')
    sys.path.insert(0,str(comfy))
    graph={'source':{'class_type':'Load','inputs':{}},
           'model':{'class_type':'Model','inputs':{}},
           'start':{'class_type':'BFSH3NativeShotLoopStart','inputs':{'images':['source',0]}},
           'sampler':{'class_type':'Sample','inputs':{'images':['start',0],'model':['model',0]}},
           'end':{'class_type':'BFSH3NativeShotLoopEnd','inputs':{'images':['sampler',0],'loop_state':['start',1]}}}
    dyn=SimpleNamespace(get_node=lambda n:graph[n],get_display_node_id=lambda n:n)
    out=m.BFSH3NativeShotLoopEnd.expand(dyn,'end',{'index':1})
    expanded=list(out['expand'].values())
    assert {n['class_type'] for n in expanded}=={'BFSH3NativeShotLoopStart','Sample','BFSH3NativeShotLoopEnd'}
    start=next(n for n in expanded if n['class_type']=='BFSH3NativeShotLoopStart')
    assert start['inputs']['images']==['source',0] and start['inputs']['loop_state']=={'index':1}
    assert next(n for n in expanded if n['class_type']=='Sample')['inputs']['model']==['model',0]


def test_example_ui_links_are_bidirectional_and_match_api_loop():
    import json
    d=json.loads((ROOT/'examples/h3_tara_native_auto_prompt_loop.comfy.json').read_text());nodes={n['id']:n for n in d['nodes']}
    for lid,src,slot,dst,input_slot,kind in d['links']:
        assert lid in nodes[src]['outputs'][slot]['links']
        assert nodes[dst]['inputs'][input_slot]['link']==lid
    api=json.loads((ROOT/'examples/h3_tara_native_auto_prompt_loop.api.json').read_text())
    assert api['22']['inputs']['images']==['42',0]
    assert api['22']['inputs']['audio']==['10',2]


def test_seconds_selection_rounding_and_legacy_frames(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    video = torch.zeros(250, 2, 2, 3)
    start = m.BFSH3NativeShotLoopStart()
    _, legacy, *_ = start.start(video)
    assert legacy['ranges'][0]['end'] == 107
    _, seconds, *_ = start.start(video, chunk_frames=107, segment_unit='seconds', chunk_seconds=5.)
    assert [(x['start'], x['end']) for x in seconds['ranges']] == [(0, 120), (120, 240), (240, 250)]
    _, rounded, *_ = start.start(video, segment_unit='seconds', chunk_seconds=1.03, camera_cuts='30')
    assert [(x['start'], x['end']) for x in rounded['ranges'][:3]] == [(0, 25), (25, 30), (30, 55)]
    _, disabled, *_ = start.start(video, enabled=False, segment_unit='seconds', chunk_seconds=5.)
    assert len(disabled['ranges']) == 1 and disabled['ranges'][0]['end'] == 250
    with pytest.raises(ValueError, match='seconds'):
        start.start(video, segment_unit='seconds', chunk_seconds=float('nan'))
