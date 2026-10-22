import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import minimax_h3_native_temporal as node

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
CORE=load('native_temporal_comfy_fixture',ROOT/'tests/fixtures/comfy_h3_layout_8cfe5e1.py')

def example(vt=7):
 refs=[dict(kind='image',latent_h=4,latent_w=8),dict(kind='video_audio',latent_h=8,latent_w=4,latent_t=vt,ref_audio_t=3)]
 l=CORE.PackedLayout(3,7,8,8,2,refs=refs)
 return l,refs

def test_layout_matches_actual_training_coordinates():
 l,refs=example();before=l.position_ids.clone()
 aligned,src,tgt=node.temporal_layout(l,refs,None)
 golden=json.loads((ROOT/'tests/fixtures/h3_native_temporal_training_positions.json').read_text())
 torch.testing.assert_close(aligned.position_ids,torch.tensor(golden['position_ids'],dtype=torch.float64),rtol=0,atol=1e-12)
 assert src.tolist()==golden['source']
 assert torch.equal(l.position_ids,before)
 assert torch.equal(aligned.position_ids[~src],before[~src])
 assert torch.equal(aligned.position_ids[src,1:],before[src,1:])
 assert not (src&tgt).any()

def test_wrong_video_index_and_temporal_mismatch_rejected():
 l,r=example()
 with pytest.raises(ValueError,match='Video 2'):node.temporal_layout(l,r,None,2)
 l,r=example(6)
 with pytest.raises(ValueError,match='identical temporal'):node.temporal_layout(l,r,None)
 with pytest.raises(ValueError,match='Add Guide'):node.temporal_layout(l,r,[{}])

def context():
 l,r=example();a,s,t=node.temporal_layout(l,r,None)
 return node.core.TemporalContext(a.position_ids[:,0].float()[None]/40,s[None],t[None],torch.ones_like(s[None]),2,.2,.25,6)

def test_attention_adapter_matches_dense_score_policy_and_shapes():
 ctx=context();torch.manual_seed(12)
 q,k,v=[torch.randn(1,2,ctx.times.shape[1],8) for _ in range(3)]
 def forbidden(*args,**kwargs):raise AssertionError('wrong backend')
 out=node.attention_override(ctx)(forbidden,q,k,v,2,skip_reshape=True)
 expected=node.core.dense_reference(q,k,v,ctx).transpose(1,2).flatten(2)
 torch.testing.assert_close(out,expected)
 raw=[x.transpose(1,2).flatten(2) for x in (q,k,v)]
 torch.testing.assert_close(node.attention_override(ctx)(forbidden,*raw,2),expected)

def test_token_refiner_uses_existing_attention():
 ctx=context();q=torch.randn(1,2,3,8)
 def original(*args,**kwargs):return 'original'
 assert node.attention_override(ctx)(original,q,q,q,2,skip_reshape=True)=='original'

class Model:
 def __init__(self):self.model_options={}
 def clone(self):m=Model();m.model_options=copy.deepcopy(self.model_options);return m
 def get_model_object(self,key):return type('MiniMaxH3Model',(),{})()
 def set_model_unet_function_wrapper(self,w):self.model_options['model_function_wrapper']=w

def test_per_model_wrapper_clones_payload_and_applies_attention():
 m=Model();patched=node.patch_temporal_model(m);assert not m.model_options
 l,r=example();payload=dict(layout=l,refs=r)
 def fn(x,t,**c):
  assert c['minimax_payload']['layout'] is not l
  assert 'optimized_attention_override' in c['transformer_options']
  return 'ok'
 args=dict(input=torch.zeros(1),timestep=torch.ones(1),c=dict(minimax_payload=payload))
 assert patched.model_options['model_function_wrapper'](fn,args)=='ok'
 assert payload['layout'] is l and 'transformer_options' not in args['c']
 with pytest.raises(ValueError,match='already installed'):node.patch_temporal_model(patched)

def test_incompatible_attention_rejected():
 m=Model();m.model_options={'transformer_options':{'optimized_attention_override':object()}}
 with pytest.raises(ValueError,match='VSA'):node.patch_temporal_model(m)
 with pytest.raises(ValueError,match='MODEL input'):node.patch_temporal_model(None)


def test_second_video_selection_preserves_first_video_and_images():
 refs=[dict(kind='image',latent_h=4,latent_w=8),
       dict(kind='video',latent_h=4,latent_w=4,latent_t=3,ref_audio_t=0),
       dict(kind='video',latent_h=8,latent_w=4,latent_t=7,ref_audio_t=0)]
 native=CORE.PackedLayout(3,7,8,8,2,refs=refs)
 aligned,source,target=node.temporal_layout(native,refs,None,2)
 assert int(source.sum())==56
 assert torch.equal(aligned.position_ids[~source],native.position_ids[~source])
 assert not source[3:23].any()
