import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minimax_h3_temporal_sol as sol
import minimax_h3_temporal_fast as fast
import minimax_h3_native_temporal_core as core
import minimax_h3_sparse_compat as compat

@pytest.mark.parametrize('strength,cap',[(2.,6.),(0.,6.),(2.,0.),(10.,6.)])
def test_temporal_sol_bias_matches_dense_policy(strength,cap):
 torch.manual_seed(14);frames,sf,tf=7,2,3;n=44
 pos=torch.zeros(n,3,dtype=torch.float64)
 times=torch.tensor([0,1,5,9,13,17,18],dtype=torch.float64)*5/3
 pos[4:18,0]=times.repeat_interleave(sf);pos[20:41,0]=times.repeat_interleave(tf)
 source=torch.zeros(n,dtype=torch.bool);source[4:18]=True
 target=torch.zeros(n,dtype=torch.bool);target[20:41]=True
 plan=fast.make_plan(pos,source,target,strength,.2,.25,cap)
 ctx=core.TemporalContext(pos[:,0].float()[None]/40,source[None],target[None],torch.ones(1,n,dtype=torch.bool),strength,.2,.25,cap)
 q,k,v=[torch.randn(1,2,n,128) for _ in range(3)]
 calls=[]
 def oracle(q,k,v,key_bias=None,**kw):
  assert kw['sink_blocks'][1]*64>=plan.source_stop
  assert kw['topk_ratio']==0. and kw['tail'] is True
  if key_bias is not None:
   assert key_bias.shape==(n,)
   assert torch.all(key_bias[~source]==0)
  calls.append(key_bias)
  return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=None if key_bias is None else key_bias.view(1,1,1,-1)).transpose(1,2)
 actual=sol.attention(q,k,v,plan,ctx,tau=1.3,kernel=oracle)
 expected=core.dense_reference(q,k,v,ctx)
 torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
 assert len(calls)==(frames+1 if strength and cap else 1)
 # Repeat uses the compact cached biases and does not change the policy.
 torch.testing.assert_close(sol.attention(q,k,v,plan,ctx,tau=1.3,kernel=oracle),actual)


def make_sparse(vsa=False,topk=0.):
 SparseAttnPatch=type('SparseAttnPatch',(),{'__module__':'comfy_extras.nodes_sparse_attention'})
 patch=SparseAttnPatch();patch.vsa=vsa;patch.topk_ratio=topk
 previous=None
 def override(*args):return patch,previous,args
 override.__module__='comfy_extras.nodes_sparse_attention'
 override.__qualname__='make_attention_override.<locals>.override'
 def block_patch(*args):return patch,args
 block_patch.__module__=override.__module__
 block_patch.__qualname__='make_h3_block_patch.<locals>.block_patch'
 return patch,override,block_patch


def test_only_sol_owned_hooks_are_bypassed_without_mutating_model():
 patch,override,block=make_sparse()
 options={'optimized_attention_override':override,'patches_replace':{'dit':{('double_block',0):block},'other':{'keep':1}}}
 dense,actual=compat.selection(options)
 assert dense is None and actual is patch
 copied=compat.without_owned_blocks(options,patch)
 assert copied['patches_replace']['dit']=={}
 assert copied['patches_replace']['other']=={'keep':1}
 assert options['patches_replace']['dit']
 # A dense selection applied after Sparse can hide the sparse closure; recover
 # the exact same state from the node's owned H3 block hook.
 hidden={**options,'optimized_attention_override':None}
 assert compat.selection(hidden)==(None,patch)
 unknown={**options,'patches_replace':{'dit':{'unknown':lambda:None}}}
 with pytest.raises(ValueError,match='another attention/block'):compat.selection(unknown)

@pytest.mark.parametrize('vsa,topk',[(True,0.),(False,.1)])
def test_vsa_and_sla_still_rejected(vsa,topk):
 _,override,block=make_sparse(vsa,topk)
 with pytest.raises(ValueError,match='only sol-attn'):compat.selection({'optimized_attention_override':override})
 with pytest.raises(ValueError,match='only sol-attn'):compat.selection({'patches_replace':{'dit':{'block':block}}})
