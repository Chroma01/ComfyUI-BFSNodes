import sys
from pathlib import Path
from types import ModuleType
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minimax_h3_temporal_kitchen as kitchen
import minimax_h3_temporal_fast as fast
import minimax_h3_native_temporal_core as core
import minimax_h3_native_temporal as node

@pytest.mark.parametrize('strength,cap',[(2.,6.),(0.,6.),(2.,0.),(10.,6.)])
def test_frame_bias_matches_full_temporal_rule(strength,cap):
    torch.manual_seed(42)
    frames,sf,tf=7,2,3
    n=4+frames*sf+2+frames*tf+3
    pos=torch.zeros(n,3,dtype=torch.float64)
    times=torch.tensor([0,1,5,9,13,17,18],dtype=torch.float64)*5/3
    pos[4:18,0]=times.repeat_interleave(sf);pos[20:41,0]=times.repeat_interleave(tf)
    source=torch.zeros(n,dtype=torch.bool);source[4:18]=True
    target=torch.zeros(n,dtype=torch.bool);target[20:41]=True
    plan=fast.make_plan(pos,source,target,strength,.2,.25,cap)
    ctx=core.TemporalContext(pos[:,0].float()[None]/40,source[None],target[None],torch.ones(1,n,dtype=torch.bool),strength,.2,.25,cap)
    q,k,v=[torch.randn(1,2,n,8) for _ in range(3)]
    calls=[]
    def oracle(q,k,v,attn_mask=None):
        if attn_mask is not None:
            assert attn_mask.shape==(1,1,1,n)
            calls.append(attn_mask)
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask)
    actual=kitchen.attention(q,k,v,plan,ctx,kernel=oracle)
    expected=core.dense_reference(q,k,v,ctx)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(kitchen.attention(q,k,v,plan,ctx,kernel=oracle),actual)
    if strength and cap:
        assert len(calls)==2*frames
        assert plan._kitchen_biases.numel()==frames*n
        assert torch.all(plan._kitchen_biases[:,~source]==0)


def test_detects_only_official_model_attention_backend(monkeypatch):
    mod=ModuleType('comfy.ldm.modules.attention')
    kernels={'pytorch':lambda:None,'comfy_kitchen_int8':lambda:None}
    mod.get_attention_function=lambda name,default=None:kernels.get(name,default)
    monkeypatch.setitem(sys.modules,'comfy.ldm.modules.attention',mod)
    def setter(optimized_attention):
        def optimized_attention_override(_, *args, **kwargs):
            return optimized_attention(*args,**kwargs)
        optimized_attention_override.__module__='comfy.model_patcher'
        optimized_attention_override.__qualname__='ModelPatcher.set_model_optimized_attention.<locals>.optimized_attention_override'
        return optimized_attention_override
    for key,fn in kernels.items():assert node.dense_backend_selection(setter(fn))==key
    assert node.dense_backend_selection(None) is None
    with pytest.raises(ValueError):node.dense_backend_selection(lambda *args:None)
    with pytest.raises(ValueError):node.dense_backend_selection(setter(lambda:None))
