import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import minimax_h3_reference_binding as binding
import minimax_h3_native_temporal as temporal
import minimax_h3_temporal_fast as fast
spec=importlib.util.spec_from_file_location('binding_layout',ROOT/'tests/fixtures/comfy_h3_layout_8cfe5e1.py');layout_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout_module)


def fixture(strength=2.):
    refs=[dict(kind='image',latent_h=4,latent_w=4),dict(kind='image',latent_h=4,latent_w=4),dict(kind='video',latent_t=2,latent_h=4,latent_w=4,ref_audio_t=0)]
    native=layout_module.PackedLayout(3,2,4,4,0,refs=refs)
    layout,source,target=temporal.temporal_layout(native,refs,None)
    plan=fast.make_plan(layout.position_ids,source,target,strength,.2,.25,6.)
    return layout,refs,plan


def flash_oracle(q,k,v):
    logits=(q.double()@k.double().transpose(-1,-2))/math.sqrt(q.shape[-1])
    lse=logits.logsumexp(-1)
    return (logits.softmax(-1)@v.double()).to(q.dtype),lse


@pytest.mark.parametrize('strength',[0.,2.])
def test_reference_and_temporal_biases_match_exact_dense_attention(strength):
    layout,refs,plan=fixture(strength)
    mask=torch.tensor([[[1.,0.],[.5,1.]]])
    controls=[dict(picture=1,strength=1.5,other_strength=.75,mask=mask)]
    biases=binding.build_biases(layout,refs,plan,controls,'cpu')
    torch.manual_seed(3);q,k,v=[torch.randn(1,2,layout.seq_len,8) for _ in range(3)]
    out=fast.attention(q,k,v,plan,flash=flash_oracle,reference_biases=biases,frame_chunk=1)
    logits=(q.double()@k.double().transpose(-1,-2))/math.sqrt(8)
    ta,tb,sa,sb=plan.target_start,plan.target_stop,plan.source_start,plan.source_stop
    times=layout.position_ids[:,0]/40
    penalty=(strength*(torch.relu((times[ta:tb,None]-times[None,sa:sb]).abs()-.2)/.25).square()).clamp(max=6.)
    logits[:,:,ta:tb,sa:sb]-=penalty
    for a,b,gain in biases:logits[:,:,ta:tb,a:b]+=gain[None,None,:,None]
    expected=(logits.softmax(-1)@v.double()).float()
    torch.testing.assert_close(out,expected,atol=2e-6,rtol=2e-5)
    # Non-target queries preserve their native attention.
    torch.testing.assert_close(out[:,:,:ta],flash_oracle(q[:,:,:ta],k,v)[0],atol=2e-6,rtol=2e-5)


def test_masks_follow_latent_spans_and_reject_wrong_timeline():
    layout,refs,plan=fixture()
    mask=torch.cat([torch.zeros(1,2,2),torch.ones(4,2,2)])
    values=binding.target_mask(mask,layout,plan).reshape(2,4)
    assert not values[0].any() and values[1].eq(1).all()
    with pytest.raises(ValueError,match='5 frames'):binding.target_mask(torch.zeros(6,2,2),layout,plan)
    with pytest.raises(ValueError,match='not attached'):
        binding.build_biases(layout,refs,plan,[dict(picture=3,strength=2.,other_strength=1.,mask=None)],'cpu')


def test_node_clones_and_disabled_or_neutral_is_exact_bypass():
    class Model:
        def __init__(self):self.model_options={'bfs_h3_native_temporal_v1':{}}
        def clone(self):m=Model();m.model_options=dict(self.model_options);return m
    m=Model();m.model_options['bfs_h3_native_temporal_v1']={'version':'v1'};node=binding.BFSMiniMaxH3ReferenceBinding()
    assert node.apply(m,enabled=False)[0] is m
    assert node.apply(m,reference_strength=1.,other_reference_strength=1.)[0] is m
    first=node.apply(m)[0];second=node.apply(first,picture_index=2)[0]
    assert binding.OPTION not in m.model_options.get('transformer_options',{})
    assert len(first.model_options['transformer_options'][binding.OPTION])==1
    assert len(second.model_options['transformer_options'][binding.OPTION])==2


def test_wrapper_reads_controls_and_refreshes_when_same_layout_is_reused(monkeypatch):
    class Model:
        def __init__(self):self.model_options={}
        def clone(self):n=Model();n.model_options=dict(self.model_options);return n
        def get_model_object(self,key):return type('MiniMaxH3Model',(),{})()
        def set_model_unet_function_wrapper(self,fn):self.model_options['model_function_wrapper']=fn
    layout,refs,plan=fixture();seen=[]
    def capture(ctx,**kwargs):
        seen.append(kwargs['reference_biases']);return 'controlled'
    monkeypatch.setattr(temporal,'attention_override',capture)
    patched=temporal.patch_temporal_model(Model());fn=patched.model_options['model_function_wrapper']
    def model(x,t,**conds):
        assert conds['transformer_options']['optimized_attention_override']=='controlled'
        return 'ok'
    for gain in (1.25,2.):
        control=[dict(picture=1,strength=gain,other_strength=1.,mask=None)]
        args=dict(input=torch.zeros(1),timestep=torch.ones(1),c=dict(minimax_payload=dict(layout=layout,refs=refs),transformer_options={binding.OPTION:control}))
        assert fn(model,args)=='ok'
    assert seen[0][0][2][0]==pytest.approx(math.log(1.25))
    assert seen[1][0][2][0]==pytest.approx(math.log(2.))


def test_text_and_picture_logit_gains_compose_exactly():
    layout,refs,plan=fixture(2.)
    picture=binding.build_biases(layout,refs,plan,[dict(picture=1,strength=1.25,other_strength=1.,mask=None)],'cpu')
    text=[(0,2,torch.full((plan.target_stop-plan.target_start,),math.log(.75)))]
    all_biases=picture+text
    torch.manual_seed(33);q,k,v=[torch.randn(1,2,layout.seq_len,8) for _ in range(3)]
    out=fast.attention(q,k,v,plan,flash=flash_oracle,reference_biases=all_biases,frame_chunk=1)
    logits=(q.double()@k.double().transpose(-1,-2))/math.sqrt(8)
    ta,tb,sa,sb=plan.target_start,plan.target_stop,plan.source_start,plan.source_stop
    times=layout.position_ids[:,0]/40
    penalty=(2.*(torch.relu((times[ta:tb,None]-times[None,sa:sb]).abs()-.2)/.25).square()).clamp(max=6.)
    logits[:,:,ta:tb,sa:sb]-=penalty
    for a,b,g in all_biases:logits[:,:,ta:tb,a:b]+=g[None,None,:,None]
    torch.testing.assert_close(out,(logits.softmax(-1)@v.double()).float(),atol=2e-6,rtol=2e-5)
