import sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minimax_h3_temporal_kernel as bridge


def fake_api(monkeypatch):
 api=ModuleType('comfy_kitchen.temporal_attention');api.VERSION='h3_native_temporal_kernel_v1';api.is_available=lambda d:True
 records=[]
 api.TemporalPolicy=lambda *args:args
 def dense(q,k,v,p,**kw):records.append(('dense',p,kw));return torch.ones(1,dtype=torch.bfloat16)
 def sol(q,k,v,p,**kw):records.append(('sol',p,kw));return torch.ones(1,dtype=torch.bfloat16)
 api.int8_attention=dense;api.sol_attn=sol
 monkeypatch.setitem(sys.modules,'comfy_kitchen.temporal_attention',api)
 return api,records


def test_bridge_uses_one_native_call_and_preserves_policy(monkeypatch):
 api,records=fake_api(monkeypatch)
 q=SimpleNamespace(device=torch.device('cuda'),dtype=torch.bfloat16)
 plan=SimpleNamespace(source_start=4,source_stop=10,target_start=12,target_stop=21,source_frame_rows=2,target_frame_rows=3,frame_times=(0.,.7,1.4))
 ctx=SimpleNamespace(strength=2.,window=.2,falloff=.25,cap=6.)
 bridge.run('dense',q,q,q,plan,ctx)
 bridge.run('sol',q,q,q,plan,ctx,tau=1.3,token_aug=256,sink_q=(0,1))
 assert len(records)==2 and [r[0] for r in records]==['dense','sol']
 assert records[0][1]==(4,10,12,21,2,3,(0.,.7,1.4),2.,.2,.25,6.)
 assert records[1][2]=={'tau':1.3,'token_aug':256,'sink_q':(0,1)}
 ctx.strength=3.
 assert bridge.policy_for(api,plan,ctx)[7]==3.


def test_bridge_rejects_wrong_extension_version(monkeypatch):
 api,_=fake_api(monkeypatch);api.VERSION='unknown'
 q=SimpleNamespace(device=torch.device('cuda'))
 with pytest.raises(RuntimeError,match='version'):bridge.api_for(q)


def test_cpu_and_missing_binary_keep_compatibility_path(monkeypatch):
 assert bridge.api_for(torch.zeros(1)) is None
 api,_=fake_api(monkeypatch);api.is_available=lambda d:False
 q=SimpleNamespace(device=torch.device('cuda'))
 assert bridge.api_for(q) is None
