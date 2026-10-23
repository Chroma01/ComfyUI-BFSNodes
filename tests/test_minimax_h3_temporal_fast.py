import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import minimax_h3_temporal_fast as fast
import minimax_h3_native_temporal_core as core


def oracle(q,k,v):
    scores=q.double()@k.double().transpose(-1,-2)/q.shape[-1]**.5
    return (scores.softmax(-1)@v.double()).to(q.dtype),scores.logsumexp(-1).float()


def test_exact_partition_matches_original_temporal_policy():
    torch.manual_seed(5)
    frames=7;sf=2;tf=3;n=4+frames*sf+2+frames*tf
    pos=torch.zeros(n,3,dtype=torch.float64)
    times=torch.tensor([0,1,5,9,13,17,18],dtype=torch.float64)*5/3
    pos[4:18,0]=times.repeat_interleave(sf)
    pos[20:,0]=times.repeat_interleave(tf)
    source=torch.zeros(n,dtype=torch.bool);source[4:18]=True
    target=torch.zeros(n,dtype=torch.bool);target[20:]=True
    q,k,v=[torch.randn(1,2,n,8) for _ in range(3)]
    for strength,cap in [(2.,6.),(0.,6.),(2.,0.),(10.,6.)]:
        plan=fast.make_plan(pos,source,target,strength,.2,.25,cap)
        ctx=core.TemporalContext(pos[:,0].float()[None]/40,source[None],target[None],torch.ones(1,n,dtype=torch.bool),strength,.2,.25,cap)
        actual=fast.attention(q,k,v,plan,flash=oracle,frame_chunk=3)
        expected=core.dense_reference(q,k,v,ctx)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
