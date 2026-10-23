"""Exact bounded temporal bias through FlashAttention partitions; inference only.

The guide's weights are exp(-cap) globally plus positive local corrections
exp(-penalty)-exp(-cap). Other keys have weight 1. Merge partitions by logsumexp.
No compiled score modifier, dense score matrix, or subtraction of attention sums.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class Plan:
    source_start: int
    source_stop: int
    target_start: int
    target_stop: int
    source_frame_rows: int
    target_frame_rows: int
    frames: int
    cap: float
    corrections: list
    frame_times: tuple = ()


def make_plan(positions, source, target, strength, window, falloff, cap):
    si=torch.where(source)[0];ti=torch.where(target)[0]
    sa,sb=int(si[0]),int(si[-1])+1;ta,tb=int(ti[0]),int(ti[-1])+1
    if si.numel()!=sb-sa or ti.numel()!=tb-ta:
        raise ValueError('Fast temporal attention requires contiguous source and target video rows')
    st,sc=torch.unique_consecutive(positions[sa:sb,0],return_counts=True)
    tt,tc=torch.unique_consecutive(positions[ta:tb,0],return_counts=True)
    if not torch.equal(sc,sc[:1].expand_as(sc)) or not torch.equal(tc,tc[:1].expand_as(tc)):
        raise ValueError('Fast temporal attention requires a fixed spatial grid per video')
    if st.shape!=tt.shape or not torch.allclose(st,tt,rtol=0,atol=1e-9):
        raise ValueError('Fast temporal attention requires matching temporal grids')
    frames=st.numel();corrections=[]
    if strength>0 and cap>0:
        times=st.double()/40
        for offset in range(1-frames,frames):
            lo=max(0,-offset);hi=min(frames,frames-offset)
            dt=(times[lo:hi]-times[lo+offset:hi+offset]).abs()
            penalty=(strength*(torch.relu(dt-window)/falloff).square()).clamp(max=cap)
            # Stable log(exp(-penalty)-exp(-cap)); zero correction becomes -inf.
            logs=-penalty+torch.log(-torch.expm1(penalty-cap))
            if torch.isfinite(logs).any():corrections.append((offset,lo,hi,logs.float()))
    return Plan(sa,sb,ta,tb,int(sc[0]),int(tc[0]),frames,cap if strength>0 else 0.,corrections,tuple((st.float()/40).tolist()))


def flash_with_lse(q,k,v):
    # PyTorch's native FlashAttention primitive returns natural-log normalizers.
    if q.device.type!='cuda' or q.dtype not in (torch.float16,torch.bfloat16):
        raise RuntimeError('Fast temporal attention requires CUDA FP16/BF16. Choose flex backend for other configurations.')
    result=torch.ops.aten._scaled_dot_product_flash_attention.default(q,k,v,0.0,False,False)
    return result[0],result[1]


def merge(a,la,b,lb):
    """Stable positive weighted merge. Outputs retain model dtype; LSE stays FP32."""
    total=torch.logaddexp(la,lb)
    result=a.float()*torch.exp(la-total)[...,None]+b.float()*torch.exp(lb-total)[...,None]
    return result.to(a.dtype),total


def attention(q,k,v,plan,flash=flash_with_lse,frame_chunk=8):
    if q.shape[0]!=1 or q.shape[2]!=k.shape[2] or q.shape!=v.shape:
        raise ValueError('Fast native temporal attention currently requires one packed self-attention sequence')
    if plan.cap==0:
        return flash(q,k,v)[0]
    out=torch.empty_like(q)
    sa,sb,ta,tb=plan.source_start,plan.source_stop,plan.target_start,plan.target_stop
    # Queries outside the target retain ordinary attention, including token refs/audio.
    for lo,hi in ((0,ta),(tb,q.shape[2])):
        if hi>lo:out[:,:,lo:hi]=flash(q[:,:,lo:hi],k,v)[0]
    heads,dim=q.shape[1],q.shape[-1]
    qframes=q[:,:,ta:tb].reshape(1,heads,plan.frames,plan.target_frame_rows,dim)[0].permute(1,0,2,3)
    kframes=k[:,:,sa:sb].reshape(1,heads,plan.frames,plan.source_frame_rows,dim)[0].permute(1,0,2,3)
    vframes=v[:,:,sa:sb].reshape(1,heads,plan.frames,plan.source_frame_rows,dim)[0].permute(1,0,2,3)
    if getattr(plan, '_correction_device', None) != q.device:
        plan._device_corrections=[(off,lo,hi,logs.to(device=q.device)) for off,lo,hi,logs in plan.corrections]
        plan._correction_device=q.device
    corrections=plan._device_corrections
    for first in range(0,plan.frames,frame_chunk):
        last=min(plan.frames,first+frame_chunk)
        start=ta+first*plan.target_frame_rows;stop=ta+last*plan.target_frame_rows
        query=q[:,:,start:stop]
        # Positive baseline: all motion keys downweighted by exp(-cap).
        acc,lse=flash(query,k[:,:,sa:sb],v[:,:,sa:sb]);lse=lse-plan.cap
        for lo,hi in ((0,sa),(sb,k.shape[2])):
            if hi>lo:
                part,pl=flash(query,k[:,:,lo:hi],v[:,:,lo:hi]);acc,lse=merge(acc,lse,part,pl)
        acc=acc.reshape(1,heads,last-first,plan.target_frame_rows,dim)[0].permute(1,0,2,3)
        lse=lse.reshape(1,heads,last-first,plan.target_frame_rows)[0].permute(1,0,2)
        for off,lo,hi,logs in corrections:
            a=max(first,lo);b=min(last,hi)
            if b<=a:continue
            part,pl=flash(qframes[a:b],kframes[a+off:b+off],vframes[a+off:b+off])
            pl=pl+logs[a-lo:b-lo,None,None]
            merged,ml=merge(acc[a-first:b-first],lse[a-first:b-first],part,pl)
            acc[a-first:b-first]=merged;lse[a-first:b-first]=ml
        out[:,:,start:stop]=acc.permute(1,0,2,3).reshape(1,heads,stop-start,dim)
    return out
