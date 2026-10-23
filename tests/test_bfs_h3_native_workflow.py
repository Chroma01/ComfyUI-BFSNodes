import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import pytest
import torch

spec=importlib.util.spec_from_file_location('h3_native_workflow_tested',Path(__file__).resolve().parents[1]/'bfs_h3_native_workflow.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

def test_auto_geometry_uses_actual_video_and_preserves_24fps_timing():
    images=torch.arange(150).float()[:,None,None,None].expand(150,96,160,3)/150
    video,width,height,length,meta=w.prepare_video(images,30.,1.,.4,.5)
    assert (width,height,length)==(160,96,107)
    assert video.shape==(107,32,64,3)
    assert video[-1,0,0,0].item()==pytest.approx(132/150)
    assert meta['source_frames']==150 and meta['grid_tail_frames']==13
    assert meta['duration_seconds']==107/24

@pytest.mark.parametrize('shape,expected', [((90,160),(160,64)),((160,90),(64,160))])
def test_portrait_landscape_from_source_not_fixed(shape,expected):
    _,a,b,n,_=w.prepare_video(torch.zeros(22,*shape,3),24.)
    assert (a,b)==expected and n==22

def test_explicit_frame_cap_and_canvas_budget_are_reported():
    video,a,b,n,meta=w.prepare_video(torch.zeros(141,96,160,3),24.,2.,.02,1.,107)
    assert a*b<=20000 and n==107 and meta['capped']
    assert len(video)==107

@pytest.mark.parametrize('fps',[0,-2,float('nan')])
def test_bad_fps_rejected(fps):
    with pytest.raises(ValueError):w.prepare_video(torch.zeros(22,32,32,3),fps)

def test_too_short_clip_fails_instead_of_padding_fake_motion():
    with pytest.raises(ValueError,match='at least 5'):w.prepare_video(torch.zeros(4,32,32,3),24.)

def test_frame_sheet_uses_original_pixels_and_sampled_timestamps():
    sheet=w.contact_sheet(torch.rand(90,48,64,3),6,30.,73)
    assert sheet.shape==(1,416,1280,3) and torch.isfinite(sheet).all()

def test_references_stable_order_and_gaps_fail():
    a,b=torch.zeros(1,32,48,3),torch.ones(1,64,32,3)
    assert w.collect_references(dict(ref_image_2=b,ref_image_1=a))==(a,b)
    with pytest.raises(ValueError,match='consecutive'):w.collect_references(dict(ref_image_1=a,ref_image_3=b))
    with pytest.raises(ValueError,match='one image'):w.collect_references(dict(ref_image_1=a.expand(2,-1,-1,-1)))

def test_each_reference_is_inspected_separately_and_returned_for_same_encoder(monkeypatch):
    a,b=torch.zeros(1,32,48,3),torch.ones(1,64,32,3);calls=[]
    def gen(clip,image,prompt,tokens):
        calls.append((image,prompt));return 'red coat' if image is a else 'beach'
    monkeypatch.setattr(w,'generate_reference',gen)
    refs,inventory=w.BFSH3DescribeReferences().describe(object(),'identity 1; setting 2',ref_image_2=b,ref_image_1=a)
    assert len(calls)==2 and calls[0][0] is a and calls[1][0] is b
    assert '<Picture 1>: red coat' in inventory and '<Picture 2>: beach' in inventory
    assert refs[0] is a and refs[1] is b

def test_template_does_not_recursively_expand_user_braces():
    result=w.build_prompt('{request}|{inventory}|{guide_analysis}|{metadata}','keep {inventory}','one','motion','{}')
    assert result=='keep {inventory}|one|motion|{}'


def full_prompt():return 'motion_transfer: '+ '\n'.join(s+': N/A' for s in w.SECTIONS)

def test_h3_format_validation_rejects_truncation():
    w.validate_prompt(full_prompt())
    with pytest.raises(ValueError):w.validate_prompt(full_prompt().split('overall_soundscape')[0])

def test_encoding_calls_native_ref_video_and_all_n_refs_without_add_guide(monkeypatch):
    module=ModuleType('comfy_extras.nodes_minimax_h3');seen={}
    class Native:
        @classmethod
        def execute(cls,**kwargs):seen.update(kwargs);return ('positive','latent')
    module.MiniMaxH3ReferenceToVideo=Native
    package=ModuleType('comfy_extras');package.__path__=[]
    monkeypatch.setitem(sys.modules,'comfy_extras',package)
    monkeypatch.setitem(sys.modules,'comfy_extras.nodes_minimax_h3',module)
    refs=tuple(torch.zeros(1,32,32,3) for _ in range(3));guide=torch.zeros(22,32,32,3)
    result=w.BFSH3NativeReferenceSetToVideo().encode(object(),object(),guide,refs,full_prompt(),64,32,22)
    assert result[:2]==('positive','latent')
    assert seen['ref_videos']['ref_video_1'] is guide
    assert list(seen['ref_images'])==['ref_image_1','ref_image_2','ref_image_3']
    with pytest.raises(ValueError,match='unattached'):
        w.BFSH3NativeReferenceSetToVideo().encode(None,None,guide,refs,full_prompt()+' <Picture 4>',64,32,22)


def test_generated_reference_uses_native_textgenerate_and_old_mtp_signature(monkeypatch):
    module=ModuleType('comfy_extras.nodes_textgen');calls=[]
    class Old:
        @classmethod
        def execute(cls,clip,image,prompt,max_length,sampling_mode,thinking,use_default_template):
            calls.append((image,thinking));return ('Answer: description','')
    module.TextGenerate=Old
    package=ModuleType('comfy_extras');package.__path__=[]
    monkeypatch.setitem(sys.modules,'comfy_extras',package);monkeypatch.setitem(sys.modules,'comfy_extras.nodes_textgen',module)
    image=torch.zeros(1,32,32,3)
    assert w.generate_reference(None,image,'test',128)=='description'
    assert calls[0][0] is image and calls[0][1] is False


def test_pad_last_frame_keeps_full_source_duration_for_decoded_trim():
    images=torch.arange(24).float()[:,None,None,None].expand(24,32,32,3)/24
    result=w.BFSH3PrepareNativeVideo().prepare(images,source_fps=24.,request='swap')
    video,_,_,_,length,metadata,_,_,trim=result
    assert length==39 and trim==24
    assert torch.equal(video[-1],images[-1])
    import json
    assert json.loads(metadata)['padded_tail_frames']==15
