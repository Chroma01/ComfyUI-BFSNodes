"""CPU tests against the audited native ComfyUI layout and training fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

NODE = load('bfs_h3_guide_test', ROOT / 'minimax_h3_downscaled_guide.py')
CORE = load('bfs_h3_native_layout_test', ROOT / 'tests/fixtures/comfy_h3_layout_8cfe5e1.py')


def guide(factor, vt=2, h=8, w=8, index=0):
    return {'resolved_frame_index': index, 'latent': torch.zeros(1, 24, vt, h//factor, w//factor),
            NODE.MARKER: {'downscale_factor': factor}}


class GuideLayoutTest(unittest.TestCase):
    def test_positions_match_training_for_factors_1_2_4(self):
        golden = json.loads((ROOT/'tests/fixtures/h3_training_guide_positions.json').read_text())
        for factor in (1, 2, 4):
            with self.subTest(factor=factor):
                keyframes = [guide(factor)]
                native = CORE.PackedLayout(3, 2, 8, 8, 2, keyframes=keyframes)
                layout = NODE.downscaled_layout(native, keyframes)
                expected = golden['factors'][str(factor)]
                torch.testing.assert_close(layout.position_ids, torch.tensor(expected['positions'], dtype=torch.float64), rtol=0, atol=1e-14)
                self.assertEqual(layout.img_pos.tolist(), expected['video_indices'])
                self.assertEqual(layout.audio_pos.tolist(), expected['audio_indices'])
                self.assertEqual(layout.seq_len, len(expected['positions']))
                self.assertEqual(layout.segments[-1][1], layout.seq_len)
                self.assertEqual(len(layout.img_pos) + len(layout.audio_pos) + 3, layout.seq_len)
                self.assertEqual(native.seq_len, 3+32+4+32)

    def test_73_frame_guide_counts_and_frozen_rows(self):
        keyframes = [guide(4, vt=22)]
        native = CORE.PackedLayout(3, 22, 8, 8, 122, keyframes=keyframes)
        layout = NODE.downscaled_layout(native, keyframes)
        self.assertEqual(layout.segments[1], (3, 25, 'cond'))
        self.assertFalse(layout.img_update[:22].any())
        self.assertTrue(layout.img_update[22:].all())
        self.assertEqual(layout.seq_len, 3+22+244+22*16)
        original_target = native.position_ids[native.segments[-1][0]:]
        torch.testing.assert_close(layout.position_ids[layout.segments[-1][0]:], original_target)

    def test_multiple_guides_native_refs_and_audio_keep_their_coordinates(self):
        keyframes = [guide(4, vt=1), guide(2, vt=2, index=1)]
        keyframes[0]['audio_latent'] = torch.zeros(1,32,2,2)
        refs = [{'kind':'image','latent_h':4,'latent_w':8}]
        native = CORE.PackedLayout(4,7,8,8,40,keyframes=keyframes,refs=refs)
        result = NODE.downscaled_layout(native,keyframes)
        for orig, changed in zip(native.segments,result.segments):
            if orig[2] != 'cond':
                torch.testing.assert_close(native.position_ids[orig[0]:orig[1]],result.position_ids[changed[0]:changed[1]])
        self.assertEqual(result.segments[-1][1],result.seq_len)
        self.assertTrue((result.position_ids[result.segments[1][0],0] == 5).item())

    def test_invalid_guide_shape_fails(self):
        keyframes=[guide(4)]
        keyframes[0]['latent']=torch.zeros(1,24,2,4,4)
        native=CORE.PackedLayout(3,2,8,8,2,keyframes=keyframes)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            NODE.downscaled_layout(native,keyframes)

    def test_standard_guide_keeps_full_resolution(self):
        keyframes=[guide(4),{'resolved_frame_index':0,'latent':torch.zeros(1,24,1,8,8)}]
        native=CORE.PackedLayout(3,2,8,8,2,keyframes=keyframes)
        result=NODE.downscaled_layout(native,keyframes)
        self.assertEqual(result.segments[2][1]-result.segments[2][0],16)


class PreprocessingTest(unittest.TestCase):
    def test_exact_training_crop_and_resize(self):
        source=np.random.default_rng(7).integers(0,256,(80,120,3),dtype=np.uint8)
        images=torch.from_numpy(source.copy()).float().unsqueeze(0)/255
        result=NODE.prepare_guide_frames(images,128,128,4)
        expected=ImageOps.fit(Image.fromarray(source),(128,128),Image.Resampling.BICUBIC).resize((32,32),Image.Resampling.LANCZOS)
        self.assertEqual(tuple(result.shape),(1,32,32,3))
        np.testing.assert_array_equal((result[0].numpy()*255).round().astype(np.uint8),np.array(expected))

    def test_bad_canvas_and_factor(self):
        image=torch.zeros(1,8,8,3)
        for factor in (0,True,1.5):
            with self.assertRaises(ValueError): NODE.prepare_guide_frames(image,128,128,factor)
        with self.assertRaisesRegex(ValueError,'multiples'): NODE.prepare_guide_frames(image,160,128,4)

    def test_posterior_seed_and_fp16_recipe_without_global_rng_changes(self):
        moments=torch.zeros(1,48,1,2,2)
        moments[:,:24]=.123456
        mean=torch.full((24,),.25); std=torch.full((24,),1.5)
        before=torch.random.get_rng_state()
        result=NODE.sample_posterior(moments,mean,std,fp16_round=True)
        generator=torch.Generator('cpu').manual_seed(42)
        expected=(.123456+torch.randn((1,24,1,2,2),generator=generator)).half().float()
        expected=(expected-.25)/1.5
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        torch.testing.assert_close(result,NODE.sample_posterior(moments,mean,std,fp16_round=True))


class EncoderMomentsTest(unittest.TestCase):
    def stage(self):
        calls = []
        def normalize(x):
            self.assertEqual(x.dtype, torch.float32)
            return x + .123456
        def adaptive(x):
            calls.append(x.clone())
            self.assertEqual(x.dtype, torch.float16)
            return torch.zeros(1,48,5,2,2)
        return SimpleNamespace(clip_length=17, token_drop=3,
                               _normalize_pixels=normalize, _adaptive_encode=adaptive), calls

    def test_image_uses_last_posterior_slice(self):
        stage,calls = self.stage()
        result = NODE.encode_moments(stage,torch.zeros(1,3,1,32,32),'cpu',torch.float16)
        self.assertEqual(tuple(result.shape),(1,48,1,2,2))
        self.assertEqual(len(calls),1)

    def test_video_chunk_padding_and_tail_drop_match_native_h3(self):
        stage,calls = self.stage()
        pixels = torch.arange(73,dtype=torch.float32).view(1,1,73,1,1).expand(1,3,73,2,2)
        result = NODE.encode_moments(stage,pixels,'cpu',torch.float16)
        self.assertEqual(result.shape[2],22)
        self.assertEqual(len(calls),5)
        self.assertEqual(calls[-1].shape[2],17)
        torch.testing.assert_close(calls[-1][:,:,4],calls[-1][:,:,16])


class FakeModel:
    def __init__(self, options=None): self.model_options=options or {}
    def clone(self): return FakeModel(dict(self.model_options))
    def set_model_unet_function_wrapper(self,fn): self.model_options['model_function_wrapper']=fn


class WrapperTest(unittest.TestCase):
    def test_scoped_wrapper_composes_and_does_not_mutate_payload(self):
        calls=[]
        def previous(fn,args):
            calls.append('previous')
            return fn(args['input'],args['timestep'],**args['c'])
        original=FakeModel({'model_function_wrapper':previous})
        patched=NODE.patch_guide_model(original)
        keyframes=[guide(4)]
        native=CORE.PackedLayout(3,2,8,8,2,keyframes=keyframes)
        payload={'keyframes':keyframes,'layout':native}
        args={'input':'x','timestep':'t','c':{'minimax_payload':payload}}
        def model_fn(x,t,**c): return c['minimax_payload']['layout']
        result=patched.model_options['model_function_wrapper'](model_fn,args)
        self.assertEqual(result.seq_len,41)
        self.assertIs(payload['layout'],native)
        self.assertIs(original.model_options['model_function_wrapper'],previous)
        self.assertEqual(calls,['previous'])
        again=NODE.patch_guide_model(patched)
        self.assertIs(again.model_options['model_function_wrapper'],patched.model_options['model_function_wrapper'])
        self.assertIs(patched.model_options['model_function_wrapper'](model_fn,args),result)

    def test_other_models_and_conditioning_are_untouched(self):
        model=NODE.patch_guide_model(FakeModel())
        args={'input':1,'timestep':2,'c':{'other':3}}
        result=model.model_options['model_function_wrapper'](lambda x,t,**c:(x,t,c),args)
        self.assertEqual(result,(1,2,{'other':3}))



class NodeExecutionTest(unittest.TestCase):
    def setUp(self):
        import types
        self.comfy=types.ModuleType('comfy')
        ldm=types.ModuleType('comfy.ldm'); minimax=types.ModuleType('comfy.ldm.minimax')
        core=types.ModuleType('comfy.ldm.minimax.model')
        class Diffusion:
            patch_size=(1,2,2)
        core.MiniMaxH3Model=Diffusion
        core.FRAME_PER_TOKEN=(1,4,4,4,4); core.FRAME_RESCALE=5/3
        minimax.model=core; ldm.minimax=minimax; self.comfy.ldm=ldm
        self.modules={'comfy':self.comfy,'comfy.ldm':ldm,'comfy.ldm.minimax':minimax,'comfy.ldm.minimax.model':core}
        self.diffusion=Diffusion()
        self.model=FakeModel()
        self.model.get_model_object=lambda key:self.diffusion

    def target(self, vt):
        return {'samples':SimpleNamespace(is_nested=True,tensors=[torch.zeros(1,24,vt,8,8),torch.zeros(1,32,2,122)])}

    def test_image_node_encodes_reduced_guide_and_keeps_inputs(self):
        embedding=torch.zeros(1,3,4)
        positive=[[embedding,{'minimax_token_tags':torch.ones(3)}]]
        with patch.dict(sys.modules,self.modules), patch.object(NODE,'encode_guide',return_value=torch.zeros(1,24,1,2,2)) as encode:
            model,cond,preview=NODE.BFSMiniMaxH3DownscaledGuide().apply(self.model,positive,None,self.target(1),torch.zeros(1,128,128,3))
        self.assertEqual(tuple(preview.shape),(1,32,32,3))
        self.assertEqual(tuple(encode.call_args[0][1].shape),(1,32,32,3))
        self.assertNotIn('minimax_keyframes',positive[0][1])
        self.assertIs(cond[0][0],embedding)
        self.assertEqual(cond[0][1]['minimax_keyframes'][0][NODE.MARKER]['downscale_factor'],4)
        self.assertTrue(model.model_options[NODE.WRAPPER_MARKER])

    def test_video_node_validates_73_frames_without_truncation(self):
        with patch.dict(sys.modules,self.modules), patch.object(NODE,'encode_guide',return_value=torch.zeros(1,24,22,2,2)):
            _,cond,preview=NODE.BFSMiniMaxH3DownscaledGuide().apply(self.model,[[None,{}]],None,self.target(22),torch.zeros(73,32,32,3))
        self.assertEqual(preview.shape[0],73)
        self.assertEqual(cond[0][1]['minimax_keyframes'][0]['latent'].shape[2],22)

    def test_invalid_video_or_temporal_window_fails_before_encoding(self):
        for frames,index in [(3,0),(72,0),(73,1),(1,-74)]:
            with patch.dict(sys.modules,self.modules), patch.object(NODE,'encode_guide') as encode:
                with self.assertRaises(ValueError):
                    NODE.BFSMiniMaxH3DownscaledGuide().apply(self.model,[[None,{}]],None,self.target(22),torch.zeros(frames,32,32,3),frame_idx=index)
                encode.assert_not_called()

    def test_true_single_frame_and_video_targets(self):
        import types
        management=types.ModuleType('comfy.model_management')
        management.intermediate_device=lambda:torch.device('cpu')
        nested=types.ModuleType('comfy.nested_tensor')
        nested.NestedTensor=lambda tensors:SimpleNamespace(tensors=tensors,is_nested=True)
        self.comfy.model_management=management
        modules=dict(self.modules,**{'comfy.model_management':management,'comfy.nested_tensor':nested})
        with patch.dict(sys.modules,modules):
            for frames,vt,at in [(1,1,2),(5,2,8),(73,22,122),(124,37,207)]:
                result=NODE.BFSMiniMaxH3GuideTarget().create(128,128,frames)[0]['samples']
                self.assertEqual(tuple(result.tensors[0].shape),(1,24,vt,8,8))
                self.assertEqual(tuple(result.tensors[1].shape),(1,32,2,at))
            with self.assertRaises(ValueError): NODE.BFSMiniMaxH3GuideTarget().create(128,128,72)

    def test_metadata_factor_and_version_validation(self):
        import tempfile, types
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'guide.safetensors')
            save_file({'x':torch.zeros(1)},path,metadata={
                'reference_downscale_factor':'4','guide_latent_only':'true',
                'minimax_h3_guide_spatial_version':NODE.SPATIAL_VERSION,
                'minimax_h3_guide_position_version':NODE.POSITION_VERSION})
            folder=types.ModuleType('folder_paths'); folder.get_full_path_or_raise=lambda *args:path
            with patch.dict(sys.modules,{'folder_paths':folder}):
                NODE.check_lora_metadata('guide.safetensors',4)
                with self.assertRaisesRegex(ValueError,'factor'): NODE.check_lora_metadata('guide.safetensors',2)
            save_file({'x':torch.zeros(1)},path,metadata={})
            with patch.dict(sys.modules,{'folder_paths':folder}):
                with self.assertRaisesRegex(ValueError,'spatial_version'): NODE.check_lora_metadata('guide.safetensors',4)


if __name__ == "__main__":
    unittest.main()
