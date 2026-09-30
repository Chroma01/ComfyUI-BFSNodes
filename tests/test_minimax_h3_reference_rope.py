import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from test_minimax_h3_downscaled_guide import NODE,CORE,FakeModel,NodeExecutionTest,guide

class PhaseModel(FakeModel):
    def __init__(self,options=None,objects=None):
        super().__init__(options);self.objects=objects or {}
    def clone(self):return PhaseModel(dict(self.model_options),dict(self.objects))
    def get_model_object(self,key):
        return self.objects.get(key,self.original)
    def add_object_patch(self,key,value):self.objects[key]=value
    @staticmethod
    def original(positions,device):
        inv=10000**(-torch.arange(16)/16)
        half=(positions.float().unsqueeze(-1)*inv).flatten(1)
        return torch.cat((half,half),-1).to(device)

class SourcePhaseTest(unittest.TestCase):
    def test_patch_is_scoped_composable_and_does_not_leak_to_unmarked_payload(self):
        original=PhaseModel();patched=NODE.patch_guide_model(original,True)
        kf=guide(4);kf[NODE.MARKER].update(source_phase=True,source_id=3,phase_scale=.7)
        native=CORE.PackedLayout(3,2,8,8,2,keyframes=[kf])
        def forward(x,t,**kwargs):
            layout=kwargs['minimax_payload']['layout']
            return patched.get_model_object('diffusion_model.rope_freqs')(layout.position_ids,'cpu'),layout
        fn=patched.model_options['model_function_wrapper']
        args={'input':0,'timestep':0,'c':{'minimax_payload':{'layout':native,'keyframes':[kf]}}}
        angles,layout=fn(forward,args)
        expected=NODE.add_source_phase(PhaseModel.original(layout.position_ids,'cpu'),layout.bfs_source_phase_values)
        torch.testing.assert_close(angles,expected)
        self.assertFalse(original.objects)
        again=NODE.patch_guide_model(patched,True)
        self.assertIs(again.objects['diffusion_model.rope_freqs'],patched.objects['diffusion_model.rope_freqs'])
        blank=CORE.PackedLayout(3,2,8,8,2)
        angles,_=fn(forward,{'input':0,'timestep':0,'c':{'minimax_payload':{'layout':blank}}})
        torch.testing.assert_close(angles,PhaseModel.original(blank.position_ids,'cpu'))

    def test_default_and_experimental_metadata_are_checked(self):
        import types
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'lora.safetensors')
            meta={'minimax_h3_guide_spatial_version':NODE.SPATIAL_VERSION,'minimax_h3_guide_position_version':NODE.POSITION_VERSION,
                  'reference_downscale_factor':'4','guide_latent_only':'true'}
            opts=NODE.validate_options('overlap','sidecar',True,.7,2)
            meta['minimax_h3_reference_rope']=json.dumps(dict(opts,phase_version=NODE.PHASE_VERSION))
            save_file({'w':torch.zeros(1)},path,metadata=meta)
            folders=types.SimpleNamespace(get_full_path_or_raise=lambda *args:path)
            with patch.dict('sys.modules',{'folder_paths':folders}):
                NODE.check_lora_metadata('lora',4,opts)
                with self.assertRaisesRegex(ValueError,'RoPE options'):NODE.check_lora_metadata('lora',4)

    def test_identity_node_uses_own_aspect_and_training_image_bucket(self):
        harness=NodeExecutionTest();harness.setUp()
        positive=[[torch.zeros(1,3,4),{}]]
        image=torch.zeros(1,192,64,3)
        with patch.dict('sys.modules',harness.modules),patch.object(NODE,'encode_guide',return_value=torch.zeros(1,24,1,12,4)) as encoder:
            model,conditioning,preview=NODE.BFSMiniMaxH3IdentityReference().apply(harness.model,positive,None,harness.target(2),image)
        self.assertEqual(tuple(preview.shape),(1,192,64,3))
        block=conditioning[0][1]['minimax_refs'][0]
        self.assertEqual(block[NODE.MARKER]['source_id'],2)
        self.assertEqual((block['latent_h'],block['latent_w']),(12,4))
        self.assertNotIn('minimax_refs',positive[0][1])
        self.assertEqual(encoder.call_count,1)

if __name__=='__main__':unittest.main()
