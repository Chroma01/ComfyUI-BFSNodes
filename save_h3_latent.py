"""Salva um latente do MiniMax-H3 em disco.

O `SaveLatent` nativo quebra com o latente do H3: ele chega como NestedTensor (video + audio)
e o node chama `.contiguous()` direto. Aqui os componentes sao desempacotados e salvos
separadamente, para o latente poder ser inspecionado ou decodificado fora do grafo.
"""

import os

import folder_paths
import torch
from safetensors.torch import save_file


class SaveH3Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "h3_latent"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "BFS/debug"

    def save(self, samples, filename_prefix):
        z = samples["samples"]
        tensores = {}
        if getattr(z, "is_nested", False):
            for i, parte in enumerate(z.unbind()):
                t = parte if parte.dim() == 5 else parte.unsqueeze(0)
                tensores[f"component_{i}"] = t.contiguous().cpu()
        else:
            t = z if z.dim() == 5 else z.unsqueeze(0)
            tensores["component_0"] = t.contiguous().cpu()

        pasta = folder_paths.get_output_directory()
        n = 0
        while os.path.exists(os.path.join(pasta, f"{filename_prefix}_{n:05d}.safetensors")):
            n += 1
        caminho = os.path.join(pasta, f"{filename_prefix}_{n:05d}.safetensors")
        meta = {k: str(tuple(v.shape)) for k, v in tensores.items()}
        save_file(tensores, caminho, metadata={"format": "pt", **meta})
        print(f"[BFSNodes] latente salvo em {caminho}")
        for k, v in tensores.items():
            print(f"[BFSNodes]   {k}: {tuple(v.shape)} {v.dtype}")
        return {}


NODE_CLASS_MAPPINGS = {"SaveH3Latent": SaveH3Latent}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveH3Latent": "Save H3 Latent / nested-safe (BFS)"}
