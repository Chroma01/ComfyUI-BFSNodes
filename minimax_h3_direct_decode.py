"""Decode direto do single-frame VAE do MiniMax-H3, na receita do card.

Motivo de existir: gerando pelo DiT e decodificando com o `VAEDecode` padrao, a saida sai
quadriculada. O caminho de referencia (card do iamkaikai, e o Space
`multimodalart/MiniMax-H3-images`) NAO usa `vae.decode()` -- chama o decoder direto. Quatro
detalhes que corrompem a saida em silencio se mudarem, todos preservados aqui:

  1. `vae.decoder(vae.post_quant_conv(z))` direto, nunca `vae.decode()`, cujo chunker temporal
     nao consegue formar chunk com poucas fatias de latente;
  2. `decoded[:, :, -1]` -- a ULTIMA fatia de saida, nao a de indice 0;
  3. desnormalizacao do latente: `z * latents_std + latents_mean`;
  4. desnormalizacao de pixel em ImageNet (mean .485/.456/.406, std .229/.224/.225) -- sem ela
     a imagem sai escura e com contraste estourado, parecendo erro de tone mapping.
"""

import comfy.model_management
import torch

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


class MiniMaxH3DirectDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "VAE do H3 com o decoder single-frame (use o loader deste pacote)."}),
                "samples": ("LATENT", {"tooltip": "Latente do H3. Funciona com o latente gerado pelo DiT, que e onde o VAEDecode padrao falha."}),
                "latent_index": ("INT", {"default": 0, "min": -64, "max": 64, "tooltip": "Qual fatia temporal decodificar. 0 = primeira, -1 = ultima."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax-H3"

    def decode(self, vae, samples, latent_index):
        model = getattr(vae, "first_stage_model", None)
        if model is None or not hasattr(model, "decoder"):
            raise ValueError(
                "este VAE nao e o do MiniMax-H3 (sem first_stage_model.decoder). "
                "Carregue com o loader single-frame deste pacote."
            )

        z = samples["samples"]
        # O latente do H3 chega aninhado (video + audio). Os nodes nativos desempacotam e ficam
        # com o primeiro componente, que e o video -- ver VAEDecode em nodes.py:335.
        if getattr(z, "is_nested", False):
            z = z.unbind()[0]
            if z.dim() == 4:  # unbind entrega o item sem a dimensao de batch
                z = z.unsqueeze(0)
        if z.dim() == 4:
            z = z.unsqueeze(0)
        if z.dim() != 5:
            raise ValueError(f"latente esperado [B, C, T, H, W]; veio {tuple(z.shape)}")

        # Sem isto o decode roda onde os pesos estiverem -- e com o gerenciamento de memoria do
        # ComfyUI o VAE fica na CPU ate ser usado, o que torna o decode lentissimo. O vae.decode()
        # nativo chama load_models_gpu antes de rodar; aqui e o mesmo passo.
        comfy.model_management.load_models_gpu([vae.patcher])
        device = vae.device
        dtype = vae.vae_dtype
        i = latent_index if latent_index >= 0 else z.shape[2] + latent_index
        i = max(0, min(i, z.shape[2] - 1))
        z = z[:, :, i : i + 1].to(device=device, dtype=torch.float32)

        # (3) desnormalizacao do latente
        mean = model.latents_mean.view(1, -1, 1, 1, 1).to(device=device, dtype=torch.float32)
        std = model.latents_std.view(1, -1, 1, 1, 1).to(device=device, dtype=torch.float32)
        z = z * std + mean

        # (1) decoder direto, sem vae.decode()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            decoded = model.decoder(model.post_quant_conv(z.to(dtype)))

        # (2) ultima fatia de saida + (4) desnormalizacao ImageNet
        pm = torch.tensor(PIXEL_MEAN, device=decoded.device).view(1, 3, 1, 1)
        ps = torch.tensor(PIXEL_STD, device=decoded.device).view(1, 3, 1, 1)
        frame = (decoded[:, :, -1].float() * ps + pm).clamp(0, 1)
        print(f"[BFSNodes] H3 direct decode: latente {tuple(z.shape)} -> {tuple(frame.shape)} (fatia {i}, {device})")
        saida = frame.permute(0, 2, 3, 1).to(comfy.model_management.intermediate_device())
        return (saida,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3DirectDecode": MiniMaxH3DirectDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3DirectDecode": "MiniMax-H3 Single-Frame Decode / direct (BFS)"}
