"""Loader do decoder single-frame do MiniMax-H3 (iamkaikai/MiniMax-H3-Single-Frame-VAE-500K).

O checkpoint publicado e decoder-only e usa o naming do diffusers, entao o VAELoader padrao
do ComfyUI nao o carrega: a deteccao em comfy/sd.py exige tambem uma chave de encoder
(`encoder.down.5.block.0.conv1.weight`), e sem ela o arquivo cai numa arquitetura errada e a
saida vira blocos.

Este node resolve os dois lados: converte as chaves para o layout do ComfyUI e completa o que
falta (encoder, quant_conv, latents_mean/std, mask_token) a partir do VAE oficial do H3.

Duas transformacoes NAO sao dedutiveis pelo nome, e errar qualquer uma produz saida errada
sem nenhum erro de load. Ambas foram determinadas medindo correlacao contra o VAE oficial:

  to_qkv  e intercalado por cabeca -- (heads, 3, dim_head), nao [q;k;v] empilhado.
          Confirmado em comfy/ldm/minimax/vae.py: `qkv.view(B, S, -1, 3 * dim_head)`.
  ff.w1   tem as metades trocadas em relacao ao diffusers, porque o ComfyUI le
          `gate, x = w1(x).chunk(2)` -- gate primeiro.

Validado contra o caminho de referencia do autor (diffusers + load_decoder.py dele) com o
mesmo latente: PSNR 72,92 dB entre as duas saidas, ou seja, ruido de fp16.
"""

import comfy.sd
import comfy.utils
import folder_paths
import torch

HEADS = 32
DIM_HEAD = 64
# Chaves que o comfy/sd.py usa para reconhecer o VAE do MiniMax-H3.
DETECT_KEYS = ("decoder.transformer_blocks.0.scale1", "encoder.down.5.block.0.conv1.weight")


def _convert_decoder_keys(decoder_sd):
    """Naming diffusers -> naming ComfyUI. Devolve (state_dict, n_blocos)."""
    out = {}
    blocks = set()
    for key, tensor in decoder_sd.items():
        if ".attn.to_q." in key or ".attn.to_k." in key or ".attn.to_v." in key:
            blocks.add(int(key.split("transformer_blocks.")[1].split(".")[0]))
            continue
        new_key = (
            key.replace(".ff.net.0.proj.", ".ff.w1.")
            .replace(".ff.net.2.", ".ff.w2.")
            .replace(".attn.to_out.0.", ".attn.to_out.")
            .replace("decoder.proj_in.", "decoder.x_embedder.")
        )
        if ".ff.w1." in new_key:  # ComfyUI le (gate, valor); diffusers grava (valor, gate)
            half = tensor.shape[0] // 2
            tensor = torch.cat([tensor[half:], tensor[:half]], dim=0)
        out[new_key] = tensor

    for b in sorted(blocks):
        for suffix in ("weight", "bias"):
            parts = [decoder_sd[f"decoder.transformer_blocks.{b}.attn.to_{n}.{suffix}"] for n in ("q", "k", "v")]
            tail = parts[0].shape[1:]
            stacked = torch.stack([p.view(HEADS, DIM_HEAD, *tail) for p in parts], dim=1)  # [H, 3, DH, ...]
            out[f"decoder.transformer_blocks.{b}.attn.to_qkv.{suffix}"] = stacked.reshape(HEADS * 3 * DIM_HEAD, *tail)
    return out, len(blocks)


class MiniMaxH3SingleFrameVAELoader:
    """Monta um VAE do H3 com o decoder single-frame por cima do VAE oficial."""

    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "base_vae": (vaes, {"tooltip": "VAE oficial do MiniMax-H3 (fornece encoder, quant_conv e as estatisticas de latente)."}),
                "single_frame_decoder": (vaes, {"tooltip": "Checkpoint decoder-only do autor (585 tensores, naming diffusers). Tambem aceita um ja convertido."}),
                "tiling": ("BOOLEAN", {"default": True, "tooltip": "Tiling espacial. MANTENHA LIGADO acima de ~768px: o curriculo do autor parou em 1024 e a imagem inteira acima disso produz grade na fronteira de 32px (blocagem 2.9x em 1024, 3.5x em 1536, contra ~1.5x com tiling). Desligar so compensa em imagens pequenas, onde rende um pouco mais de nitidez."}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "tooltip": "So vale com tiling ligado. 512 e o otimo MEDIDO, nao um chute: numa imagem 1056x640, PSNR 26.00 dB com costura 0.91 (indistinguivel), contra 22.17/1.49 com 256 (o default do ComfyUI, que multiplica costuras) e 21.35/2.42 com 1024 (acima da resolucao treinada). O autor treinou 475k das 500k imagens em <=512px, entao o tile de 512 mantem cada pedaco na regiao com massa de treino."}),
                "dtype": (["float16", "float32", "bfloat16"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3"

    def load(self, base_vae, single_frame_decoder, tiling, tile_size, dtype):
        base_sd = comfy.utils.load_torch_file(folder_paths.get_full_path("vae", base_vae))
        dec_sd = comfy.utils.load_torch_file(folder_paths.get_full_path("vae", single_frame_decoder))

        missing_detect = [k for k in DETECT_KEYS if k not in base_sd]
        if missing_detect:
            raise ValueError(
                f"'{base_vae}' nao parece o VAE oficial do MiniMax-H3 (faltam {missing_detect}). "
                "Escolha o VAE completo, com encoder."
            )

        if any(".attn.to_q." in k for k in dec_sd):
            converted, n_blocks = _convert_decoder_keys(dec_sd)
            print(f"[BFSNodes] single-frame decoder convertido do naming diffusers ({n_blocks} blocos)")
        else:
            converted = dict(dec_sd)  # ja esta no layout do ComfyUI
            print("[BFSNodes] single-frame decoder ja estava no naming do ComfyUI")

        sd = dict(base_sd)
        replaced, unknown = 0, []
        for key, tensor in converted.items():
            if key in sd:
                if tuple(sd[key].shape) != tuple(tensor.shape):
                    raise ValueError(f"shape divergente em '{key}': base {tuple(sd[key].shape)} vs decoder {tuple(tensor.shape)}")
                sd[key] = tensor.to(sd[key].dtype)
                replaced += 1
            else:
                unknown.append(key)
        if unknown:
            raise ValueError(f"chaves do decoder que nao existem no VAE base: {unknown[:5]} (total {len(unknown)})")
        # Relatorio honesto: um checkpoint decoder-only substitui so decoder/post_quant_conv, mas um
        # arquivo ja convertido carrega o VAE inteiro e sobrescreve tambem o encoder do base.
        from collections import Counter
        por_prefixo = Counter(k.split(".")[0] for k in converted)
        do_base = sorted({k.split(".")[0] for k in sd} - set(por_prefixo))
        print(f"[BFSNodes] {replaced} tensores substituidos: "
              + ", ".join(f"{p}={n}" for p, n in sorted(por_prefixo.items())))
        print(f"[BFSNodes] preservado do base: {', '.join(do_base) if do_base else 'nada (o arquivo cobriu o VAE inteiro)'}")
        if "encoder" in por_prefixo:
            print("[BFSNodes] aviso: o arquivo tambem trouxe encoder, entao o encoder do base foi sobrescrito")

        torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        vae = comfy.sd.VAE(sd=sd, dtype=torch_dtype)

        model = getattr(vae, "first_stage_model", None)
        if model is not None and hasattr(model, "tiling"):
            model.tiling = bool(tiling)
            model.tile_size = int(tile_size)
            print(f"[BFSNodes] tiling={'on' if tiling else 'off'} tile_size={tile_size}")
        return (vae,)

class MiniMaxH3VAELoaderTiled:
    """Carrega um VAE do H3 ja completo (por exemplo um single-frame convertido) expondo o tiling.

    Igual ao VAELoader nativo, com uma diferenca que muda o resultado: o construtor do
    MiniMaxH3VideoVAE usa tile_size=256, e o loader nativo nao expoe esse campo. 256 e o pior
    valor medido -- multiplica costuras. Aqui da para por 512.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "VAE completo do H3 (com encoder). Para o checkpoint decoder-only do autor, use o outro node."}),
                "tiling": ("BOOLEAN", {"default": True, "tooltip": "Tiling espacial. MANTENHA LIGADO acima de ~768px: o curriculo do autor parou em 1024 e a imagem inteira acima disso produz grade na fronteira de 32px."}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "tooltip": "512 e o otimo MEDIDO: numa imagem 1056x640, PSNR 26.00 dB / costura 0.91, contra 22.17/1.49 com 256 (o default do ComfyUI) e 21.35/2.42 com 1024. O autor treinou 475k das 500k imagens em <=512px."}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3"

    def load(self, vae_name, tiling, tile_size):
        path = folder_paths.get_full_path("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)

        missing = [k for k in DETECT_KEYS if k not in sd]
        if missing:
            raise ValueError(
                f"'{vae_name}' nao sera reconhecido como MiniMax-H3 pelo ComfyUI (faltam {missing}). "
                "Um checkpoint decoder-only cai em geometria de Stable Diffusion e produz blocos -- "
                "use o node 'MiniMax-H3 Single-Frame VAE Loader' para combina-lo com o VAE oficial."
            )

        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        model = getattr(vae, "first_stage_model", None)
        if model is not None and hasattr(model, "tiling"):
            model.tiling = bool(tiling)
            model.tile_size = int(tile_size)
            print(f"[BFSNodes] {vae_name}: tiling={'on' if tiling else 'off'} tile_size={tile_size}")
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SingleFrameVAELoader": MiniMaxH3SingleFrameVAELoader,
    "MiniMaxH3VAELoaderTiled": MiniMaxH3VAELoaderTiled,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SingleFrameVAELoader": "MiniMax-H3 Single-Frame VAE Loader (BFS)",
    "MiniMaxH3VAELoaderTiled": "MiniMax-H3 VAE Loader / tile control (BFS)",
}
