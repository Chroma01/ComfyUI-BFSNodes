"""Escolhe o frame mais nitido de um lote, por variancia do Laplaciano.

Vem do IF-Edit (arXiv:2511.19435): frames decodificados carregam quantidades diferentes de
borrao residual, entao pontuar e ficar com o mais crocante bate pegar um indice fixo.

Util em particular com o MiniMax-H3 usado como gerador de imagem: o clipe minimo do VAE
(5 frames = 2 latentes) sai do decode com nitidez desigual, e o melhor frame nao e sempre
o mesmo indice.

CUIDADO -- e uma heuristica de BORRAO, nao uma metrica de qualidade. Variancia do
Laplaciano mede energia de alta frequencia, e artefato tambem e alta frequencia. Medido
num decode do H3: entre 4 frames do mesmo bloco, o Laplaciano escolheu o de PIOR PSNR
(19.72 dB contra 23.19 do vizinho), porque ele era o mais artefatado. Use quando os
candidatos forem o MESMO conteudo diferindo so em borrao -- que e o caso do IF-Edit e do
clipe estatico gerado. Nao use para escolher entre frames de momentos diferentes.
"""

import torch


def _laplacian_variance(img: torch.Tensor) -> float:
    """img: [H, W, C] em [0,1]. Variancia do Laplaciano 4-vizinhos sobre a luminancia."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=torch.float32)
    g = (img[..., :3].float() * weights).sum(-1)
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


class SharpestFrame:
    """Recebe um lote de imagens e devolve a mais nitida, com o indice e as notas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Lote de frames que sejam o MESMO conteudo diferindo em borrao (ex.: clipe estatico do H3 em modo imagem). Para frames de momentos diferentes isso nao faz sentido."}),
            },
            "optional": {
                "skip_first": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Ignora os N primeiros frames antes de pontuar. No H3 os primeiros frames de cada bloco decodificado costumam ser preenchimento temporal."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "index", "scores")
    FUNCTION = "pick"
    CATEGORY = "BFS/image"

    def pick(self, images, skip_first=0):
        total = images.shape[0]
        start = min(skip_first, max(total - 1, 0))
        scores = [_laplacian_variance(images[i]) for i in range(start, total)]
        best = start + max(range(len(scores)), key=scores.__getitem__)
        texto = " | ".join(f"{start + i}:{s:.5f}{' <-' if start + i == best else ''}" for i, s in enumerate(scores))
        print(f"[BFSNodes] SharpestFrame: {total} frames, escolhido {best} -> {texto}")
        return (images[best:best + 1], best, texto)


NODE_CLASS_MAPPINGS = {"SharpestFrame": SharpestFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"SharpestFrame": "Sharpest Frame / Laplacian (BFS)"}
