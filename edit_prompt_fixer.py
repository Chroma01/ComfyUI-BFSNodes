"""Corrige a saida do VLM para o formato que o LoRA foi treinado.

Duas regras que o VLM erra de forma consistente, por mais explicito que o prompt seja, e que
sao deterministicas -- quem chama ja sabe se anexou referencia:

  sem referencia  ->  a frase-marcador nao pode existir, e "trocar" e um Replace simples.
                      O VLM tende a emitir Add + marcador + Remove, que e a forma da referencia.
  com referencia  ->  o marcador so pode seguir uma frase que comeca com Add. Se vier num
                      Replace, vira Add + Remove.
"""

import re

MARCADOR = "The object to add is shown in the reference image."
TRIGGER = "edit_anything: "


def _frases(texto):
    corpo = texto[len(TRIGGER):] if texto.startswith(TRIGGER) else texto
    return [f.strip() for f in re.split(r"(?<=\.)\s+", corpo.strip()) if f.strip()]


def _juntar(frases):
    return TRIGGER + " ".join(frases)


def corrigir(prompt, tem_referencia):
    frases = _frases(prompt)
    mudou = []

    if not tem_referencia:
        antes = len(frases)
        frases = [f for f in frases if f.rstrip() != MARCADOR]
        if len(frases) != antes:
            mudou.append("marcador removido (sem referencia)")

        # Add(X) ... Remove(Y) sem referencia = forma da referencia aplicada no lugar errado.
        # Recompoe como Replace Y with X.
        i_add = next((i for i, f in enumerate(frases) if f.startswith("Add ")), None)
        i_rem = next((i for i, f in enumerate(frases) if f.startswith("Remove ")), None)
        if i_add is not None and i_rem is not None:
            novo = re.sub(r"^Add\s+", "", frases[i_add]).rstrip(".")
            velho = re.sub(r"^Remove\s+", "", frases[i_rem]).rstrip(".")
            # descarta localizacao redundante herdada do Add
            novo = re.sub(r"\s+(where|parked|standing|placed)\b.*$", "", novo).strip()
            frases = [f for k, f in enumerate(frases) if k not in (i_add, i_rem)]
            frases.insert(min(i_add, i_rem), f"Replace {velho} with {novo}.")
            mudou.append("Add+Remove recomposto como Replace")
    else:
        for i, f in enumerate(frases):
            if f.rstrip() == MARCADOR and i > 0 and not frases[i - 1].startswith("Add "):
                anterior = frases[i - 1]
                m = re.match(r"^Replace\s+(.*?)\s+with\s+(.*)\.$", anterior)
                if m:
                    velho, novo = m.group(1), m.group(2)
                    frases[i - 1] = f"Add {novo}."
                    frases.insert(i + 1, f"Remove {velho}.")
                    mudou.append("Replace com marcador virou Add + Remove")
                break

    return _juntar(frases), mudou


class EditAnythingPromptFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "has_reference": ("BOOLEAN", {"default": False, "tooltip": "Ligue quando uma imagem de referencia for anexada. Este e o mesmo booleano que decide anexar a imagem."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "fix"
    CATEGORY = "BFS/EditAnything"

    def fix(self, prompt, has_reference):
        saida, mudou = corrigir(prompt, has_reference)
        if mudou:
            print(f"[BFSNodes] prompt corrigido: {', '.join(mudou)}")
            print(f"[BFSNodes]   {saida}")
        return (saida,)


NODE_CLASS_MAPPINGS = {"EditAnythingPromptFixer": EditAnythingPromptFixer}
NODE_DISPLAY_NAME_MAPPINGS = {"EditAnythingPromptFixer": "Edit Anything Prompt Fixer (BFS)"}
