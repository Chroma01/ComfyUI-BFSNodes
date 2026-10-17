"""Sapiens2 body-part segmentation (Meta, facebook/sapiens2-seg-*): one class per pixel out of 29 (28 parts +
background), lips, teeth and tongue separate from the face. Used by the source blur for parts SAM 3.1 can't
find reliably from a text prompt (lips, tongue). It marks every person in the frame (no instances).

The weights (0.4B, 1.6 GB) go to models/sapiens2/sapiens2-seg-0.4b, downloaded from Hugging Face the first time.
Sapiens2 License: https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
"""
from __future__ import annotations

import os

import torch

REPO = "facebook/sapiens2-seg-0.4b"
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
CLASSES = ("Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm",
           "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
           "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
           "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth",
           "Upper_Teeth", "Tongue")
ID = {n: i for i, n in enumerate(CLASSES)}
PIXELS = 1024 * 768   # the size it was trained at; other shapes keep this area and their aspect ratio
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_MODEL: dict = {}


def model_dir() -> str:
    try:
        import folder_paths
        base = folder_paths.models_dir
    except Exception:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")
    return os.path.join(base, "sapiens2", REPO.split("/")[-1])


def _load():
    if "m" in _MODEL:
        return _MODEL["m"]
    d = model_dir()
    if not all(os.path.exists(os.path.join(d, f)) for f in FILES):
        print(f"[BFS Sapiens2] downloading {REPO} (1.6 GB) to {d}", flush=True)
        from huggingface_hub import hf_hub_download
        for f in FILES:
            hf_hub_download(REPO, f, local_dir=d)
    from transformers import Sapiens2ForSemanticSegmentation
    _MODEL["m"] = Sapiens2ForSemanticSegmentation.from_pretrained(d, dtype=torch.float32).eval()
    return _MODEL["m"]


def _device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def input_size(h: int, w: int) -> tuple[int, int]:
    """(H, W) with the trained area, the frame's aspect ratio, multiples of the 16 px patch."""
    s = (PIXELS / (h * w)) ** 0.5
    return max(16, round(h * s / 16) * 16), max(16, round(w * s / 16) * 16)


@torch.no_grad()
def segment(frames: torch.Tensor, out_size: tuple[int, int] | None = None, batch: int = 4) -> torch.Tensor:
    """frames [N,H,W,3] 0-1 -> class ids [N,h,w] uint8 at `out_size` (default the frames' size)."""
    m = _load()
    dev = _device()
    half = dev.type == "cuda"
    m.to(dev, dtype=torch.float16 if half else torch.float32)
    N, H, W = frames.shape[:3]
    oh, ow = out_size or (H, W)
    ih, iw = input_size(H, W)
    mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
    out = torch.empty(N, oh, ow, dtype=torch.uint8)
    try:
        for a in range(0, N, batch):
            x = frames[a:a + batch, ..., :3].to(dev).float().movedim(-1, 1)
            x = torch.nn.functional.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False, antialias=True)
            x = ((x - mean) / std).to(torch.float16 if half else torch.float32)
            lg = m(pixel_values=x).logits.float()
            lg = torch.nn.functional.interpolate(lg, size=(oh, ow), mode="bilinear", align_corners=False)
            out[a:a + batch] = lg.argmax(1).to(torch.uint8).cpu()
            del x, lg
    finally:
        m.to("cpu")   # 0.4B: back to RAM, the VRAM is the video model's
        if half:
            torch.cuda.empty_cache()
    return out


def parts(classes: torch.Tensor, names) -> torch.Tensor:
    """[N,h,w] float mask of the pixels whose class is one of `names`."""
    ids = torch.tensor([ID[n] for n in names], dtype=classes.dtype)
    return torch.isin(classes, ids).float()
