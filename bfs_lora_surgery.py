"""BFS LoRA Surgery: read any LoRA's layers and blocks, then scale or drop them.

Model-agnostic: the structure is discovered from the key names, not hardcoded. Works with
the diffusers style (``lora_A``/``lora_B``), the kohya style (``lora_down``/``lora_up`` plus
``alpha``) and anything that follows either convention.

Why this exists: a defect a LoRA learned (smoothed skin, identity drifting at distance, a
pose it refuses to copy) usually lives in a specific module family and a specific range of
blocks. Scaling or dropping that group and regenerating tells you where it lives, in minutes,
without retraining. See the node's tooltip for the workflow.

Scaling a module by ``s`` multiplies only the up/B factor, since ``(sB)A = s(BA)``. That keeps
negative values usable and is exact regardless of the alpha convention.
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Any

import torch

import comfy.sd
import comfy.utils
import folder_paths

# ---------------------------------------------------------------------------- parsing

_SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight",
    ".lora_A.default.weight", ".lora_B.default.weight",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
    ".diff", ".diff_b", ".alpha", ".dora_scale",
)
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight", ".lora_B.default.weight")
_BLOCK_RE = re.compile(r"(?<=[._])(\d+)(?=[._])")


def _strip_suffix(key: str) -> str | None:
    for s in _SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)]
    return None


def _split_block(path: str) -> tuple[str, int | None, str]:
    """Return (family, block_index, module_type) using the LAST numeric path component.

    ``transformer_blocks.12.attn.to_q`` -> ("transformer_blocks", 12, "attn.to_q")
    Keys with no numeric component come back with block ``None``.
    """
    matches = list(_BLOCK_RE.finditer(path))
    if not matches:
        return path, None, ""
    m = matches[-1]
    family = path[: m.start()].rstrip("._")
    module = path[m.end():].lstrip("._")
    return family, int(m.group(1)), module


def read_structure(lora_path: str, with_norms: bool = True) -> dict[str, Any]:
    """Describe a LoRA: families, block indices, module types, ranks and ΔW norms.

    Reads the safetensors header first (instant) and only loads tensors when norms are asked
    for. Norms are what tell you where training actually invested.
    """
    with open(lora_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)

    modules: dict[str, dict[str, Any]] = {}
    for key, info in header.items():
        base = _strip_suffix(key)
        if base is None:
            continue
        entry = modules.setdefault(base, {"rank": None, "keys": []})
        entry["keys"].append(key)
        shape = info.get("shape") or []
        if len(shape) == 2 and entry["rank"] is None:
            entry["rank"] = int(min(shape))

    norms: dict[str, float] = {}
    if with_norms and modules:
        norms = _module_norms(lora_path, modules)

    families: dict[str, dict[str, Any]] = {}
    loose: list[dict[str, Any]] = []
    for base, entry in modules.items():
        family, block, mtype = _split_block(base)
        if block is None:
            loose.append({"path": base, "rank": entry["rank"], "norm": norms.get(base)})
            continue
        fam = families.setdefault(family, {"name": family, "blocks": set(), "types": {}})
        fam["blocks"].add(block)
        t = fam["types"].setdefault(mtype, {"name": mtype, "count": 0, "rank": entry["rank"],
                                            "norm_sum": 0.0, "norm_n": 0, "blocks": []})
        t["count"] += 1
        t["blocks"].append(block)
        if base in norms:
            t["norm_sum"] += norms[base]
            t["norm_n"] += 1

    out_families = []
    for fam in families.values():
        types = []
        for t in fam["types"].values():
            types.append({
                "name": t["name"], "count": t["count"], "rank": t["rank"],
                "norm": (t["norm_sum"] / t["norm_n"]) if t["norm_n"] else None,
                "blocks": sorted(t["blocks"]),
            })
        types.sort(key=lambda x: -(x["norm"] or 0))
        out_families.append({
            "name": fam["name"],
            "blocks": sorted(fam["blocks"]),
            "types": types,
        })
    out_families.sort(key=lambda f: -sum(t["count"] for t in f["types"]))

    return {
        "file": os.path.basename(lora_path),
        "total_modules": len(modules),
        "families": out_families,
        "loose": sorted(loose, key=lambda x: x["path"])[:64],
        "per_module_norms": norms,
    }


def _module_norms(lora_path: str, modules: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Exact ||ΔW||_F per module.

    ΔW = B·A has rank <= r, so QR both factors and take the singular values of the small
    r x r product. No need to materialize the full d x d matrix.
    """
    try:
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    except Exception:
        return {}
    out: dict[str, float] = {}
    for base in modules:
        down = sd.get(base + ".lora_down.weight", sd.get(base + ".lora_A.weight"))
        up = sd.get(base + ".lora_up.weight", sd.get(base + ".lora_B.weight"))
        if down is None or up is None or down.ndim != 2 or up.ndim != 2:
            continue
        try:
            a = down.float()
            b = up.float()
            scale = 1.0
            alpha = sd.get(base + ".alpha")
            if alpha is not None:
                scale = float(alpha) / a.shape[0]
            qb, rb = torch.linalg.qr(b)
            qa, ra = torch.linalg.qr(a.T)
            s = torch.linalg.svdvals(rb @ ra.T)
            out[base] = float(s.pow(2).sum().sqrt()) * scale
        except Exception:
            continue
    del sd
    return out


# ---------------------------------------------------------------------------- rules

def _match_blocks(spec: str | None, block: int) -> bool:
    """``spec`` is a block selector: "" / "all" / "8-15" / "0,4,7" / "8-15,24-31"."""
    if not spec or spec.strip().lower() in ("all", "*"):
        return True
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                if int(lo) <= block <= int(hi):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == block:
                    return True
            except ValueError:
                continue
    return False


def build_regex(family: str | None, mtype: str | None, blocks: str | None) -> str:
    r"""Build the regex a selection stands for. This is what the UI's "to regex" button calls.

    Selecting family ``transformer_blocks``, type ``img_mlp.gate_up`` and blocks ``8-15``
    gives ``^.*transformer_blocks\.(8|9|10|11|12|13|14|15)\.img_mlp\.gate_up$``.
    Leaving blocks empty matches every block.
    """
    fam = re.escape(family) if family else r"[\\w.]+"
    if blocks and blocks.strip().lower() not in ("all", "*", ""):
        idx: list[int] = []
        for part in blocks.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, _, hi = part.partition("-")
                try:
                    idx.extend(range(int(lo), int(hi) + 1))
                except ValueError:
                    continue
            else:
                try:
                    idx.append(int(part))
                except ValueError:
                    continue
        blk = "(" + "|".join(str(i) for i in sorted(set(idx))) + ")" if idx else r"\\d+"
    else:
        blk = r"\\d+"
    if not mtype or mtype.strip() in ("*", ""):
        mod = r".*"
    elif mtype.endswith("*"):
        mod = re.escape(mtype[:-1]) + r".*"
    else:
        mod = re.escape(mtype)
    return rf"^.*{fam}\.{blk}\.{mod}$"


def _match_type(spec: str | None, mtype: str) -> bool:
    if not spec or spec.strip() in ("*", ""):
        return True
    spec = spec.strip()
    if spec.endswith("*"):
        return mtype.startswith(spec[:-1])
    return mtype == spec


_REGEX_CACHE: dict[str, Any] = {}


def _compiled(pattern: str):
    rx = _REGEX_CACHE.get(pattern)
    if rx is None:
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = False          # invalid pattern matches nothing, and never raises mid-run
        _REGEX_CACHE[pattern] = rx
    return rx


def resolve_scale(rules: list[dict[str, Any]], family: str, block: int | None, mtype: str,
                  full_path: str | None = None) -> float:
    """Last matching rule wins, which is what makes the UI predictable.

    A rule matches either structurally (family / type / blocks) or by ``regex`` against the
    module's full path. A rule carrying a regex ignores the structural fields.
    """
    scale = 1.0
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        m = rule.get("match", {})
        pattern = m.get("regex")
        if pattern:
            rx = _compiled(pattern)
            if not rx or full_path is None or not rx.search(full_path):
                continue
            try:
                scale = float(rule.get("scale", 1.0))
            except (TypeError, ValueError):
                pass
            continue
        if m.get("family") and m["family"] != family:
            continue
        if not _match_type(m.get("type"), mtype):
            continue
        if block is not None and not _match_blocks(m.get("blocks"), block):
            continue
        if block is None and m.get("blocks"):
            continue
        try:
            scale = float(rule.get("scale", 1.0))
        except (TypeError, ValueError):
            continue
    return scale


def apply_rules(sd: dict[str, torch.Tensor], rules: list[dict[str, Any]]) -> tuple[dict, dict]:
    """Return (new state dict, stats). Scale 0 drops the module entirely."""
    if not rules:
        return sd, {"kept": None, "dropped": 0, "scaled": 0}
    out: dict[str, torch.Tensor] = {}
    dropped = scaled = kept = 0
    seen: dict[str, float] = {}
    for key, tensor in sd.items():
        base = _strip_suffix(key)
        if base is None:
            out[key] = tensor
            continue
        if base not in seen:
            family, block, mtype = _split_block(base)
            seen[base] = resolve_scale(rules, family, block, mtype, full_path=base)
        s = seen[base]
        if s == 0.0:
            dropped += 1 if key.endswith(_UP_SUFFIXES) else 0
            continue
        if s != 1.0 and key.endswith(_UP_SUFFIXES):
            # (sB)A == s(BA): exact, and negative values stay usable
            out[key] = tensor.float().mul(s).to(tensor.dtype)
            scaled += 1
        else:
            out[key] = tensor
        if key.endswith(_UP_SUFFIXES):
            kept += 1
    return out, {"kept": kept, "dropped": dropped, "scaled": scaled}


# ---------------------------------------------------------------------------- node

class BFSLoraSurgery:
    """Load a LoRA, scale or drop parts of it, and apply the result to the model.

    Nothing is written to disk: the edited LoRA lives only in this run, so you can move a
    slider and re-queue. Use the panel to pick a module family, a block range and a scale,
    or write a regex when you want something the tree cannot express.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0,
                                             "step": 0.01, "tooltip": "Overall LoRA strength, as usual."}),
                "rules": ("STRING", {"default": "[]", "multiline": True,
                                     "tooltip": "JSON list of rules. The panel writes this for you."}),
            },
            "optional": {"clip": ("CLIP",)},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "report")
    FUNCTION = "apply"
    CATEGORY = "BFS/lora"
    DESCRIPTION = ("Scale or drop a LoRA's module groups (attention, MLP, per block range) and "
                   "apply it live. Finds which layers carry a defect without retraining.")

    def apply(self, model, lora_name, strength_model, rules, clip=None):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"LoRA not found: {lora_name}")
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        parsed: list[dict[str, Any]] = []
        if rules and rules.strip():
            try:
                loaded = json.loads(rules)
                if isinstance(loaded, dict):
                    loaded = loaded.get("rules", [])
                if isinstance(loaded, list):
                    parsed = loaded
            except json.JSONDecodeError as exc:
                raise ValueError(f"'rules' is not valid JSON: {exc}") from exc

        edited, stats = apply_rules(sd, parsed)
        total = sum(1 for k in sd if k.endswith(_UP_SUFFIXES))
        kept = stats["kept"] if stats["kept"] is not None else total
        report = (f"{lora_name}: {kept}/{total} modules kept, {stats['dropped']} dropped, "
                  f"{stats['scaled']} scaled, {len(parsed)} rule(s), strength {strength_model}")
        print(f"[BFSNodes] LoRA surgery, {report}")

        new_model, new_clip = comfy.sd.load_lora_for_models(
            model, clip, edited, strength_model, strength_model if clip is not None else 0.0)
        return (new_model, new_clip if clip is not None else clip, report)


NODE_CLASS_MAPPINGS = {"BFSLoraSurgery": BFSLoraSurgery}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSLoraSurgery": "BFS LoRA Surgery"}

# ---------------------------------------------------------------------------- http api

try:
    from aiohttp import web
    from server import PromptServer

    _STRUCT_CACHE: dict[tuple[str, float], dict] = {}

    @PromptServer.instance.routes.get("/bfs/lora/structure")
    async def _bfs_lora_structure(request):
        name = request.query.get("name", "")
        with_norms = request.query.get("norms", "1") not in ("0", "false", "no")
        path = folder_paths.get_full_path("loras", name)
        if path is None:
            return web.json_response({"error": f"LoRA not found: {name}"}, status=404)
        key = (path, os.path.getmtime(path))
        cached = _STRUCT_CACHE.get(key)
        if cached is None or (with_norms and not cached.get("per_module_norms")):
            try:
                cached = read_structure(path, with_norms=with_norms)
            except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
                return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            _STRUCT_CACHE[key] = cached
        return web.json_response(cached)

    @PromptServer.instance.routes.post("/bfs/lora/regex")
    async def _bfs_lora_regex(request):
        body = await request.json()
        return web.json_response({
            "regex": build_regex(body.get("family"), body.get("type"), body.get("blocks"))})
except Exception as _exc:  # noqa: BLE001 - the nodes still work without the panel
    print(f"[BFSNodes] LoRA surgery HTTP routes not registered: {_exc!r}")
