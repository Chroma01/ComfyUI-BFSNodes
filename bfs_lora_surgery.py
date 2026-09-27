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

import hashlib
import json
import os
from datetime import datetime, timezone
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


_NUM_RE = re.compile(r"\d+")


def _template(path: str) -> tuple[str, list[int], list[tuple[int, int]]]:
    """``a.12.attn1.q`` -> ("a.{}.attn{}.q", [12, 1], spans)."""
    nums, spans, out, last = [], [], [], 0
    for m in _NUM_RE.finditer(path):
        out.append(path[last:m.start()])
        out.append("{}")
        nums.append(int(m.group()))
        spans.append((m.start(), m.end()))
        last = m.end()
    out.append(path[last:])
    return "".join(out), nums, spans


def choose_block_axis(paths: list[str]) -> dict[str, int]:
    """For each path template, decide which numeric slot is the block index.

    Picking the last number breaks on names like ``transformer_blocks.12.attn1.to_q``, where
    the last one is the ``1`` of ``attn1``, and on nested layouts such as
    ``down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj``. The block axis is the
    slot that varies most across the whole LoRA, so decide it from the file, not from one key.
    """
    seen: dict[str, list[set]] = {}
    for path in paths:
        tpl, nums, _ = _template(path)
        slots = seen.setdefault(tpl, [set() for _ in nums])
        for i, v in enumerate(nums):
            if i < len(slots):
                slots[i].add(v)
    axis: dict[str, int] = {}
    for tpl, slots in seen.items():
        if not slots:
            continue
        counts = [len(x) for x in slots]
        best = max(range(len(counts)), key=lambda i: (counts[i], -i))
        axis[tpl] = best if counts[best] > 1 else -1
    return axis


def _split_block(path: str, axis: dict[str, int] | None = None) -> tuple[str, int | None, str]:
    """Return (family, block_index, module_type).

    ``transformer_blocks.12.attn.to_q`` -> ("transformer_blocks", 12, "attn.to_q").
    Numbers that are not the block axis stay in the name, so ``attn1`` and ``attn2`` remain
    distinct module types instead of being folded together.
    """
    tpl, nums, spans = _template(path)
    if not nums:
        return path, None, ""
    idx = axis.get(tpl, -1) if axis is not None else (len(nums) - 1)
    if idx < 0 or idx >= len(nums):
        return path.rstrip("._"), None, ""
    start, end = spans[idx]
    family = path[:start].rstrip("._")
    module = path[end:].lstrip("._")
    return family, nums[idx], module


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
    axis = choose_block_axis(list(modules))

    families: dict[str, dict[str, Any]] = {}
    loose: list[dict[str, Any]] = []
    all_modules: list[dict[str, Any]] = []
    for base, entry in modules.items():
        family, block, mtype = _split_block(base, axis)
        all_modules.append({"path": base, "family": family, "block": block, "type": mtype,
                            "rank": entry["rank"], "norm": norms.get(base)})
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

    all_modules.sort(key=lambda m: (m["family"], m["block"] if m["block"] is not None else -1, m["type"]))
    return {
        "file": os.path.basename(lora_path),
        "total_modules": len(modules),
        "families": out_families,
        "loose": sorted(loose, key=lambda x: x["path"])[:64],
        "modules": all_modules,
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
    bases = {b for b in (_strip_suffix(k) for k in sd) if b}
    axis = choose_block_axis(sorted(bases))
    out: dict[str, torch.Tensor] = {}
    dropped = scaled = kept = 0
    seen: dict[str, float] = {}
    for key, tensor in sd.items():
        base = _strip_suffix(key)
        if base is None:
            out[key] = tensor
            continue
        if base not in seen:
            family, block, mtype = _split_block(base, axis)
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

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "report", "rules")
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
        # `rules` comes back out so it can feed BFS LoRA Surgery (save) without copying JSON
        return (new_model, new_clip if clip is not None else clip, report, rules)


def describe_rules(rules: list[dict[str, Any]]) -> str:
    """Short, readable slug for a rule list, used when no filename is given.

    ``gate_up-b8_15-off__all-b16_23-off__attn-x1.15``. A file named after what was done to it
    beats one named v2_final_real.
    """
    parts: list[str] = []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        m = rule.get("match", {})
        try:
            scale = float(rule.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        if m.get("regex"):
            what = "re" + hashlib.sha1(m["regex"].encode()).hexdigest()[:6]
        else:
            core = (m.get("type") or "").replace("*", "").strip(".")
            what = re.sub(r"[^A-Za-z0-9_-]", "", core.split(".")[-1]) or "all"
        blocks = re.sub(r"[^0-9,_-]", "", (m.get("blocks") or "")).replace(",", "_").replace("-", "_")
        tag = what if not blocks else f"{what}-b{blocks}"
        parts.append(f"{tag}-{'off' if scale == 0 else f'x{scale:g}'}")
    slug = "__".join(parts) if parts else "unchanged"
    return re.sub(r"[^A-Za-z0-9._-]", "", slug)


# Linux caps a single filename at 255 bytes (NAME_MAX); Windows caps the whole path near 260
# unless long paths are enabled. Stay well under both, since the folder above us is not ours.
MAX_FILENAME = 160


def fit_filename(stem: str, slug: str, ext: str = ".safetensors", limit: int = MAX_FILENAME) -> str:
    """Build ``stem__slug.ext`` and shrink it to ``limit`` bytes without losing uniqueness.

    When it does not fit, both halves are trimmed and a short hash of the full slug is
    appended, so two different recipes never collapse onto the same name.
    """
    name = f"{stem}__{slug}{ext}"
    if len(name.encode("utf-8")) <= limit:
        return name
    digest = hashlib.sha1(f"{stem}__{slug}".encode()).hexdigest()[:8]
    room = limit - len(ext) - len(digest) - 3          # "__" between halves, "-" before hash
    keep_stem = min(len(stem), max(16, room // 2))
    keep_slug = max(8, room - keep_stem)
    return f"{stem[:keep_stem]}__{slug[:keep_slug]}-{digest}{ext}"


def summarize_rules(rules: list[dict[str, Any]]) -> str:
    """One sentence a human can read months later, without parsing JSON."""
    drops, scales = [], []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        m = rule.get("match", {})
        try:
            scale = float(rule.get("scale", 1.0))
        except (TypeError, ValueError):
            continue
        if m.get("regex"):
            what = f"modules matching /{m['regex']}/"
        else:
            t = m.get("type") or "*"
            what = "all modules" if t in ("*", "") else t
            blocks = m.get("blocks")
            if blocks and blocks.strip().lower() not in ("all", "*"):
                what += f" in blocks {blocks}"
        if scale == 0.0:
            drops.append(what)
        elif scale != 1.0:
            scales.append(f"{what} x{scale:g}")
    bits = []
    if drops:
        bits.append("dropped " + "; ".join(drops))
    if scales:
        bits.append("scaled " + "; ".join(scales))
    return ", ".join(bits) if bits else "no changes"


def _file_digest(path: str) -> str:
    """sha256 of the source, so provenance survives a rename."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(4 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return ""


class BFSLoraSurgerySave:
    """Write the edited LoRA to models/loras as a real file.

    Same rules as BFS LoRA Surgery. Leave ``filename`` empty and the name is built from the
    rules themselves, so the file says what was done to it. The recipe is also stored in the
    safetensors metadata under ``bfs_surgery``, which survives being shared.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "rules": ("STRING", {"default": "[]", "multiline": True,
                                     "tooltip": "Same JSON the surgery panel writes."}),
                "filename": ("STRING", {"default": "", "multiline": False,
                                        "tooltip": "Leave empty to name it after the rules."}),
                "subfolder": ("STRING", {"default": "surgery", "multiline": False}),
                "save_dtype": (["keep", "float16", "bfloat16", "float32"],),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "BFS/lora"
    DESCRIPTION = "Apply the same surgery rules and write the result to models/loras."

    def save(self, lora_name, rules, filename, subfolder, save_dtype, overwrite):
        dest, msg = write_surgery(lora_name, rules, filename, subfolder, save_dtype, overwrite)
        print(f"[BFSNodes] LoRA surgery save: {msg}")
        return {"ui": {"text": [msg]}, "result": (dest,)}


def write_surgery(lora_name, rules, filename="", subfolder="surgery", save_dtype="keep",
                  overwrite=False) -> tuple[str, str]:
    """Apply the rules and write the file. Shared by the save node and the panel's button."""
    src = folder_paths.get_full_path("loras", lora_name)
    if src is None:
        raise ValueError(f"LoRA not found: {lora_name}")
    parsed: list[dict[str, Any]] = []
    if rules and rules.strip():
        loaded = json.loads(rules)
        if isinstance(loaded, dict):
            loaded = loaded.get("rules", [])
        if isinstance(loaded, list):
            parsed = loaded

    sd = comfy.utils.load_torch_file(src, safe_load=True)
    edited, stats = apply_rules(sd, parsed)

    if save_dtype != "keep":
        want = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[save_dtype]
        edited = {k: (v.to(want) if v.is_floating_point() else v) for k, v in edited.items()}

    stem = os.path.splitext(os.path.basename(src))[0]
    typed = re.sub(r"[^A-Za-z0-9._ -]", "", filename).strip()
    if typed:
        if typed.endswith(".safetensors"):
            typed = typed[: -len(".safetensors")]
        name = fit_filename(typed, "", "").rstrip("_") + ".safetensors"
    else:
        name = fit_filename(stem, describe_rules(parsed))
    root = folder_paths.get_folder_paths("loras")[0]
    out_dir = os.path.join(root, subfolder.strip()) if subfolder.strip() else root
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, name)
    if len(dest) > 240:
        print(f"[BFSNodes] warning: the path is {len(dest)} characters, which some systems reject")
    if os.path.exists(dest) and not overwrite:
        base, ext = os.path.splitext(dest)
        n = 1
        while os.path.exists(f"{base}_{n}{ext}"):
            n += 1
        dest = f"{base}_{n}{ext}"

    meta = {}
    try:
        with open(src, "rb") as fh:
            hn = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(hn)).get("__metadata__", {}) or {}
    except Exception:  # noqa: BLE001 - metadata is a nicety, not a requirement
        meta = {}
    meta = {str(k): str(v) for k, v in meta.items()}
    total_modules = sum(1 for k in sd if k.endswith(_UP_SUFFIXES))
    kept_modules = stats["kept"] if stats["kept"] is not None else total_modules
    summary = summarize_rules(parsed)
    meta["bfs_surgery"] = json.dumps({
        "tool": "ComfyUI-BFSNodes / BFS LoRA Surgery",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": os.path.basename(src),
        "source_sha256": _file_digest(src),
        "recipe": describe_rules(parsed),
        "summary": summary,
        "rules": parsed,
        "modules_total": total_modules,
        "modules_kept": kept_modules,
        "modules_dropped": stats["dropped"],
        "modules_scaled": stats["scaled"],
        "save_dtype": save_dtype,
    }, ensure_ascii=False)
    # a line any viewer shows without knowing about this tool
    meta["bfs_surgery_summary"] = (f"{kept_modules}/{total_modules} modules kept: {summary} "
                                   f"(from {os.path.basename(src)})")
    meta["ss_output_name"] = os.path.splitext(name)[0]

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in edited.items()}, dest, metadata=meta)
    total = total_modules
    kept = kept_modules
    msg = f"saved {os.path.basename(dest)} ({kept}/{total} modules, {stats['dropped']} dropped)"
    return dest, msg


NODE_CLASS_MAPPINGS = {"BFSLoraSurgery": BFSLoraSurgery, "BFSLoraSurgerySave": BFSLoraSurgerySave}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSLoraSurgery": "BFS LoRA Surgery",
                              "BFSLoraSurgerySave": "BFS LoRA Surgery (save)"}

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

    @PromptServer.instance.routes.post("/bfs/lora/save")
    async def _bfs_lora_save(request):
        body = await request.json()
        try:
            dest, msg = write_surgery(
                body.get("name", ""), json.dumps(body.get("rules", [])),
                body.get("filename", ""), body.get("subfolder", "surgery"),
                body.get("save_dtype", "keep"), bool(body.get("overwrite", False)))
        except Exception as exc:  # noqa: BLE001 - the panel shows the reason
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response({"path": dest, "message": msg})

    @PromptServer.instance.routes.post("/bfs/lora/regex")
    async def _bfs_lora_regex(request):
        body = await request.json()
        return web.json_response({
            "regex": build_regex(body.get("family"), body.get("type"), body.get("blocks"))})
except Exception as _exc:  # noqa: BLE001 - the nodes still work without the panel
    print(f"[BFSNodes] LoRA surgery HTTP routes not registered: {_exc!r}")
