"""BFS Video Compare: two videos in one player (wipe, overlay or side by side, either one on top) plus a per-pixel
difference map. Made for BFS Shot Join's `guide` and `images` outputs, but any two IMAGE batches work.

The difference is measured here on the frames themselves (not on the browser's compressed preview): per pixel, the mean
absolute RGB difference, drawn black (equal) -> red -> yellow -> white (very different), scaled by `gain`. The player
gets three temporary MP4s (guide, target, difference) and the per-frame difference, so it can jump to the worst frames.
"""
import math
import os
import subprocess
import uuid

import torch
import torch.nn.functional as F

import folder_paths

CHUNK = 16


def _scale(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if tuple(x.shape[1:3]) == (h, w):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False,
                         antialias=True).permute(0, 2, 3, 1).clamp(0, 1)


def heat(v: torch.Tensor) -> torch.Tensor:
    """[..., H, W] in 0..1 -> RGB: black (0), red, yellow, white (1)."""
    return torch.stack([(3 * v).clamp(0, 1), (3 * v - 1).clamp(0, 1), (3 * v - 2).clamp(0, 1)], -1)


def difference(guide: torch.Tensor, target: torch.Tensor, gain: float = 4.0):
    """Both [N, H, W, 3] in 0..1 (same size). Returns (map [N, H, W, 3], mean abs difference per frame in 0..255,
    mean squared error per frame in 0..1)."""
    d = (guide[..., :3].float() - target[..., :3].float()).abs()
    mae = d.mean(dim=(1, 2, 3)) * 255
    mse = (d * d).mean(dim=(1, 2, 3))
    return heat((d.mean(-1) * float(gain)).clamp(0, 1)), mae, mse


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else 10 * math.log10(1.0 / mse)


class _Mp4:
    """Frames piped to ffmpeg as H.264 yuv420p (what every browser plays), optionally with a soundtrack."""

    def __init__(self, path: str, w: int, h: int, fps: float, audio: dict | None = None, crf: int = 14):
        import tempfile
        import wave
        self.wav = None
        inputs = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
        maps, outs = ["-map", "0:v"], []
        if audio and isinstance(audio.get("waveform"), torch.Tensor) and audio["waveform"].numel():
            fd, self.wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            wf = audio["waveform"][0].clamp(-1, 1)
            with wave.open(self.wav, "wb") as f:
                f.setnchannels(int(wf.shape[0]))
                f.setsampwidth(2)
                f.setframerate(int(audio["sample_rate"]))
                f.writeframes((wf.t().contiguous() * 32767).round().short().numpy().tobytes())
            inputs += ["-i", self.wav]
            maps += ["-map", "1:a"]
            outs += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd = ["ffmpeg", "-v", "error", "-y", *inputs, *maps, *outs, "-c:v", "libx264", "-preset", "fast",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-movflags", "+faststart", path]
        self.path = path
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frames: torch.Tensor) -> None:
        self.proc.stdin.write((frames[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy().tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        self.proc.wait()
        if self.wav and os.path.exists(self.wav):
            os.remove(self.wav)
        if self.proc.returncode != 0:
            raise RuntimeError(f"BFS Video Compare: ffmpeg could not write {self.path}: {err.strip()[-400:]}")


class BFSVideoCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": ("IMAGE", {"tooltip": "The original video (e.g. BFS Shot Join's 'guide' output)."}),
                "target": ("IMAGE", {"tooltip": "The result to compare with it (e.g. BFS Shot Join's 'images')."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001, "tooltip":
                                  "Frame rate of both videos (connect BFS Shot Join's 'fps')."}),
                "audio": ("AUDIO", {"tooltip": "Soundtrack for the player (e.g. BFS Shot Join's 'audio')."}),
                "gain": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 64.0, "step": 0.5, "tooltip":
                                   "How strongly differences light up in the map: a mean RGB difference of 1/gain "
                                   "(128/255 at 2) is already white. Raise it to see small changes."}),
                "size": (["guide", "target"], {"default": "guide", "tooltip":
                                               "Which video's size the other is scaled to before comparing. "
                                               "The Join's guide is at comparison_height, so 'guide' is lighter."}),
                "preview_height": ("INT", {"default": 720, "min": 0, "max": 4096, "step": 16, "tooltip":
                                           "Height of the player's videos (0 = the compared size). Only the preview: "
                                           "the difference is measured before it."}),
                "label_guide": ("STRING", {"default": "guide"}),
                "label_target": ("STRING", {"default": "target"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("difference", "report", "mean_difference")
    OUTPUT_TOOLTIPS = ("The difference map, one frame per compared frame (black = equal, white = very different).",
                       "Frame count, sizes, mean difference, PSNR and the most / least different frames.",
                       "Mean absolute RGB difference over all frames, 0-255.")
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Play two videos in one player (wipe, overlay or side by side; choose which is on top) and see "
                   "where they differ, pixel by pixel.")

    def compare(self, guide, target, fps=24.0, audio=None, gain=2.0, size="guide", preview_height=720,
                label_guide="guide", label_target="target"):
        notes = []
        n = min(guide.shape[0], target.shape[0])
        if guide.shape[0] != target.shape[0]:
            notes.append(f"lengths differ (guide {guide.shape[0]}, target {target.shape[0]} frames): "
                         f"compared the first {n}")
        ref = guide if size == "guide" else target
        H, W = int(ref.shape[1]), int(ref.shape[2])
        for name, v in (("guide", guide), ("target", target)):
            if tuple(v.shape[1:3]) != (H, W):
                notes.append(f"{name} scaled from {v.shape[2]}x{v.shape[1]} to {W}x{H}")
        ph = min(H, int(preview_height)) if preview_height and preview_height > 0 else H
        PH, PW = max(16, ph // 2 * 2), max(16, int(round(W * ph / H)) // 2 * 2)

        out_dir = os.path.join(folder_paths.get_temp_directory(), "bfs_compare", uuid.uuid4().hex[:12])
        os.makedirs(out_dir, exist_ok=True)
        names = {"guide": "guide.mp4", "target": "target.mp4", "difference": "difference.mp4"}
        writers = {k: _Mp4(os.path.join(out_dir, f), PW, PH, float(fps), audio if k == "target" else None)
                   for k, f in names.items()}
        maps, mae, mse = [], [], []
        try:
            for a in range(0, n, CHUNK):
                e = min(a + CHUNK, n)   # the longer video stops where the shorter one ends
                g = _scale(guide[a:e, ..., :3].float(), H, W)
                t = _scale(target[a:e, ..., :3].float(), H, W)
                m, e, s = difference(g, t, gain)
                maps.append(m)
                mae.append(e)
                mse.append(s)
                for k, v in (("guide", g), ("target", t), ("difference", m)):
                    writers[k].write(_scale(v, PH, PW))
        finally:
            for w in writers.values():
                w.close()
        diff_map = torch.cat(maps, 0)
        mae, mse = torch.cat(mae), torch.cat(mse)
        mean = float(mae.mean())
        worst, best = int(mae.argmax()), int(mae.argmin())
        report = "\n".join([
            f"{label_guide} vs {label_target}: {n} frames at {fps:g} fps, compared at {W}x{H}",
            f"mean difference {mean:.2f}/255, PSNR {psnr(float(mse.mean())):.2f} dB",
            f"most different: frame {worst} ({worst / fps:.2f} s) {float(mae[worst]):.2f}/255",
            f"least different: frame {best} ({best / fps:.2f} s) {float(mae[best]):.2f}/255",
            *notes])
        print(f"[BFS Video Compare] {report}", flush=True)
        sub = os.path.relpath(out_dir, folder_paths.get_temp_directory())
        files = {k: {"filename": f, "subfolder": sub, "type": "temp"} for k, f in names.items()}
        ui = {"bfs_compare": [{**files, "fps": float(fps), "frames": n, "width": PW, "height": PH,
                               "labels": [label_guide, label_target], "mae": [round(float(x), 2) for x in mae],
                               "report": report}]}
        return {"ui": ui, "result": (diff_map, report, mean)}


NODE_CLASS_MAPPINGS = {"BFSVideoCompare": BFSVideoCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"BFSVideoCompare": "BFS Video Compare"}
