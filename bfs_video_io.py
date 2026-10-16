"""Video files for the BFS nodes, with or without an ffmpeg program.

The ffmpeg program is found on PATH, else the one bundled by imageio-ffmpeg; with neither (e.g. a Windows portable
ComfyUI), MP4s are written and read through PyAV, the library ComfyUI's own Save Video uses, so it is always there.
"""
import json
import math
import os
import re
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

_BIN: dict = {}


def ffmpeg_bin() -> str | None:
    """The ffmpeg program to run, or None."""
    if "ffmpeg" not in _BIN:
        exe = shutil.which("ffmpeg")
        if exe is None:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:  # noqa: BLE001 - optional
                exe = None
        _BIN["ffmpeg"] = exe
    return _BIN["ffmpeg"]


def ffprobe_bin() -> str | None:
    if "ffprobe" not in _BIN:
        _BIN["ffprobe"] = shutil.which("ffprobe")
    return _BIN["ffprobe"]


def _wav(audio: dict) -> str:
    import tempfile
    import wave
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wf = audio["waveform"][0].clamp(-1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(int(wf.shape[0]))
        f.setsampwidth(2)
        f.setframerate(int(audio["sample_rate"]))
        f.writeframes((wf.t().contiguous() * 32767).round().short().numpy().tobytes())
    return path


def _has_audio(audio) -> bool:
    try:
        return audio is not None and audio["waveform"].ndim == 3 and audio["waveform"].shape[-1] > 0
    except Exception:  # noqa: BLE001
        return False


class Mp4Writer:
    """Frames streamed into an MP4 (H.264, yuv420p, even size) with the soundtrack, one chunk at a time.
    metadata: tags like Save Video's (prompt, workflow...), values JSON-encoded when not strings."""

    def __init__(self, path: str, w: int, h: int, fps: float, audio: dict | None = None, crf: int = 17,
                 metadata: dict | None = None, who: str = "BFS"):
        self.path, self.who, self.wav, self.meta, self.proc, self.av = path, who, None, None, None, None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        exe = ffmpeg_bin()
        if exe is None:
            self._open_av(w, h, fps, audio, crf, metadata)
            return
        inputs = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
        maps, outs = ["-map", "0:v"], []
        if _has_audio(audio):
            self.wav = _wav(audio)
            inputs += ["-i", self.wav]
            maps += ["-map", "1:a"]
            outs += ["-c:a", "aac", "-b:a", "192k"]
        if metadata:   # through an ffmetadata file
            import tempfile
            fd, self.meta = tempfile.mkstemp(suffix=".txt")
            esc = lambda v: re.sub(r"([=;#\\\n])", r"\\\1", v)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(";FFMETADATA1\n")
                for k, v in metadata.items():
                    f.write(f"{esc(str(k))}={esc(v if isinstance(v, str) else json.dumps(v))}\n")
            maps += ["-map_metadata", str(inputs.count("-i"))]
            inputs += ["-f", "ffmetadata", "-i", self.meta]
        cmd = [exe, "-v", "error", "-y", *inputs, *maps, *outs,
               "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+use_metadata_tags+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _open_av(self, w, h, fps, audio, crf, metadata):
        import av
        self.av = av.open(self.path, mode="w", options={"movflags": "use_metadata_tags+faststart"})
        for k, v in (metadata or {}).items():
            self.av.metadata[str(k)] = v if isinstance(v, str) else json.dumps(v)
        self.vs = self.av.add_stream("h264", rate=Fraction(round(float(fps) * 1000), 1000))
        self.vs.width, self.vs.height = w + w % 2, h + h % 2
        self.vs.pix_fmt = "yuv420p"
        self.vs.options = {"crf": str(int(crf)), "preset": "medium"}
        self.audio = audio if _has_audio(audio) else None
        if self.audio is not None:
            ch = int(self.audio["waveform"].shape[1])
            self.layout = {1: "mono", 2: "stereo"}.get(ch, "stereo")
            self.as_ = self.av.add_stream("aac", rate=int(self.audio["sample_rate"]), layout=self.layout)

    def write(self, frames: torch.Tensor) -> None:
        data = (frames[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy()
        if self.proc is not None:
            self.proc.stdin.write(data.tobytes())
            return
        import av
        h, w = data.shape[1:3]
        for fr in data:
            if (h % 2) or (w % 2):
                fr = np.pad(fr, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
            for p in self.vs.encode(av.VideoFrame.from_ndarray(fr, format="rgb24").reformat(format="yuv420p")):
                self.av.mux(p)

    def close(self) -> str:
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="replace")
            self.proc.wait()
            self._cleanup()
            if self.proc.returncode != 0:
                raise RuntimeError(f"{self.who}: ffmpeg could not write {self.path}: {err.strip()[-400:]}")
            return self.path
        import av
        try:
            for p in self.vs.encode(None):
                self.av.mux(p)
            if self.audio is not None:
                wf = self.audio["waveform"][0, :2].float().clamp(-1, 1).contiguous().numpy()
                fr = av.AudioFrame.from_ndarray(wf, format="fltp", layout=self.layout)
                fr.sample_rate, fr.pts = int(self.audio["sample_rate"]), 0
                for p in self.as_.encode(fr):
                    self.av.mux(p)
                for p in self.as_.encode(None):
                    self.av.mux(p)
        finally:
            self.av.close()
        return self.path

    def abort(self) -> None:
        try:
            if self.proc is not None:
                self.proc.kill()
            elif self.av is not None:
                self.av.close()
        except Exception:  # noqa: BLE001
            pass
        self._cleanup()

    def _cleanup(self) -> None:
        for f in (self.wav, self.meta):
            if f and os.path.exists(f):
                os.remove(f)


def read_video_scaled(path: str, budget: float) -> tuple[torch.Tensor, float]:
    """A video file as IMAGE, scaled down (never up) so it takes at most `budget` bytes; streamed into one tensor.
    Returns (frames, scale)."""
    exe, probe = ffmpeg_bin(), ffprobe_bin()
    if exe is None or probe is None:
        return _read_video_av(path, budget)
    info = json.loads(subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                      "stream=width,height,nb_read_packets", "-of", "json", path],
                                     capture_output=True, check=True).stdout)["streams"][0]
    w, h, n = int(info["width"]), int(info["height"]), int(info["nb_read_packets"])
    s = min(1.0, math.sqrt(max(budget, 1.0) / max(1, n * h * w * 3 * 4)))
    W, H = max(16, int(w * s) // 2 * 2), max(16, int(h * s) // 2 * 2)
    out = torch.empty((n, H, W, 3), dtype=torch.float32)
    proc = subprocess.Popen([exe, "-v", "error", "-i", path, "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    fb, i = W * H * 3, 0
    while i < n:
        raw = proc.stdout.read(fb * 16)
        if not raw:
            break
        k = len(raw) // fb
        out[i:i + k] = torch.frombuffer(bytearray(raw[:k * fb]), dtype=torch.uint8).view(k, H, W, 3).float() / 255.0
        i += k
    proc.stdout.close()
    proc.wait()
    return out[:max(1, i)], s


def _read_video_av(path: str, budget: float) -> tuple[torch.Tensor, float]:
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        w, h = int(st.codec_context.width), int(st.codec_context.height)
        n = int(st.frames or 0) or int(round(float(st.duration * st.time_base) * float(st.average_rate)) if st.duration
                                       else 0)
        s = min(1.0, math.sqrt(max(budget, 1.0) / max(1, max(n, 1) * h * w * 3 * 4)))
        W, H = max(16, int(w * s) // 2 * 2), max(16, int(h * s) // 2 * 2)
        chunks, buf = [], []
        for fr in c.decode(st):
            buf.append(torch.from_numpy(fr.to_ndarray(format="rgb24")))
            if len(buf) == 16:
                chunks.append(_shrink(torch.stack(buf), H, W))
                buf = []
        if buf:
            chunks.append(_shrink(torch.stack(buf), H, W))
    return (torch.cat(chunks) if chunks else torch.zeros(1, H, W, 3)), s


def _shrink(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x = x.float() / 255.0
    if tuple(x.shape[1:3]) == (h, w):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode="area").permute(0, 2, 3, 1)
