"""Assemble the shots a BFS Shot Planner run stored on disk into one MP4, one shot at a time (low RAM).

For a run whose final join did not finish (ComfyUI killed, out of memory...). Every shot is stored by BFS Shot Join as
<run folder>/shot_0000.pt (+ shot_0000.mkv), shot_0001.pt ... already pasted back / uncropped, without its continuity head; this reads
them in order, trims each to its length, cross-fades joins that are not camera cuts (like BFS Shot Join), and streams
the frames to ffmpeg, so memory stays at about one shot whatever the video length.

    python bfs_assemble_shots.py <run folder> out.mp4 [--audio source.mp4] [--crossfade 4] [--crf 17]

--audio: the source video; its sound is cut to the stored shots' time range (shot 1's start to the last shot's end).
Shots that did not run (disabled / filtered) are not stored, so they are not in the output.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run folder with shot_0000.pt ... (ComfyUI output/bfs_shotloop/<run id>; "
                                "temp/bfs_shotloop/<run id> before 1.104)")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--audio", help="source video to take the sound from", default="")
    ap.add_argument("--crossfade", type=int, default=4, help="frames blended at joins that are not cuts (0 = hard)")
    ap.add_argument("--crf", type=int, default=17)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run, "shot_*.pt")))
    if not files:
        print(f"no shot_*.pt in {a.run}", file=sys.stderr)
        return 1
    metas = [torch.load(f, map_location="cpu", weights_only=False)["shot"] for f in files[:1]]
    m0 = metas[0]
    count = int(m0.get("count") or len(files))
    idx = sorted(int(os.path.basename(f)[5:9]) for f in files)
    missing = [i for i in range(count) if i not in idx]
    if missing:
        print(f"warning: shots {[i + 1 for i in missing]} of {count} are not stored; the video skips them", file=sys.stderr)
    fps = float(m0.get("fps") or 24.0)

    proc, prev_tail, first_start, last_end, n_out = None, None, None, None, 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        meta = d["shot"]
        if "frames" in d:                              # raw frames (older runs, Shot storage = raw)
            frames = d["frames"]                       # uint8 [N, H, W, 3]
        else:                                          # the frames are in shot_NNNN.mkv next to the .pt
            n, h, w, _ = d["shape"]
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(a.run, d["video"]), "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            frames = torch.frombuffer(bytearray(raw[:n * h * w * 3]), dtype=torch.uint8).view(n, h, w, 3)
        L = int(meta["length"])
        body = frames[:L]
        if body.shape[0] < L:                          # shorter than planned: hold the last frame
            body = torch.cat([body, body[-1:].expand(L - body.shape[0], -1, -1, -1)], 0)
        if proc is None:
            H, W = int(body.shape[1]), int(body.shape[2])
            cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                   "-r", f"{fps:g}", "-i", "-"]
            if a.audio:
                t0 = float(meta["start"]) / fps
                cmd += ["-ss", f"{t0:.4f}", "-i", a.audio, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
            cmd += ["-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        elif body.shape[1:3] != (H, W):
            body = torch.nn.functional.interpolate(body.permute(0, 3, 1, 2).float(), size=(H, W), mode="bilinear",
                                                   align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        if a.crossfade > 0 and prev_tail is not None and not meta.get("cut_before"):
            n = min(a.crossfade, prev_tail.shape[0], body.shape[0])
            if n > 0:
                w = torch.linspace(0, 1, n + 2)[1:-1].view(-1, 1, 1, 1)
                body = body.clone()
                body[:n] = (prev_tail[:n].float() * (1 - w) + body[:n].float() * w).round().byte()
        proc.stdin.write(body.contiguous().numpy().tobytes())
        prev_tail = frames[L:].clone() if frames.shape[0] > L else None
        first_start = meta["start"] if first_start is None else first_start
        last_end, n_out = meta["end"], n_out + L
        print(f"shot {int(meta['index']) + 1}/{count}: {L} frames", flush=True)
        del d, frames, body
    proc.stdin.close()
    proc.wait()
    print(f"{a.out}: {n_out} frames ({n_out / fps:.2f} s at {fps:g} fps)")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
