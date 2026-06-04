"""Frame extraction: duration-aware budget + ffmpeg command + runner."""
from __future__ import annotations

import subprocess
from pathlib import Path

MAX_FPS = 2.0
MAX_FRAMES = 100


def frame_count_for_duration(duration_s: float, max_frames: int = MAX_FRAMES) -> int:
    if duration_s <= 30:
        n = 30
    elif duration_s <= 60:
        n = 40
    elif duration_s <= 180:
        n = 60
    elif duration_s <= 600:
        n = 80
    else:
        n = 100
    return min(n, max_frames)


def fps_for(duration_s: float, frame_budget: int) -> float:
    if duration_s <= 0:
        return MAX_FPS
    # No lower floor: a floor would override the frame budget on long videos and
    # extract far more frames than requested. Cap only at MAX_FPS.
    return min(MAX_FPS, frame_budget / duration_s)


def build_ffmpeg_cmd(input_path: str, output_template: str, fps: float, width: int,
                     start: float | None = None, end: float | None = None) -> list[str]:
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", input_path,
            "-vf", f"fps={fps},scale={width}:-2",
            "-q:v", "3", output_template]
    return cmd


def extract_frames(input_path: str, out_dir: str, duration_s: float, width: int = 512,
                   start: float | None = None, end: float | None = None,
                   max_frames: int = MAX_FRAMES, fps_override: float | None = None) -> list[dict]:
    """Extract frames; return [{"path", "t"}] with absolute timestamps (seconds)."""
    span = duration_s
    base = start or 0.0
    if start is not None or end is not None:
        span = (end if end is not None else duration_s) - base
    budget = frame_count_for_duration(span, max_frames)
    fps = fps_override if fps_override else fps_for(span, budget)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    template = str(Path(out_dir) / "frame_%04d.jpg")
    cmd = build_ffmpeg_cmd(input_path, template, fps, width, start, end)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(Path(out_dir).glob("frame_*.jpg"))
    # Guarantee the budget cap: ffmpeg's fps rounding can yield a few extra frames.
    # Keep absolute timestamps tied to the original extraction index (idx / fps).
    keep_idx = set(range(len(frames)))
    if len(frames) > budget and budget > 0:
        step = len(frames) / budget
        keep_idx = {int(i * step) for i in range(budget)}
        for idx, fp in enumerate(frames):
            if idx not in keep_idx:
                fp.unlink(missing_ok=True)
    out = []
    for idx, fp in enumerate(frames):
        if idx not in keep_idx:
            continue
        t = base + idx / fps
        out.append({"path": str(fp), "t": round(t, 1)})
    return out
