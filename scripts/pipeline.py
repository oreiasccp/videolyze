"""videolyze orchestrator: media -> frames -> caption-first transcript -> report.

Transcript is optional: pass no_transcript=True (CLI --no-transcript) for visual-only
analysis (frames only, no caption fetch, no whisper).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from scripts import captions, download, frames, transcript, whisper_local


def _parse_ts(v: str | None) -> float | None:
    if v is None:
        return None
    parts = [float(x) for x in v.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _probe_duration(video_path: str) -> float:
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video_path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def run(source: str, lang: str | None, start: str | None, end: str | None,
        width: int, max_frames: int, fps: float | None, mode: str | None,
        no_transcript: bool = False) -> str:
    work = Path(tempfile.mkdtemp(prefix="videolyze_"))
    media = download.get_media(source, work)
    video_path = media["video_path"]
    info = media["info"]
    duration = info.get("duration") or _probe_duration(video_path)

    start_s, end_s = _parse_ts(start), _parse_ts(end)

    frame_list = frames.extract_frames(
        video_path, str(work / "frames"), duration or 0.0,
        width=width, start=start_s, end=end_s, max_frames=max_frames, fps_override=fps,
    )

    source_kind = "skipped"
    segs: list[dict] = []
    if not no_transcript:
        # caption-first
        if media["downloaded"]:
            cap = captions.fetch_captions(source, lang)
            if cap:
                segs, source_kind = cap["segments"], f"caption:{cap['kind']}"
        if not segs:
            audio = whisper_local.extract_audio(video_path, str(work / "audio.mp3"))
            tr = whisper_local.transcribe(audio, language=lang)
            segs, source_kind = tr["segments"], "whisper"
        segs = transcript.filter_range(segs, start_s, end_s)

    return _report(info, source_kind, mode, frame_list, segs, str(work), no_transcript)


def _report(info, source_kind, mode, frame_list, segs, workdir, no_transcript) -> str:
    lines = ["=== VIDEOLYZE ==="]
    lines.append(f"title: {info.get('title')}")
    lines.append(f"channel: {info.get('channel')}")
    lines.append(f"duration: {info.get('duration')}")
    lines.append(f"transcript_source: {source_kind}")
    lines.append(f"mode: {mode or 'freeform/general'}")
    lines.append(f"--- FRAMES ({len(frame_list)}) ---")
    for f in frame_list:
        t = int(f["t"])
        lines.append(f"{f['path']}  t={t // 60:02d}:{t % 60:02d}")
    lines.append("--- TRANSCRIPT ---")
    if no_transcript:
        lines.append("(skipped — visual-only mode)")
    else:
        lines.append(transcript.format_transcript(segs) if segs else "(none available)")
    lines.append(f"--- WORKDIR --- {workdir}")
    return "\n".join(lines)


def main():
    import sys
    # Force UTF-8 stdout so unicode in transcripts/titles doesn't crash on Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--no-transcript", action="store_true",
                    help="visual-only: skip caption fetch and whisper (frames only)")
    a = ap.parse_args()
    print(run(a.source, a.lang, a.start, a.end, a.resolution, a.max_frames, a.fps, a.mode,
              no_transcript=a.no_transcript))


if __name__ == "__main__":
    main()
