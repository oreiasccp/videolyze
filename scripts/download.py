"""Download a video via yt-dlp, or resolve a local file path.

Like the caption fetch, the video download escalates: try cookieless first, and on a
YouTube bot/login block retry once with cookies from the browser.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

COOKIES_BROWSER = os.environ.get("VIDEOLYZE_COOKIES_BROWSER", "chrome").strip()
BOT_MARKERS = ("sign in to confirm", "confirm you're not a bot", "login_required", "http error 403")


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    p = urlparse(source)
    return p.scheme in ("http", "https") and bool(p.netloc)


def build_download_cmd(url: str, output_template: str, cookies_browser: str | None = None) -> list[str]:
    cmd = [
        "yt-dlp", "-N", "8",
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json", "--no-playlist", "--ignore-errors",
        "-o", output_template,
    ]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    cmd += ["--", url]
    return cmd


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        for c in out_dir.glob(f"video*{ext}"):
            return c
    return None


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    return {"video_path": str(p), "info": {"title": p.name, "url": str(p)}, "downloaded": False}


def _run(cmd: list[str]) -> str:
    """Run yt-dlp, stream stderr to ours AND capture it so we can detect bot blocks."""
    proc = subprocess.run(cmd, stdout=sys.stderr, stderr=subprocess.PIPE, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.stderr or ""


def download_url(url: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "video.%(ext)s")

    # 1) cookieless
    stderr = _run(build_download_cmd(url, template))
    video = _pick_video(out_dir)

    # 2) escalate to browser cookies on a bot/login block
    if video is None and COOKIES_BROWSER and any(m in stderr.lower() for m in BOT_MARKERS):
        _run(build_download_cmd(url, template, cookies_browser=COOKIES_BROWSER))
        video = _pick_video(out_dir)

    if video is None:
        raise SystemExit(f"yt-dlp produced no video in {out_dir}")

    info = {"url": url}
    info_path = out_dir / "video.info.json"
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "channel": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception:
            pass
    return {"video_path": str(video), "info": info, "downloaded": True}


def get_media(source: str, out_dir: Path) -> dict:
    return download_url(source, out_dir) if is_url(source) else resolve_local(source)
