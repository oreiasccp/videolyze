"""Preflight + installer for videolyze. Ensures uv venv + ffmpeg + yt-dlp."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _status() -> dict:
    venv = (ROOT / ".venv").exists()
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "venv": venv,
    }


def check() -> int:
    s = _status()
    return 0 if all(s.values()) else 2


def install() -> None:
    subprocess.run(["uv", "sync"], cwd=str(ROOT), check=True)
    if not shutil.which("ffmpeg"):
        print("videolyze: install ffmpeg manually (https://ffmpeg.org/download.html)", file=sys.stderr)


def main():
    if "--check" in sys.argv:
        raise SystemExit(check())
    if "--json" in sys.argv:
        print(json.dumps(_status()))
        return
    install()


if __name__ == "__main__":
    main()
