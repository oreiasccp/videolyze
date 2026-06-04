# videolyze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `videolyze`, a standalone Claude Code plugin that lets Claude watch and analyze a video (URL or local file) using ffmpeg frame extraction plus a caption-first, local-faster-whisper transcript, with free-form-driven analysis.

**Architecture:** A self-contained plugin repo. Pure-logic modules (`transcript.py`, frame-budget, URL/language helpers) are built TDD-first with real unit tests. yt-dlp / ffmpeg / faster-whisper integration is wrapped behind thin, command-builder functions that are unit-tested for the parts that don't need network/GPU, plus smoke-tested end-to-end. `pipeline.py` is the only orchestrator. Everything runs locally — no cloud, no API keys.

**Tech Stack:** Python 3.10+, uv (own venv), yt-dlp, faster-whisper (CUDA), ffmpeg, defusedxml, pytest. Plugin packaging mirrors `bradautomates/claude-video` conventions.

**Working directory:** `d:/code/claudecode/videolyze` (already git-init'd; spec at `docs/superpowers/specs/2026-06-04-videolyze-design.md`).

**Env var prefix:** `VIDEOLYZE_` (e.g. `VIDEOLYZE_WHISPER_MODEL`, `VIDEOLYZE_COOKIES_BROWSER`).

---

## File Structure

```
videolyze/
├── .claude-plugin/plugin.json          # plugin manifest (name, version, commands, hooks)
├── .claude-plugin/marketplace.json     # marketplace entry for /plugin install
├── commands/videolyze.md               # slash-command entry, delegates to SKILL.md
├── SKILL.md                            # pipeline + freeform/mode analysis prompt
├── hooks/hooks.json                    # SessionStart preflight
├── hooks/scripts/check-setup.sh        # one-line setup status
├── scripts/transcript.py               # parse json3/vtt, dedupe, range, format (pure)
├── scripts/frames.py                   # frame budget (pure) + ffmpeg extraction
├── scripts/download.py                 # is_url/resolve_local (pure) + yt-dlp download
├── scripts/captions.py                 # language/format pick (pure) + yt-dlp caption fetch
├── scripts/whisper_local.py            # faster-whisper transcription
├── scripts/pipeline.py                 # orchestrator → prints report
├── scripts/setup.py                    # uv venv + ffmpeg/yt-dlp checks
├── tests/test_transcript.py
├── tests/test_frames.py
├── tests/test_download.py
├── tests/test_captions.py
├── pyproject.toml                      # deps + pytest config
├── README.md
└── LICENSE                             # MIT
```

Each script = one responsibility. `pipeline.py` imports the others; nothing else cross-imports except the shared `transcript.Segment` dict shape.

**Segment shape (used across modules):** `{"start": float, "end": float, "text": str}`.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `scripts/__init__.py`, `tests/__init__.py`, `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.db
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "videolyze"
version = "0.1.0"
description = "Watch and analyze a video locally: ffmpeg frames + caption-first multi-language transcript with local faster-whisper fallback, freeform-driven analysis."
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Rafael Lopes" }]
requires-python = ">=3.10"
dependencies = [
    "yt-dlp>=2025.11.0",
    "faster-whisper>=1.1.0",
    "defusedxml>=0.7.1",
    "nvidia-cublas-cu12; platform_system != 'Darwin'",
    "nvidia-cudnn-cu12>=9.0.0; platform_system != 'Darwin'",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Create empty package markers**

Create `scripts/__init__.py` and `tests/__init__.py` (both empty files).

- [ ] **Step 4: Create the venv and install**

Run: `cd d:/code/claudecode/videolyze && uv sync`
Expected: resolves and installs yt-dlp, faster-whisper, defusedxml, pytest, CUDA libs.

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `uv run pytest -q`
Expected: "no tests ran" (exit 5) — confirms the runner works.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml scripts/__init__.py tests/__init__.py uv.lock
git commit -m "chore: scaffold videolyze project"
```

---

## Task 2: transcript.py (pure parsing — full TDD)

**Files:**
- Create: `scripts/transcript.py`
- Test: `tests/test_transcript.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transcript.py
from scripts.transcript import parse_json3, parse_vtt, dedupe, filter_range, format_transcript


def test_parse_json3_basic():
    raw = '{"events":[{"tStartMs":1360,"dDurationMs":1680,"segs":[{"utf8":"hello"}]},' \
          '{"tStartMs":3040,"dDurationMs":1000,"segs":[{"utf8":"world"}]}]}'
    segs = parse_json3(raw)
    assert segs == [
        {"start": 1.36, "end": 3.04, "text": "hello"},
        {"start": 3.04, "end": 4.04, "text": "world"},
    ]


def test_parse_json3_skips_empty():
    raw = '{"events":[{"tStartMs":0,"dDurationMs":500,"segs":[{"utf8":"\\n"}]}]}'
    assert parse_json3(raw) == []


def test_parse_vtt_basic():
    raw = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello\n\n00:00:03.000 --> 00:00:05.000\nworld\n"
    segs = parse_vtt(raw)
    assert segs == [
        {"start": 1.0, "end": 3.0, "text": "hello"},
        {"start": 3.0, "end": 5.0, "text": "world"},
    ]


def test_dedupe_collapses_rolling_duplicates():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 3.0, "text": "hello world"},
    ]
    out = dedupe(segs)
    assert out == [{"start": 0.0, "end": 3.0, "text": "hello world"}]


def test_filter_range_overlap():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.0, "end": 4.0, "text": "b"},
        {"start": 4.0, "end": 6.0, "text": "c"},
    ]
    assert [s["text"] for s in filter_range(segs, 2.0, 4.0)] == ["a", "b", "c"]
    assert [s["text"] for s in filter_range(segs, 4.5, None)] == ["c"]


def test_format_transcript_timestamps():
    segs = [{"start": 63.0, "end": 65.0, "text": "hi"}]
    assert format_transcript(segs) == "[01:03] hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcript.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.transcript`.

- [ ] **Step 3: Implement `scripts/transcript.py`**

```python
"""Parse YouTube caption payloads (json3 / vtt) into segments and format them.

Segment shape: {"start": float, "end": float, "text": str}.
"""
from __future__ import annotations

import json
import re

TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_json3(raw: bytes | str) -> list[dict]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = json.loads(raw)
    out: list[dict] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        end = start + ev.get("dDurationMs", 0) / 1000.0
        out.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return out


def parse_vtt(raw: bytes | str) -> list[dict]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    lines = raw.splitlines()
    out: list[dict] = []
    i = 0
    while i < len(lines):
        m = TS_RE.match(lines[i])
        if not m:
            i += 1
            continue
        start = _to_seconds(*m.groups()[:4])
        end = _to_seconds(*m.groups()[4:])
        i += 1
        cue = []
        while i < len(lines) and lines[i].strip():
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue.append(cleaned)
            i += 1
        text = " ".join(cue).strip()
        if text:
            out.append({"start": round(start, 2), "end": round(end, 2), "text": text})
        i += 1
    return dedupe(out)


def dedupe(segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        out.append(dict(seg))
    return out


def filter_range(segments, start_seconds, end_seconds) -> list[dict]:
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [s for s in segments if s["end"] >= lo and s["start"] <= hi]


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        t = int(s["start"])
        lines.append(f"[{t // 60:02d}:{t % 60:02d}] {s['text']}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcript.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/transcript.py tests/test_transcript.py
git commit -m "feat: transcript parsing (json3/vtt, dedupe, range, format)"
```

---

## Task 3: frames.py (budget pure-TDD + ffmpeg extraction)

**Files:**
- Create: `scripts/frames.py`
- Test: `tests/test_frames.py`

- [ ] **Step 1: Write failing tests for the frame budget**

```python
# tests/test_frames.py
from scripts.frames import frame_count_for_duration, build_ffmpeg_cmd


def test_frame_budget_buckets():
    assert frame_count_for_duration(20) == 30
    assert frame_count_for_duration(45) == 40
    assert frame_count_for_duration(120) == 60
    assert frame_count_for_duration(400) == 80
    assert frame_count_for_duration(1200) == 100


def test_frame_budget_respects_max():
    assert frame_count_for_duration(20, max_frames=10) == 10


def test_build_ffmpeg_cmd_has_fps_and_scale():
    cmd = build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", fps=0.5, width=512)
    joined = " ".join(cmd)
    assert "fps=0.5" in joined
    assert "scale=512:-2" in joined
    assert cmd[0] == "ffmpeg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.frames`.

- [ ] **Step 3: Implement `scripts/frames.py`**

```python
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
    return min(MAX_FPS, max(0.1, frame_budget / duration_s))


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
    out = []
    for idx, fp in enumerate(frames):
        t = base + idx / fps
        out.append({"path": str(fp), "t": round(t, 1)})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_frames.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/frames.py tests/test_frames.py
git commit -m "feat: frame budget + ffmpeg extraction"
```

---

## Task 4: download.py (pure helpers TDD + yt-dlp download)

**Files:**
- Create: `scripts/download.py`
- Test: `tests/test_download.py`

- [ ] **Step 1: Write failing tests for pure helpers**

```python
# tests/test_download.py
from scripts.download import is_url, build_download_cmd


def test_is_url():
    assert is_url("https://youtu.be/abc")
    assert is_url("http://x.com/v")
    assert not is_url("video.mp4")
    assert not is_url("--start")
    assert not is_url("/home/user/clip.mov")


def test_build_download_cmd_has_format_and_output():
    cmd = build_download_cmd("https://youtu.be/abc", "/tmp/out/video.%(ext)s")
    joined = " ".join(cmd)
    assert cmd[0] == "yt-dlp"
    assert "height<=720" in joined
    assert "--write-info-json" in joined
    assert "/tmp/out/video.%(ext)s" in joined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_download.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.download`.

- [ ] **Step 3: Implement `scripts/download.py`**

```python
"""Download a video via yt-dlp, or resolve a local file path."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    p = urlparse(source)
    return p.scheme in ("http", "https") and bool(p.netloc)


def build_download_cmd(url: str, output_template: str) -> list[str]:
    return [
        "yt-dlp", "-N", "8",
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json", "--no-playlist", "--ignore-errors",
        "-o", output_template, "--", url,
    ]


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


def download_url(url: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_download_cmd(url, str(out_dir / "video.%(ext)s"))
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_download.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/download.py tests/test_download.py
git commit -m "feat: video download + local file resolve"
```

---

## Task 5: captions.py (pure pick logic TDD + yt-dlp fetch)

**Files:**
- Create: `scripts/captions.py`
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write failing tests for language/format selection**

```python
# tests/test_captions.py
from scripts.captions import pick_track_lang, order_formats


def test_pick_track_lang_prefers_requested():
    store = {"en": ["a"], "pt": ["b"], "pt-BR": ["c"]}
    assert pick_track_lang(store, "pt") == "pt"


def test_pick_track_lang_prefix_match():
    store = {"en-US": ["a"], "pt-BR": ["b"]}
    assert pick_track_lang(store, "pt") == "pt-BR"


def test_pick_track_lang_default_english_then_any():
    assert pick_track_lang({"fr": ["a"], "en": ["b"]}, None) == "en"
    assert pick_track_lang({"fr": ["a"], "de": ["b"]}, None) == "fr"
    assert pick_track_lang({}, None) is None


def test_order_formats_prefers_json3():
    track = [{"ext": "vtt"}, {"ext": "json3"}, {"ext": "srv3"}]
    assert [f["ext"] for f in order_formats(track)] == ["json3", "srv3", "vtt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_captions.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.captions`.

- [ ] **Step 3: Implement `scripts/captions.py`**

```python
"""Fetch native captions via yt-dlp (multi-language), cookieless then cookie escalation."""
from __future__ import annotations

import os

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from scripts import transcript

COOKIES_BROWSER = os.environ.get("VIDEOLYZE_COOKIES_BROWSER", "chrome").strip()
SUB_FORMAT_PREF = ("json3", "srv3", "srv1", "vtt")
BOT_MARKERS = ("sign in to confirm", "confirm you're not a bot", "login_required", "http error 403")


def pick_track_lang(store: dict, lang: str | None) -> str | None:
    if not store:
        return None
    if lang:
        if lang in store:
            return lang
        for k in store:
            if k.split("-")[0] == lang.split("-")[0]:
                return k
    for k in store:
        if k.startswith("en"):
            return k
    return next(iter(store))


def order_formats(track: list[dict]) -> list[dict]:
    def rank(fmt):
        ext = fmt.get("ext", "")
        return SUB_FORMAT_PREF.index(ext) if ext in SUB_FORMAT_PREF else len(SUB_FORMAT_PREF)
    return sorted(track, key=rank)


def _is_bot_block(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in BOT_MARKERS)


def _base_opts(use_cookies: bool) -> dict:
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android_vr", "web_safari"]}},
    }
    if use_cookies and COOKIES_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_BROWSER,)
    return opts


def fetch_captions(url: str, lang: str | None = None) -> dict | None:
    """Return {"segments", "lang", "kind", "used_cookies"} or None if no captions."""
    def run(ydl, used_cookies):
        info = ydl.extract_info(url, download=False)
        for kind in ("subtitles", "automatic_captions"):
            store = info.get(kind) or {}
            track_lang = pick_track_lang(store, lang)
            if not track_lang:
                continue
            for fmt in order_formats(store[track_lang]):
                sub_url = fmt.get("url")
                if not sub_url:
                    continue
                try:
                    raw = ydl.urlopen(sub_url).read()
                except Exception:
                    continue
                if not raw or len(raw.strip()) < 3:
                    continue
                try:
                    segs = (transcript.parse_json3(raw) if fmt.get("ext") == "json3"
                            else transcript.parse_vtt(raw))
                except Exception:
                    continue
                if segs:
                    return {"segments": segs, "lang": track_lang,
                            "kind": "manual" if kind == "subtitles" else "auto",
                            "used_cookies": used_cookies}
        return None

    try:
        with YoutubeDL(_base_opts(False)) as ydl:
            return run(ydl, False)
    except DownloadError as e:
        if not _is_bot_block(e):
            raise
        with YoutubeDL(_base_opts(True)) as ydl:
            return run(ydl, True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_captions.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Smoke test against a real captioned video**

Run:
```bash
uv run python -c "from scripts.captions import fetch_captions; c=fetch_captions('https://www.youtube.com/watch?v=aircAruvnKk','en'); print(c['lang'], c['kind'], len(c['segments']))"
```
Expected: prints a language, `manual` or `auto`, and a segment count > 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/captions.py tests/test_captions.py
git commit -m "feat: multi-language caption fetch with cookie escalation"
```

---

## Task 6: whisper_local.py (faster-whisper)

**Files:**
- Create: `scripts/whisper_local.py`

(No unit test — GPU/model-dependent; smoke-tested.)

- [ ] **Step 1: Implement `scripts/whisper_local.py`**

```python
"""faster-whisper local transcription. Lazy-loads the model once."""
from __future__ import annotations

import os

_MODEL = None

MODEL = os.environ.get("VIDEOLYZE_WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("VIDEOLYZE_WHISPER_DEVICE", "cuda")
COMPUTE = os.environ.get("VIDEOLYZE_WHISPER_COMPUTE", "int8_float16")
# VAD off: bundled silero VAD over-filters and can drop all segments. Set =1 to enable.
VAD = os.environ.get("VIDEOLYZE_WHISPER_VAD", "0") == "1"


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from faster_whisper import WhisperModel
    try:
        _MODEL = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
    except Exception:
        _MODEL = WhisperModel(MODEL, device="cpu", compute_type="int8")
    return _MODEL


def transcribe(audio_path: str, language: str | None = None) -> dict:
    """Return {"segments":[{"start","end","text"}], "lang"}."""
    model = _load()
    seg_iter, info = model.transcribe(audio_path, language=language, vad_filter=VAD, beam_size=5)
    out = []
    for s in seg_iter:
        text = s.text.strip()
        if text:
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
    return {"segments": out, "lang": info.language}


def extract_audio(video_path: str, out_path: str) -> str:
    """Extract mono 16kHz mp3 from a video file via ffmpeg."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "64k", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out_path
```

- [ ] **Step 2: Smoke test on a short caption-less clip (or any audio)**

Run:
```bash
uv run python -c "import tempfile; from pathlib import Path; from scripts import download, whisper_local; d=download.download_url('https://www.youtube.com/watch?v=dQw4w9WgXcQ', Path(tempfile.mkdtemp())); a=whisper_local.extract_audio(d['video_path'], d['video_path']+'.mp3'); print(len(whisper_local.transcribe(a)['segments']), 'segments')"
```
Expected: prints a segment count > 0 (model downloads on first run; use `VIDEOLYZE_WHISPER_MODEL=tiny` to speed up the smoke test).

- [ ] **Step 3: Commit**

```bash
git add scripts/whisper_local.py
git commit -m "feat: local faster-whisper transcription + audio extraction"
```

---

## Task 7: pipeline.py (orchestrator)

**Files:**
- Create: `scripts/pipeline.py`

- [ ] **Step 1: Implement `scripts/pipeline.py`**

```python
"""videolyze orchestrator: media -> frames -> caption-first transcript -> report."""
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


def run(source: str, lang: str | None, start: str | None, end: str | None,
        width: int, max_frames: int, fps: float | None, mode: str | None) -> str:
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

    # caption-first
    source_kind = "none"
    segs: list[dict] = []
    used_cookies = False
    if media["downloaded"]:
        cap = captions.fetch_captions(source, lang)
        if cap:
            segs, source_kind, used_cookies = cap["segments"], f"caption:{cap['kind']}", cap["used_cookies"]
    if not segs:
        audio = whisper_local.extract_audio(video_path, str(work / "audio.mp3"))
        tr = whisper_local.transcribe(audio, language=lang)
        segs, source_kind = tr["segments"], "whisper"

    segs = transcript.filter_range(segs, start_s, end_s)
    return _report(info, source_kind, mode, frame_list, segs, str(work))


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


def _report(info, source_kind, mode, frame_list, segs, workdir) -> str:
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
    lines.append(transcript.format_transcript(segs) if segs else "(none available)")
    lines.append(f"--- WORKDIR --- {workdir}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--mode", default=None)
    a = ap.parse_args()
    print(run(a.source, a.lang, a.start, a.end, a.resolution, a.max_frames, a.fps, a.mode))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: End-to-end smoke (captioned URL)**

Run:
```bash
uv run python -m scripts.pipeline "https://www.youtube.com/watch?v=aircAruvnKk" --max-frames 10
```
Expected: prints the report — FRAMES list with paths, TRANSCRIPT with `transcript_source: caption:...`, WORKDIR path.

- [ ] **Step 3: End-to-end smoke (local file)**

Run: download any short `.mp4` to `d:/tmp/clip.mp4` first, then:
```bash
uv run python -m scripts.pipeline "d:/tmp/clip.mp4" --max-frames 10
```
Expected: report with `transcript_source: whisper`.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline.py
git commit -m "feat: pipeline orchestrator with report output"
```

---

## Task 8: setup.py + check-setup.sh

**Files:**
- Create: `scripts/setup.py`, `hooks/scripts/check-setup.sh`, `hooks/hooks.json`

- [ ] **Step 1: Implement `scripts/setup.py`**

```python
"""Preflight + installer for videolyze. Ensures uv venv + ffmpeg + yt-dlp."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _status() -> dict:
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "yt_dlp": (ROOT / ".venv").exists(),
        "venv": (ROOT / ".venv").exists(),
    }


def check() -> int:
    s = _status()
    missing = [k for k, v in s.items() if not v]
    return 0 if not missing else 2


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
```

- [ ] **Step 2: Implement `hooks/scripts/check-setup.sh`**

```bash
#!/usr/bin/env bash
# SessionStart hook: one-line status. Silent when ready.
set -euo pipefail
command -v ffmpeg >/dev/null 2>&1 || { echo "/videolyze: needs ffmpeg on PATH."; exit 0; }
command -v yt-dlp >/dev/null 2>&1 || true
if [[ ! -d "${CLAUDE_PLUGIN_ROOT:-.}/.venv" ]]; then
  echo "/videolyze: run \`python3 \$CLAUDE_PLUGIN_ROOT/scripts/setup.py\` once to install deps."
fi
exit 0
```

- [ ] **Step 3: Implement `hooks/hooks.json`**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/check-setup.sh", "timeout": 5 }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Verify the check exits 0 when ready**

Run: `uv run python scripts/setup.py --check; echo "exit=$?"`
Expected: `exit=0` (ffmpeg + venv present).

- [ ] **Step 5: Commit**

```bash
git add scripts/setup.py hooks/scripts/check-setup.sh hooks/hooks.json
git commit -m "feat: setup preflight + SessionStart hook"
```

---

## Task 9: Plugin metadata + command

**Files:**
- Create: `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `commands/videolyze.md`

- [ ] **Step 1: Create `.claude-plugin/plugin.json`**

```json
{
  "name": "videolyze",
  "version": "0.1.0",
  "description": "Watch and analyze a video locally: ffmpeg frames + caption-first multi-language transcript with local faster-whisper fallback, freeform-driven analysis.",
  "author": "Rafael Lopes",
  "license": "MIT",
  "commands": ["./commands/videolyze.md"],
  "hooks": "./hooks/hooks.json"
}
```

- [ ] **Step 2: Create `.claude-plugin/marketplace.json`**

```json
{
  "name": "videolyze",
  "owner": "oreiasccp",
  "plugins": [
    {
      "name": "videolyze",
      "source": "./",
      "description": "Watch and analyze a video locally with frames + caption-first transcript and local Whisper."
    }
  ]
}
```

- [ ] **Step 3: Create `commands/videolyze.md`**

```markdown
---
description: Watch and analyze a video (URL or local path). Extracts frames + caption-first transcript (local Whisper fallback), then analyzes per your focus.
argument-hint: <video-url-or-path> [focus/question]
allowed-tools: [Bash, Read, AskUserQuestion]
---

Invoke the `videolyze` skill (defined in SKILL.md) with the user's arguments: $ARGUMENTS

Follow the skill's pipeline: preflight → run pipeline.py (download/frames/caption-first transcript) → Read each frame → analyze grounded in frames + transcript, driven by the user's focus. If no arguments were provided, ask for a video URL or local path.
```

- [ ] **Step 4: Commit**

```bash
git add .claude-plugin commands/videolyze.md
git commit -m "feat: plugin manifest, marketplace entry, slash command"
```

---

## Task 10: SKILL.md (pipeline + analysis prompt)

**Files:**
- Create: `SKILL.md`

- [ ] **Step 1: Create `SKILL.md`**

````markdown
---
name: videolyze
description: Watch and analyze a video (URL or local path). Extracts frames with ffmpeg, pulls a caption-first multi-language transcript (local faster-whisper fallback), and analyzes the video driven by the user's focus — transitions, composition, hooks, bugs, or a general summary.
argument-hint: "<video-url-or-path> [focus/question]"
allowed-tools: Bash, Read, AskUserQuestion
license: MIT
user-invocable: true
---

# /videolyze — watch and analyze a video locally

You don't have a video input; this skill gives you one. A pipeline downloads the video (or reads a
local file), extracts frames as JPEGs, gets a timestamped transcript (native captions first, local
faster-whisper fallback — no cloud, no API key), and prints frame paths. You `Read` each frame and
combine frames + transcript to analyze the video.

## Step 0 — Preflight (every run, silent on success)

On Windows use `python`; on macOS/Linux use `python3`.

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/setup.py" --check
```

Exit 0 → silent, proceed. Non-zero → run `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/setup.py"` to
create the venv and check ffmpeg/yt-dlp, then continue. Do NOT announce "setup complete".

## Step 1 — Parse input

Separate the video source (URL or local path) from the user's focus/question.
Example: `/videolyze https://youtu.be/abc focus on the transitions` →
source = `https://youtu.be/abc`, focus = `focus on the transitions`.

## Step 2 — Run the pipeline

```bash
uv run --directory "${CLAUDE_PLUGIN_ROOT}" python -m scripts.pipeline "<source>" [flags]
```

Flags: `--lang pt` (preferred caption language), `--start MM:SS` / `--end MM:SS` (focus a section,
denser frames), `--resolution 1024` (read on-screen text), `--max-frames N`, `--mode creator|technical|motion`.

For any video over ~10 minutes, prefer `--start/--end` on the relevant section over a sparse full scan.

## Step 3 — Read every frame

The pipeline prints frame paths with `t=MM:SS`. `Read` all of them in a single message (parallel
tool calls) so you see them together, in order, aligned to the transcript.

## Step 4 — Analyze (focus-driven)

This is the core. **The user's focus directs your analysis.**

- **If the user gave a specific focus** (e.g. "transitions and frame composition, Remotion style"),
  first expand it into a concrete checklist, then analyze the frames + transcript against it. Example
  for that focus:
  - transition type per cut (cut / fade / wipe / morph) with `[MM:SS]`
  - transition timing and duration
  - composition: rule of thirds, visual hierarchy, motion direction
  - Remotion-style patterns: spring, interpolation, apparent easing
- **If `--mode` was passed**, apply that lens:
  - `creator` — hook (first 3s), narrative structure, pacing/retention, CTA, visual aesthetic, verdict
  - `technical` — find the frame where the issue appears, describe on-screen state, likely cause
  - `motion` — transitions, composition, easing/timing (motion design)
- **If no focus and no mode**, give a structured general summary: structure, key moments, on-screen
  vs spoken content, takeaway.

Always cite `[MM:SS]` timestamps. Ground every claim in a frame you saw or a transcript line — never
the title or a guess.

## Step 5 — Clean up

The pipeline prints `--- WORKDIR --- <dir>`. If the user won't ask follow-ups, remove it with
`rm -rf <dir>`. If they might, leave it.

## Transcription (how it works)

1. **Native captions (preferred, free, local).** yt-dlp pulls manual or auto captions in the
   requested language (or best available — not English-only). json3 preferred, vtt fallback. On a
   "confirm you're not a bot" block, it retries with your browser cookies.
2. **Local faster-whisper fallback.** When no caption exists (or for local files), it extracts mono
   16 kHz audio and transcribes on your GPU (`large-v3`, falls back to CPU). Nothing leaves the
   machine. No API key.

## Token efficiency

Frames dominate cost (~50-80k image tokens for 80 frames at 512px). Don't bump `--resolution` unless
you need to read on-screen text. If you already watched a video this session, answer follow-ups from
context — don't re-run.

## Security & permissions

- yt-dlp + ffmpeg + faster-whisper all run **locally**; nothing is uploaded; no API keys.
- Cookies are only read from your browser on a bot block; never written or logged.
- Working files go under the system temp dir; clean up at the end.
- `allowed-tools: Bash, Read, AskUserQuestion`.
- Bundled scripts: `pipeline.py`, `download.py`, `captions.py`, `frames.py`, `whisper_local.py`,
  `transcript.py`, `setup.py`. Review before first use.
````

- [ ] **Step 2: Commit**

```bash
git add SKILL.md
git commit -m "feat: SKILL.md pipeline + focus-driven analysis prompt"
```

---

## Task 11: README + LICENSE

**Files:**
- Create: `README.md`, `LICENSE`

- [ ] **Step 1: Create `LICENSE`** (MIT, copyright 2026 Rafael Lopes — same text as the yt-transcript-mcp LICENSE).

- [ ] **Step 2: Create `README.md`** documenting: what it does, install (`/plugin marketplace add oreiasccp/videolyze` → `/plugin install videolyze@videolyze`), requirements (uv, ffmpeg, NVIDIA GPU optional), usage examples (`/videolyze <url> focus on the transitions`), env vars (`VIDEOLYZE_WHISPER_MODEL`, `VIDEOLYZE_COOKIES_BROWSER`, `VIDEOLYZE_WHISPER_VAD`), and the comparison-vs-claude-video table from the spec.

- [ ] **Step 3: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: README + MIT license"
```

---

## Task 12: Publish + register locally

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all unit tests pass (transcript, frames, download, captions).

- [ ] **Step 2: Create the public GitHub repo and push**

```bash
gh repo create videolyze --public --source=. --remote=origin --push
```

- [ ] **Step 3: Install the plugin locally to verify**

```bash
claude plugin marketplace add oreiasccp/videolyze
claude plugin install videolyze@videolyze
```
Expected: installs at user scope.

- [ ] **Step 4: End-to-end via the skill**

Restart Claude Code, then run `/videolyze https://www.youtube.com/watch?v=aircAruvnKk summarize the structure`.
Expected: frames Read, transcript from captions, a structured answer citing timestamps.

---

## Self-Review

**Spec coverage:** purpose → Tasks 7+10; architecture/file-structure → all tasks; pipeline+matrix →
Task 7; freeform analysis + shortcut modes → Task 10; captions(multi-lang/cookies/json3) → Task 5;
whisper(local/VAD-off) → Task 6; frames(budget) → Task 3; download/local file → Task 4; error
handling → Tasks 4-7 (SystemExit + fallbacks) and Task 10 (guidance); output contract → Task 7
`_report`; security → Task 10; testing → Tasks 2-5 + 12; distribution → Tasks 9, 11, 12;
comparison table → Task 11. No gaps.

**Placeholder scan:** Task 11 describes README/LICENSE content rather than inlining the full text —
acceptable (LICENSE is the standard MIT text already used in yt-transcript-mcp; README is prose). All
code steps contain complete code.

**Type consistency:** Segment shape `{"start","end","text"}` is identical across transcript.py,
captions.py, whisper_local.py, and pipeline.py. `get_media`/`download_url`/`resolve_local` all return
`{"video_path","info","downloaded"}`. `fetch_captions` returns `{"segments","lang","kind","used_cookies"}`.
`extract_frames` returns `[{"path","t"}]`. Consistent.
