# videolyze — Design Spec

**Date:** 2026-06-04
**Status:** Approved (brainstorming complete)
**Type:** Standalone Claude Code plugin (skill) — independent, no shared code with `yt-transcript-mcp`.

## 1. Purpose

Give Claude the ability to **watch and analyze a video** (YouTube/other URL or a local file) with
a transcription engine superior to existing video skills: multi-language native captions first,
local `faster-whisper` (GPU) fallback — free, private, no cloud API. Analysis is **driven by the
user's free-form focus** (e.g. "focus on the transitions and frame composition"), with optional
preset shortcuts.

Positioned against `bradautomates/claude-video` (`/watch`), whose weak point is transcript
collection: English-only captions, cloud Whisper API (paid, audio leaves the machine), no cookie
escalation, and a 3-sentence analysis prompt. videolyze fixes all four.

## 2. Non-goals

- Not a pure-transcript tool — that is what `yt-transcript-mcp` is for. videolyze **always**
  extracts frames; the visual analysis is its reason to exist.
- No cloud transcription. No API keys. Everything runs locally.
- No reuse of `yt-transcript-mcp` files. All code is written fresh and self-contained.

## 3. Architecture

Standalone plugin repo, conventions mirroring claude-video:

```
videolyze/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── commands/
│   └── videolyze.md            # slash command entry (/videolyze)
├── SKILL.md                    # multi-mode + freeform analysis prompt + pipeline instructions
├── hooks/
│   ├── hooks.json              # SessionStart preflight (lightweight status)
│   └── scripts/check-setup.sh
├── scripts/
│   ├── setup.py                # create uv venv, install deps, check ffmpeg/yt-dlp
│   ├── pipeline.py             # orchestrator: glue; prints the analysis report
│   ├── download.py             # yt-dlp: download video / resolve local file
│   ├── captions.py             # multi-language caption fetch, json3, cookie escalation
│   ├── frames.py               # ffmpeg frame extraction (duration-aware auto-fps)
│   ├── whisper_local.py        # faster-whisper local transcription
│   └── transcript.py           # parse json3/vtt, dedupe, range-filter, format
├── pyproject.toml              # own deps: yt-dlp, faster-whisper, defusedxml, CUDA libs
├── README.md
└── LICENSE                     # MIT
```

Each script has a single purpose and is testable in isolation. `pipeline.py` is the only glue.

## 4. Pipeline (pipeline.py)

Inputs: `source` (URL or path), `question`/focus (free text), `--mode` (optional), `--start`/`--end`,
`--resolution`, `--max-frames`, `--fps`.

```
0. PREFLIGHT  → setup.py --check ensures venv + ffmpeg + yt-dlp
1. RESOLVE    → URL vs local file
2. MEDIA      → URL: download.py fetches video (≤720p mp4) | local: use file in place
3. FRAMES     → frames.py: ffmpeg extracts at duration-aware fps; filter to --start/--end   [ALWAYS]
4. TRANSCRIPT (caption-first):
   ├─ URL: captions.py fetches native captions (preferred lang, cookieless→cookies)
   │        └─ found? use it, SKIP whisper
   └─ no caption / local file → extract audio (ffmpeg) → whisper_local.py
5. OUTPUT     → print frame paths (t=MM:SS) + timestamped transcript + metadata + source + mode/focus
6. Claude     → Read frames + apply analysis prompt (mode/focus) → answer, citing timestamps
```

### Path matrix

| Scenario | Download video | Frames | Transcription |
|---|---|---|---|
| URL + caption available | yes | yes | caption (no whisper) |
| URL, no caption | yes | yes | whisper local |
| Local file | already present | yes | whisper local |

**Frames always. Whisper only when no caption exists.** Caption-first is a transcription
optimization (faster/better than whispering when captions exist), not a video-skip optimization.

## 5. Analysis: freeform-driven, with optional shortcuts

The **user's instruction is the analysis director.** SKILL.md tells Claude:

1. Read the user's focus ("transitions, frame composition, remotion style").
2. If specific, **expand it into a self-generated specialized checklist**, e.g. for the above:
   - transition type per cut (cut/fade/wipe/morph) + timestamp
   - transition timing/duration
   - composition: rule of thirds, visual hierarchy, motion
   - Remotion-style patterns (spring, interpolation, apparent easing)
3. Analyze frames + transcript against the checklist.
4. Answer citing `[MM:SS]` for each finding.

If the focus is generic / absent → structured general summary (structure, key moments, on-screen
vs spoken, takeaway), always with timestamps.

### Optional shortcut modes (`--mode`)
Convenience starting lenses, not required:
- `creator` — hook (first 3s), narrative structure, pacing/retention, CTA, visual aesthetic, verdict.
- `technical` — screen-recording/bug: locate the frame where the issue appears, on-screen state, likely cause.
- `motion` — transitions, composition, easing/timing (Remotion-style motion design).

Auto-detect hints from question keywords; default to freeform/general.

## 6. Components

### captions.py
- yt-dlp Python API → `subtitles` + `automatic_captions`.
- Prefer manual over automatic; language: requested → prefix match → English-ish → any (NOT
  English-hardcoded).
- Format preference: json3 → srv3 → vtt.
- Cookieless first; on bot/login block, retry with `cookiesfrombrowser`. Browser via env
  (`VIDEOLYZE_COOKIES_BROWSER`, default chrome).
- Defensive against empty/zero-byte tracks (yt-dlp #13443): iterate formats/kinds.

### whisper_local.py
- `faster-whisper`, default `large-v3`, `int8_float16`, `cuda`; fallback CPU `int8` if CUDA unavailable.
- VAD **off** by default (bundled silero VAD over-filters → drops all segments in current
  faster-whisper); plain decode `beam_size=5`. Env toggle `VIDEOLYZE_WHISPER_VAD=1`.
- Lazy-load the model (only when whisper path is hit).
- Env overrides: `VIDEOLYZE_WHISPER_MODEL`, `_DEVICE`, `_COMPUTE`.

### frames.py — ffmpeg, duration-aware budget
| Duration | Frames |
|---|---|
| ≤30s | ~30 |
| 30–60s | ~40 |
| 1–3min | ~60 |
| 3–10min | ~80 |
| >10min | 100 (sparse, warning printed) |

Hard caps: 2 fps, 100 frames. JPEG 512px default (`--resolution 1024` to read on-screen text).
Focused mode via `--start/--end` uses denser per-second budgets. Frame timestamps are absolute.

### download.py
- URL → yt-dlp downloads `bv*[height<=720]+ba/...` merged to mp4 + info.json (title/uploader/duration).
- Local file → resolve/validate path and extension; no download.

### transcript.py
- Parse json3 and vtt → `{start, end, text}` segments.
- Dedupe rolling-duplicate cues (YouTube auto-subs).
- `filter_range(start, end)`; `format_transcript` → `[MM:SS] text`.

### Dependencies / environment
Own `pyproject.toml`: `yt-dlp`, `faster-whisper`, `defusedxml`, `nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`. `setup.py` runs `uv sync` into the skill's own `.venv` on first run; all
scripts are invoked through that venv's interpreter. ~1–2 GB (CUDA libs), accepted for full
self-containment.

## 7. Error handling

| Failure | Action |
|---|---|
| ffmpeg/yt-dlp missing | setup.py installs or prints exact command |
| Download fails (login/region-lock) | report clearly; do not retry-loop |
| Empty caption track (#13443) | try next lang/format → else whisper |
| Bot/login block | retry with cookies-from-browser |
| Whisper fails (no GPU) | CPU int8 fallback; if that fails, frames-only + warn |
| Video >10min | print sparse-scan warning, suggest `--start/--end` |
| No caption AND no whisper output | frames-only, tell the user |

## 8. Output contract (pipeline.py → stdout for Claude)

```
=== VIDEOLYZE ===
title / channel / duration / source(caption|whisper) / mode / focus
--- FRAMES (N) ---
<path>/frame_0001.jpg  t=00:03
... (Claude Reads each)
--- TRANSCRIPT ---
[00:03] text...
--- WORKDIR --- <tmp dir>   (Claude removes when done)
```

## 9. Security & permissions

- yt-dlp + ffmpeg run locally; whisper runs locally — **nothing leaves the machine**.
- Cookies: only reads the browser session on a bot block; never written or logged.
- No API key, no cloud, no upload (key advantage over claude-video).
- Workdir under system temp; cleaned (`rm -rf`) at end.
- `allowed-tools: Bash, Read, AskUserQuestion`.
- Bundled scripts documented in SKILL.md; reviewable before first use.

## 10. Testing

- `transcript.py`: json3/vtt parse + dedupe (unit, no network).
- `frames.py`: duration→budget mapping (unit).
- `captions.py` / `whisper_local.py`: smoke test on one captioned video + one caption-less video.
- `pipeline.py`: end-to-end on one URL and one local file.

## 11. Distribution

Standalone public GitHub repo with `.claude-plugin/marketplace.json`. Install:
`/plugin marketplace add oreiasccp/videolyze` then `/plugin install videolyze@videolyze`.
MIT licensed.

## 12. Differences vs claude-video (the wins)

| | claude-video | videolyze |
|---|---|---|
| Whisper fallback | cloud (Groq/OpenAI, key, $, audio leaves) | local faster-whisper (free, private) |
| Caption languages | English-hardcoded | any / preferred-lang |
| Bot/cookie handling | yt-dlp default | cookieless→cookies escalation |
| Caption format | vtt | json3 → vtt |
| Analysis prompt | ~3 sentences | freeform-driven + self-generated checklist + shortcuts |
| Local file | yes | yes (local whisper) |
