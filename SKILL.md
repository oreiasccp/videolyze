---
name: videolyze
description: Watch and analyze a video (URL or local path). Extracts frames with ffmpeg, pulls a caption-first multi-language transcript (local faster-whisper fallback), and analyzes the video driven by the user's focus — transitions, composition, hooks, bugs, scenes, or a general summary. Supports visual-only analysis (no transcript) and local video files.
argument-hint: "<video-url-or-path> [focus/question]"
allowed-tools: Bash, Read, AskUserQuestion
license: MIT
user-invocable: true
---

# /videolyze — watch and analyze a video locally

You don't have a video input; this skill gives you one. A pipeline downloads the video (or reads a
local file), extracts frames as JPEGs, optionally gets a timestamped transcript (native captions
first, local faster-whisper fallback — no cloud, no API key), and prints frame paths. You `Read` each
frame and combine frames + transcript to analyze the video.

## Step 0 — Preflight (every run, silent on success)

On Windows use `python`; on macOS/Linux use `python3`.

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/setup.py" --check
```

Exit 0 → silent, proceed. Non-zero → run `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/setup.py"` to
create the venv and check ffmpeg/yt-dlp, then continue. Do NOT announce "setup complete".

## Step 1 — Parse input

Separate the video source (URL or local path) from the user's focus/question, and decide whether the
user wants the transcript at all.
- `/videolyze https://youtu.be/abc focus on the transitions` → source = URL, focus = "transitions".
- **Visual-only signals** ("just the scenes", "only the visuals", "no audio needed", "frames only",
  "composição visual apenas") → add `--no-transcript` so the run skips caption fetch and whisper.

## Step 2 — Run the pipeline

```bash
uv run --directory "${CLAUDE_PLUGIN_ROOT}" python -m scripts.pipeline "<source>" [flags]
```

Flags:
- `--no-transcript` — visual-only: frames only, no caption fetch, no whisper (faster, cheaper).
- `--lang pt` — preferred caption language.
- `--start MM:SS` / `--end MM:SS` — focus a section (denser frames).
- `--resolution 1024` — bump frame width to read on-screen text.
- `--max-frames N` — cap the number of frames (token budget).
- `--mode creator|technical|motion` — optional starting lens (see Step 4).

For any video over ~10 minutes, prefer `--start/--end` on the relevant section over a sparse full scan.

## Step 3 — Read every frame

The pipeline prints frame paths with `t=MM:SS`. `Read` all of them in a single message (parallel
tool calls) so you see them together, in order, aligned to the transcript (if any).

## Step 4 — Analyze (focus-driven)

This is the core. **The user's focus directs your analysis.**

- **If the user gave a specific focus** (e.g. "transitions and frame composition, Remotion style"),
  first expand it into a concrete checklist, then analyze the frames (and transcript, if present)
  against it. Example for that focus:
  - transition type per cut (cut / fade / wipe / morph) with `[MM:SS]`
  - transition timing and duration
  - composition: rule of thirds, visual hierarchy, motion direction
  - Remotion-style patterns: spring, interpolation, apparent easing
- **If `--mode` was passed**, apply that lens:
  - `creator` — hook (first 3s), narrative structure, pacing/retention, CTA, visual aesthetic, verdict
  - `technical` — find the frame where the issue appears, describe on-screen state, likely cause
  - `motion` — transitions, composition, easing/timing (motion design)
- **If no focus and no mode**, give a structured general summary: structure, key moments, on-screen
  (and spoken, if transcript present) content, takeaway.

Always cite `[MM:SS]` timestamps. Ground every claim in a frame you saw or a transcript line — never
the title or a guess. In visual-only runs (`--no-transcript`), rely on the frames alone and say so.

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
3. **Skipped entirely** when `--no-transcript` is set (visual-only).

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
