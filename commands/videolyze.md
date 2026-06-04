---
description: Watch and analyze a video (URL or local path). Extracts frames + caption-first transcript (local Whisper fallback), then analyzes per your focus. Add "visual only" to skip transcript.
argument-hint: <video-url-or-path> [focus/question]
allowed-tools: [Bash, Read, AskUserQuestion]
---

Invoke the `videolyze` skill (defined in SKILL.md) with the user's arguments: $ARGUMENTS

Follow the skill's pipeline: preflight → run pipeline.py (download/frames/caption-first transcript) → Read each frame → analyze grounded in frames + transcript, driven by the user's focus. If the user only wants visual analysis (e.g. "just the scenes", "visual only"), pass `--no-transcript`. If no arguments were provided, ask for a video URL or local path.
