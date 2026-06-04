# videolyze

**Watch and analyze a video locally.** A Claude Code plugin that gives Claude eyes (frames) and ears
(transcript) on any video — a URL (YouTube, TikTok, X, Vimeo, …) or a local file — and analyzes it
driven by your focus. Everything runs on your machine: no cloud transcription, no API keys.

## Why videolyze

Built to beat cloud-dependent video skills on the weak spot — transcript collection:

| | Typical video skill | **videolyze** |
|---|---|---|
| Whisper fallback | cloud API (key, $, audio leaves machine) | **local faster-whisper** (free, private) |
| Caption languages | often English-only | **any / preferred language** |
| Bot/cookie handling | yt-dlp defaults | **cookieless → cookies escalation** |
| Caption format | vtt | **json3 → vtt** |
| Analysis | a few generic sentences | **freeform-driven + self-generated checklist + shortcut modes** |
| Visual-only mode | — | **`--no-transcript` (frames only)** |
| Local files | sometimes | **yes (local whisper)** |

## How it works

```
1. download video (URL) or read local file
2. ffmpeg extracts frames at a duration-aware rate          [always]
3. transcript (unless --no-transcript):
     caption-first (multi-language, cookie escalation)
     → local faster-whisper when no caption exists
4. prints frame paths (t=MM:SS) + timestamped transcript
5. Claude Reads the frames and analyzes per your focus
```

## Requirements

| Need | Why | Required? |
|------|-----|-----------|
| Python 3.10+ and [uv](https://docs.astral.sh/uv/) | runtime + deps | **Yes** |
| `ffmpeg` / `ffprobe` on PATH | frame extraction + audio for Whisper | **Yes** |
| NVIDIA GPU + CUDA driver | fast local Whisper (falls back to CPU int8) | optional |
| Browser logged into YouTube (Chrome) | cookie escalation when YouTube flags a request | optional |
| [Deno](https://deno.com) or another JS runtime | yt-dlp uses it for some YouTube formats | optional (recommended) |

## Install

```sh
/plugin marketplace add oreiasccp/videolyze
/plugin install videolyze@videolyze
```

First run, let it create its own venv:

```sh
python3 "$CLAUDE_PLUGIN_ROOT/scripts/setup.py"
```

## Usage

```
/videolyze <video-url-or-path> [focus or question]
```

Examples:

```
/videolyze https://youtu.be/<id> summarize the structure
/videolyze https://youtu.be/<id> focus on the transitions and frame composition, Remotion style
/videolyze bug-repro.mov what's going wrong on screen?
/videolyze https://youtu.be/<id> just the scenes, visual only        # skips transcript
/videolyze https://youtu.be/<id> --start 2:15 --end 2:45             # focus a section
```

Optional flags: `--no-transcript` (visual only), `--lang pt`, `--start/--end MM:SS`,
`--resolution 1024` (read on-screen text), `--max-frames N`, `--mode creator|technical|motion`.

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `VIDEOLYZE_COOKIES_BROWSER` | `chrome` | Browser yt-dlp reads cookies from on a bot block. `firefox`, `edge`, `brave`… or empty to disable. |
| `VIDEOLYZE_WHISPER_MODEL` | `large-v3` | faster-whisper model. `turbo`, `medium`, `small`, `tiny`. |
| `VIDEOLYZE_WHISPER_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `VIDEOLYZE_WHISPER_COMPUTE` | `int8_float16` | Quantization. `float16`, `int8`. |
| `VIDEOLYZE_WHISPER_VAD` | `0` | Voice-activity filter. Off by default — the bundled VAD over-filters and can drop all segments. Set `1` to try it. |

## Notes

- **Privacy:** nothing is uploaded. Captions and metadata come from yt-dlp; transcription runs locally.
- **Token cost** is dominated by frames (images). Use `--max-frames` or `--start/--end` on long videos.
- **Account risk:** the cookie escalation reuses your logged-in browser session; for low-volume manual
  use the risk is small, but automated downloads with your main account violate YouTube ToS.
- **JS runtime:** yt-dlp warns if no JS runtime is found; installing Deno avoids missing-format issues.

## License

MIT — see [LICENSE](LICENSE).
