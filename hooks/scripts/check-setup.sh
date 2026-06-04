#!/usr/bin/env bash
# SessionStart hook: one-line status. Silent when ready.
set -euo pipefail
command -v ffmpeg >/dev/null 2>&1 || { echo "/videolyze: needs ffmpeg on PATH."; exit 0; }
command -v yt-dlp >/dev/null 2>&1 || true
if [[ ! -d "${CLAUDE_PLUGIN_ROOT:-.}/.venv" ]]; then
  echo "/videolyze: run \`python3 \$CLAUDE_PLUGIN_ROOT/scripts/setup.py\` once to install deps."
fi
exit 0
