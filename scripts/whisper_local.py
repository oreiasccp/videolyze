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
