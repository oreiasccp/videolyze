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
