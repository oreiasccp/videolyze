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
