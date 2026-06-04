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
