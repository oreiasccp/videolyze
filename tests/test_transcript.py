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
