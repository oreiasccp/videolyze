from scripts.download import is_url, build_download_cmd


def test_is_url():
    assert is_url("https://youtu.be/abc")
    assert is_url("http://x.com/v")
    assert not is_url("video.mp4")
    assert not is_url("--start")
    assert not is_url("/home/user/clip.mov")


def test_build_download_cmd_has_format_and_output():
    cmd = build_download_cmd("https://youtu.be/abc", "/tmp/out/video.%(ext)s")
    joined = " ".join(cmd)
    assert cmd[0] == "yt-dlp"
    assert "height<=720" in joined
    assert "--write-info-json" in joined
    assert "/tmp/out/video.%(ext)s" in joined
