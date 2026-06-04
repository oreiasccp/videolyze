from scripts.frames import frame_count_for_duration, build_ffmpeg_cmd


def test_frame_budget_buckets():
    assert frame_count_for_duration(20) == 30
    assert frame_count_for_duration(45) == 40
    assert frame_count_for_duration(120) == 60
    assert frame_count_for_duration(400) == 80
    assert frame_count_for_duration(1200) == 100


def test_frame_budget_respects_max():
    assert frame_count_for_duration(20, max_frames=10) == 10


def test_build_ffmpeg_cmd_has_fps_and_scale():
    cmd = build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", fps=0.5, width=512)
    joined = " ".join(cmd)
    assert "fps=0.5" in joined
    assert "scale=512:-2" in joined
    assert cmd[0] == "ffmpeg"
