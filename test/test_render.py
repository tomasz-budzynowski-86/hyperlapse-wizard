import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hyperlapse import render as render_mod
from hyperlapse.config import TransitionsConfig, parse_config, parse_transitions
from hyperlapse.models import HyperlapseError, Segment
from hyperlapse.render import (
    FFmpegRenderer,
    audio_segment_args,
    concat_list_text,
    effect_filter,
    mux_args,
    music_fade_filter,
    video_filter,
    video_segment_args,
)

FFMPEG = "ffmpeg-bin"
SIZE = (1920, 1080)


def cfg(tmp_path, **video):
    return parse_config({"video": video, "render": {"workers": 2}}, tmp_path)


def seg(i=0, speed=1.0, transition=None, n=15, src=Path("a.mp4"), src_in=1.0):
    t0 = i * n / 30
    return Segment(t0, t0 + n / 30, src, src_in, src_in + n / 30 * speed, speed, n, transition)


def test_video_args_order_and_filter(tmp_path):
    args = video_segment_args(FFMPEG, seg(n=45, src_in=2.5), Path("out.mp4"), SIZE, cfg(tmp_path), TransitionsConfig())
    assert args[0] == FFMPEG
    assert args.index("-ss") < args.index("-i")
    assert args[args.index("-ss") + 1] == "2.5"
    assert args[args.index("-frames:v") + 1] == "45"
    assert "-an" in args and "libx264" in args and "yuv420p" in args
    assert args[args.index("-crf") + 1] == "20" and args[args.index("-preset") + 1] == "veryfast"
    vf = args[args.index("-vf") + 1]
    assert "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080" in vf
    assert "fps=30" in vf and "setsar=1" in vf and "setpts" not in vf
    assert args[-1] == "out.mp4"
    assert all(isinstance(a, str) for a in args)


def test_speedup_uses_setpts_and_longer_input_window(tmp_path):
    slow = video_segment_args(FFMPEG, seg(n=30), Path("o.mp4"), SIZE, cfg(tmp_path), TransitionsConfig())
    fast = video_segment_args(FFMPEG, seg(n=30, speed=2.0), Path("o.mp4"), SIZE, cfg(tmp_path), TransitionsConfig())
    assert "setpts=PTS/2" in fast[fast.index("-vf") + 1]
    assert float(fast[fast.index("-t") + 1]) > 1.9 * float(slow[slow.index("-t") + 1])
    assert fast[fast.index("-frames:v") + 1] == "30"


def test_nvenc_encoder_args(tmp_path):
    conf = parse_config({"render": {"encoder": "h264_nvenc", "crf": 23}}, tmp_path)
    args = video_segment_args(FFMPEG, seg(), Path("o.mp4"), SIZE, conf, TransitionsConfig())
    assert "h264_nvenc" in args and "libx264" not in args
    assert args[args.index("-cq") + 1] == "23"


@pytest.mark.parametrize(
    ("effect", "needle"),
    [
        ("flash", "eq=brightness='0.8*max(0,1-t/0.12)':eval=frame"),
        ("zoom_punch", "scale=w='1920*(1+(1.15-1)*max(0,1-t/0.2))'"),
        ("dip_to_black", "fade=t=in:st=0:d=0.15"),
        ("shake", "scale=w='1920+24*max(0,1-t/0.2)'"),
        ("rgb_split", "rgbashift=rh=8:bh=-8:enable='between(t,0,0.15)'"),
    ],
)
def test_effect_filters_use_segment_local_time(effect, needle, tmp_path):
    tr = TransitionsConfig(effects={effect: True})
    vf = video_filter(seg(transition=effect, n=90), SIZE, 30, tr)
    assert needle in vf
    assert vf.index("setsar=1") < vf.index(needle)


def test_effect_duration_is_capped_in_filter():
    tr = TransitionsConfig()
    vf = video_filter(seg(transition="dip_to_black", n=6), SIZE, 30, tr)  # 0.2 s segment -> 0.06 s effect
    assert "d=0.06" in vf
    assert "d=0.06" in effect_filter("dip_to_black", tr.params["dip_to_black"], 0.06, 1920, 1080)


def test_audio_segment_args_variants():
    with_src = audio_segment_args(FFMPEG, seg(n=30), Path("o.wav"), 30, has_audio=True)
    assert with_src.index("-ss") < with_src.index("-i") and "-vn" in with_src
    assert "afade=t=in:st=0:d=0.015" in with_src[with_src.index("-af") + 1]
    assert "apad=whole_dur=1" in with_src[with_src.index("-af") + 1]
    assert with_src[with_src.index("-ar") + 1] == "48000" and "pcm_s16le" in with_src
    for silent in (
        audio_segment_args(FFMPEG, seg(n=30), Path("o.wav"), 30, has_audio=False),
        audio_segment_args(FFMPEG, seg(n=30, speed=2.0), Path("o.wav"), 30, has_audio=True),
    ):
        assert any(a.startswith("anullsrc") for a in silent) and "-ss" not in silent


def test_mux_args_music_only_and_mixed():
    plain = mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, None, 0.25)
    assert plain[plain.index("-c:v") + 1] == "copy" and plain[plain.index("-c:a") + 1] == "aac"
    assert plain[plain.index("-t") + 1] == "12.5" and "+faststart" in plain
    assert "-filter_complex" not in plain
    mixed = mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, Path("s.wav"), 0.25)
    graph = mixed[mixed.index("-filter_complex") + 1]
    assert "volume=0.25" in graph and "volume=1.0" in graph and "normalize=0" in graph
    assert "[a]" in mixed


def test_mux_args_music_fades():
    plain = mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, None, 0.25, fade_in=3, fade_out=3)
    assert plain[plain.index("-af") + 1] == "afade=t=in:st=0:d=3,afade=t=out:st=9.5:d=3"
    mixed = mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, Path("s.wav"), 0.25, fade_in=3, fade_out=3)
    graph = mixed[mixed.index("-filter_complex") + 1]
    assert "volume=1.0,afade=t=in:st=0:d=3,afade=t=out:st=9.5:d=3[m]" in graph
    assert "-af" not in mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, None, 0.25)


def test_music_fade_shrinks_when_longer_than_montage():
    assert music_fade_filter(4, 3, 3) == "afade=t=in:st=0:d=2,afade=t=out:st=2:d=2"
    assert music_fade_filter(10, 0, 2) == "afade=t=out:st=8:d=2"
    assert music_fade_filter(10, 0, 0) == ""


def test_blur_pad_filter_wraps_effects_after_overlay():
    tr = TransitionsConfig(effects={"flash": True})
    vf = video_filter(seg(transition="flash", n=90), SIZE, 30, tr, pad=True)
    assert "split[a][b];" in vf and "overlay=(W-w)/2:(H-h)/2,setsar=1" in vf
    assert "force_original_aspect_ratio=decrease[fg]" in vf
    assert vf.index("overlay") < vf.index("eq=brightness")
    assert "overlay" not in video_filter(seg(), SIZE, 30, tr)


def test_mux_args_seeks_into_the_music_for_a_fragment():
    args = mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, None, 0.25, music_offset=92.0)
    i = args.index("m.mp3")
    assert args[i - 3 : i] == ["-ss", "92", "-i"]
    assert "-ss" not in mux_args(FFMPEG, Path("v.txt"), Path("m.mp3"), Path("o.mp4"), 12.5, None, 0.25)


def test_concat_list_text():
    assert concat_list_text([Path("x/seg_000001.mp4"), Path("x/seg_000002.mp4")]) == "file 'seg_000001.mp4'\nfile 'seg_000002.mp4'\n"
    assert concat_list_text([Path("s_1.mp4")], [1.5]) == "file 's_1.mp4'\nduration 1.500000\n"


class FakeRun:
    def __init__(self, audio_sources=()):
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.lists: list[str] = []
        self.audio_sources = set(audio_sources)

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        self.kwargs.append(kwargs)
        if "-f" in args and args[args.index("-f") + 1] == "concat":
            self.lists.append(Path(args[args.index("-i") + 1]).read_text(encoding="utf-8"))
        if args[-1].endswith(".part"):
            Path(args[-1]).write_bytes(b"x")
        stderr = "  Stream #0:1: Audio: aac" if args[-1] in self.audio_sources else "  Stream #0:0: Video: h264"
        return subprocess.CompletedProcess(args, 0, stdout="", stderr=stderr)


def test_render_builds_expected_commands_without_running_ffmpeg(tmp_path, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(render_mod.subprocess, "run", fake)
    segments = [seg(0, src=Path("a.mp4")), seg(1, src=Path("b.mp4"), transition="flash")]
    out = tmp_path / "out" / "montage.mp4"
    FFmpegRenderer(FFMPEG, TransitionsConfig(effects={"flash": True}), SIZE).render(segments, Path("music.mp3"), out, cfg(tmp_path))

    assert out.is_file()
    assert len(fake.calls) == 3  # two clips + concat/mux
    assert all(kw["shell"] is False for kw in fake.kwargs)
    clips = [c for c in fake.calls if "-vf" in c]
    assert sorted(c[c.index("-ss") + 1] for c in clips) == ["1", "1"]
    assert any("eq=brightness" in c[c.index("-vf") + 1] for c in clips)
    assert fake.lists == ["file 'seg_000000.mp4'\nduration 0.500000\nfile 'seg_000001.mp4'\nduration 0.500000\n"]
    mux = fake.calls[-1]
    assert mux[mux.index("-t") + 1] == "1" and not any(a.endswith("source_audio.wav") for a in mux)
    assert not list(out.parent.glob("*.part"))


def test_render_with_source_audio_builds_pcm_and_mix(tmp_path, monkeypatch):
    fake = FakeRun(audio_sources=["a.mp4"])
    monkeypatch.setattr(render_mod.subprocess, "run", fake)
    segments = [seg(0, src=Path("a.mp4")), seg(1, src=Path("b.mp4"))]
    out = tmp_path / "montage.mp4"
    FFmpegRenderer(FFMPEG, TransitionsConfig(), SIZE).render(segments, Path("music.mp3"), out, cfg(tmp_path, remove_audio=False))

    pcm = [c for c in fake.calls if "pcm_s16le" in c]
    assert len(pcm) == 2
    assert sum(any(a.startswith("anullsrc") for a in c) for c in pcm) == 1  # b.mp4 has no audio
    assert any("-filter_complex" in c for c in fake.calls)
    assert len(fake.lists) == 2


def test_failed_ffmpeg_raises_and_cleans_up(tmp_path, monkeypatch):
    def failing(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom: invalid data")

    monkeypatch.setattr(render_mod.subprocess, "run", failing)
    out = tmp_path / "montage.mp4"
    with pytest.raises(HyperlapseError, match="boom"):
        FFmpegRenderer(FFMPEG, TransitionsConfig(), SIZE).render([seg()], Path("m.mp3"), out, cfg(tmp_path))
    assert not out.exists()


def test_empty_plan_is_error(tmp_path):
    with pytest.raises(HyperlapseError):
        FFmpegRenderer(FFMPEG, TransitionsConfig(), SIZE).render([], Path("m.mp3"), tmp_path / "o.mp4", cfg(tmp_path))


@pytest.fixture
def music(tmp_path):
    path = tmp_path / "music.wav"
    t = np.arange(int(22050 * 4)) / 22050
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 22050)
    return path


def test_real_render_all_effects_with_music_only(tmp_path, make_video, probe, music):
    from hyperlapse.utils import find_ffmpeg

    src = make_video(tmp_path / "src.mp4", "moving", duration=6, audio=True)
    effects = ["flash", "zoom_punch", "dip_to_black", "shake", "rgb_split"]
    segments = [
        Segment(i * 0.5, (i + 1) * 0.5, src, 0.5 + i, 1.0 + i, 1.0, 15, e) for i, e in enumerate(effects)
    ]
    tr = parse_transitions({"effects": dict.fromkeys(effects, True)})
    out = tmp_path / "output" / "montage.mp4"
    FFmpegRenderer(find_ffmpeg(), tr, SIZE).render(segments, music, out, cfg(tmp_path))

    info = probe(out)
    assert (info["width"], info["height"]) == SIZE
    assert info["fps"] == pytest.approx(30)
    assert info["frames"] == 75
    assert abs(info["video_s"] - 2.5) <= 1 / 30
    assert abs(info["audio_s"] - info["video_s"]) <= 1 / 30


def test_real_render_many_cuts_do_not_drift(tmp_path, make_video, probe, music):
    from hyperlapse.utils import find_ffmpeg

    src = make_video(tmp_path / "src.mp4", "moving", duration=6)
    n_segments, n = 40, 13
    segments = [
        Segment(i * n / 30, (i + 1) * n / 30, src, (i % 20) * 0.25, (i % 20) * 0.25 + n / 30, 1.0, n)
        for i in range(n_segments)
    ]
    out = tmp_path / "montage.mp4"
    FFmpegRenderer(find_ffmpeg(), TransitionsConfig(), (640, 360)).render(segments, music, out, cfg(tmp_path))

    info = probe(out)
    total = n_segments * n
    assert info["frames"] == total
    # Timestamps, not just frame counts: container durations must not add a gap at every cut.
    assert abs(info["video_time_s"] - total / 30) <= 1 / 30 + 1e-3


def test_real_render_with_source_audio_and_speedup(tmp_path, make_video, probe, music):
    from hyperlapse.utils import find_ffmpeg

    with_audio = make_video(tmp_path / "a.mp4", "moving", duration=6, audio=True)
    silent = make_video(tmp_path / "b.mp4", "moving", duration=6)
    segments = [
        Segment(0.0, 1.0, with_audio, 1.0, 2.0, 1.0, 30),
        Segment(1.0, 2.0, silent, 1.0, 3.0, 2.0, 30),
        Segment(2.0, 2.5, with_audio, 3.0, 3.5, 1.0, 15),
    ]
    out = tmp_path / "montage.mp4"
    conf = parse_config({"video": {"remove_audio": False}, "render": {"workers": 2}}, tmp_path)
    FFmpegRenderer(find_ffmpeg(), TransitionsConfig(), (1280, 720)).render(segments, music, out, conf)

    info = probe(out)
    assert (info["width"], info["height"]) == (1280, 720)
    assert info["frames"] == 75
    assert abs(info["audio_s"] - info["video_s"]) <= 1 / 30
