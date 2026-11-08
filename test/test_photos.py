import itertools
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from hyperlapse.cli import main
from hyperlapse.config import TransitionsConfig, parse_config, parse_transitions
from hyperlapse.models import Photo, Segment, Slot
from hyperlapse.motion import prepare_photos, prepare_sources
from hyperlapse.photos import load_photos, plan_photo_shots, shot_counts, split_slot
from hyperlapse.planner import plan
from hyperlapse.render import FFmpegRenderer, run_ffmpeg, video_filter, video_segment_args
from hyperlapse.utils import find_ffmpeg
from test_audio import click_track
from test_planner import make_audio, make_sources

FPS = 30


def make_photos(n=10, w=1920, h=1080) -> list[Photo]:
    return [Photo(Path(f"p{i:02d}.jpg"), w, h) for i in range(n)]


def config(tmp_path, ratio="random", reuse=False, lo=1, hi=3):
    data = {
        "global": {"video_to_photo_ratio": ratio},
        "photos": {"time_shown_min": lo, "time_shown_max": hi},
        "video": {"allow_segment_reusage": reuse},
        "planning": {"seed": 5},
    }
    return parse_config(data, tmp_path)


def run_plan(tmp_path, photos, ratio="random", reuse=False, chronological=False, order=None, sources=None, **kw):
    cfg = config(tmp_path, ratio, reuse, **kw)
    sources = sources or make_sources(6)
    return plan(
        make_audio(n_bars=24), sources, cfg, parse_transitions({}),
        chronological=chronological, photos=photos, media_order=order,
    )  # fmt: skip


def photo_shots(segs):
    return [s for s in segs if s.is_photo]


def test_shot_counts():
    assert shot_counts(90, FPS, 1, 3) == (1, 3)
    assert shot_counts(240, FPS, 1, 3) == (3, 8)
    assert shot_counts(30, FPS, 2, 3) is None  # 1 s slot, photos need 2 s
    assert shot_counts(270, FPS, 2, 2.5) == (4, 4)  # 9 s only splits into 4 x 2.25 s


def test_split_slot_cuts_on_beats_within_limits():
    slot = Slot(0.0, 4.0, 120)
    grid = np.arange(0, 8, 0.5)
    parts = split_slot(slot, 2, grid, FPS, 1.0, 3.0)
    assert [(p.t_start, p.t_end) for p in parts] == [(0.0, 2.0), (2.0, 4.0)]
    assert sum(p.n_frames for p in parts) == 120

    # An off-grid slot falls back to equal parts when the nearest beat would break the limits.
    parts = split_slot(Slot(0.1, 3.2, 93), 3, np.array([0.4, 3.0]), FPS, 1.0, 1.1)
    assert sum(p.n_frames for p in parts) == 93 and len(parts) == 3
    assert all(p.n_frames >= 1 for p in parts)


def test_photos_fill_the_timeline_without_gaps_and_respect_shown_time(tmp_path):
    segs = run_plan(tmp_path, make_photos(), lo=1, hi=2)
    shots = photo_shots(segs)
    assert shots and any(not s.is_photo for s in segs)
    for s in shots:
        assert 1 - 1.5 / FPS <= s.n_frames / FPS <= 2 + 1.5 / FPS
        assert s.speed == 1.0 and s.src_in == s.src_out == 0.0
    for a, b in itertools.pairwise(segs):
        assert a.t_end == pytest.approx(b.t_start)
    assert sum(s.n_frames for s in segs) / FPS == pytest.approx(segs[-1].t_end, abs=1 / FPS)
    assert segs == run_plan(tmp_path, make_photos(), lo=1, hi=2)


def test_long_slot_is_split_into_several_photos(tmp_path):
    segs = run_plan(tmp_path, make_photos(40), ratio="1:9", reuse=True, lo=1, hi=2)
    assert any(b.t_start == a.t_end and a.is_photo and b.is_photo for a, b in itertools.pairwise(segs))
    assert max(s.n_frames for s in photo_shots(segs)) / FPS <= 2 + 1.5 / FPS


def test_fixed_ratio_controls_the_photo_share(tmp_path):
    def share(ratio):
        segs = run_plan(tmp_path, make_photos(60), ratio=ratio, reuse=True)
        return len(photo_shots(segs)) / len(segs)

    assert share("1:9") > share("1:1") > share("9:1")
    assert share("9:1") < 0.35


def test_proportional_follows_available_photos(tmp_path):
    few = len(photo_shots(run_plan(tmp_path, make_photos(2), ratio="proportional")))
    many = len(photo_shots(run_plan(tmp_path, make_photos(60), ratio="proportional", reuse=True)))
    assert many > few


def test_without_reuse_every_photo_is_shown_at_most_once(tmp_path):
    segs = run_plan(tmp_path, make_photos(4), ratio="1:9")
    used = [s.source for s in photo_shots(segs)]
    assert 0 < len(used) <= 4 and len(set(used)) == len(used)
    reused = run_plan(tmp_path, make_photos(4), ratio="1:9", reuse=True)
    assert len(photo_shots(reused)) > 4


def test_no_photos_leaves_the_plan_unchanged(tmp_path):
    cfg = config(tmp_path)
    audio, sources = make_audio(n_bars=24), make_sources(6)
    tr = parse_transitions({})
    assert plan(audio, sources, cfg, tr) == plan(audio, sources, cfg, tr, photos=[])


def test_slots_too_short_for_photos_stay_video(tmp_path):
    segs = run_plan(tmp_path, make_photos(), lo=30, hi=60)
    assert not photo_shots(segs)


def test_chronological_photo_sits_between_the_videos_it_was_taken_between(tmp_path):
    sources = make_sources(6)
    photos = [Photo(Path("mid.jpg"), 1920, 1080)]
    order = [s.path for s in sources[:3]] + [photos[0].path] + [s.path for s in sources[3:]]
    segs = run_plan(tmp_path, photos, ratio="1:3", chronological=True, order=order, sources=sources)
    shots = [i for i, s in enumerate(segs) if s.is_photo]
    assert shots
    index = shots[0]
    before = {s.source for s in segs[:index] if not s.is_photo}
    after = {s.source for s in segs[index:] if not s.is_photo}
    assert not before & after
    assert not before & {s.path for s in sources[3:]}
    assert not after & {s.path for s in sources[:2]}


def test_chronological_photos_keep_their_date_order(tmp_path):
    sources = make_sources(4)
    photos = make_photos(8)
    order = []
    for i, s in enumerate(sources):
        order += [s.path, photos[2 * i].path, photos[2 * i + 1].path]
    segs = run_plan(tmp_path, list(reversed(photos)), ratio="1:3", chronological=True, order=order, sources=sources)
    names = [s.source.name for s in photo_shots(segs)]
    assert names == sorted(names)


def test_plan_photo_shots_rng_is_the_only_randomness():
    slots = [Slot(i * 2.0, i * 2.0 + 2.0, 60) for i in range(20)]
    cfg = parse_config({}, Path(".")).photos
    args = (slots, np.empty(0), make_photos(), cfg, "random", FPS, 10.0, False)
    assert plan_photo_shots(*args, random.Random(1)) == plan_photo_shots(*args, random.Random(1))


# ---- loading, filtering, rendering -------------------------------------------------


def test_load_and_prepare_photos(tmp_path, make_photo):
    land = make_photo(tmp_path / "land.jpg", 640, 360)
    port = make_photo(tmp_path / "port.png", 360, 640)
    small = make_photo(tmp_path / "small.jpg", 200, 100)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    photos = load_photos([land, port, small, broken])
    assert [(p.path.name, p.width, p.height) for p in photos] == [
        ("land.jpg", 640, 360), ("port.png", 360, 640), ("small.jpg", 200, 100),
    ]  # fmt: skip
    assert [p.path.name for p in prepare_photos(photos, (640, 360), "skip")] == ["land.jpg"]
    assert [p.path.name for p in prepare_photos(photos, (640, 360), "crop")] == ["land.jpg"]
    assert [p.path.name for p in prepare_photos(photos, (640, 360), "blur_pad")] == ["land.jpg", "port.png"]


def test_photos_take_part_in_the_orientation_vote(tmp_path, make_video, monkeypatch):
    from hyperlapse import motion
    from hyperlapse.config import MotionConfig

    monkeypatch.setitem(motion.TARGET_SIZES, "landscape", (640, 360))
    monkeypatch.setitem(motion.TARGET_SIZES, "portrait", (360, 640))
    land = make_video(tmp_path / "l.mp4", "moving", w=640, h=360, duration=2)
    port = make_video(tmp_path / "p.mp4", "moving", w=360, h=640, duration=2)
    cfg = MotionConfig(0.4, 1.0, None)
    with pytest.raises(Exception, match="Mixed orientations"):
        prepare_sources([land, port], cfg)
    portrait_photo = [Photo(tmp_path / "x.jpg", 360, 640)]
    assert prepare_sources([land, port], cfg, photos=portrait_photo)[1] == (360, 640)


def test_photo_filter_holds_one_frame_and_keeps_effects():
    photo = Segment(0, 2, Path("p.jpg"), 0.0, 0.0, 1.0, 60, "flash", True)
    vf = video_filter(photo, (1920, 1080), FPS, TransitionsConfig(effects={"flash": True}))
    assert "tpad=stop_mode=clone" in vf and "setpts" not in vf and "eq=brightness" in vf
    assert vf.index("tpad") < vf.index("eq=brightness")


def test_photo_segment_args_loop_the_image(tmp_path):
    photo = Segment(0, 2, Path("p.jpg"), 0.0, 0.0, 1.0, 60, None, True)
    args = video_segment_args("ff", photo, Path("o.mp4"), (1920, 1080), config(tmp_path), TransitionsConfig())
    assert "-ss" not in args and args[args.index("-framerate") + 1] == "30"
    assert args[args.index("-i") + 1] == "p.jpg" and args[args.index("-frames:v") + 1] == "60"


@pytest.mark.parametrize(("photo_size", "pad"), [((1600, 900), False), ((900, 1600), False), ((900, 1600), True)])
def test_photo_segments_render_to_exact_frame_counts(tmp_path, make_photo, make_video, probe, photo_size, pad):
    photo = make_photo(tmp_path / "p.jpg", *photo_size)
    clip = make_video(tmp_path / "c.mp4", "moving", w=640, h=360, duration=4)
    segments = [
        Segment(0.0, 1.0, clip, 0.5, 1.5, 1.0, 30),
        Segment(1.0, 2.5, photo, 0.0, 0.0, 1.0, 45, "zoom_punch", True),
        Segment(2.5, 3.0, photo, 0.0, 0.0, 1.0, 15, "flash", True),
        Segment(3.0, 4.0, clip, 2.0, 3.0, 1.0, 30),
    ]
    tr = TransitionsConfig(effects={"zoom_punch": True, "flash": True})
    out = tmp_path / "out.mp4"
    music = tmp_path / "m.wav"
    click_track(music, 4, 2, lambda bar: 0.5)
    cfg = parse_config({"render": {"workers": 2}, "video": {"remove_audio": False}}, tmp_path)
    FFmpegRenderer(find_ffmpeg(), tr, (640, 360), frozenset({photo}) if pad else frozenset()).render(
        segments, music, out, cfg
    )
    info = probe(out)
    assert (info["width"], info["height"]) == (640, 360)
    assert info["frames"] == 120


def _jpeg_with_exif_orientation(path: Path, orientation: int) -> Path:
    image = np.zeros((360, 640, 3), np.uint8)
    image[:, 320:] = 255  # left half black, right half white
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    tiff = b"MM\x00*\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01" + orientation.to_bytes(2, "big") + b"\x00\x00\x00\x00\x00\x00"
    app1 = b"\xff\xe1" + (len(tiff) + 8).to_bytes(2, "big") + b"Exif\x00\x00" + tiff
    data = bytes(buf)
    path.write_bytes(data[:2] + app1 + data[2:])
    return path


def test_exif_rotated_photo_is_measured_and_rendered_upright(tmp_path):
    photo = _jpeg_with_exif_orientation(tmp_path / "rot.jpg", 6)  # display: rotate 90 degrees clockwise
    (loaded,) = load_photos([photo])
    assert (loaded.width, loaded.height) == (360, 640)

    seg = Segment(0.0, 1.0, photo, 0.0, 0.0, 1.0, 30, None, True)
    out = tmp_path / "seg.mp4"
    cfg = parse_config({}, tmp_path)
    run_ffmpeg(video_segment_args(find_ffmpeg(), seg, out, (360, 640), cfg, TransitionsConfig()))
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (640, 360)
    # Clockwise rotation turns the black left half into the top half.
    assert frame[:200].mean() < 40 and frame[-200:].mean() > 215


def test_end_to_end_with_photos(tmp_path, make_video, make_photo, probe):
    audio_dir, video_dir, photo_dir, out_dir = (tmp_path / n for n in ("audio", "video", "photos", "out"))
    for d in (audio_dir, video_dir, photo_dir):
        d.mkdir()
    click_track(audio_dir / "track.wav", 4, 8, lambda bar: 0.5)
    for i in range(4):
        make_video(video_dir / f"DJI_2025080217{i:02d}05_00{i}_D.mp4", "moving", w=1920, h=1080, duration=8)
    for i in range(5):
        make_photo(photo_dir / f"DJI_2025080217{i:02d}30_01{i}_D.jpg", 1920, 1080, seed=i)
    make_photo(photo_dir / "DJI_20250802170000_0200_D.jpg", 1080, 1920, seed=9)
    config_data = {
        "paths": {
            "video_input": str(video_dir), "photo_input": str(photo_dir),
            "audio_input": str(audio_dir), "output": str(out_dir),
        },
        "global": {"orientation": "landscape", "mismatch": "blur_pad", "video_to_photo_ratio": "1:2"},
        "audio": {"cut_interval": 3, "min_segment_s": 1},
        "video": {"chronologic": True},
        "planning": {"seed": 3},
        "render": {"workers": 2},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config_data), encoding="utf-8")
    (tmp_path / "transitions.yaml").write_text(
        yaml.safe_dump({"probability": 0.7, "min_gap_cuts": 0, "effects": {"flash": True, "shake": True}}), encoding="utf-8"
    )
    assert main(["--config", str(tmp_path / "config.yaml")]) == 0
    info = probe(out_dir / "montage.mp4")
    assert (info["width"], info["height"]) == (1920, 1080)
    assert info["frames"] > 0 and info["audio_s"] is not None
