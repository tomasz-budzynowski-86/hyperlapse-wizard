import numpy as np
import pytest

from hyperlapse import motion
from hyperlapse.config import MotionConfig
from hyperlapse.models import HyperlapseError
from hyperlapse.motion import (
    SCENE_CUT_SCORE,
    analyze_video,
    find_good_intervals,
    mismatched_sources,
    motion_timeline,
    prepare_sources,
    resolve_min_score,
    window_motion,
)

CFG = MotionConfig(step_s=0.4, window_s=3, min_score=None)


def test_window_motion_uses_half_open_window():
    times = np.array([0.4, 0.8, 1.2, 1.6])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert window_motion(times, scores, 0.4, 1.2) == pytest.approx(0.25)
    assert window_motion(times, scores, 0.0, 1.6) == pytest.approx(0.25)
    assert window_motion(times, scores, 5.0, 6.0) == 0.0


def test_resolve_min_score():
    scores = np.linspace(0.0, 1.0, 101)
    assert resolve_min_score(scores, None) == pytest.approx(0.4)
    assert resolve_min_score(np.zeros(10), None) == motion.MIN_SCORE_FLOOR
    assert resolve_min_score(scores, 0.123) == 0.123


def test_good_intervals_never_span_a_scene_cut():
    times = np.arange(1, 31) * 0.4
    scores = np.full(30, 0.05)
    scores[14] = 0.95
    intervals, cuts = find_good_intervals(times, scores, 12.4, 0.4, 3.0, 0.01)
    assert list(cuts) == [pytest.approx(times[14])]
    assert len(intervals) == 2
    for a, b in intervals:
        assert not (a < times[14] and b > times[13])
    assert scores[14] >= SCENE_CUT_SCORE


def test_good_intervals_empty_for_static():
    times = np.arange(1, 21) * 0.4
    intervals, cuts = find_good_intervals(times, np.zeros(20), 8.4, 0.4, 3.0, motion.MIN_SCORE_FLOOR)
    assert intervals == [] and len(cuts) == 0


def test_unopenable_video_raises(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(HyperlapseError):
        motion_timeline(bad, 0.4)
    with pytest.raises(HyperlapseError):
        motion_timeline(tmp_path / "missing.mp4", 0.4)


def test_static_vs_moving(tmp_path, make_video):
    moving = make_video(tmp_path / "moving.mp4", "moving")
    static = make_video(tmp_path / "static.mp4", "static")
    tm = motion_timeline(moving, 0.4)
    ts = motion_timeline(static, 0.4)
    assert tm.duration == pytest.approx(6.0, abs=0.1)
    assert (tm.width, tm.height) == (640, 360)
    assert tm.fps == pytest.approx(30)
    assert np.diff(tm.times).mean() == pytest.approx(0.4, abs=0.05)
    assert ts.scores.max() < 0.002
    assert tm.scores.mean() > 0.02

    cache = tmp_path / "cache"
    assert analyze_video(static, CFG, cache).good_intervals == []
    moving_profile = analyze_video(moving, CFG, cache)
    covered = sum(b - a for a, b in moving_profile.good_intervals)
    assert covered > 3.0


def test_second_run_uses_cache(tmp_path, make_video, monkeypatch):
    video = make_video(tmp_path / "v.mp4", "moving", duration=3)
    calls = []
    real = motion.motion_timeline
    monkeypatch.setattr(motion, "motion_timeline", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = tmp_path / "cache"
    first = analyze_video(video, CFG, cache)
    second = analyze_video(video, CFG, cache)
    assert len(calls) == 1
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.good_intervals == second.good_intervals
    # Changing the sampling step invalidates the cache.
    analyze_video(video, MotionConfig(0.5, 3, None), cache)
    assert len(calls) == 2


def test_scene_cut_detected_in_source(tmp_path, make_video):
    video = make_video(tmp_path / "cut.mp4", "cut", duration=6)
    profile = analyze_video(video, MotionConfig(0.4, 1.0, 0.0), tmp_path / "cache")
    assert len(profile.scene_cuts) >= 1
    assert abs(profile.scene_cuts[0] - 3.0) < 0.5
    for a, b in profile.good_intervals:
        assert not (a < 2.8 and b > 3.2)


@pytest.fixture
def small_targets(monkeypatch):
    monkeypatch.setattr(motion, "TARGET_SIZES", {"landscape": (640, 360), "portrait": (360, 640)})


def test_prepare_sources_filters_and_picks_majority(tmp_path, make_video, small_targets, caplog):
    paths = [
        make_video(tmp_path / "a.mp4", w=640, h=360, duration=2),
        make_video(tmp_path / "b.mp4", w=640, h=360, duration=2),
        make_video(tmp_path / "c.mp4", w=360, h=640, duration=2),
        make_video(tmp_path / "d.mp4", w=320, h=180, duration=2),
    ]
    kept, size = prepare_sources(paths, CFG, mismatch="skip")
    assert size == (640, 360)
    assert [p.path.name for p in kept] == ["a.mp4", "b.mp4"]
    assert "Skipping c.mp4" in caplog.text and "Skipping d.mp4" in caplog.text


def test_prepare_sources_keeps_mismatched_for_crop_and_blur_pad(tmp_path, make_video, small_targets):
    paths = [
        make_video(tmp_path / "a.mp4", w=640, h=360, duration=2),
        make_video(tmp_path / "b.mp4", w=640, h=360, duration=2),
        make_video(tmp_path / "c.mp4", w=720, h=1280, duration=2),
    ]
    for mode in ("crop", "blur_pad"):
        kept, size = prepare_sources(paths, CFG, mismatch=mode)
        assert [p.path.name for p in kept] == ["a.mp4", "b.mp4", "c.mp4"]
        assert [p.name for p in mismatched_sources(kept, size)] == ["c.mp4"]
    assert [p.path.name for p in prepare_sources(paths, CFG)[0]] == ["a.mp4", "b.mp4", "c.mp4"]


def test_prepare_sources_errors(tmp_path, make_video, small_targets):
    land = make_video(tmp_path / "l.mp4", w=640, h=360, duration=2)
    port = make_video(tmp_path / "p.mp4", w=360, h=640, duration=2)
    tiny = make_video(tmp_path / "t.mp4", w=320, h=180, duration=2)
    with pytest.raises(HyperlapseError, match="Mixed orientations"):
        prepare_sources([land, port], CFG)
    assert prepare_sources([land, port], CFG, orientation="portrait")[1] == (360, 640)
    with pytest.raises(HyperlapseError, match="No usable"):
        prepare_sources([tiny], CFG)
