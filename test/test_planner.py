import itertools
import random
from pathlib import Path

import numpy as np
import pytest

from hyperlapse.config import AppConfig, TransitionsConfig, parse_config, parse_transitions
from hyperlapse.models import AudioAnalysis, HyperlapseError, MotionProfile, Section, Segment
from hyperlapse.planner import (
    SOURCE_MARGIN_S,
    allocate_sources,
    build_slots,
    cut_period_beats,
    plan,
)
from hyperlapse.transitions import assign_transitions, effect_duration

FPS = 30


def make_config(tmp_path: Path, **audio) -> AppConfig:
    data = {"audio": {"cut_interval": 2, "min_segment_s": 1, **audio}, "planning": {"seed": 7}}
    return parse_config(data, tmp_path)


def make_audio(bpb=4, n_bars=24, bpm=120, offset=0.3, sections=None) -> AudioAnalysis:
    beat_s = 60 / bpm
    beats = offset + np.arange(n_bars * bpb) * beat_s
    sections = sections or [
        Section(0, n_bars // 3, "low"),
        Section(n_bars // 3, 2 * n_bars // 3, "high", True),
        Section(2 * n_bars // 3, n_bars, "mid"),
    ]
    duration = float(beats[-1] + beat_s + 0.4)
    return AudioAnalysis(duration, beats, bpb, beats[::bpb], np.linspace(0, 1, n_bars), sections)


def make_sources(n=6, length=120.0, score=0.05) -> list[MotionProfile]:
    times = np.arange(1, int(length / 0.4)) * 0.4
    return [
        MotionProfile(Path(f"src{i}.mp4"), length, 30.0, 1920, 1080, times, np.full(len(times), score), np.empty(0), [(0.0, length)])
        for i in range(n)
    ]


def test_ladder_table_4_4_and_3_4():
    expected = {
        4: {1: (2, 2, 4), 2: (2, 4, 8), 3: (4, 8, 16)},
        3: {1: (3, 3, 6), 2: (3, 6, 12), 3: (6, 12, 12)},
    }
    for bpb, table in expected.items():
        for interval, (high, mid, low) in table.items():
            got = tuple(cut_period_beats(bpb, c, interval, 0.5, 0, None) for c in ("high", "mid", "low"))
            assert got == (high, mid, low), (bpb, interval)


def test_min_raises_and_max_lowers_the_rung():
    assert cut_period_beats(4, "high", 1, 0.5, 2, None) == 4  # 2 beats = 1 s < 2 s
    assert cut_period_beats(4, "high", 1, 0.5, 5, None) == 16  # 8 beats = 4 s < 5 s
    assert cut_period_beats(4, "low", 3, 0.5, 0, 3) == 4  # 16/8 beats exceed 3 s
    assert cut_period_beats(4, "low", 3, 0.5, 100, None) == 16  # ladder top


@pytest.mark.parametrize("bpb", [3, 4])
@pytest.mark.parametrize("min_s", [0, 1, 3])
def test_slot_properties(tmp_path, bpb, min_s):
    audio = make_audio(bpb=bpb, n_bars=30)
    cfg = make_config(tmp_path, min_segment_s=min_s).audio
    slots = build_slots(audio, cfg, FPS)

    assert slots[0].t_start == 0.0 and slots[-1].t_end == audio.duration
    for a, b in itertools.pairwise(slots):
        assert a.t_end == b.t_start
    total = sum(s.n_frames for s in slots) / FPS
    assert abs(total - audio.duration) <= 1 / FPS
    grid = audio.grid_beats()
    for s in slots[1:]:
        assert np.min(np.abs(grid - s.t_start)) < 1e-9
    if min_s:
        assert all(s.t_end - s.t_start >= min_s - 1e-6 for s in slots)
    assert all(s.n_frames >= 1 for s in slots)


def test_drop_forces_a_cut_at_section_start(tmp_path):
    audio = make_audio(n_bars=24)
    slots = build_slots(audio, make_config(tmp_path, cut_interval=3).audio, FPS)
    drop_time = audio.bar_starts[8]
    assert any(abs(s.t_start - drop_time) < 1e-9 for s in slots)


def test_cut_interval_1_to_3_reduces_segment_count(tmp_path):
    audio = make_audio(n_bars=48)
    counts = [len(build_slots(audio, make_config(tmp_path, cut_interval=i, min_segment_s=0).audio, FPS)) for i in (1, 2, 3)]
    assert counts[0] > counts[1] > counts[2]


def test_min_and_max_segment_change_the_plan(tmp_path):
    audio = make_audio(n_bars=48)
    short = build_slots(audio, make_config(tmp_path, cut_interval=1, min_segment_s=0).audio, FPS)
    long_ = build_slots(audio, make_config(tmp_path, cut_interval=1, min_segment_s=4).audio, FPS)
    assert len(long_) < len(short)
    capped = build_slots(audio, make_config(tmp_path, cut_interval=3, min_segment_s=0, max_segment_s=3).audio, FPS)
    interior = capped[:-1]
    assert all(s.t_end - s.t_start <= 3 + 1e-6 for s in interior)


def test_short_tail_is_merged_into_previous_segment(tmp_path):
    # 3 bars (6 s) with 4/4 period of 8 beats leaves nothing; use 5 beats of remainder instead.
    audio = make_audio(n_bars=3, sections=[Section(0, 3, "low")])
    slots = build_slots(audio, make_config(tmp_path, cut_interval=2, min_segment_s=4).audio, FPS)
    assert all(s.t_end - s.t_start >= 4 - 1e-6 for s in slots)


def segments_for(tmp_path, seed=7, n_sources=6, length=120.0, **audio):
    cfg = make_config(tmp_path, **audio)
    audio_a = make_audio(n_bars=24)
    slots = build_slots(audio_a, cfg.audio, FPS)
    return audio_a, allocate_sources(slots, make_sources(n_sources, length), cfg.motion, FPS, seed)


def test_no_overlapping_source_ranges_and_no_consecutive_sources(tmp_path):
    _, segs = segments_for(tmp_path)
    by_source: dict[Path, list[tuple[float, float]]] = {}
    for s in segs:
        assert s.speed == 1.0
        assert s.src_out - s.src_in == pytest.approx(s.n_frames / FPS)
        by_source.setdefault(s.source, []).append((s.src_in, s.src_out))
    for ranges in by_source.values():
        ranges.sort()
        for (_, out_a), (in_b, _) in itertools.pairwise(ranges):
            assert in_b - out_a >= SOURCE_MARGIN_S - 1e-6
    assert all(a.source != b.source for a, b in itertools.pairwise(segs))


def test_frames_sum_matches_audio(tmp_path):
    audio, segs = segments_for(tmp_path)
    assert abs(sum(s.n_frames for s in segs) / FPS - audio.duration) <= 1 / FPS


def test_deterministic_with_seed_and_varies_between_seeds(tmp_path):
    _, a = segments_for(tmp_path, seed=1)
    _, b = segments_for(tmp_path, seed=1)
    _, c = segments_for(tmp_path, seed=2)
    assert a == b
    assert a != c


def test_capacity_error_reports_missing_seconds(tmp_path):
    cfg = make_config(tmp_path)
    audio = make_audio(n_bars=24)
    slots = build_slots(audio, cfg.audio, FPS)
    with pytest.raises(HyperlapseError, match=r"missing:\s+\d+\.\d+s"):
        allocate_sources(slots, make_sources(1, length=20.0), cfg.motion, FPS, 1)


def test_reusage_fills_the_montage_from_too_little_footage(tmp_path):
    cfg = make_config(tmp_path)
    slots = build_slots(make_audio(n_bars=24), cfg.audio, FPS)
    sources = make_sources(2, length=20.0)
    with pytest.raises(HyperlapseError):
        allocate_sources(slots, sources, cfg.motion, FPS, 1)
    for chronological in (False, True):
        segs = allocate_sources(slots, sources, cfg.motion, FPS, 1, chronological, reuse=True)
        assert [s.n_frames for s in segs] == [s.n_frames for s in slots]
        assert all(s.speed == 1.0 for s in segs)


def test_extension_slows_down_footage_that_is_too_short(tmp_path):
    cfg = make_config(tmp_path)
    slots = build_slots(make_audio(n_bars=8, sections=[Section(0, 8, "mid")]), cfg.audio, FPS)
    longest = max(s.n_frames for s in slots) / FPS
    src = make_sources(1, length=100.0)
    src[0].good_intervals = [(0.0, longest * 0.8), (30.0, 30.0 + longest * 0.8)] * 1
    with pytest.raises(HyperlapseError):
        allocate_sources(slots, src, cfg.motion, FPS, 2)
    segs = allocate_sources(slots, src, cfg.motion, FPS, 2, extend=True, reuse=True)
    slowed = [s for s in segs if s.speed < 1.0]
    assert slowed
    for s in slowed:
        assert s.src_out - s.src_in == pytest.approx(s.n_frames / FPS * s.speed)
        assert 0.5 <= s.speed < 1.0


def test_single_source_relaxes_consecutive_rule(tmp_path, caplog):
    cfg = make_config(tmp_path)
    slots = build_slots(make_audio(n_bars=8, sections=[Section(0, 8, "mid")]), cfg.audio, FPS)
    segs = allocate_sources(slots, make_sources(1, length=200.0), cfg.motion, FPS, 3)
    assert len(segs) > 1 and "Single source" in caplog.text


def test_motion_candidates_prefer_moving_windows(tmp_path):
    cfg = make_config(tmp_path)
    src = make_sources(2, length=200.0, score=0.0)
    # Only the second half of each source has motion.
    for p in src:
        p.scores = np.where(p.times > 100, 0.2, 0.0)
    slots = build_slots(make_audio(n_bars=8, sections=[Section(0, 8, "mid")]), cfg.audio, FPS)
    segs = allocate_sources(slots, src, cfg.motion, FPS, 5)
    assert sum(s.src_in > 95 for s in segs) >= len(segs) * 0.9


def test_full_plan_is_deterministic(tmp_path):
    cfg = make_config(tmp_path)
    tr = parse_transitions({"probability": 0.5, "effects": {"flash": True, "shake": True}})
    audio, sources = make_audio(n_bars=24), make_sources()
    assert plan(audio, sources, cfg, tr) == plan(audio, sources, cfg, tr)


def _segments(n=60):
    return [Segment(i * 2.0, i * 2.0 + 2.0, Path("a.mp4"), 0, 2, 1.0, 60) for i in range(n)]


def test_transitions_all_false_means_hard_cuts():
    out = assign_transitions(_segments(), parse_transitions({}), random.Random(1))
    assert all(s.transition is None for s in out)
    off = parse_transitions({"enabled": False, "effects": {"flash": True}})
    assert all(s.transition is None for s in assign_transitions(_segments(), off, random.Random(1)))


def test_transitions_rules():
    tr = parse_transitions({"probability": 0.9, "min_gap_cuts": 1, "effects": {"flash": True, "shake": True, "rgb_split": True}})
    out = assign_transitions(_segments(), tr, random.Random(4))
    marked = [(i, s.transition) for i, s in enumerate(out) if s.transition]
    assert marked and out[0].transition is None
    assert {e for _, e in marked} <= {"flash", "shake", "rgb_split"}
    assert all(b[1] != a[1] for a, b in itertools.pairwise(marked))
    assert all(b[0] - a[0] >= 2 for a, b in itertools.pairwise(marked))
    assert out == assign_transitions(_segments(), tr, random.Random(4))


def _chronological(tmp_path, n_sources=6, length=120.0, seed=7):
    cfg = make_config(tmp_path)
    audio = make_audio(n_bars=24)
    slots = build_slots(audio, cfg.audio, FPS)
    sources = make_sources(n_sources, length)
    return audio, sources, allocate_sources(slots, sources, cfg.motion, FPS, seed, chronological=True)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_chronological_follows_source_order_and_moves_forward(tmp_path, seed):
    audio, sources, segs = _chronological(tmp_path, seed=seed)
    order = [p.path for p in sources]
    idx = [order.index(s.source) for s in segs]
    assert idx == sorted(idx)
    assert len(set(idx)) > 1
    for a, b in itertools.pairwise(segs):
        if a.source == b.source:
            assert b.src_in - a.src_out >= SOURCE_MARGIN_S - 1e-6
    assert abs(sum(s.n_frames for s in segs) / FPS - audio.duration) <= 1 / FPS


def test_chronological_single_source_is_ascending(tmp_path):
    _, _, segs = _chronological(tmp_path, n_sources=1, length=200.0)
    assert [s.src_in for s in segs] == sorted(s.src_in for s in segs)


def test_chronological_is_deterministic_and_plan_flag_is_used(tmp_path):
    cfg = make_config(tmp_path)
    tr = parse_transitions({})
    audio, sources = make_audio(n_bars=24), make_sources()
    a = plan(audio, sources, cfg, tr, chronological=True)
    assert a == plan(audio, sources, cfg, tr, chronological=True)
    order = [p.path for p in sources]
    idx = [order.index(s.source) for s in a]
    assert idx == sorted(idx)


def test_chronological_spills_to_next_source_when_one_is_full(tmp_path):
    cfg = make_config(tmp_path)
    audio = make_audio(n_bars=24)
    slots = build_slots(audio, cfg.audio, FPS)
    sources = make_sources(2, length=60.0)
    sources[0].good_intervals = [(0.0, 15.0)]  # far too small for its proportional share
    segs = allocate_sources(slots, sources, cfg.motion, FPS, 4, chronological=True)
    assert sum(s.src_out - s.src_in for s in segs if s.source == sources[0].path) <= 15.0


def test_transitions_zero_gap_and_single_effect():
    tr = parse_transitions({"probability": 1.0, "min_gap_cuts": 0, "effects": {"flash": True}})
    out = assign_transitions(_segments(10), tr, random.Random(0))
    assert [s.transition for s in out] == [None] + ["flash"] * 9


def test_effect_duration_capped_to_30_percent_of_segment():
    tr = TransitionsConfig(effects={"flash": True})
    assert effect_duration("flash", tr, 10.0) == pytest.approx(0.12)
    assert effect_duration("flash", tr, 0.2) == pytest.approx(0.06)
