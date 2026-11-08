import numpy as np
import pytest
import soundfile as sf

from hyperlapse.audio import analyze_audio
from hyperlapse.models import HyperlapseError

SR = 22050
BPM = 120
BEAT_S = 60 / BPM
OFFSET_S = 0.25


def click_track(path, beats_per_bar, n_bars, gain_for_bar):
    """Accented click track: downbeats get a strong 80 Hz thump, other beats a weak one."""
    n = int((OFFSET_S + n_bars * beats_per_bar * BEAT_S + 1.0) * SR)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(int(0.12 * SR)) / SR
    env = np.exp(-t / 0.03)
    thump = np.sin(2 * np.pi * 80 * t) * env
    tick = np.sin(2 * np.pi * 1500 * t) * env
    for beat in range(n_bars * beats_per_bar):
        bar = beat // beats_per_bar
        accent = 1.0 if beat % beats_per_bar == 0 else 0.3
        burst = (accent * thump + 0.5 * tick) * gain_for_bar(bar)
        i = int((OFFSET_S + beat * BEAT_S) * SR)
        y[i : i + len(burst)] += burst.astype(np.float32)
    sf.write(path, y, SR)
    return path


def loud_middle(bar):
    # 7 quiet bars, 7 loud bars (drop), then a medium tail.
    return 0.15 if bar < 7 else 0.8 if bar < 14 else 0.45


@pytest.fixture(scope="module")
def track_44(tmp_path_factory):
    return click_track(tmp_path_factory.mktemp("a") / "t44.wav", 4, 20, loud_middle)


@pytest.fixture(scope="module")
def track_34(tmp_path_factory):
    return click_track(tmp_path_factory.mktemp("a") / "t34.wav", 3, 21, loud_middle)


def assert_bars_on_grid(analysis, bar_s):
    offset = (analysis.bar_starts - OFFSET_S) / bar_s
    assert np.all(np.abs(offset - np.round(offset)) < 0.05 / bar_s)


def test_4_4_detection(track_44):
    a = analyze_audio(track_44)
    assert a.beats_per_bar == 4
    assert abs(a.n_bars - 20) <= 1
    assert_bars_on_grid(a, 4 * BEAT_S)
    assert len(a.bar_energy) == a.n_bars
    assert 0.0 <= a.bar_energy.min() and a.bar_energy.max() <= 1.0
    assert sections_cover_all_bars(a)

    def cls(bar_time):
        idx = int(np.searchsorted(a.bar_starts, bar_time + 0.01) - 1)
        return next(s.energy_class for s in a.sections if s.start_bar <= idx < s.end_bar)

    assert cls(OFFSET_S + 2 * 4 * BEAT_S) == "low"
    assert cls(OFFSET_S + 10 * 4 * BEAT_S) == "high"
    assert cls(OFFSET_S + 17 * 4 * BEAT_S) == "mid"


def test_drop_at_loud_section_start(track_44):
    a = analyze_audio(track_44)
    drops = [s for s in a.sections if s.is_drop]
    assert drops and all(s.energy_class == "high" for s in drops)
    drop_bar = drops[0].start_bar
    expected = int(np.argmin(np.abs(a.bar_starts - (OFFSET_S + 7 * 4 * BEAT_S))))
    assert abs(drop_bar - expected) <= 1
    assert drops[0].end_bar - drops[0].start_bar >= 2


def test_3_4_detection(track_34):
    a = analyze_audio(track_34)
    assert a.beats_per_bar == 3
    assert abs(a.n_bars - 21) <= 1
    assert_bars_on_grid(a, 3 * BEAT_S)
    assert {s.energy_class for s in a.sections} == {"low", "mid", "high"}


def test_manual_meter_overrides_detection(track_34):
    a = analyze_audio(track_34, beats_per_bar=4)
    assert a.beats_per_bar == 4
    assert a.n_bars <= 20


def test_grid_beats_start_at_first_downbeat(track_44):
    a = analyze_audio(track_44)
    assert a.grid_beats()[0] == pytest.approx(a.bar_starts[0])
    assert a.grid_beats()[a.beats_per_bar] == pytest.approx(a.bar_starts[1])


def test_silence_is_a_clear_error(tmp_path):
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros(SR * 3, dtype=np.float32), SR)
    with pytest.raises(HyperlapseError):
        analyze_audio(p)


def test_fixed_time_analyses_only_the_fragment(track_44):
    whole = analyze_audio(track_44)
    a = analyze_audio(track_44, fixed_time=(8.0, 24.0))
    assert a.offset == 8.0
    assert a.duration == pytest.approx(16.0, abs=0.05)
    assert a.duration < whole.duration
    assert float(a.beat_times.max()) <= a.duration
    assert whole.offset == 0.0


def test_fixed_time_open_end_and_clamped_end(track_44, caplog):
    whole = analyze_audio(track_44)
    tail = analyze_audio(track_44, fixed_time=(8.0, None))
    assert tail.duration == pytest.approx(whole.duration - 8.0, abs=0.05)
    analyze_audio(track_44, fixed_time=(8.0, 500.0))
    assert "ends after the end" in caplog.text


def test_fixed_time_past_the_end_is_an_error(track_44):
    with pytest.raises(HyperlapseError, match="past the end"):
        analyze_audio(track_44, fixed_time=(500.0, 600.0))


def sections_cover_all_bars(a):
    covered = [b for s in a.sections for b in range(s.start_bar, s.end_bar)]
    return covered == list(range(a.n_bars))
