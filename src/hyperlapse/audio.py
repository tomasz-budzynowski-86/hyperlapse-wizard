from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .models import AudioAnalysis, HyperlapseError, Section

logger = logging.getLogger(__name__)

_SR = 22050
_HOP = 512
_N_FFT = 2048
_TIGHTNESS = 100
_LOW_BAND_HZ = 150.0
_BEAT_WINDOW_FRAMES = 4
_NORM_WINDOW_BEATS = 12  # lcm(3, 4): phase-neutral for both candidate meters
_METER_RATIO = 1.2  # 3/4 must beat 4/4 by this factor ...
_METER_MARGIN = 1.0  # ... and by this absolute z-score margin
_DROP_JUMP = 0.25
_HYSTERESIS = 0.05
_DROP_HOLD_BARS = 2
_BAR_LEAD_S = 0.06  # tracked beats lag the real attack by a few tens of ms
_MIN_BEATS = 4


def _beat_low_energy(y: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    spec = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    freqs = librosa.fft_frequencies(sr=_SR, n_fft=_N_FFT)
    low = spec[freqs <= _LOW_BAND_HZ].sum(axis=0)
    out = np.empty(len(beat_frames))
    for i, f in enumerate(beat_frames):
        out[i] = low[f : f + _BEAT_WINDOW_FRAMES].mean() if f < len(low) else 0.0
    return out


def _phase_scores(rel: np.ndarray, meter: int) -> tuple[float, int]:
    """Return (z-score of the strongest phase, that phase) for a candidate meter."""
    std = float(rel.std())
    best_phase = int(np.argmax([rel[p::meter].mean() for p in range(meter)]))
    if std < 1e-12 or len(rel) < 2 * meter:
        return 0.0, best_phase
    group = rel[best_phase::meter]
    z = (float(group.mean()) - float(rel.mean())) / (std / np.sqrt(len(group)))
    return z, best_phase


def _detect_meter(beat_energy: np.ndarray, forced: int | None) -> tuple[int, int]:
    """Return (beats_per_bar, downbeat_phase)."""
    baseline = uniform_filter1d(beat_energy, size=_NORM_WINDOW_BEATS, mode="nearest") + 1e-12
    rel = beat_energy / baseline
    if forced is not None:
        return forced, _phase_scores(rel, forced)[1]
    z3, phase3 = _phase_scores(rel, 3)
    z4, phase4 = _phase_scores(rel, 4)
    logger.debug("Meter z-scores: 3/4=%.2f 4/4=%.2f", z3, z4)
    if z3 > z4 * _METER_RATIO and z3 - z4 > _METER_MARGIN:
        return 3, phase3
    return 4, phase4


def _bar_energy(y: np.ndarray, bar_starts: np.ndarray, duration: float) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=_N_FFT, hop_length=_HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=_SR, hop_length=_HOP)
    edges = np.append(bar_starts - _BAR_LEAD_S, duration)
    energy = np.zeros(len(bar_starts))
    for i in range(len(bar_starts)):
        mask = (times >= edges[i]) & (times < edges[i + 1])
        if mask.any():
            energy[i] = np.sqrt(np.mean(rms[mask] ** 2))
    return energy


def _normalise(energy: np.ndarray) -> np.ndarray:
    smoothed = median_filter(energy, size=3, mode="nearest")
    lo, hi = np.percentile(smoothed, [5, 95])
    if hi - lo < 1e-9:
        return np.full_like(smoothed, 0.5)
    return np.clip((smoothed - lo) / (hi - lo), 0.0, 1.0)


def _classify(norm: np.ndarray) -> list[str]:
    q_lo, q_hi = np.quantile(norm, [0.33, 0.66])
    if q_hi - q_lo < 1e-9:
        return ["mid"] * len(norm)
    cls = "low" if norm[0] < q_lo else "high" if norm[0] >= q_hi else "mid"
    out = []
    for e in norm:
        for _ in range(2):
            if cls == "low" and e > q_lo + _HYSTERESIS:
                cls = "mid"
            elif cls == "mid" and e > q_hi + _HYSTERESIS:
                cls = "high"
            elif cls == "high" and e < q_hi - _HYSTERESIS:
                cls = "mid"
            elif cls == "mid" and e < q_lo - _HYSTERESIS:
                cls = "low"
            else:
                break
        out.append(cls)
    return out


def _build_sections(classes: list[str], drops: set[int]) -> list[Section]:
    sections: list[Section] = []
    start = 0
    for i in range(1, len(classes) + 1):
        if i == len(classes) or classes[i] != classes[start] or i in drops:
            sections.append(Section(start, i, classes[start], start in drops))
            start = i
    return sections


def analyze_audio(
    path: Path, beats_per_bar: int | None = None, fixed_time: tuple[float, float | None] | None = None
) -> AudioAnalysis:
    offset, span = 0.0, None
    if fixed_time is not None:
        offset = fixed_time[0]
        span = None if fixed_time[1] is None else fixed_time[1] - offset
    try:
        y, _ = librosa.load(path, sr=_SR, mono=True, offset=offset, duration=span)
    except Exception as exc:  # decoder errors vary by backend
        raise HyperlapseError(f"Cannot decode audio file {path.name}: {exc}") from exc
    duration = len(y) / _SR
    if duration <= 0:
        if fixed_time is not None:
            raise HyperlapseError(f"audio.fixed_time starts at {offset:.1f}s, past the end of {path.name}")
        raise HyperlapseError(f"Audio file {path.name} is empty")
    if span is not None and duration < span - 0.1:
        logger.warning("audio.fixed_time ends after the end of %s; using %.1fs instead of %.1fs", path.name, duration, span)

    onset_env = librosa.onset.onset_strength(y=y, sr=_SR, hop_length=_HOP)
    # trim=False: the default drops quiet intros/outros, which would shift the bar grid.
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=_SR, hop_length=_HOP, tightness=_TIGHTNESS, trim=False
    )
    if len(beat_frames) < _MIN_BEATS:
        raise HyperlapseError(f"Detected only {len(beat_frames)} beat(s) in {path.name}; cannot build a rhythmic montage")
    beat_times = librosa.frames_to_time(beat_frames, sr=_SR, hop_length=_HOP)

    meter, phase = _detect_meter(_beat_low_energy(y, beat_frames), beats_per_bar)
    bar_starts = beat_times[phase::meter]
    energy = _bar_energy(y, bar_starts, duration)
    norm = _normalise(energy)

    jumps = np.diff(norm, prepend=norm[0])
    drops: set[int] = set()
    for i in np.flatnonzero(jumps >= _DROP_JUMP):
        # A ramp spanning two bars is one drop, not two.
        if not drops or i >= max(drops) + _DROP_HOLD_BARS:
            drops.add(int(i))
    classes = _classify(norm)
    for d in drops:
        for b in range(d, min(d + _DROP_HOLD_BARS, len(classes))):
            classes[b] = "high"

    logger.info(
        "Audio: %.1fs, ~%.0f BPM, %d/4, %d bars, %d drop(s)",
        duration, float(np.atleast_1d(tempo)[0]), meter, len(bar_starts), len(drops),
    )  # fmt: skip
    return AudioAnalysis(
        duration=duration,
        beat_times=beat_times,
        beats_per_bar=meter,
        bar_starts=bar_starts,
        bar_energy=norm,
        sections=_build_sections(classes, drops),
        offset=offset,
    )
