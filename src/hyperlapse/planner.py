from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import AppConfig, AudioConfig, MotionConfig, TransitionsConfig
from .models import AudioAnalysis, HyperlapseError, MotionProfile, Photo, Segment, Slot
from .motion import resolve_min_score, window_motion
from .photos import order_by_media, plan_photo_shots
from .transitions import assign_transitions

logger = logging.getLogger(__name__)

LADDERS = {4: [2, 4, 8, 16], 3: [3, 6, 12, 12]}
# cut_interval -> energy class -> rung on the ladder
RUNGS = {
    1: {"high": 0, "mid": 0, "low": 1},
    2: {"high": 0, "mid": 1, "low": 2},
    3: {"high": 1, "mid": 2, "low": 3},
}
SOURCE_MARGIN_S = 0.5
# Source-time per timeline second tried in order; values below 1 are slow motion that stretches shorter footage.
SLOWDOWN_STEPS = (1.0, 0.75, 0.5)
MAX_CANDIDATES = 30
ALLOCATION_ATTEMPTS = 20
EPS = 1e-6


def cut_period_beats(
    beats_per_bar: int, energy_class: str, cut_interval: int, beat_s: float, min_s: int, max_s: int | None
) -> int:
    ladder = LADDERS[beats_per_bar]
    rung = RUNGS[cut_interval][energy_class]
    if min_s > 0:
        while rung < len(ladder) - 1 and ladder[rung] * beat_s < min_s:
            rung += 1
    if max_s is not None:
        while rung > 0 and ladder[rung] * beat_s > max_s:
            rung -= 1
    return ladder[rung]


def _merge_short(spans: list[list], min_s: int, fps: int) -> list[list]:
    """spans: [t_start, t_end, forced]; absorb too-short spans into a neighbour."""

    def too_short(s: list) -> bool:
        frames = round(s[1] * fps) - round(s[0] * fps)
        return frames < 1 or (min_s > 0 and s[1] - s[0] < min_s - EPS)

    while len(spans) > 1:
        i = next((k for k, s in enumerate(spans) if too_short(s)), None)
        if i is None:
            break
        if (spans[i][2] and i + 1 < len(spans)) or i == 0:
            spans[i + 1][0] = spans[i][0]
            spans[i + 1][2] = spans[i][2] or spans[i + 1][2]
        else:
            spans[i - 1][1] = spans[i][1]
        del spans[i]
    return spans


def build_slots(audio: AudioAnalysis, cfg: AudioConfig, fps: int) -> list[Slot]:
    grid = audio.grid_beats()
    n_beats = len(grid)
    bpb = audio.beats_per_bar
    global_beat_s = float(np.median(np.diff(grid))) if n_beats > 1 else 0.5

    cuts: list[tuple[int, bool]] = []
    for sec in audio.sections:
        start, end = sec.start_bar * bpb, min(sec.end_bar * bpb, n_beats)
        if start >= n_beats:
            break
        local = np.diff(grid[start : end + 1])
        beat_s = float(local.mean()) if len(local) else global_beat_s
        period = cut_period_beats(bpb, sec.energy_class, cfg.cut_interval, beat_s, cfg.min_segment_s, cfg.max_segment_s)
        cuts.append((start, sec.is_drop))
        cuts.extend((i, False) for i in range(start + period, end, period))

    if not cuts:
        raise HyperlapseError("Audio analysis produced no bars to cut on")
    # The intro joins the first segment; the last one runs to the end of the audio.
    starts = [0.0] + [float(grid[i]) for i, _ in cuts[1:]]
    ends = starts[1:] + [audio.duration]
    spans = [[s, e, forced] for s, e, (_, forced) in zip(starts, ends, cuts, strict=True)]
    spans = _merge_short(spans, cfg.min_segment_s, fps)
    return [Slot(s, e, round(e * fps) - round(s * fps), bool(f)) for s, e, f in spans]


def _subtract(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in intervals:
        if hi <= a or lo >= b:
            out.append((a, b))
            continue
        if lo > a:
            out.append((a, lo))
        if hi < b:
            out.append((hi, b))
    return out


def _free_time(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)


def check_capacity(slots: list[Slot], sources: list[MotionProfile], fps: int, extend: bool = False) -> None:
    needed = sum(s.n_frames for s in slots) / fps + len(slots) * SOURCE_MARGIN_S
    if extend:
        needed = sum(s.n_frames for s in slots) / fps * SLOWDOWN_STEPS[-1] + len(slots) * SOURCE_MARGIN_S
    available = sum(_free_time(p.good_intervals) for p in sources)
    if available + EPS < needed:
        raise HyperlapseError(
            "Not enough usable source material for a montage without repeated footage.\n"
            f"  needed:    {needed:.1f}s ({sum(s.n_frames for s in slots) / fps:.1f}s of video + "
            f"{len(slots)} x {SOURCE_MARGIN_S}s safety margins)\n"
            f"  available: {available:.1f}s of moving footage in {len(sources)} source(s)\n"
            f"  missing:   {needed - available:.1f}s\n"
            "Add more footage or lower 'motion.min_score' to admit more of the existing footage."
        )


class _AllocationError(Exception):
    pass


def _pick_start(
    free: list[tuple[float, float]], length: float, profile: MotionProfile, threshold: float, rng: random.Random
) -> float:
    fitting = [(a, b - length) for a, b in free if b - a >= length - EPS]
    weights = [max(hi - lo, 1e-3) for lo, hi in fitting]
    best_start, best_score = None, -1.0
    for _ in range(MAX_CANDIDATES):
        lo, hi = rng.choices(fitting, weights)[0]
        start = rng.uniform(lo, max(lo, hi))
        score = window_motion(profile.times, profile.scores, start, start + length)
        if score >= threshold:
            return start
        if score > best_score:
            best_start, best_score = start, score
    logger.warning("%s: no candidate with enough motion (best %.4f < %.4f)", profile.path.name, best_score, threshold)
    assert best_start is not None
    return best_start


def _choose_source(
    free: dict[Path, list[tuple[float, float]]], length: float, steps: tuple[float, ...], previous: Path | None, n_sources: int
) -> tuple[list[Path], float] | None:
    """Sources able to supply `length` seconds (slowed by the first workable step); None when there are none."""
    for speed in steps:
        window = length * speed
        eligible = [p for p in free if any(b - a >= window - EPS for a, b in free[p])]
        choices = [p for p in eligible if p != previous]
        if choices:
            return choices, speed
        if previous in eligible and n_sources == 1:
            return eligible, speed
    return None


def _allocate_once(
    slots: list[Slot],
    sources: list[MotionProfile],
    motion_cfg: MotionConfig,
    fps: int,
    rng: random.Random,
    reuse: bool = False,
    extend: bool = False,
) -> list[Segment]:
    full = {p.path: list(p.good_intervals) for p in sources}
    free = {k: list(v) for k, v in full.items()}
    by_path = {p.path: p for p in sources}
    thresholds = {p.path: resolve_min_score(p.scores, motion_cfg.min_score) for p in sources}
    steps = SLOWDOWN_STEPS if extend else (1.0,)
    previous: Path | None = None
    warned_single = False
    segments: list[Segment] = []

    for slot in slots:
        length = slot.n_frames / fps
        pick = _choose_source(free, length, steps, previous, len(sources))
        if pick is None and reuse:
            logger.debug("Footage exhausted at t=%.2fs: starting another pass over the sources", slot.t_start)
            free = {k: list(v) for k, v in full.items()}
            pick = _choose_source(free, length, steps, previous, len(sources))
        if pick is None:
            raise _AllocationError(
                f"no source has a free {length:.2f}s range for the segment at t={slot.t_start:.2f}s"
            )
        choices, speed = pick
        if previous in choices:
            if not warned_single:
                logger.warning("Single source: consecutive segments come from the same file")
                warned_single = True
        window = length * speed
        path = rng.choices(choices, [_free_time(free[p]) for p in choices])[0]
        start = _pick_start(free[path], window, by_path[path], thresholds[path], rng)
        end = start + window
        free[path] = _subtract(free[path], start - SOURCE_MARGIN_S, end + SOURCE_MARGIN_S)
        segments.append(Segment(slot.t_start, slot.t_end, path, start, end, speed, slot.n_frames))
        previous = path
    return segments


def _partition_chronological(slots: list[Slot], sources: list[MotionProfile], fps: int) -> list[list[Slot]]:
    """Split the slots, in timeline order, into one consecutive block per source (in source order)."""
    caps = [_free_time(p.good_intervals) for p in sources]
    weights = [s.n_frames / fps + SOURCE_MARGIN_S for s in slots]
    total_w, total_c = sum(weights), sum(caps)
    boundaries = list(itertools.accumulate(total_w * c / total_c for c in caps))
    groups: list[list[Slot]] = [[] for _ in sources]
    used = [0.0] * len(sources)
    i, done = 0, 0.0
    for slot, w in zip(slots, weights, strict=True):
        # A block ends at its proportional share of the timeline, or earlier when its source is full.
        while i < len(sources) - 1 and (done + w / 2 > boundaries[i] or used[i] + w > caps[i] + EPS):
            i += 1
        groups[i].append(slot)
        used[i] += w
        done += w
    return groups


def _ascending_window(
    free: list[tuple[float, float]], cursor: float, length: float, reserve: float
) -> list[tuple[float, float]]:
    """Free ranges after `cursor` where a clip can start while `reserve` seconds stay available behind it."""
    avail = [(max(a, cursor), b) for a, b in free if b > cursor]
    target = length + reserve
    acc, latest_start = 0.0, None
    for a, b in reversed(avail):
        if acc + (b - a) >= target - EPS:
            latest_start = b - (target - acc)
            break
        acc += b - a
    if latest_start is None:
        return []
    clipped = [(a, min(b, latest_start + length)) for a, b in avail]
    return [(a, b) for a, b in clipped if b - a >= length - EPS]


def _chronological_window(
    free: list[tuple[float, float]], cursor: float, length: float, reserve: float, steps: tuple[float, ...]
) -> tuple[list[tuple[float, float]], float] | None:
    for speed in steps:
        window = _ascending_window(free, cursor, length * speed, reserve)
        if window:
            return window, speed
    return None


def _allocate_chronological(
    slots: list[Slot],
    sources: list[MotionProfile],
    motion_cfg: MotionConfig,
    fps: int,
    rng: random.Random,
    reuse: bool = False,
    extend: bool = False,
) -> list[Segment]:
    steps = SLOWDOWN_STEPS if extend else (1.0,)
    segments: list[Segment] = []
    for profile, group in zip(sources, _partition_chronological(slots, sources, fps), strict=True):
        threshold = resolve_min_score(profile.scores, motion_cfg.min_score)
        cursor = 0.0
        reserve = sum(s.n_frames / fps + SOURCE_MARGIN_S for s in group)
        for slot in group:
            length = slot.n_frames / fps
            reserve = max(0.0, reserve - (length + SOURCE_MARGIN_S))
            pick = _chronological_window(profile.good_intervals, cursor, length, reserve, steps)
            if pick is None and reuse:
                cursor, reserve = 0.0, 0.0  # start another pass over this source
                pick = _chronological_window(profile.good_intervals, cursor, length, reserve, steps)
            if pick is None:
                raise _AllocationError(
                    f"{profile.path.name} has no free {length:.2f}s range left for the segment at t={slot.t_start:.2f}s"
                )
            window, speed = pick
            span = length * speed
            start = _pick_start(window, span, profile, threshold, rng)
            cursor = start + span + SOURCE_MARGIN_S
            segments.append(Segment(slot.t_start, slot.t_end, profile.path, start, start + span, speed, slot.n_frames))
    return segments


def allocate_sources(
    slots: list[Slot],
    sources: list[MotionProfile],
    motion_cfg: MotionConfig,
    fps: int,
    seed: int,
    chronological: bool = False,
    reuse: bool = False,
    extend: bool = False,
) -> list[Segment]:
    if not reuse:
        check_capacity(slots, sources, fps, extend)
    last_error = ""
    for attempt in range(ALLOCATION_ATTEMPTS):
        rng = random.Random(f"{seed}:alloc:{attempt}")
        try:
            if chronological:
                return _allocate_chronological(slots, sources, motion_cfg, fps, rng, reuse, extend)
            return _allocate_once(slots, sources, motion_cfg, fps, rng, reuse, extend)
        except _AllocationError as exc:
            last_error = str(exc)
            logger.debug("Allocation attempt %d failed: %s", attempt, last_error)
    raise HyperlapseError(
        f"Could not place all segments ({last_error}). The usable footage is too fragmented or too short; "
        "add more source material or set video.allow_segment_reusage / video.allow_segment_extension to true."
    )


def plan(
    audio: AudioAnalysis,
    sources: list[MotionProfile],
    config: AppConfig,
    transitions: TransitionsConfig,
    chronological: bool = False,
    photos: Sequence[Photo] = (),
    media_order: Sequence[Path] | None = None,
) -> list[Segment]:
    """Plan the montage. `media_order` (videos and photos by date/number) positions photos in chronological mode."""
    seed = config.planning.seed
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    logger.info("Planning seed: %d", seed)

    slots = build_slots(audio, config.audio, config.video.fps)
    logger.info("Planned %d segment(s)", len(slots))
    reuse = config.video.allow_segment_reusage
    photo_segments: list[Segment] = []
    if photos:
        anchors = None
        if chronological:
            if media_order is None:
                anchors = [i / len(photos) for i in range(len(photos))]
            else:
                free = {p.path: _free_time(p.good_intervals) for p in sources}
                photos, anchors = order_by_media(photos, free, media_order)
        mean_slot_s = sum(s.n_frames for s in slots) / config.video.fps / len(slots)
        clip_estimate = sum(_free_time(p.good_intervals) for p in sources) / (mean_slot_s + SOURCE_MARGIN_S)
        slots, photo_segments = plan_photo_shots(
            slots, audio.grid_beats(), photos, config.photos, config.global_.video_to_photo_ratio,
            config.video.fps, clip_estimate, reuse, random.Random(f"{seed}:photos"), anchors,
        )  # fmt: skip
    segments = allocate_sources(
        slots, sources, config.motion, config.video.fps, seed, chronological, reuse,
        config.video.allow_segment_extension,
    ) if slots else []  # fmt: skip
    segments = sorted([*segments, *photo_segments], key=lambda s: s.t_start)
    return assign_transitions(segments, transitions, random.Random(f"{seed}:transitions"))

