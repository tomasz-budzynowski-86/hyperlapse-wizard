from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MotionConfig
from .models import HyperlapseError, MotionProfile, Photo
from .utils import temp_root

logger = logging.getLogger(__name__)

SCENE_CUT_SCORE = 0.6
PIXEL_THRESHOLD = 25
ANALYSIS_WIDTH = 320
AUTO_PERCENTILE = 40
MIN_SCORE_FLOOR = 0.002
CACHE_VERSION = 1
TARGET_SIZES = {"landscape": (1920, 1080), "portrait": (1080, 1920)}


@dataclass
class RawTimeline:
    times: np.ndarray
    scores: np.ndarray
    fps: float
    duration: float
    width: int
    height: int


def motion_timeline(path: Path, step_s: float) -> RawTimeline:
    """Fraction of changed pixels between consecutive time-sampled frames (score k belongs to times[k])."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HyperlapseError(f"Cannot open video file: {path.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise HyperlapseError(f"Video {path.name} reports an invalid frame rate ({fps})")
        stride = max(1, round(fps * step_s))

        times: list[float] = []
        scores: list[float] = []
        prev: np.ndarray | None = None
        width = height = 0
        idx = 0
        while cap.grab():
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    if width == 0:
                        height, width = frame.shape[:2]
                    small = cv2.resize(
                        frame, (ANALYSIS_WIDTH, max(1, round(height * ANALYSIS_WIDTH / width))), interpolation=cv2.INTER_AREA
                    )
                    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                    if prev is not None:
                        delta = cv2.absdiff(gray, prev)
                        times.append(idx / fps)
                        scores.append(float(np.count_nonzero(delta > PIXEL_THRESHOLD)) / delta.size)
                    prev = gray
            idx += 1
    finally:
        cap.release()

    if width == 0 or idx == 0:
        raise HyperlapseError(f"Video {path.name} contains no decodable frames")
    return RawTimeline(np.asarray(times), np.asarray(scores), fps, idx / fps, width, height)


def window_motion(times: np.ndarray, scores: np.ndarray, t_in: float, t_out: float) -> float:
    lo = int(np.searchsorted(times, t_in, side="right"))
    hi = int(np.searchsorted(times, t_out, side="right"))
    if hi <= lo:
        return 0.0
    return float(np.mean(scores[lo:hi]))


def resolve_min_score(scores: np.ndarray, configured: float | None) -> float:
    if configured is not None:
        return configured
    if len(scores) == 0:
        return MIN_SCORE_FLOOR
    return max(float(np.percentile(scores, AUTO_PERCENTILE)), MIN_SCORE_FLOOR)


def find_good_intervals(
    times: np.ndarray, scores: np.ndarray, duration: float, step_s: float, window_s: float, min_score: float
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Return (good_intervals, scene_cuts); intervals never span a scene cut."""
    if len(times) == 0:
        return [], np.empty(0)
    is_cut = scores >= SCENE_CUT_SCORE
    half = window_s / 2
    smoothed = np.array([window_motion(times, scores, t - half, t + half) for t in times])
    active = (smoothed >= min_score) & ~is_cut

    # Sample k covers (times[k-1], times[k]]; the first sample's reference frame is at times[0] - step.
    starts = np.concatenate(([max(0.0, times[0] - step_s)], times[:-1]))
    min_len = 2 * step_s
    intervals: list[tuple[float, float]] = []
    run_start: int | None = None
    for k in range(len(times) + 1):
        if k < len(times) and active[k]:
            if run_start is None:
                run_start = k
            continue
        if run_start is not None:
            a, b = float(starts[run_start]), min(float(times[k - 1]), duration)
            if b - a >= min_len:
                intervals.append((a, b))
            run_start = None
    return intervals, times[is_cut]


def _cache_path(path: Path, step_s: float, cache_dir: Path) -> Path:
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{step_s}|{PIXEL_THRESHOLD}|{ANALYSIS_WIDTH}|{CACHE_VERSION}"
    return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.npz"


def _load_timeline(path: Path, step_s: float, cache_dir: Path) -> RawTimeline:
    cache_file = _cache_path(path, step_s, cache_dir)
    if cache_file.is_file():
        try:
            with np.load(cache_file) as data:
                meta = data["meta"]
                logger.debug("Motion cache hit: %s", path.name)
                return RawTimeline(data["times"], data["scores"], float(meta[0]), float(meta[1]), int(meta[2]), int(meta[3]))
        except (OSError, ValueError, KeyError):
            logger.warning("Ignoring corrupt motion cache for %s", path.name)
    started = time.monotonic()
    logger.info("Analysing motion of %s (not cached, this can take a long time for 4K footage)", path.name)
    timeline = motion_timeline(path, step_s)
    logger.info("Analysed %s in %.0fs", path.name, time.monotonic() - started)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp.npz")
    meta = np.array([timeline.fps, timeline.duration, timeline.width, timeline.height])
    np.savez(tmp, times=timeline.times, scores=timeline.scores, meta=meta)
    tmp.replace(cache_file)
    return timeline


def analyze_video(path: Path, cfg: MotionConfig, cache_dir: Path | None = None) -> MotionProfile:
    timeline = _load_timeline(path, cfg.step_s, cache_dir or temp_root() / "motion_cache")
    threshold = resolve_min_score(timeline.scores, cfg.min_score)
    intervals, cuts = find_good_intervals(
        timeline.times, timeline.scores, timeline.duration, cfg.step_s, cfg.window_s, threshold
    )
    return MotionProfile(
        path=path,
        duration=timeline.duration,
        fps=timeline.fps,
        width=timeline.width,
        height=timeline.height,
        times=timeline.times,
        scores=timeline.scores,
        scene_cuts=cuts,
        good_intervals=intervals,
    )


def _orientation(profile: MotionProfile | Photo) -> str:
    return "landscape" if profile.width >= profile.height else "portrait"


def mismatched_sources(sources: Sequence[MotionProfile | Photo], size: tuple[int, int]) -> frozenset[Path]:
    """Sources whose orientation differs from the montage format given by `size`."""
    target = "landscape" if size[0] >= size[1] else "portrait"
    return frozenset(p.path for p in sources if _orientation(p) != target)


def _is_usable(item: MotionProfile | Photo, orientation: str, size: tuple[int, int], mismatch: str) -> bool:
    width, height = size
    if _orientation(item) == orientation:
        ok = item.width >= width and item.height >= height
    elif mismatch == "skip":
        logger.warning("Skipping %s: %s source in a %s montage", item.path.name, _orientation(item), orientation)
        return False
    elif mismatch == "blur_pad":
        ok = item.width >= width or item.height >= height
    else:
        ok = item.width >= width and item.height >= height
    if not ok:
        logger.warning("Skipping %s: resolution %dx%d too low for %dx%d", item.path.name, item.width, item.height, width, height)
        return False
    if _orientation(item) != orientation:
        logger.info("%s is %s in a %s montage: using %s", item.path.name, _orientation(item), orientation, mismatch)
    return True


def prepare_sources(
    paths: list[Path],
    cfg: MotionConfig,
    orientation: str = "auto",
    workers: int = 1,
    mismatch: str = "crop",
    photos: Sequence[Photo] = (),
) -> tuple[list[MotionProfile], tuple[int, int]]:
    """Analyse all sources and keep the usable ones.

    Sources in the other orientation are dropped for mismatch="skip"; otherwise they are kept and later
    cropped to fill the frame ("crop") or fitted over a blurred copy of themselves ("blur_pad").
    `photos` only take part in the "auto" orientation vote; filter them with `prepare_photos`.
    """
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        logger.info("Analysing %d video(s) with %d worker(s)", len(paths), max(1, workers))
        profiles = list(pool.map(lambda p: analyze_video(p, cfg), paths))

    counts = {o: sum(_orientation(p) == o for p in (*profiles, *photos)) for o in TARGET_SIZES}
    if orientation == "auto":
        if counts["landscape"] == counts["portrait"]:
            raise HyperlapseError(
                f"Mixed orientations without a majority ({counts['landscape']} landscape, "
                f"{counts['portrait']} portrait); set global.orientation explicitly."
            )
        orientation = max(counts, key=counts.__getitem__)
    size = TARGET_SIZES[orientation]
    logger.info("Target format: %s %dx%d", orientation, *size)

    kept = [p for p in profiles if _is_usable(p, orientation, size, mismatch)]
    if not kept:
        raise HyperlapseError(
            f"No usable video sources: none is {orientation} with at least {size[0]}x{size[1]} pixels"
        )
    return kept, size


def prepare_photos(photos: Sequence[Photo], size: tuple[int, int], mismatch: str = "crop") -> list[Photo]:
    """Photos that fit the montage format `size` under the given mismatch mode."""
    orientation = "landscape" if size[0] >= size[1] else "portrait"
    return [p for p in photos if _is_usable(p, orientation, size, mismatch)]
