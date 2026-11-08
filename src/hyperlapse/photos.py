from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .config import PhotosConfig
from .models import Photo, Segment, Slot

logger = logging.getLogger(__name__)

EPS = 1e-6


def _load_photo(path: Path) -> Photo | None:
    try:
        # imdecode instead of imread: imread cannot open non-ASCII paths on Windows; both apply the EXIF rotation.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None:
        logger.warning("Skipping %s: cannot read the image", path.name)
        return None
    height, width = image.shape[:2]
    return Photo(path, width, height)


def load_photos(paths: Sequence[Path], workers: int = 1) -> list[Photo]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        loaded = list(pool.map(_load_photo, paths))
    return [p for p in loaded if p is not None]


def order_by_media(
    photos: Sequence[Photo], free_time: dict[Path, float], media_order: Sequence[Path]
) -> tuple[list[Photo], list[float]]:
    """Photos in `media_order` plus, for each, the share (0..1) of the video footage that precedes it."""
    total = sum(free_time.values()) or 1.0
    by_path = {p.path: p for p in photos}
    ordered: list[Photo] = []
    anchors: list[float] = []
    preceding = 0.0
    for path in media_order:
        if path in free_time:
            preceding += free_time[path]
        elif path in by_path:
            ordered.append(by_path[path])
            anchors.append(preceding / total)
    return ordered, anchors


def shot_counts(n_frames: int, fps: int, lo: float, hi: float) -> tuple[int, int] | None:
    """Range (min, max) of equal parts a slot can be split into so every photo is shown lo..hi seconds."""
    duration = n_frames / fps
    k_min = max(1, math.ceil(duration / hi - EPS))
    k_max = min(math.floor(duration / lo + EPS), n_frames)
    return (k_min, k_max) if k_min <= k_max else None


def split_slot(slot: Slot, k: int, grid: np.ndarray, fps: int, lo: float, hi: float) -> list[Slot]:
    """Split into `k` consecutive shots, cutting on beats when that keeps every shot within lo..hi seconds."""
    if k == 1:
        return [slot]
    span = slot.t_end - slot.t_start
    ideal = [slot.t_start + span * j / k for j in range(1, k)]
    edges = [slot.t_start, *ideal, slot.t_end]
    if len(grid):
        snapped = [slot.t_start, *(float(grid[int(np.argmin(np.abs(grid - t)))]) for t in ideal), slot.t_end]
        tol = 1.5 / fps
        lengths = np.diff(snapped)
        frames = np.diff([round(t * fps) for t in snapped])
        if (lengths > 0).all() and (frames >= 1).all() and (lengths >= lo - tol).all() and (lengths <= hi + tol).all():
            edges = snapped
    return [
        Slot(a, b, round(b * fps) - round(a * fps), slot.forced and i == 0)
        for i, (a, b) in enumerate(zip(edges, edges[1:], strict=False))
    ]


class _PhotoQueue:
    """Hands out photos: every one once before any repeats; shuffled, or in the given order when chronological."""

    def __init__(self, photos: Sequence[Photo], rng: random.Random, in_order: bool) -> None:
        self.photos = list(photos)
        self.rng = rng
        self.in_order = in_order
        self.queue = list(range(len(self.photos)))
        if not in_order:
            rng.shuffle(self.queue)
        self.pos = 0
        self.last: int | None = None

    def remaining(self) -> int:
        return len(self.queue) - self.pos

    def next_index(self) -> int | None:
        return self.queue[self.pos] if self.pos < len(self.queue) else None

    def take(self, k: int) -> list[Photo]:
        out = []
        for _ in range(k):
            if self.pos >= len(self.queue):
                self.pos = 0
                if not self.in_order:
                    self.rng.shuffle(self.queue)
                    if len(self.queue) > 1 and self.queue[0] == self.last:
                        self.queue[0], self.queue[-1] = self.queue[-1], self.queue[0]
            self.last = self.queue[self.pos]
            out.append(self.photos[self.last])
            self.pos += 1
        return out


def plan_photo_shots(
    slots: list[Slot],
    grid: np.ndarray,
    photos: Sequence[Photo],
    cfg: PhotosConfig,
    ratio: str | tuple[int, int],
    fps: int,
    clip_estimate: float,
    reuse: bool,
    rng: random.Random,
    anchors: Sequence[float] | None = None,
) -> tuple[list[Slot], list[Segment]]:
    """Decide which slots show photos; returns (slots left for video, photo segments).

    A photo slot longer than `time_shown_max` is split into several shots, each with its own photo.
    `ratio` is the wanted videos:photos shot ratio ("random" draws every slot independently, with the
    photo chance following the share of available photos). With `anchors` (chronological mode) photos
    go out in the given order and a photo is shown only once the video footage before it has played.
    """
    n = len(photos)
    if n == 0 or not slots:
        return list(slots), []
    if isinstance(ratio, tuple):
        videos_w, photos_w = float(ratio[0]), float(ratio[1])
    else:
        videos_w, photos_w = max(clip_estimate, 1.0), float(n)
    share = photos_w / (videos_w + photos_w)
    total_s = sum(s.n_frames for s in slots) / fps
    photo_time = min(n * (cfg.time_shown_min + cfg.time_shown_max) / 2, total_s * share)
    video_time_est = max(total_s - photo_time, 0.1 * total_s)

    queue = _PhotoQueue(photos, rng, in_order=anchors is not None)
    video_slots: list[Slot] = []
    segments: list[Segment] = []
    n_videos = n_photos = 0
    video_time = 0.0
    for slot in slots:
        draw = rng.random()
        wants = draw < share if ratio == "random" else n_photos * videos_w < n_videos * photos_w
        counts = shot_counts(slot.n_frames, fps, cfg.time_shown_min, cfg.time_shown_max)
        if wants and counts and not reuse:
            counts = (counts[0], min(counts[1], queue.remaining())) if queue.remaining() >= counts[0] else None
        if wants and counts and anchors is not None:
            upcoming = queue.next_index()
            wants = upcoming is None or anchors[upcoming] <= video_time / video_time_est + EPS
        if not (wants and counts):
            video_slots.append(slot)
            n_videos += 1
            video_time += slot.n_frames / fps
            continue
        k = rng.randint(*counts)
        shots = split_slot(slot, k, grid, fps, cfg.time_shown_min, cfg.time_shown_max)
        for shot, photo in zip(shots, queue.take(len(shots)), strict=True):
            segments.append(Segment(shot.t_start, shot.t_end, photo.path, 0.0, 0.0, 1.0, shot.n_frames, None, True))
        n_photos += len(shots)

    if not segments:
        logger.warning(
            "No photo was placed: none was drawn or no slot fits time_shown %g-%g s; the montage has videos only",
            cfg.time_shown_min, cfg.time_shown_max,
        )  # fmt: skip
    else:
        logger.info("Photos: %d shot(s) from %d photo(s) next to %d video segment(s)", n_photos, n, n_videos)
    return video_slots, segments
