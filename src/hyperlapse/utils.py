from __future__ import annotations

import logging
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from .models import HyperlapseError

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = frozenset({".mp3", ".wav"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov"})
PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# 20240615_143012, 2024-06-15 14.30.12, VID-20240615-WA0001; time part optional
_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})(?:[-_. T]?(\d{2})[-_.: ]?(\d{2})[-_.: ]?(\d{2}))?"
)
_LEADING_NUMBER_RE = re.compile(r"^\s*(\d+)")
_TRAILING_NUMBER_RE = re.compile(r"(\d+)\s*$")


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise HyperlapseError(
            "FFmpeg not found: install it on PATH or install the 'imageio-ffmpeg' package."
        ) from exc


def temp_root() -> Path:
    # Outside the project dir, which lives in OneDrive.
    path = Path(tempfile.gettempdir()) / "hyperlapse_wizard"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_audio(directory: Path) -> Path:
    if not directory.is_dir():
        raise HyperlapseError(f"Audio input directory does not exist: {directory}")
    candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    if not candidates:
        raise HyperlapseError(f"No audio files (.mp3/.wav) found in {directory}")
    newest = max(candidates, key=lambda p: (p.stat().st_mtime, p.name))
    logger.info("Using audio file: %s", newest.name)
    return newest


def list_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise HyperlapseError(f"Video input directory does not exist: {directory}")
    videos = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    if not videos:
        raise HyperlapseError(f"No video files (.mp4/.mov) found in {directory}")
    logger.info("Found %d video file(s)", len(videos))
    return videos


def list_photos(directory: Path | None) -> list[Path]:
    """Photos are optional: a missing or empty folder simply means a video-only montage."""
    if directory is None or not directory.is_dir():
        return []
    photos = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    if photos:
        logger.info("Found %d photo file(s)", len(photos))
    return photos


def _name_date(stem: str) -> datetime | None:
    for match in _DATE_RE.finditer(stem):
        year, month, day, hour, minute, second = (int(g) if g else 0 for g in match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            continue
    return None


def _name_number(stem: str) -> int | None:
    match = _LEADING_NUMBER_RE.search(stem) or _TRAILING_NUMBER_RE.search(stem)
    return int(match.group(1)) if match else None


def chronological_order(videos: list[Path]) -> list[Path] | None:
    """Sort by the date in the file names, else by manual numbering; None if not every file has one."""
    for label, extract in (("date in the file name", _name_date), ("number in the file name", _name_number)):
        keys = [extract(v.stem) for v in videos]
        if all(k is not None for k in keys):
            logger.info("Chronological order from the %s", label)
            ranked = sorted(zip(keys, (v.name.lower() for v in videos), range(len(videos)), strict=True))
            return [videos[i] for _, _, i in ranked]
    logger.warning("video.chronologic: not every file name has a date or number; using random order")
    return None
