from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .config import AppConfig


class HyperlapseError(Exception):
    """Critical, user-facing error: the CLI prints it and exits with a non-zero code."""


class ConfigError(HyperlapseError):
    pass


@dataclass(frozen=True)
class Section:
    start_bar: int  # inclusive
    end_bar: int  # exclusive
    energy_class: str  # "low" | "mid" | "high"
    is_drop: bool = False


@dataclass
class AudioAnalysis:
    duration: float
    beat_times: np.ndarray
    beats_per_bar: int
    bar_starts: np.ndarray  # downbeat times, one per bar
    bar_energy: np.ndarray  # 0..1, one per bar
    sections: list[Section] = field(default_factory=list)
    offset: float = 0.0  # start of the analysed fragment in the audio file; all times above are relative to it

    @property
    def n_bars(self) -> int:
        return len(self.bar_starts)

    @property
    def first_downbeat_index(self) -> int:
        """Index in `beat_times` of the first bar start."""
        if len(self.bar_starts) == 0:
            return 0
        return int(np.argmin(np.abs(self.beat_times - self.bar_starts[0])))

    def grid_beats(self) -> np.ndarray:
        """Beat times starting at the first downbeat; bar `i` owns grid beats [i*bpb, (i+1)*bpb)."""
        return self.beat_times[self.first_downbeat_index :]


@dataclass
class MotionProfile:
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    times: np.ndarray
    scores: np.ndarray
    scene_cuts: np.ndarray
    good_intervals: list[tuple[float, float]]


@dataclass(frozen=True)
class Slot:
    t_start: float
    t_end: float
    n_frames: int
    forced: bool = False  # starts at a drop


@dataclass(frozen=True)
class Photo:
    path: Path
    width: int  # after applying EXIF rotation
    height: int


@dataclass(frozen=True)
class Segment:
    t_start: float  # timeline position (seconds)
    t_end: float
    source: Path
    src_in: float
    src_out: float
    speed: float
    n_frames: int
    transition: str | None = None
    is_photo: bool = False  # `source` is a still image; src_in/src_out are unused


class Renderer(Protocol):
    def render(
        self,
        segments: list[Segment],
        audio_path: Path,
        output_path: Path,
        config: AppConfig,
        audio_offset: float = 0.0,
    ) -> None: ...
