from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import ConfigError

logger = logging.getLogger(__name__)

X264_PRESETS = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)  # fmt: skip
ENCODERS = ("libx264", "h264_nvenc")
MISMATCH_MODES = ("crop", "blur_pad", "skip")
EFFECT_DEFAULTS: dict[str, dict[str, float]] = {
    "flash": {"duration_s": 0.12, "strength": 0.8},
    "zoom_punch": {"duration_s": 0.2, "scale": 1.15},
    "dip_to_black": {"duration_s": 0.15},
    "shake": {"duration_s": 0.2, "amplitude_px": 12.0},
    "rgb_split": {"duration_s": 0.15, "shift_px": 8.0},
}
EFFECT_NAMES = tuple(EFFECT_DEFAULTS)
DEFAULT_OUTPUT_NAME = "montage.mp4"
_TIME_RE = re.compile(r"^\d+(?::\d{1,2}){0,2}(?:\.\d+)?$")
_RATIO_RE = re.compile(r"^([1-9]):([1-9])$")
_BAD_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class PathsConfig:
    video_input: Path
    audio_input: Path
    output: Path
    output_name: str = DEFAULT_OUTPUT_NAME
    photo_input: Path | None = None


@dataclass(frozen=True)
class GlobalConfig:
    orientation: str = "auto"  # montage format; auto = majority of all videos and photos
    mismatch: str = "crop"  # videos and photos in the other orientation
    # "random" | "proportional" | (videos, photos) shots, e.g. (2, 5)
    video_to_photo_ratio: str | tuple[int, int] = "random"


@dataclass(frozen=True)
class PhotosConfig:
    time_shown_min: float = 1.0
    time_shown_max: float = 3.0


@dataclass(frozen=True)
class AudioConfig:
    beats_per_bar: int | None  # None = auto-detect
    cut_interval: int
    min_segment_s: int
    max_segment_s: int | None
    fixed_time: tuple[float, float | None] | None = None  # (start_s, end_s | None = to the end); None = whole file
    fade_in_s: float = 3.0
    fade_out_s: float = 3.0


@dataclass(frozen=True)
class VideoConfig:
    fps: int
    remove_audio: bool
    source_audio_gain: float
    chronologic: bool = False
    allow_segment_reusage: bool = False
    allow_segment_extension: bool = False


@dataclass(frozen=True)
class MotionConfig:
    step_s: float
    window_s: float
    min_score: float | None  # None = auto (percentile per file)


@dataclass(frozen=True)
class PlanningConfig:
    allow_speedup: bool
    max_speedup: float
    seed: int | None


@dataclass(frozen=True)
class RenderConfig:
    encoder: str
    crf: int
    preset: str
    workers: int


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    audio: AudioConfig
    video: VideoConfig
    motion: MotionConfig
    planning: PlanningConfig
    render: RenderConfig
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    photos: PhotosConfig = field(default_factory=PhotosConfig)


@dataclass(frozen=True)
class TransitionsConfig:
    enabled: bool = False
    probability: float = 0.4
    min_gap_cuts: int = 1  # plain cuts required between two transitions
    effects: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(EFFECT_NAMES, False))
    params: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in EFFECT_DEFAULTS.items()}
    )

    @property
    def active_effects(self) -> list[str]:
        if not self.enabled:
            return []
        return [name for name in EFFECT_NAMES if self.effects.get(name)]


def _mapping(
    where: str, value: Any, allowed: tuple[str, ...], moved: dict[str, str] | None = None
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(value).__name__}")
    for key in value:
        if moved and key in moved:
            raise ConfigError(f"'{where}.{key}' has moved to '{moved[key]}'")
        if key not in allowed:
            raise ConfigError(f"Unknown key '{where}.{key}' (allowed: {', '.join(allowed)})")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(where: str, value: Any, lo: int | None = None, hi: int | None = None) -> int:
    if not _is_int(value):
        raise ConfigError(f"'{where}' must be an integer, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        bounds = f"{lo if lo is not None else '-inf'}..{hi if hi is not None else 'inf'}"
        raise ConfigError(f"'{where}' must be in range {bounds}, got {value}")
    return value


def _number(
    where: str, value: Any, lo: float | None = None, hi: float | None = None, lo_open: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{where}' must be a number, got {value!r}")
    value = float(value)
    too_low = lo is not None and (value <= lo if lo_open else value < lo)
    if too_low or (hi is not None and value > hi) or value != value:
        raise ConfigError(f"'{where}' is out of range (min {lo}{' exclusive' if lo_open else ''}, max {hi}), got {value}")
    return value


def _bool(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{where}' must be true or false, got {value!r}")
    return value


def _choice(where: str, value: Any, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ConfigError(f"'{where}' must be one of {', '.join(options)}, got {value!r}")
    return value


def _timestamp(where: str, text: str) -> float:
    text = text.strip()
    parts = text.split(":")
    if not _TIME_RE.match(text) or any(float(p) >= 60 for p in parts[1:]):
        raise ConfigError(f"'{where}' has an invalid time {text!r}; use m:ss, h:mm:ss or seconds")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def _time_range(where: str, value: Any) -> tuple[float, float | None] | None:
    """None = whole file; otherwise (start, end) where end None means 'to the end'."""
    if value is None or (_is_int(value) and value == 0):
        return None
    if not isinstance(value, str):
        # YAML reads unquoted 1:32 as the number 92 (sexagesimal), so only strings are accepted.
        raise ConfigError(f"'{where}' must be a quoted string like \"1:32-2:15\", got {value!r}")
    text = value.strip().replace("\u2013", "-")
    if "-" not in text:
        if _timestamp(where, text) == 0:
            return None
        raise ConfigError(f"'{where}' must be a range like \"1:32-2:15\" (or \"0:00\" for the whole file), got {value!r}")
    if text.count("-") != 1:
        raise ConfigError(f"'{where}' must contain a single range like \"1:32-2:15\", got {value!r}")
    left, right = (side.strip() for side in text.split("-"))
    start = _timestamp(where, left) if left else 0.0
    end = _timestamp(where, right) if right else None
    if end is not None and end <= start:
        raise ConfigError(f"'{where}' must end after it starts, got {value!r}")
    if start == 0 and end is None:
        return None
    return start, end


def _output_name(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{where}' must be a non-empty string, got {value!r}")
    name = value.strip()
    if _BAD_FILENAME_CHARS.search(name) or name in (".", "..") or name.endswith((".", " ")):
        raise ConfigError(f"'{where}' must be a plain file name without folders or special characters, got {value!r}")
    suffix = Path(name).suffix
    if not suffix:
        return name + ".mp4"
    if suffix.lower() != ".mp4":
        raise ConfigError(f"'{where}' must end with .mp4, got {value!r}")
    return name


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) // 2, 6))


def _ratio(where: str, value: Any) -> str | tuple[int, int]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("random", "proportional"):
            return text
        match = _RATIO_RE.match(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    elif _is_int(value) and value >= 60:
        # Unquoted 2:5 is read by YAML as the sexagesimal number 2*60+5 = 125.
        videos, photos = divmod(value, 60)
        if 1 <= videos <= 9 and 1 <= photos <= 9:
            return videos, photos
    raise ConfigError(f"'{where}' must be random, proportional or <1-9>:<1-9> like \"2:5\", got {value!r}")


def parse_config(data: Any, base_dir: Path) -> AppConfig:
    """Validate raw YAML data; relative paths are resolved against `base_dir`."""
    top = _mapping("config", data, ("paths", "global", "audio", "video", "photos", "motion", "planning", "render"))

    p = _mapping("paths", top.get("paths"), ("video_input", "photo_input", "audio_input", "output", "output_name"))
    output_name = _output_name("paths.output_name", p.get("output_name", DEFAULT_OUTPUT_NAME))
    paths = {}
    for key, default in (
        ("video_input", "input/video"), ("photo_input", "input/photos"),
        ("audio_input", "input/audio"), ("output", "output"),
    ):  # fmt: skip
        raw = p.get(key, default)
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError(f"'paths.{key}' must be a non-empty string, got {raw!r}")
        paths[key] = (base_dir / raw).resolve()

    g = _mapping("global", top.get("global"), ("orientation", "mismatch", "video_to_photo_ratio"))
    global_ = GlobalConfig(
        orientation=_choice("global.orientation", g.get("orientation", "auto"), ("auto", "landscape", "portrait")),
        mismatch=_choice("global.mismatch", g.get("mismatch", "crop"), MISMATCH_MODES),
        video_to_photo_ratio=_ratio("global.video_to_photo_ratio", g.get("video_to_photo_ratio", "random")),
    )

    ph = _mapping("photos", top.get("photos"), ("time_shown_min", "time_shown_max"))
    shown_min = _number("photos.time_shown_min", ph.get("time_shown_min", 1), 0.1)
    shown_max = _number("photos.time_shown_max", ph.get("time_shown_max", 3), 0.1)
    if shown_min > shown_max:
        raise ConfigError(
            f"'photos.time_shown_min' ({shown_min:g}) must not exceed 'photos.time_shown_max' ({shown_max:g})"
        )
    photos = PhotosConfig(shown_min, shown_max)

    a = _mapping(
        "audio",
        top.get("audio"),
        ("beats_per_bar", "cut_interval", "min_segment_s", "max_segment_s", "fixed_time", "fade_in_s", "fade_out_s"),
    )
    bpb_raw = a.get("beats_per_bar", "auto")
    if bpb_raw == "auto":
        bpb = None
    elif _is_int(bpb_raw) and bpb_raw in (3, 4):
        bpb = bpb_raw
    else:
        raise ConfigError(f"'audio.beats_per_bar' must be auto, 3 or 4, got {bpb_raw!r}")
    min_seg = _int("audio.min_segment_s", a.get("min_segment_s", 1), lo=0)
    max_raw = a.get("max_segment_s")
    max_seg = None if max_raw is None else _int("audio.max_segment_s", max_raw, lo=1)
    if max_seg is not None and min_seg > max_seg:
        raise ConfigError(f"'audio.min_segment_s' ({min_seg}) must not exceed 'audio.max_segment_s' ({max_seg})")
    audio = AudioConfig(
        bpb,
        _int("audio.cut_interval", a.get("cut_interval", 2), 1, 3),
        min_seg,
        max_seg,
        _time_range("audio.fixed_time", a.get("fixed_time", "0:00")),
        _number("audio.fade_in_s", a.get("fade_in_s", 3), 0.0),
        _number("audio.fade_out_s", a.get("fade_out_s", 3), 0.0),
    )

    v = _mapping(
        "video",
        top.get("video"),
        ("fps", "remove_audio", "source_audio_gain", "chronologic", "allow_segment_reusage", "allow_segment_extension"),
        moved={"orientation": "global.orientation", "mismatch": "global.mismatch"},
    )
    video = VideoConfig(
        fps=_int("video.fps", v.get("fps", 30), 1, 120),
        remove_audio=_bool("video.remove_audio", v.get("remove_audio", True)),
        source_audio_gain=_number("video.source_audio_gain", v.get("source_audio_gain", 0.25), 0.0, 1.0),
        chronologic=_bool("video.chronologic", v.get("chronologic", False)),
        allow_segment_reusage=_bool("video.allow_segment_reusage", v.get("allow_segment_reusage", False)),
        allow_segment_extension=_bool("video.allow_segment_extension", v.get("allow_segment_extension", False)),
    )

    m = _mapping("motion", top.get("motion"), ("step_s", "window_s", "min_score"))
    step_s = _number("motion.step_s", m.get("step_s", 0.4), 0.0, lo_open=True)
    window_s = _number("motion.window_s", m.get("window_s", 3), 0.0, lo_open=True)
    if window_s < step_s:
        raise ConfigError(f"'motion.window_s' ({window_s}) must be >= 'motion.step_s' ({step_s})")
    ms_raw = m.get("min_score", "auto")
    min_score = None if ms_raw == "auto" else _number("motion.min_score", ms_raw, 0.0, 1.0)
    motion = MotionConfig(step_s, window_s, min_score)

    pl = _mapping("planning", top.get("planning"), ("allow_speedup", "max_speedup", "seed"))
    seed_raw = pl.get("seed")
    planning = PlanningConfig(
        allow_speedup=_bool("planning.allow_speedup", pl.get("allow_speedup", True)),
        max_speedup=_number("planning.max_speedup", pl.get("max_speedup", 2.0), 1.0),
        seed=None if seed_raw is None else _int("planning.seed", seed_raw),
    )

    r = _mapping("render", top.get("render"), ("encoder", "crf", "preset", "workers"))
    workers_raw = r.get("workers", "auto")
    workers = default_workers() if workers_raw == "auto" else _int("render.workers", workers_raw, lo=1)
    render = RenderConfig(
        encoder=_choice("render.encoder", r.get("encoder", "libx264"), ENCODERS),
        crf=_int("render.crf", r.get("crf", 20), 0, 51),
        preset=_choice("render.preset", r.get("preset", "veryfast"), X264_PRESETS),
        workers=workers,
    )

    return AppConfig(
        PathsConfig(paths["video_input"], paths["audio_input"], paths["output"], output_name, paths["photo_input"]),
        audio, video, motion, planning, render, global_, photos,
    )  # fmt: skip


def parse_transitions(data: Any) -> TransitionsConfig:
    top = _mapping("transitions", data, ("enabled", "probability", "min_gap_cuts", "effects", "params"))
    effects = dict.fromkeys(EFFECT_NAMES, False)
    for name, value in _mapping("effects", top.get("effects"), EFFECT_NAMES).items():
        effects[name] = _bool(f"effects.{name}", value)

    params = {k: dict(v) for k, v in EFFECT_DEFAULTS.items()}
    for name, raw in _mapping("params", top.get("params"), EFFECT_NAMES).items():
        for key, value in _mapping(f"params.{name}", raw, tuple(EFFECT_DEFAULTS[name])).items():
            hi = 1.0 if key == "strength" else None
            lo = 1.0 if key == "scale" else 0.0
            params[name][key] = _number(f"params.{name}.{key}", value, lo, hi, lo_open=lo == 0.0)

    return TransitionsConfig(
        enabled=_bool("enabled", top.get("enabled", True)),
        probability=_number("probability", top.get("probability", 0.4), 0.0, 1.0),
        min_gap_cuts=_int("min_gap_cuts", top.get("min_gap_cuts", 1), lo=0),
        effects=effects,
        params=params,
    )


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc


def load_config(path: Path) -> AppConfig:
    try:
        return parse_config(_read_yaml(path), path.resolve().parent)
    except ConfigError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc


def load_transitions(path: Path) -> TransitionsConfig:
    if not path.is_file():
        logger.warning("%s not found; using hard cuts only", path)
        return TransitionsConfig(enabled=False)
    try:
        return parse_transitions(_read_yaml(path))
    except ConfigError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
