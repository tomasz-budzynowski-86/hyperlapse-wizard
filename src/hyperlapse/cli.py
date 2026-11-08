from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .audio import analyze_audio
from .config import load_config, load_transitions
from .models import HyperlapseError
from .motion import mismatched_sources, prepare_photos, prepare_sources
from .photos import load_photos
from .planner import plan
from .render import FFmpegRenderer
from .utils import chronological_order, find_ffmpeg, list_photos, list_videos, pick_audio

logger = logging.getLogger("hyperlapse")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperlapse", description="Beat-synced video montage generator.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument(
        "--transitions", type=Path, default=None, help="path to transitions.yaml (default: next to the config)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def run(config_path: Path, transitions_path: Path | None = None) -> Path:
    config = load_config(config_path)
    transitions = load_transitions(transitions_path or config_path.resolve().with_name("transitions.yaml"))
    ffmpeg = find_ffmpeg()
    audio_path = pick_audio(config.paths.audio_input)
    videos = list_videos(config.paths.video_input)
    photos = load_photos(list_photos(config.paths.photo_input), config.render.workers)
    media_order = None
    if config.video.chronologic:
        media_order = chronological_order([*videos, *(p.path for p in photos)])
    ordered = None
    if media_order:
        video_set = set(videos)
        ordered = [v for v in media_order if v in video_set]
    logger.info("Active transitions: %s", ", ".join(transitions.active_effects) or "none (hard cuts)")

    audio = analyze_audio(audio_path, config.audio.beats_per_bar, config.audio.fixed_time)
    sources, size = prepare_sources(
        ordered or videos, config.motion, config.global_.orientation, config.render.workers, config.global_.mismatch,
        photos,
    )  # fmt: skip
    photos = prepare_photos(photos, size, config.global_.mismatch)
    pad_sources = frozenset()
    if config.global_.mismatch == "blur_pad":
        pad_sources = mismatched_sources([*sources, *photos], size)
    segments = plan(
        audio, sources, config, transitions, chronological=ordered is not None, photos=photos, media_order=media_order
    )

    output = config.paths.output / config.paths.output_name
    FFmpegRenderer(ffmpeg, transitions, size, pad_sources).render(segments, audio_path, output, config, audio.offset)
    return output


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_MAX_BYTES = 5_000_000
LOG_BACKUPS = 5
NOISY_LOGGERS = ("numba", "matplotlib", "PIL", "llvmlite", "numexpr")


def setup_logging(log_file: Path, verbose: bool) -> None:
    """Console at INFO (DEBUG with -v); the log file always records DEBUG so past runs can be diagnosed."""
    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, "_hyperlapse", False)]:
        root.removeHandler(old)
        old.close()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console._hyperlapse = True  # type: ignore[attr-defined]
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot write log file %s: %s", log_file, exc)
        return
    handler.setLevel(logging.DEBUG)
    handler._hyperlapse = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = args.config.resolve().parent / "logs" / "hyperlapse.log"
    setup_logging(log_file, args.verbose)
    logger.info("Run started: config=%s, log file=%s", args.config, log_file)
    try:
        output = run(args.config, args.transitions)
    except HyperlapseError as exc:
        logger.error("%s", exc, exc_info=args.verbose)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Unexpected error")
        return 1
    logger.info("Montage written to %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
