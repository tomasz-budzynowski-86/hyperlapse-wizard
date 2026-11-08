from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import AppConfig, RenderConfig, TransitionsConfig
from .models import HyperlapseError, Segment
from .transitions import effect_duration
from .utils import temp_root

logger = logging.getLogger(__name__)

AUDIO_RATE = 48000
AUDIO_FADE_S = 0.015
AAC_BITRATE = "192k"
GOP_SECONDS = 2
# Extra source read past the window so rounding never yields fewer than n_frames; -frames:v trims it.
SOURCE_OVERSHOOT_FRAMES = 2
NVENC_PRESETS = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3", "fast": "p4",
    "medium": "p4", "slow": "p5", "slower": "p6", "veryslow": "p7",
}  # fmt: skip


def _num(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        logger.error("FFmpeg exit %d, full command: %s\n%s", result.returncode, " ".join(args), result.stderr.strip())
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise HyperlapseError(f"FFmpeg failed (exit {result.returncode}) running: {' '.join(args[:1] + args[-3:])}\n{tail}")
    return result


def effect_filter(effect: str, params: dict[str, float], duration: float, width: int, height: int) -> str:
    """Filter that applies `effect` to the first `duration` seconds of segment-local time."""
    d = _num(duration)
    if effect == "flash":
        return f"eq=brightness='{_num(params['strength'])}*max(0,1-t/{d})':eval=frame"
    if effect == "zoom_punch":
        zoom = f"(1+({_num(params['scale'])}-1)*max(0,1-t/{d}))"
        return f"scale=w='{width}*{zoom}':h='{height}*{zoom}':eval=frame,crop={width}:{height}"
    if effect == "dip_to_black":
        return f"fade=t=in:st=0:d={d}"
    if effect == "shake":
        a = int(round(params["amplitude_px"]))
        decay = f"max(0,1-t/{d})"
        # Overscan fades to zero with the shake, so the frame returns to exactly W x H after the effect.
        return (
            f"scale=w='{width}+{2 * a}*{decay}':h='{height}+{2 * a}*{decay}':eval=frame,"
            f"crop={width}:{height}:x='(in_w-out_w)/2+{a}*{decay}*sin(t*40)':y='(in_h-out_h)/2+{a}*{decay}*sin(t*53+1)'"
        )
    if effect == "rgb_split":
        s = int(round(params["shift_px"]))
        return f"rgbashift=rh={s}:bh=-{s}:enable='between(t,0,{d})'"
    raise ValueError(f"Unknown effect: {effect}")


def video_filter(
    seg: Segment, size: tuple[int, int], fps: int, transitions: TransitionsConfig, pad: bool = False
) -> str:
    width, height = size
    chain = []
    if seg.speed != 1.0 and not seg.is_photo:
        chain.append(f"setpts=PTS/{_num(seg.speed)}")
    chain.append(f"fps={fps}")
    if pad:
        # Whole frame fitted over a blurred, cropped copy of itself.
        chain.append(
            "split[a][b];"
            f"[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=30:3[bg];"
            f"[b]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
        chain.append("setsar=1")
    else:
        chain += [f"scale={width}:{height}:force_original_aspect_ratio=increase", f"crop={width}:{height}", "setsar=1"]
    length = seg.n_frames / fps
    if seg.is_photo:
        # One decoded frame, scaled once, then held for the whole shot.
        chain.append(f"tpad=stop_mode=clone:stop_duration={_num(length + SOURCE_OVERSHOOT_FRAMES / fps)}")
        chain.append(f"fps={fps}")
    if seg.transition:
        chain.append(effect_filter(seg.transition, transitions.params[seg.transition], effect_duration(seg.transition, transitions, length), width, height))
    return ",".join(chain)


def _video_codec_args(cfg: RenderConfig, fps: int) -> list[str]:
    gop = str(GOP_SECONDS * fps)
    if cfg.encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc", "-rc", "vbr", "-cq", str(cfg.crf), "-b:v", "0",
            "-preset", NVENC_PRESETS[cfg.preset], "-pix_fmt", "yuv420p", "-g", gop, "-no-scenecut", "1",
        ]  # fmt: skip
    return [
        "-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset, "-pix_fmt", "yuv420p",
        "-g", gop, "-keyint_min", gop, "-sc_threshold", "0",
    ]  # fmt: skip


def source_window(seg: Segment, fps: int) -> float:
    return seg.n_frames / fps * seg.speed


def video_segment_args(
    ffmpeg: str,
    seg: Segment,
    out: Path,
    size: tuple[int, int],
    config: AppConfig,
    transitions: TransitionsConfig,
    pad: bool = False,
) -> list[str]:
    fps = config.video.fps
    if seg.is_photo:
        source = ["-framerate", str(fps), "-i", str(seg.source)]
    else:
        read_s = source_window(seg, fps) + SOURCE_OVERSHOOT_FRAMES / fps * seg.speed
        source = ["-ss", _num(seg.src_in), "-t", _num(read_s), "-i", str(seg.source)]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        *source,
        "-vf", video_filter(seg, size, fps, transitions, pad),
        "-frames:v", str(seg.n_frames), "-an",
        *_video_codec_args(config.render, fps),
        "-video_track_timescale", str(fps * 1000),
        str(out),
    ]  # fmt: skip


def audio_segment_args(ffmpeg: str, seg: Segment, out: Path, fps: int, has_audio: bool) -> list[str]:
    """PCM of exactly the segment's duration; silence when the source has no audio or is sped up."""
    dur = seg.n_frames / fps
    fade = min(AUDIO_FADE_S, dur / 2)
    shape = (
        f"aresample={AUDIO_RATE},aformat=sample_fmts=s16:channel_layouts=stereo,"
        f"apad=whole_dur={_num(dur)},atrim=end={_num(dur)},"
        f"afade=t=in:st=0:d={_num(fade)},afade=t=out:st={_num(dur - fade)}:d={_num(fade)}"
    )
    if has_audio and seg.speed == 1.0:
        src = ["-ss", _num(seg.src_in), "-t", _num(dur), "-i", str(seg.source), "-vn"]
    else:
        src = ["-f", "lavfi", "-t", _num(dur), "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *src,
        "-af", shape, "-ar", str(AUDIO_RATE), "-ac", "2", "-c:a", "pcm_s16le", "-t", _num(dur), str(out),
    ]  # fmt: skip


def concat_list_text(files: list[Path], durations: list[float] | None = None) -> str:
    # Names are generated (seg_000001.ext), so no quoting/escaping is needed; paths are relative to the list file.
    # Explicit durations stop the demuxer from using container durations, which are slightly longer than
    # the frame count (B-frame delay) and would shift every later cut away from the beat.
    lines = []
    for i, f in enumerate(files):
        lines.append(f"file '{f.name}'\n")
        if durations is not None:
            lines.append(f"duration {durations[i]:.6f}\n")
    return "".join(lines)


def music_fade_filter(duration: float, fade_in: float, fade_out: float) -> str:
    """afade chain for the music; fades are shrunk proportionally when they would overlap."""
    fade_in, fade_out = max(0.0, fade_in), max(0.0, fade_out)
    total = fade_in + fade_out
    if total > duration > 0:
        fade_in, fade_out = fade_in * duration / total, fade_out * duration / total
    parts = []
    if fade_in > 0:
        parts.append(f"afade=t=in:st=0:d={_num(fade_in)}")
    if fade_out > 0:
        parts.append(f"afade=t=out:st={_num(duration - fade_out)}:d={_num(fade_out)}")
    return ",".join(parts)


def mux_args(
    ffmpeg: str,
    video_list: Path,
    music: Path,
    output: Path,
    duration: float,
    source_audio: Path | None,
    gain: float,
    music_offset: float = 0.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> list[str]:
    music_in = ["-ss", _num(music_offset)] if music_offset > 0 else []
    args = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(video_list), *music_in, "-i", str(music),
    ]  # fmt: skip
    fades = music_fade_filter(duration, fade_in, fade_out)
    if source_audio is None:
        audio_map = ["-map", "0:v:0", "-map", "1:a:0"]
        if fades:
            args += ["-af", fades]
    else:
        args += ["-i", str(source_audio)]
        music_chain = f"[1:a]aresample={AUDIO_RATE},volume=1.0" + (f",{fades}" if fades else "")
        mix = (
            f"{music_chain}[m];[2:a]aresample={AUDIO_RATE},volume={_num(gain)}[s];"
            "[m][s]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[a]"
        )
        args += ["-filter_complex", mix]
        audio_map = ["-map", "0:v:0", "-map", "[a]"]
    return [
        *args, *audio_map, "-c:v", "copy", "-c:a", "aac", "-b:a", AAC_BITRATE,
        "-t", _num(duration), "-movflags", "+faststart", "-f", "mp4", str(output),
    ]  # fmt: skip


def probe_has_audio(ffmpeg: str, path: Path) -> bool:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)], shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    return re.search(r"Stream #\d+:\d+.*: Audio:", result.stderr) is not None


class FFmpegRenderer:
    def __init__(
        self, ffmpeg: str, transitions: TransitionsConfig, size: tuple[int, int], pad_sources: frozenset[Path] = frozenset()
    ) -> None:
        self.ffmpeg = ffmpeg
        self.transitions = transitions
        self.size = size
        self.pad_sources = pad_sources

    def render(
        self, segments: list[Segment], audio_path: Path, output_path: Path, config: AppConfig, audio_offset: float = 0.0
    ) -> None:
        if not segments:
            raise HyperlapseError("Nothing to render: the plan has no segments")
        fps = config.video.fps
        duration = segments[-1].t_end
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="render_", dir=temp_root()))
        part = output_path.with_name(output_path.name + ".part")
        try:
            video_files = [work / f"seg_{i:06d}.mp4" for i in range(len(segments))]
            jobs: list[Callable[[], None]] = [
                (lambda s=s, f=f: run_ffmpeg(video_segment_args(
                    self.ffmpeg, s, f, self.size, config, self.transitions, s.source in self.pad_sources
                )))
                for s, f in zip(segments, video_files, strict=True)
            ]
            source_audio: Path | None = None
            if not config.video.remove_audio:
                has_audio = {p: probe_has_audio(self.ffmpeg, p) for p in {s.source for s in segments if not s.is_photo}}
                audio_files = [work / f"seg_{i:06d}.wav" for i in range(len(segments))]
                jobs += [
                    (lambda s=s, f=f: run_ffmpeg(audio_segment_args(self.ffmpeg, s, f, fps, has_audio.get(s.source, False))))
                    for s, f in zip(segments, audio_files, strict=True)
                ]
            self._run_parallel(jobs, config.render.workers)

            video_list = work / "video.txt"
            video_list.write_text(concat_list_text(video_files, [s.n_frames / fps for s in segments]), encoding="utf-8")
            if not config.video.remove_audio:
                audio_list = work / "audio.txt"
                audio_list.write_text(concat_list_text(audio_files), encoding="utf-8")
                source_audio = work / "source_audio.wav"
                run_ffmpeg([
                    self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(audio_list), "-c", "copy", str(source_audio),
                ])  # fmt: skip
            run_ffmpeg(
                mux_args(
                    self.ffmpeg, video_list, audio_path, part, duration, source_audio,
                    config.video.source_audio_gain, audio_offset,
                    config.audio.fade_in_s, config.audio.fade_out_s,
                )
            )  # fmt: skip
            os.replace(part, output_path)
        finally:
            part.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _run_parallel(jobs: list[Callable[[], None]], workers: int) -> None:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(job) for job in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    if done % 10 == 0 or done == len(jobs):
                        logger.info("Rendered %d/%d clip(s)", done, len(jobs))
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
