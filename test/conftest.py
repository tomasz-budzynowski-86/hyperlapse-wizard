import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from hyperlapse.models import HyperlapseError
from hyperlapse.utils import find_ffmpeg

try:
    FFMPEG: str | None = find_ffmpeg()
except HyperlapseError:
    FFMPEG = None

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="FFmpeg not available")


def _make_video(
    path: Path, kind: str = "moving", w: int = 640, h: int = 360, duration: float = 6.0, fps: int = 30, audio: bool = False
) -> Path:
    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    if kind == "moving":
        cmd += ["-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={duration}"]
    elif kind == "static":
        cmd += ["-f", "lavfi", "-i", f"color=c=black:size={w}x{h}:rate={fps}:duration={duration}"]
    elif kind == "cut":
        half = duration / 2
        cmd += [
            "-f", "lavfi", "-i", f"color=c=white:size={w}x{h}:rate={fps}:duration={half}",
            "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={half}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        ]  # fmt: skip
    else:
        raise ValueError(kind)
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "30", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, shell=False)
    return path


@pytest.fixture
def make_video():
    if FFMPEG is None:
        pytest.skip("FFmpeg not available")
    return _make_video


def _make_photo(path: Path, w: int = 1920, h: int = 1080, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    image = np.repeat(np.repeat(rng.integers(0, 255, (h // 40 + 1, w // 40 + 1, 3), dtype=np.uint8), 40, 0), 40, 1)
    cv2.imwrite(str(path), image[:h, :w])
    return path


@pytest.fixture
def make_photo():
    return _make_photo


def _probe(path: Path) -> dict:
    """Size/fps/frame count via OpenCV; audio length by decoding with FFmpeg (no ffprobe available)."""
    cap = cv2.VideoCapture(str(path))
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()
    info["video_s"] = info["frames"] / info["fps"]
    res = subprocess.run(
        [FFMPEG, "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True, shell=False
    )
    info["video_time_s"] = _last_time(res.stderr)
    res = subprocess.run(
        [FFMPEG, "-i", str(path), "-map", "0:a:0", "-f", "null", "-"], capture_output=True, text=True, shell=False
    )
    info["audio_s"] = _last_time(res.stderr)
    return info


def _last_time(stderr: str) -> float | None:
    times = re.findall(r"time=(\d+):(\d+):([\d.]+)", stderr)
    if not times:
        return None
    return int(times[-1][0]) * 3600 + int(times[-1][1]) * 60 + float(times[-1][2])


@pytest.fixture
def probe():
    if FFMPEG is None:
        pytest.skip("FFmpeg not available")
    return _probe
