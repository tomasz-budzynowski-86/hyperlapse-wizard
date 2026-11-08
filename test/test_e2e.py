import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf
import yaml

from hyperlapse.cli import main
from test_audio import click_track


@pytest.fixture
def project(tmp_path, make_video):
    audio_dir, video_dir, out_dir = tmp_path / "audio", tmp_path / "video", tmp_path / "out"
    audio_dir.mkdir()
    video_dir.mkdir()
    click_track(audio_dir / "track.wav", 4, 8, lambda bar: 0.5)
    for i in range(6):
        make_video(video_dir / f"clip{i}.mp4", "moving", w=1920, h=1080, duration=8, audio=True)
    config = {
        "paths": {"video_input": str(video_dir), "audio_input": str(audio_dir), "output": str(out_dir)},
        "audio": {"cut_interval": 3, "min_segment_s": 1},
        "planning": {"seed": 11},
        "render": {"workers": 2},
    }
    transitions = {
        "probability": 0.7,
        "min_gap_cuts": 0,
        "effects": {"flash": True, "zoom_punch": True, "dip_to_black": True, "shake": True, "rgb_split": True},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "transitions.yaml").write_text(yaml.safe_dump(transitions), encoding="utf-8")
    return tmp_path


def test_end_to_end_produces_synced_1080p_montage(project, probe):
    assert main(["--config", str(project / "config.yaml")]) == 0

    out = project / "out" / "montage.mp4"
    info = probe(out)
    audio_len = sf.info(project / "audio" / "track.wav").duration
    assert (info["width"], info["height"]) == (1920, 1080)
    assert info["fps"] == pytest.approx(30)
    assert abs(info["video_s"] - audio_len) <= 1 / 30 + 1e-3
    assert abs(info["video_time_s"] - audio_len) <= 1 / 30 + 1e-3
    assert abs(info["audio_s"] - audio_len) <= 1 / 30 + 0.03


def test_end_to_end_with_source_audio_mix(project, probe):
    cfg = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    cfg["video"] = {"remove_audio": False}
    (project / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["--config", str(project / "config.yaml")]) == 0
    info = probe(project / "out" / "montage.mp4")
    assert info["audio_s"] is not None and info["frames"] > 0


def test_missing_input_exits_nonzero_without_output(project, caplog):
    for f in (project / "video").iterdir():
        f.unlink()
    assert main(["--config", str(project / "config.yaml")]) == 1
    assert "No video files" in caplog.text
    assert not (project / "out" / "montage.mp4").exists()


def test_invalid_config_exits_nonzero(project, caplog):
    (project / "config.yaml").write_text("beat_sensitivity: 1.0\n", encoding="utf-8")
    assert main(["--config", str(project / "config.yaml")]) == 1
    assert "beat_sensitivity" in caplog.text


def test_shortage_exits_nonzero_with_explanation(project, caplog):
    for f in sorted((project / "video").iterdir())[1:]:
        f.unlink()
    assert main(["--config", str(project / "config.yaml")]) == 1
    assert "Not enough usable source material" in caplog.text
    assert not (project / "out" / "montage.mp4").exists()


def test_main_py_wrapper_reports_error_exit_code(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "main.py"), "--config", str(tmp_path / "missing.yaml")],
        capture_output=True, text=True, shell=False, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 1
    assert "missing.yaml" in result.stderr
