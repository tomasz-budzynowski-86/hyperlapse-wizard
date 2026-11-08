from pathlib import Path

import pytest
import yaml

from hyperlapse.config import load_config, load_transitions, parse_config, parse_transitions
from hyperlapse.models import ConfigError
from hyperlapse.utils import chronological_order, list_photos, list_videos, pick_audio

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base() -> dict:
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_valid_default_config(tmp_path):
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.audio.beats_per_bar is None
    assert cfg.audio.cut_interval == 2
    assert cfg.audio.min_segment_s == 1
    assert cfg.audio.max_segment_s is None
    assert cfg.motion.min_score is None
    assert cfg.render.workers >= 1
    assert cfg.paths.video_input == (REPO_ROOT / "input" / "video").resolve()

    data = _base()
    data["audio"].update(beats_per_bar=3, max_segment_s=8)
    data["motion"]["min_score"] = 0.01
    cfg = parse_config(data, tmp_path)
    assert cfg.audio.beats_per_bar == 3
    assert cfg.audio.max_segment_s == 8
    assert cfg.motion.min_score == 0.01
    assert cfg.paths.output == (tmp_path / "output").resolve()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("audio", "cut_interval", "2"),
        ("audio", "cut_interval", True),
        ("audio", "min_segment_s", 1.5),
        ("video", "fps", "30"),
        ("video", "remove_audio", "yes"),
        ("global", "orientation", "square"),
        ("global", "mismatch", "stretch"),
        ("global", "video_to_photo_ratio", "10:2"),
        ("global", "video_to_photo_ratio", "0:3"),
        ("global", "video_to_photo_ratio", "most"),
        ("photos", "time_shown_min", 0),
        ("photos", "time_shown_max", "3"),
        ("audio", "fade_in_s", -1),
        ("audio", "fade_out_s", "3"),
        ("audio", "beats_per_bar", 5),
        ("render", "encoder", "libx265"),
        ("render", "workers", 0),
    ],
)
def test_wrong_value_or_type(tmp_path, section, key, value):
    data = _base()
    data[section][key] = value
    with pytest.raises(ConfigError, match=f"{section}.{key}"):
        parse_config(data, tmp_path)


def test_unknown_key(tmp_path):
    data = _base()
    data["beat_sensitivity"] = 1.0
    with pytest.raises(ConfigError, match="beat_sensitivity"):
        parse_config(data, tmp_path)

    data = _base()
    data["audio"]["bogus"] = 1
    with pytest.raises(ConfigError, match="audio.bogus"):
        parse_config(data, tmp_path)


@pytest.mark.parametrize("value", [0, 4, -1])
def test_cut_interval_out_of_range(tmp_path, value):
    data = _base()
    data["audio"]["cut_interval"] = value
    with pytest.raises(ConfigError, match="cut_interval"):
        parse_config(data, tmp_path)


def test_negative_min_segment(tmp_path):
    data = _base()
    data["audio"]["min_segment_s"] = -1
    with pytest.raises(ConfigError, match="min_segment_s"):
        parse_config(data, tmp_path)


def test_min_greater_than_max(tmp_path):
    data = _base()
    data["audio"].update(min_segment_s=5, max_segment_s=3)
    with pytest.raises(ConfigError, match="min_segment_s"):
        parse_config(data, tmp_path)


def test_load_config_reports_yaml_error(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("audio: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad)


def test_transitions_default_file():
    tr = load_transitions(REPO_ROOT / "transitions.yaml")
    assert tr.enabled
    assert tr.active_effects == []
    assert tr.params["flash"]["duration_s"] == 0.12


def test_transitions_active_effects():
    tr = parse_transitions({"effects": {"shake": True, "flash": True}})
    assert tr.active_effects == ["flash", "shake"]
    assert parse_transitions({"enabled": False, "effects": {"flash": True}}).active_effects == []


def test_transitions_unknown_effect_is_error():
    with pytest.raises(ConfigError, match="glitch"):
        parse_transitions({"effects": {"glitch": True}})
    with pytest.raises(ConfigError, match="params.flash.bogus"):
        parse_transitions({"params": {"flash": {"bogus": 1}}})


def test_transitions_value_validation():
    with pytest.raises(ConfigError, match="probability"):
        parse_transitions({"probability": 1.5})
    with pytest.raises(ConfigError, match="effects.flash"):
        parse_transitions({"effects": {"flash": "yes"}})


def test_missing_transitions_file_means_hard_cuts(tmp_path):
    assert load_transitions(tmp_path / "nope.yaml").active_effects == []


def test_pick_audio_newest_and_case_insensitive(tmp_path):
    import os

    old, new = tmp_path / "old.mp3", tmp_path / "NEW.WAV"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    (tmp_path / "ignored.txt").write_bytes(b"x")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    assert pick_audio(tmp_path) == new


def test_missing_inputs_are_errors(tmp_path):
    from hyperlapse.models import HyperlapseError

    with pytest.raises(HyperlapseError):
        pick_audio(tmp_path)
    with pytest.raises(HyperlapseError):
        list_videos(tmp_path)
    (tmp_path / "a.MOV").write_bytes(b"x")
    (tmp_path / "b.mp4").write_bytes(b"x")
    assert [p.name for p in list_videos(tmp_path)] == ["a.MOV", "b.mp4"]


def test_new_option_defaults():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.audio.fixed_time is None
    assert cfg.paths.output_name == "montage.mp4"
    assert cfg.video.chronologic is False
    assert parse_config({}, REPO_ROOT).audio.fixed_time is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0:00", None),
        (None, None),
        (0, None),
        ("0:00-", None),
        ("1:32-2:15", (92.0, 135.0)),
        ("1:32 \u2013 2:15", (92.0, 135.0)),
        ("1:32-", (92.0, None)),
        ("-0:30", (0.0, 30.0)),
        ("1:02:03-1:02:10.5", (3723.0, 3730.5)),
        ("90-120", (90.0, 120.0)),
    ],
)
def test_fixed_time_parsing(tmp_path, raw, expected):
    cfg = parse_config({"audio": {"fixed_time": raw}}, tmp_path)
    assert cfg.audio.fixed_time == expected


@pytest.mark.parametrize("raw", ["2:15-1:32", "1:32-1:32", "1:32", "1:75-2:00", "a-b", "1-2-3", 92, 1.5, True, ["0:10"]])
def test_fixed_time_invalid(tmp_path, raw):
    with pytest.raises(ConfigError, match="audio.fixed_time"):
        parse_config({"audio": {"fixed_time": raw}}, tmp_path)


@pytest.mark.parametrize(("raw", "expected"), [("clip.mp4", "clip.mp4"), ("clip", "clip.mp4"), (" My Clip.MP4 ", "My Clip.MP4")])
def test_output_name(tmp_path, raw, expected):
    assert parse_config({"paths": {"output_name": raw}}, tmp_path).paths.output_name == expected


@pytest.mark.parametrize("raw", ["", "  ", "..", "../x.mp4", "a/b.mp4", "a\\b.mp4", "C:x.mp4", "clip.mov", "bad?.mp4", 5, None])
def test_output_name_invalid(tmp_path, raw):
    with pytest.raises(ConfigError, match="paths.output_name"):
        parse_config({"paths": {"output_name": raw}}, tmp_path)


def test_chronologic_must_be_bool(tmp_path):
    assert parse_config({"video": {"chronologic": True}}, tmp_path).video.chronologic is True
    with pytest.raises(ConfigError, match="video.chronologic"):
        parse_config({"video": {"chronologic": "yes"}}, tmp_path)


def _names(*names: str) -> list[Path]:
    return [Path(n) for n in names]


def test_chronological_order_by_date():
    videos = _names("IMG_20240615_143012.mp4", "PXL_20240101_090000123.mp4", "2024-06-15 14.30.05.mov", "VID-20231231-WA0002.mp4")
    assert [p.name for p in chronological_order(videos)] == [
        "VID-20231231-WA0002.mp4",
        "PXL_20240101_090000123.mp4",
        "2024-06-15 14.30.05.mov",
        "IMG_20240615_143012.mp4",
    ]


def test_chronological_order_by_manual_number():
    videos = _names("10_end.mp4", "2_middle.mp4", "1_start.mp4")
    assert [p.name for p in chronological_order(videos)] == ["1_start.mp4", "2_middle.mp4", "10_end.mp4"]
    assert [p.name for p in chronological_order(_names("clip10.mp4", "clip9.mp4"))] == ["clip9.mp4", "clip10.mp4"]


def test_chronological_order_falls_back_when_incomplete(caplog):
    assert chronological_order(_names("IMG_20240615_143012.mp4", "holiday.mp4")) is None
    assert "random order" in caplog.text


def test_chronological_order_dji_names():
    names = _names(
        "DJI_20250802171205_0093_D.MP4",
        "DJI_20250802171204_0095_D.MP4",
        "DJI_20250801235959_0100_D.MP4",
        "DJI_20250802171205_0092_D.JPG",
    )
    assert [p.name for p in chronological_order(names)] == [
        "DJI_20250801235959_0100_D.MP4",
        "DJI_20250802171204_0095_D.MP4",
        "DJI_20250802171205_0092_D.JPG",
        "DJI_20250802171205_0093_D.MP4",
    ]


def test_photos_and_global_defaults(tmp_path):
    cfg = parse_config({}, tmp_path)
    assert (cfg.global_.orientation, cfg.global_.mismatch, cfg.global_.video_to_photo_ratio) == ("auto", "crop", "random")
    assert (cfg.photos.time_shown_min, cfg.photos.time_shown_max) == (1.0, 3.0)
    assert cfg.paths.photo_input == (tmp_path / "input" / "photos").resolve()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("random", "random"), ("proportional", "proportional"), ("2:5", (2, 5)), ("9:1", (9, 1)), (125, (2, 5)), (69, (1, 9))],
)
def test_video_to_photo_ratio(tmp_path, raw, expected):
    cfg = parse_config({"global": {"video_to_photo_ratio": raw}}, tmp_path)
    assert cfg.global_.video_to_photo_ratio == expected


def test_photo_time_range_must_be_ordered(tmp_path):
    with pytest.raises(ConfigError, match="time_shown_min"):
        parse_config({"photos": {"time_shown_min": 4, "time_shown_max": 2}}, tmp_path)


@pytest.mark.parametrize("key", ["orientation", "mismatch"])
def test_orientation_and_mismatch_moved_to_global(tmp_path, key):
    with pytest.raises(ConfigError, match=f"global.{key}"):
        parse_config({"video": {key: "crop"}}, tmp_path)


def test_list_photos_is_optional(tmp_path):
    assert list_photos(None) == []
    assert list_photos(tmp_path / "missing") == []
    (tmp_path / "b.JPG").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    assert [p.name for p in list_photos(tmp_path)] == ["a.png", "b.JPG"]
