# Hyperlapse Wizard

Builds a beat-synced video montage from a folder of clips and one music track.

- cuts land on bars/beats of the music (the cut rate follows the energy of each section, drops force a cut)
- no footage is ever used twice, and static or scene-cut parts of the sources are avoided
- optional random beat-aligned transitions (flash, zoom punch, dip to black, shake, RGB split)
- output: `output/montage.mp4` (name configurable), 1920x1080 (landscape) or 1080x1920 (portrait), 30 fps, same length as the audio (or the chosen fragment)

## Requirements

- Python 3.11+
- FFmpeg: taken from `PATH`, otherwise from the bundled `imageio-ffmpeg` package (no manual install needed)

## Run locally

```bash
python -m venv .venv
# Windows:      .\.venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Put one audio file (`.mp3`/`.wav`; the newest by modification time is used) in `input/audio` and the clips (`.mp4`/`.mov`) in `input/video` (optionally photos in `input/photos`), then run:

```bash
python main.py                       # uses ./config.yaml and ./transitions.yaml
python main.py --config my.yaml -v   # custom config, debug logging
```

On a fatal problem (missing input, invalid config, not enough footage, FFmpeg failure) the program prints an explanation and exits with a non-zero code.

Tests: `pytest`.

## Run with Docker

```bash
docker build -t hyperlapse-wizard .
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" hyperlapse-wizard
```

The container runs as a non-root user (uid 1000); the mounted `output` directory must be writable by it.

## Configuration (`config.yaml`)

Unknown keys and wrong types/ranges are rejected with a message naming the offending key (this includes the removed `beat_sensitivity`; `video.orientation` and `video.mismatch` moved to `global.`). Relative paths are resolved against the directory of the config file.

| Key | Values | Meaning |
|---|---|---|
| `paths.video_input` / `photo_input` / `audio_input` / `output` | paths | input and output folders; `photo_input` (default `input/photos`) is optional |
| `paths.output_name` | file name | name of the result inside `output`; must end with `.mp4` (appended when missing), no folders |
| `audio.beats_per_bar` | `auto`, `3`, `4` | meter; `auto` picks 3/4 or 4/4 from the low-frequency accents (4 when unclear) |
| `audio.cut_interval` | `1`..`3` | `1` = most frequent cuts, `3` = least frequent |
| `audio.min_segment_s` | int >= 0 | minimum shot length; `0` = none |
| `audio.max_segment_s` | int or `null` | upper bound for the regular cut period |
| `audio.fixed_time` | `"0:00"`, `"1:32-2:15"`, `"1:32-"` | part of the song to use; `0:00` = whole file. Always quote it (`m:ss` or `h:mm:ss`). The montage is as long as the fragment and the beat analysis covers only the fragment |
| `audio.fade_in_s` / `audio.fade_out_s` | seconds >= 0 | music fade-in at the start / fade-out at the end (default 3); `0` = off. Applies to the music only; shrunk proportionally if the montage is shorter than both |
| `global.orientation` | `auto`, `landscape`, `portrait` | montage format for videos and photos; `auto` follows the majority of all sources (a tie is an error) |
| `global.mismatch` | `crop` (default), `blur_pad`, `skip` | videos/photos in the other orientation: `crop` fills the frame and cuts the overflow, `blur_pad` fits the whole image over a blurred copy of itself, `skip` drops them |
| `global.video_to_photo_ratio` | `random` (default), `proportional`, `"2:5"` | see "Photos" below; `a:b` = `a` video shots for every `b` photo shots (each 1..9) |
| `photos.time_shown_min` / `time_shown_max` | seconds > 0 | how long one photo stays on screen (defaults 1 and 3) |
| `video.fps` | int | output frame rate |
| `video.remove_audio` | bool | `true`: only the music; `false`: mix the clips' own sound under the music (photos are silent) |
| `video.source_audio_gain` | 0..1 | clip sound level when mixed (music = 1.0) |
| `video.allow_segment_reusage` | bool (default `false`) | when the unused footage runs out, start another pass over the sources and reuse already used ranges (and photos) instead of failing |
| `video.allow_segment_extension` | bool (default `false`) | when no free range is long enough for a segment, use a shorter range and stretch it with slow motion (0.75x, then 0.5x); the clip's own sound is muted for slowed segments |
| `video.chronologic` | bool | `true`: videos and photos follow the order of their names (see below); `false`: random order |
| `motion.step_s`, `motion.window_s` | seconds | motion sampling step and smoothing window |
| `motion.min_score` | `auto` or float | minimum motion; `auto` = 40th percentile of each file, never below 0.002 |
| `planning.seed` | int or `null` | fixed seed gives a reproducible montage (the used seed is logged) |
| `planning.allow_speedup`, `planning.max_speedup` | bool, float >= 1 | accepted but currently unused (see below) |
| `render.encoder` | `libx264`, `h264_nvenc` | video encoder |
| `render.crf`, `render.preset` | int 0..51, x264 preset | quality / speed |
| `render.workers` | `auto` or int | parallel FFmpeg jobs; `auto` = `max(1, min(cpu_count // 2, 6))` |

### How cuts are chosen

Cut period (in beats) comes from a ladder depending on the meter (4/4: 2, 4, 8, 16; 3/4: 3, 6, 12, 12) and from the energy class of the section (low/mid/high) and `cut_interval`:

| cut_interval | high | mid | low |
|---|---|---|---|
| 1 | 1st | 1st | 2nd |
| 2 | 1st | 2nd | 3rd |
| 3 | 2nd | 3rd | 4th |

If the period is shorter than `min_segment_s` it moves up the ladder; if it exceeds `max_segment_s` it moves down. Cuts never cross a section boundary, a leftover shorter than the minimum is merged into the neighbouring shot, the intro before the first bar is part of the first shot and the last shot ends with the audio.

### Transitions (`transitions.yaml`)

```yaml
enabled: true
probability: 0.4      # chance of an effect at each cut
min_gap_cuts: 1       # plain cuts required between two effects
effects:              # true = used automatically at random cuts
  flash: false
  zoom_punch: false
  dip_to_black: false
  shake: false
  rgb_split: false
params:               # per-effect settings, all optional
  flash: { duration_s: 0.12, strength: 0.8 }
  zoom_punch: { duration_s: 0.2, scale: 1.15 }
  dip_to_black: { duration_s: 0.15 }
  shake: { duration_s: 0.2, amplitude_px: 12 }
  rgb_split: { duration_s: 0.15, shift_px: 8 }
```

Effects are applied at the start of the shot following a cut; the same effect is never used twice in a row and its duration is capped at 30% of the shot. With every effect `false` (or `enabled: false`) the montage has hard cuts only. Cross-fades are deliberately not offered because they blur the rhythm. An unknown effect name is an error.

### No repeated footage and the capacity check

For every source the planner keeps a list of free time ranges (the moving parts without scene cuts). Each used range, plus a 0.5 s margin on both sides, is removed from it, so no frame range is reused. Sources are drawn at random, weighted by their remaining free time, and two consecutive shots never come from the same file (unless there is only one source). Within a source, up to 30 random start positions are tried and the first with enough motion is taken.

Before rendering, the total free time is compared with the montage length (plus the safety margins). If the footage is insufficient the program stops with a report of the missing seconds instead of repeating frames. Add footage or lower `motion.min_score`.

### Photos (`input/photos`)

Optional: put `.jpg`/`.jpeg`/`.png`/`.webp` files in `input/photos` (EXIF rotation is respected). Photos use the same beat slots as the clips: some slots become photo slots, the rest stay video. Transitions apply to photos exactly as to clips; photos are silent.

- `photos.time_shown_min` / `time_shown_max`: a photo slot longer than the maximum is split into several photos (cuts snap to beats when that keeps every photo within the limits). A slot too short for the minimum stays a video slot.
- `global.video_to_photo_ratio`:
  - `random` (default): every slot independently becomes a photo slot with a probability equal to the photo share `photos / (photos + available clips)`.
  - `proportional`: the planned shot ratio equals the ratio of available photos to available clips (usable footage divided by the mean slot length).
  - `"a:b"` (e.g. `"2:5"`): planned ratio of video shots to photo shots. A slot split into several photos counts as several shots, so the real ratio is approximate.
- Without `video.allow_segment_reusage` every photo is shown at most once (the surplus slots stay video); with it, photos are reused after all have been shown.
- No photos folder or no files: the montage is video-only and the ratio is ignored.

### Chronological order (`video.chronologic`)

The files are ordered by the date in their names (`DJI_20250802171205_0093_D`, `IMG_20240615_143012`, `PXL_20240615_...`, `VID-20240615-WA0001`, `2024-06-15 14.30.12`), or else by a manual number at the start or end of the name (`01_beach`, `clip3`). Dates and numbers are never mixed: if not every file (videos and photos together) has a date and not every file has a number, a warning is logged and the order stays random.

The montage is split into consecutive blocks, one per file, sized by the usable footage of each file; inside a file the shots also move forward in time. The "no two consecutive shots from the same file" rule is switched off in this mode.

Photos are shown in date order and placed after the footage of the videos recorded before them; the exact position is approximate because it follows the beat slots and the ratio. With few slots for photos, the later photos may not be used.

## Project layout

```
main.py              thin wrapper around hyperlapse.cli
config.yaml          main settings
transitions.yaml     transition effects
src/hyperlapse/      models, config, audio, motion, photos, planner, transitions, render, cli, utils
test/                pytest suite (synthetic media only)
```

Temporary files and the motion-analysis cache live in the system temp directory (`hyperlapse_wizard/`), not in the project.

## Logs

Every run appends a full DEBUG log (including complete FFmpeg errors and tracebacks of unexpected failures) to `logs/hyperlapse.log` next to `config.yaml` (rotated, 5 × 5 MB). The console shows INFO, or DEBUG with `-v`.

## Known limitations

- The motion analysis decodes every frame of every source; the first run over many 4K files (especially 120 fps) can take hours with only per-file progress messages (results are cached, so later runs are fast).
- Fast hyperlapse-style footage can trigger the fixed scene-cut threshold (score >= 0.6) and split good intervals.
- `planning.allow_speedup` / `max_speedup` are validated but not used: a higher speed would consume more source footage, so it cannot compensate for missing material. Shots always play at 1.0x (the renderer supports other speeds).
- Meter and drop detection are heuristics; ambiguous material falls back to 4/4.
- `h264_nvenc` is implemented but not covered by automated tests.
