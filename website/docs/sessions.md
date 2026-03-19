---
sidebar_position: 3
title: Cat Sessions
---

# Cat Sessions

When the cat detector is running, every cat visit is recorded as a **session** — a structured record of when the cat arrived, what happened during the visit, and when it left. Sessions are the primary data model for the web dashboard's session browser.

## Session Lifecycle

A session tracks a single cat visit through four phases:

```
Cat detected (3 consecutive) → Entry
  ↓
Periodic samples (every 10s) → Samples
  ↓
Video recording (every frame) → Video
  ↓
Cat gone (5 consecutive misses) → Exit
```

1. **Entry** — The [tracker](./clip-detection.md) confirms a cat is present after `enter_threshold` consecutive detections (default: 3). A new session starts, an entry image is saved, video recording begins, and an optional Signal notification is sent.

2. **Samples** — While the cat is present, a sample image is saved every `sample_interval_secs` (default: 10 seconds). These capture different moments of the visit.

3. **Video** — Every captured frame is piped to FFmpeg for H.264 MP4 recording with wallclock timestamps, so the video plays back at real-time speed.

4. **Exit** — After `exit_threshold` consecutive non-detections (default: 5), the session ends. An exit image is saved, video recording stops, and an optional Signal notification is sent with the video attached.

## Data Model

Each session is a `CatSession` struct with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | `String` | Unique ID: `session_YYYYMMDD_HHMMSS` (based on entry time, UTC) |
| `entry_time` | `DateTime<Utc>` | When the cat was confirmed present |
| `exit_time` | `Option<DateTime<Utc>>` | When the cat was confirmed gone (`null` if still active) |
| `entry_image` | `Option<PathBuf>` | Path to the image captured at entry |
| `exit_image` | `Option<PathBuf>` | Path to the image captured at exit |
| `sample_images` | `Vec<PathBuf>` | Paths to periodic sample images during the visit |
| `video_path` | `Option<PathBuf>` | Path to the recorded MP4 video |

The `duration_secs()` method computes the visit length from `entry_time` to `exit_time`.

## Storage Format

Sessions are persisted as pretty-printed JSON files in the `captures/sessions/` directory (configurable via `storage.output_dir`):

```
captures/
├── sessions/
│   ├── session_20260318_143022.json
│   ├── session_20260318_150512.json
│   └── ...
├── cat_entry_20260318_143022.jpg
├── cat_exit_20260318_143845.jpg
├── cat_sample_20260318_143032.jpg
├── cat_sample_20260318_143042.jpg
├── cat_video_20260318_143022.mp4
└── ...
```

A typical session JSON file:

```json
{
  "id": "session_20260318_143022",
  "entry_time": "2026-03-18T14:30:22.456Z",
  "exit_time": "2026-03-18T14:38:45.123Z",
  "entry_image": "captures/cat_entry_20260318_143022.jpg",
  "exit_image": "captures/cat_exit_20260318_143845.jpg",
  "sample_images": [
    "captures/cat_sample_20260318_143032.jpg",
    "captures/cat_sample_20260318_143042.jpg",
    "captures/cat_sample_20260318_143052.jpg"
  ],
  "video_path": "captures/cat_video_20260318_143022.mp4"
}
```

## Session Manager

The `SessionManager` handles the full lifecycle:

- **`start_session(entry_time)`** — Creates a new `CatSession` and holds it as the active session.
- **`set_entry_image()` / `add_sample_image()` / `set_video_path()` / `set_exit_image()`** — Attach file paths to the active session as they are created.
- **`end_session(exit_time)`** — Finalizes the active session and writes it to disk as JSON.
- **`load_sessions()`** — Reads all session JSON files, returns them sorted newest-first.
- **`load_session(id)`** — Loads a single session by ID.
- **`delete_session(id)`** — Removes the JSON file and all associated images and video.

Only one session can be active at a time. If the cat leaves and returns, a new session is created.

## Web Dashboard

The web dashboard at `/sessions` lists all completed sessions with entry/exit times, duration, and image count. Clicking a session shows the detail page with:

- Entry and exit images
- All sample images in a timeline
- Video playback (if recorded)
- Session metadata (duration, image count)

The dashboard reads sessions via `SessionManager::load_sessions()` on each request.

## Configuration

Session behavior is controlled by these config fields:

```toml
[storage]
# Base directory for captures and sessions
output_dir = "captures"

[tracking]
# Consecutive detections to confirm entry
enter_threshold = 3
# Consecutive non-detections to confirm exit
exit_threshold = 5
# Seconds between sample images during a visit
sample_interval_secs = 10
```

## Disk Usage

Each session produces:
- **2 JPEG images** (entry + exit) — ~50-100 KB each
- **N sample images** — one per `sample_interval_secs` while the cat is present
- **1 MP4 video** — size depends on visit duration and frame rate (640x480 @ 30fps)

For a 5-minute visit with 10-second sampling, expect ~30 sample images plus entry/exit images and a video file. Over time this adds up — see the storage watchdog feature for automatic cleanup.
