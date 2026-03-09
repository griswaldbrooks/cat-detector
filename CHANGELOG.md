# Changelog

All notable changes to cat-detector will be documented in this file.

## [1.0.0] - 2026-03-08

First stable release. Cat-detector is deployed and running on catbox (Dell Optiplex 3040M), monitoring a USB webcam for cats using CLIP zero-shot classification.

### Detection

- **CLIP ViT-B/32 zero-shot classification** — 3-class system ("a photo of a cat" / "a photo of an empty room" / "a photo of a person") using pre-computed text embeddings and cosine similarity with softmax (temperature=100)
- ONNX Runtime loaded dynamically via `ORT_DYLIB_PATH` for flexible deployment
- CLIP preprocessing pipeline: resize shortest side to 224, center crop, OpenAI CLIP normalization constants
- Person filtering — people detected with 73-95% probability, no false cat triggers
- 97%+ accuracy on overhead cat images, zero false positives on empty rooms
- Hysteresis state machine (Absent/Present) with configurable enter/exit thresholds to prevent flickering

### Camera

- V4L2 camera capture behind `real-camera` feature flag
- Auto-detect mode scans `/dev/video*` for first available capture device
- Configurable resolution (default 640x480) and frame rate (default 30fps)
- Decoupled capture/detection loop: captures at camera FPS, runs inference every `detection_interval_ms` (default 500ms)
- Exponential backoff on camera and detector errors

### Video Recording

- FFmpeg pipe-based recording: raw RGB24 frames piped to ffmpeg stdin, producing H.264 MP4 with faststart
- Wallclock timestamps (`-use_wallclock_as_timestamps 1`, `-vsync vfr`) for real-time playback speed
- Automatic recording start on cat entry, stop on cat exit
- Video files stored alongside session data

### Notifications

- Signal messenger notifications via signal-cli native binary (no Java required)
- Linked device support with `-a` account flag
- Cat entry notification (text message) and cat exit notification (with video attachment)
- Configurable timeouts: 30s for text messages, 120s for attachments
- Recipient validation and `test-notification` CLI subcommand for setup verification

### Web Dashboard

- Axum-based web server behind `web` feature flag
- Live MJPEG stream with configurable `stream_fps` throttling
- Dashboard page with real-time status indicator, recent captures grid, and system info panel
- Session list page with card layout showing entry/exit images, duration, and image counts
- Session detail page with image lightbox, key entry/exit images, sample gallery, and video playback
- REST API: `/api/status`, `/api/system-info`, `/api/captures`, `/api/sessions`, `/api/sessions/:id`, `/api/frame`, `/api/stream`
- Version displayed in dashboard header and system info panel

### Sessions

- Cat session model tracking entry time, exit time, entry/exit images, sample images, and video path
- JSON persistence in `captures/sessions/` directory
- Automatic session creation on cat entry, finalization on cat exit
- Session browsing via web dashboard

### Configuration

- TOML configuration file (`config.toml`) with sensible defaults
- Sections: camera, detector, tracking, storage, notifications, web
- Example config provided as `config.example.toml`

### CLI

- `run` — Start the detection daemon
- `capture` — Capture a single frame and run detection (requires `real-camera` feature)
- `install-service` / `uninstall-service` — Manage systemd service
- `test-notification` — Verify Signal notification setup
- `--version` flag shows version from Cargo.toml

### Deployment

- Deploy scripts: `scripts/deploy.sh` (build + rsync + restart), `scripts/remote-run.sh`, `scripts/remote-stop.sh`, `scripts/install-service.sh`
- Systemd service with auto-start on boot, running as user `griswald`
- Target: Dell Optiplex 3040M (catbox) via SSH

### Infrastructure

- Trait-based dependency injection: `CameraCapture`, `CatDetector`, `ImageStorage`, `Notifier`, `VideoRecorder`
- Generic `App<C, D, S, N>` orchestrator accepting any trait implementation
- Mock implementations for all traits, enabling comprehensive unit testing
- 87 unit tests + 15 CLIP integration tests
- GitHub Actions CI: format check, clippy lints, unit tests (Rust 1.92)
- MIT license

### Python Tooling

- `scripts/generate_clip_embeddings.py` — Regenerate text embeddings for CLIP classes
- `scripts/test_clip_detection.py` — Test CLIP detection pipeline on test images
- Managed via pixi (`pyproject.toml` + `pixi.lock`)
