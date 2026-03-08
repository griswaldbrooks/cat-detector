# Cat Detector

Rust application that monitors a USB webcam for cats using YOLO11n (ONNX). Saves images on cat entry/exit, records video of visits, sends Signal notifications, and serves a web dashboard with live MJPEG stream and session browsing.

## Quick Commands

```bash
# Build
cargo build --release --features real-camera,web
cargo clippy --all-features -- -D warnings
cargo fmt --check

# Unit tests (no external deps)
cargo test --lib

# Integration tests (needs model + ORT runtime)
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test

# Run
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so ./target/release/cat-detector
```

## Architecture

13 modules with trait-based dependency injection:

| Module | Purpose |
|---|---|
| `main.rs` | CLI (clap), daemon loop (30fps capture, periodic detection), web server launch |
| `app.rs` | Generic `App<C,D,S,N>` orchestrating the detection loop |
| `config.rs` | TOML config parsing with defaults and validation |
| `camera.rs` | `CameraCapture` trait + V4L2Camera (auto-detect) / StubCamera |
| `detector.rs` | `CatDetector` trait + ClipDetector (zero-shot) / OnnxDetector (YOLO11/YOLOX) |
| `tracker.rs` | Hysteresis state machine (Absent/Present) |
| `storage.rs` | `ImageStorage` trait + FileSystemStorage |
| `notifier.rs` | `Notifier` trait + SignalNotifier (signal-cli) |
| `recorder.rs` | `VideoRecorder` trait + FfmpegRecorder (pipes raw frames to ffmpeg) |
| `session.rs` | `CatSession` model + `SessionManager` (JSON persistence) |
| `service.rs` | Systemd service install/uninstall |
| `web.rs` | Axum dashboard, MJPEG stream, session list/detail pages, capture serving |
| `lib.rs` | Library exports |

## Key Details

- **Model**: CLIP ViT-B/32 (default, zero-shot, 224x224, 3-class: cat/room/person) or YOLO11n (640x640). Cat = COCO class 15. Also supports YOLOX via `ModelFormat` auto-detection
- **ONNX Runtime**: loaded dynamically via `ORT_DYLIB_PATH` env var
- **Cargo features**: `real-camera` (v4l2), `web` (axum), `nokhwa-camera` (unused)
- **Error handling**: `thiserror` for module errors, `anyhow` only in `main.rs`
- **Async**: tokio runtime, `async_trait` for trait definitions
- **Logging**: `tracing` crate
- **Video**: FFmpeg pipe (requires `ffmpeg` on system). Records with wallclock timestamps for real-time playback
- **Tests**: 77 unit + 9 integration; all traits have Mock* implementations
- **Deploy target**: Dell Optiplex 3040M (catbox) via `scripts/deploy.sh`

## Issue Tracking

Uses `bd` (beads) for task tracking. Use `bd` instead of markdown TODOs.

```bash
bd ready              # Unblocked tasks by priority
bd show <id>          # Details + dependencies
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

## Rules & Agents

- `.claude/rules/` - git workflow, Rust style, testing conventions
- `.claude/agents/` - coder, code-reviewer, test-runner
- `.claude/skills/` - build, test, review, deploy, capture, wrapup
