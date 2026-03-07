# Cat Detector

Rust application that monitors a USB webcam for cats using YOLOX-S (ONNX). Saves images on cat entry/exit, sends Signal notifications, and serves a web dashboard with live MJPEG stream.

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

11 modules with trait-based dependency injection:

| Module | Purpose |
|---|---|
| `main.rs` | CLI (clap), daemon loop, web server launch |
| `app.rs` | Generic `App<C,D,S,N>` orchestrating the detection loop |
| `config.rs` | TOML config parsing with defaults and validation |
| `camera.rs` | `CameraCapture` trait + V4L2Camera / StubCamera |
| `detector.rs` | `CatDetector` trait + OnnxDetector (YOLOX-S, ort crate v2) |
| `tracker.rs` | Hysteresis state machine (Absent/Present) |
| `storage.rs` | `ImageStorage` trait + FileSystemStorage |
| `notifier.rs` | `Notifier` trait + SignalNotifier (signal-cli) |
| `service.rs` | Systemd service install/uninstall |
| `web.rs` | Axum web dashboard, MJPEG stream, status/captures API |
| `lib.rs` | Library exports |

## Key Details

- **Model**: YOLOX-S, 640x640 input, cat = COCO class 15
- **ONNX Runtime**: loaded dynamically via `ORT_DYLIB_PATH` env var
- **Cargo features**: `real-camera` (v4l2), `web` (axum), `nokhwa-camera` (unused)
- **Error handling**: `thiserror` for module errors, `anyhow` only in `main.rs`
- **Async**: tokio runtime, `async_trait` for trait definitions
- **Logging**: `tracing` crate
- **Tests**: 62 unit + 9 integration; all traits have Mock* implementations

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
- `.claude/skills/` - build, test, review
