---
sidebar_position: 1
slug: /
title: Introduction
---

# Cat Detector

A Rust application that monitors a USB webcam for cats using CLIP ViT-B/32 zero-shot classification via ONNX Runtime. It saves images on cat entry/exit, records video of visits, sends Signal notifications, and serves a web dashboard with live MJPEG stream and session browsing.

## Features

- **CLIP zero-shot detection** — no training required, works from any camera angle
- **Live web dashboard** — MJPEG stream, session browsing, system info
- **Video recording** — H.264 MP4 of each cat visit via FFmpeg
- **Signal notifications** — text + video attachments via signal-cli
- **Session tracking** — JSON-persisted visit history with entry/exit images
- **Systemd integration** — install as a system service with auto-start

## Architecture

13 modules with trait-based dependency injection:

| Module | Purpose |
|---|---|
| `main.rs` | CLI (clap), daemon loop (30fps capture, periodic detection), web server launch |
| `app.rs` | Generic `App<C,D,S,N>` orchestrating the detection loop |
| `config.rs` | TOML config parsing with defaults and validation |
| `camera.rs` | `CameraCapture` trait + V4L2Camera (auto-detect) / StubCamera |
| `detector.rs` | `CatDetector` trait + ClipDetector (zero-shot classification) |
| `tracker.rs` | Hysteresis state machine (Absent/Present) |
| `storage.rs` | `ImageStorage` trait + FileSystemStorage |
| `notifier.rs` | `Notifier` trait + SignalNotifier (signal-cli with linked device support) |
| `recorder.rs` | `VideoRecorder` trait + FfmpegRecorder (pipes raw frames to ffmpeg) |
| `session.rs` | `CatSession` model + `SessionManager` (JSON persistence) |
| `service.rs` | Systemd service install/uninstall |
| `web.rs` | Axum dashboard, MJPEG stream, session list/detail pages |
| `lib.rs` | Library exports |

## Quick Start

```bash
# Build
cargo build --release --features real-camera,web

# Run
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so ./target/release/cat-detector run --config config.toml
```

See the [Deployment Guide](./deployment.md) for full setup instructions.
