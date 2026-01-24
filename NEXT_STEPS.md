# Cat Detector - Next Steps for Development

## Project Status: Ready for Use

The cat-detector application is fully implemented with real camera support and a web dashboard. All 62 unit tests + 9 integration tests pass. The application can detect cats using a webcam and stream live video with bounding boxes.

## What's Been Built

### Core Architecture
```
src/
├── main.rs      - CLI entry point (clap-based)
├── lib.rs       - Library exports
├── config.rs    - TOML configuration parsing
├── camera.rs    - CameraCapture trait + V4L2Camera + StubCamera
├── detector.rs  - CatDetector trait + OnnxDetector (YOLOX)
├── tracker.rs   - Hysteresis-based enter/exit detection
├── storage.rs   - ImageStorage trait + FileSystemStorage
├── notifier.rs  - Notifier trait + SignalNotifier
├── app.rs       - Main detection loop with dependency injection
├── service.rs   - Systemd service management
└── web.rs       - Web dashboard with live MJPEG stream (feature: web)

tests/
└── integration_test.rs - Real model integration tests
```

### Key Features
1. **YOLOX-S Detection** - 90%+ confidence on clear cat images, ~100ms inference
2. **Real Camera Support** - V4L2 capture via pure-Rust `v4l` crate
3. **Hysteresis Tracking** - Prevents false triggers from brief detections
4. **Signal Notifications** - Alert when cat enters/exits via signal-cli
5. **Systemd Integration** - Run as a background service
6. **Web Dashboard** - Live MJPEG stream with bounding boxes, status API, captures list (feature: `web`)

## Quick Start

### 1. Download ONNX Runtime (v1.23+)
```bash
mkdir -p onnxruntime && cd onnxruntime
curl -sL https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz | tar xz --strip-components=1
cd ..
export ORT_DYLIB_PATH=$PWD/onnxruntime/lib/libonnxruntime.so
```

### 2. Download YOLOX Model
```bash
mkdir -p models
wget -O models/yolox_s.onnx "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_s.onnx"
```

### 3. Build with Camera Support
```bash
cargo build --release --features real-camera
```

### 4. Test Camera
```bash
./target/release/cat-detector test-camera -o test_frame.jpg
```

### 5. Test Detection on Image
```bash
./target/release/cat-detector test-image path/to/image.jpg
```

### 6. Run the Daemon
```bash
cp config.example.toml config.toml
# Edit config.toml as needed
./target/release/cat-detector run --config config.toml
```

## What Needs Work

### Medium Priority

1. **Signal-CLI Integration Testing**
   - `SignalNotifier` shells out to `signal-cli`
   - Needs testing with actual signal-cli installation
   - Consider adding setup verification command

2. **Systemd Service Testing**
   - `service.rs` generates service file but untested
   - Needs testing on actual Linux system with systemd
   - May need user/group configuration

3. **Error Recovery**
   - App continues on camera/detector errors but could be more robust
   - Consider exponential backoff on repeated failures

### Low Priority / Nice to Have

4. **Multiple Cat Tracking**
   - Current implementation just detects "cat present"
   - Could track individual cats with bounding boxes
   - Count distinct cats

5. **Cloud Storage**
   - Add S3/GCS upload option
   - Implement `ImageStorage` trait for cloud

6. **Raspberry Pi Support**
   - Test with `yolox_tiny.onnx` for faster inference
   - May need ARM-specific ONNX Runtime build

## Model Options

| Model | Size | Input | Inference | Use Case |
|-------|------|-------|-----------|----------|
| yolox_s.onnx | 35MB | 640x640 | ~100ms | Desktop (default) |
| yolox_tiny.onnx | 20MB | 416x416 | ~30ms | Raspberry Pi |

Download from: https://huggingface.co/hr16/yolox-onnx

## Test Coverage

| Module | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| config.rs | 7 | - |
| camera.rs | 4 | - |
| detector.rs | 7 | 9 (with real model) |
| tracker.rs | 12 | - |
| storage.rs | 5 | - |
| notifier.rs | 8 | - |
| app.rs | 5 | - |
| service.rs | 2 | - |
| web.rs | 12 | - |
| **Total** | **62** | **9** |

Run tests:
```bash
# Unit tests (no dependencies)
cargo test --lib

# Integration tests (needs model + ONNX Runtime)
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so cargo test --test integration_test
```

## Configuration

See `config.example.toml` for all options. Key settings:

```toml
[camera]
device_path = "/dev/video0"
frame_width = 640
frame_height = 480

[detector]
model_path = "models/yolox_s.onnx"
input_size = 640
confidence_threshold = 0.5

[tracking]
enter_threshold = 3      # Consecutive detections to confirm entry
exit_threshold = 5       # Consecutive non-detections to confirm exit
sample_interval_secs = 10

[notification]
enabled = false
# recipient = "+1234567890"
```

## Architecture Notes

### Detection Flow
```
Camera Frame -> YOLOX Inference -> Cat Detected? -> Tracker State Machine
                                                          |
                                   +----------------------+----------------------+
                                   v                      v                      v
                             CatEntered              SampleDue              CatExited
                                   |                      |                      |
                             Save Image              Save Image              Save Image
                             + Notify                                        + Notify
```

### Hysteresis Logic
- **Enter**: Requires N consecutive detections (default: 3)
- **Exit**: Requires M consecutive non-detections (default: 5)
- **Sample**: Capture image every S seconds while cat present (default: 10s)

This prevents:
- Brief detections (cat walking past quickly) from triggering
- Brief occlusions (cat behind furniture momentarily) from triggering exit
