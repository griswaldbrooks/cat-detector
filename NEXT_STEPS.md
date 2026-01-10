# Cat Detector - Next Steps for Development

## Project Status: MVP Complete

The cat-detector application is fully implemented and compiles successfully. All 50 unit tests pass. The codebase is ready for integration testing with real hardware.

## What's Been Built

### Core Architecture
```
src/
├── main.rs      - CLI entry point (clap-based)
├── config.rs    - TOML configuration parsing
├── camera.rs    - CameraCapture trait + StubCamera
├── detector.rs  - CatDetector trait + OnnxDetector (YOLOv8)
├── tracker.rs   - Hysteresis-based enter/exit detection
├── storage.rs   - ImageStorage trait + FileSystemStorage
├── notifier.rs  - Notifier trait + SignalNotifier
├── app.rs       - Main detection loop with dependency injection
└── service.rs   - Systemd service management
```

### Key Design Decisions
1. **Trait-based abstractions** - All external dependencies (camera, ML, storage, notifications) are behind traits for testability
2. **Dynamic ONNX loading** - Uses `ort` crate with `load-dynamic` feature to avoid glibc linking issues
3. **Hysteresis tracking** - Configurable thresholds prevent false triggers from brief detections/gaps
4. **No real camera in default build** - `nokhwa` is behind `real-camera` feature flag (wasn't needed for MVP)

## To Run the Application

### 1. Get the ONNX Model
The project uses YOLOX-tiny, which is already downloaded to `models/yolox_tiny.onnx` (20MB).

If you need to re-download it:
```bash
wget -O models/yolox_tiny.onnx "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_tiny.onnx"
```

Alternative models available from the same repo:
- `yolox_nano.onnx` - Smaller, better for Raspberry Pi
- `yolox_s.onnx` - More accurate, good for desktop

### 2. Install ONNX Runtime
Since we use `load-dynamic`, you need the ONNX Runtime shared library:
```bash
# Ubuntu/Debian
# Download from https://github.com/microsoft/onnxruntime/releases
# Or install via package manager if available

# Set library path
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

### 3. Create Config
```bash
cp config.example.toml config.toml
# Edit config.toml - especially:
#   - camera.device_path (e.g., /dev/video0)
#   - detector.model_path (models/yolov8n.onnx)
#   - storage.output_dir (where to save images)
```

### 4. Run
```bash
cargo run --release -- run --config config.toml
```

## What Needs Work

### High Priority

1. **Real Camera Integration**
   - Currently using `StubCamera` which returns blank images
   - Need to enable `real-camera` feature and fix `nokhwa` integration
   - Alternative: implement V4L2 capture directly or use `opencv` crate
   ```rust
   // In main.rs, replace StubCamera with real camera:
   // let camera = NokhwaCamera::new(...)?;
   ```

2. **ONNX Runtime Setup**
   - Document exact ONNX Runtime version needed
   - Consider bundling or providing download script
   - Test with actual model inference

### Medium Priority

3. **Signal-CLI Integration Testing**
   - `SignalNotifier` shells out to `signal-cli`
   - Needs testing with actual signal-cli installation
   - Consider adding setup verification command

4. **Systemd Service**
   - `service.rs` generates service file but untested
   - Needs testing on actual Linux system with systemd
   - May need user/group configuration

5. **Error Recovery**
   - App continues on camera/detector errors but could be more robust
   - Consider exponential backoff on repeated failures
   - Add health check endpoint?

### Low Priority / Nice to Have

6. **Real Camera Feature**
   - Fix `nokhwa` compilation issues or
   - Switch to `v4l` crate for Linux-only support
   - Add camera auto-detection

7. **Web Dashboard**
   - Add simple HTTP server to view captures
   - Real-time detection status
   - Configuration UI

8. **Multiple Cat Tracking**
   - Current implementation just detects "cat present"
   - Could track individual cats with bounding boxes
   - Count distinct cats

9. **Cloud Storage**
   - Add S3/GCS upload option
   - Implement `ImageStorage` trait for cloud

## Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| config.rs | 7 | Config parsing, validation, defaults |
| camera.rs | 4 | Mock camera cycling, availability |
| detector.rs | 7 | Mock detector, IOU calculation |
| tracker.rs | 12 | All state transitions, hysteresis |
| storage.rs | 5 | Mock storage, filename format |
| notifier.rs | 8 | Mock notifier, message format |
| app.rs | 5 | Integration with mocks |
| service.rs | 2 | Service file generation |

## Known Issues

1. **ONNX Runtime Linking** - Requires `load-dynamic` feature due to glibc version mismatch with pre-built binaries

2. **Camera Feature Disabled** - `nokhwa` had compilation issues; disabled by default

## File Locations

- **Config example**: `config.example.toml`
- **Documentation**: `README.md`
- **Model directory**: `models/` (create and add yolov8n.onnx)
- **Output directory**: Configured in config.toml, default `captures/`

## Build Commands

```bash
# Development build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with real camera (if nokhwa issues are fixed)
cargo build --features real-camera

# Check for issues
cargo clippy
```

## Architecture Notes

### Detection Flow
```
Camera Frame → YOLOv8 Inference → Cat Detected? → Tracker State Machine
                                                        ↓
                                    ┌───────────────────┴───────────────────┐
                                    ↓                   ↓                   ↓
                              CatEntered          SampleDue            CatExited
                                    ↓                   ↓                   ↓
                              Save Image          Save Image          Save Image
                              + Notify                                 + Notify
```

### Hysteresis Logic
- **Enter**: Requires N consecutive detections (default: 3)
- **Exit**: Requires M consecutive non-detections (default: 5)
- **Sample**: Capture image every S seconds while cat present (default: 10s)

This prevents:
- Brief detections (cat walking past quickly) from triggering
- Brief occlusions (cat behind furniture momentarily) from triggering exit
