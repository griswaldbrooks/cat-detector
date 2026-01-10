# Cat Detector

A Rust application that monitors a USB webcam for cats using YOLOv8 machine learning model. When a cat is detected, it logs images and can send notifications via Signal.

## Features

- Real-time cat detection using YOLOv8-nano (optimized for CPU)
- Hysteresis-based tracking to avoid false triggers
- Automatic image capture on cat entry/exit and periodic samples
- Signal messenger notifications
- Runs as a CLI daemon or systemd service
- Fully configurable via TOML

## Requirements

- Linux (tested on Ubuntu 22.04+)
- USB webcam (V4L2 compatible)
- Rust 1.70+ (for building)

## Quick Start

### 1. Build the Application

```bash
cargo build --release
```

### 2. Download and Convert the YOLOv8 Model

Ultralytics provides PyTorch models that need to be converted to ONNX format.

**Option A: Convert using Python (recommended)**

```bash
# Install ultralytics
pip install ultralytics

# Download and convert to ONNX
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
model.export(format='onnx', imgsz=640)
"

# Move to models directory
mkdir -p models
mv yolov8n.onnx models/
```

**Option B: Download pre-converted from Hugging Face**

Some community members provide pre-converted ONNX models:

```bash
mkdir -p models
# Check https://huggingface.co/models?search=yolov8+onnx for available conversions
```

**Note:** The PyTorch model is available at:
```bash
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 3. Create Configuration

```bash
cp config.example.toml config.toml
# Edit config.toml with your settings
```

### 4. Run

```bash
./target/release/cat-detector run --config config.toml
```

## Configuration

See `config.example.toml` for all available options. Key settings:

| Section | Option | Description |
|---------|--------|-------------|
| camera | device_path | Webcam device (e.g., /dev/video0) |
| detector | model_path | Path to YOLOv8 ONNX model |
| detector | confidence_threshold | Detection confidence (0.0-1.0) |
| storage | output_dir | Where to save captured images |
| notification | enabled | Enable Signal notifications |
| notification | recipient | Phone number for notifications |
| tracking | sample_interval_secs | How often to capture while cat present |
| tracking | enter_threshold | Detections needed to confirm entry |
| tracking | exit_threshold | Non-detections needed to confirm exit |

## Signal Notifications Setup

To enable Signal notifications:

### 1. Install signal-cli

```bash
# Download latest release from https://github.com/AsamK/signal-cli/releases
wget https://github.com/AsamK/signal-cli/releases/download/v0.13.2/signal-cli-0.13.2-Linux.tar.gz
tar xf signal-cli-0.13.2-Linux.tar.gz
sudo mv signal-cli-0.13.2 /opt/signal-cli
sudo ln -s /opt/signal-cli/bin/signal-cli /usr/local/bin/signal-cli
```

### 2. Register Your Phone Number

```bash
signal-cli -u +1YOURPHONENUMBER register
# You'll receive an SMS with a verification code
signal-cli -u +1YOURPHONENUMBER verify CODE
```

### 3. Configure cat-detector

Edit `config.toml`:

```toml
[notification]
enabled = true
recipient = "+1RECIPIENTPHONENUMBER"
notify_on_enter = true
notify_on_exit = true
```

## Systemd Service

### Install as Service

```bash
sudo ./target/release/cat-detector install-service --config /path/to/config.toml
```

### Manage Service

```bash
# Start
sudo systemctl start cat-detector

# Stop
sudo systemctl stop cat-detector

# View logs
journalctl -u cat-detector -f

# Check status
./target/release/cat-detector status
```

### Uninstall Service

```bash
sudo ./target/release/cat-detector uninstall-service
```

## CLI Commands

```
cat-detector run              Run the detector daemon
cat-detector install-service  Install as systemd service
cat-detector uninstall-service Remove systemd service
cat-detector start            Start the systemd service
cat-detector stop             Stop the systemd service
cat-detector status           Show service status
```

## Image Storage

Captured images are saved to the configured `output_dir` with the following naming convention:

- `cat_entry_YYYYMMDD_HHMMSS.XXX.jpg` - When cat enters
- `cat_exit_YYYYMMDD_HHMMSS.XXX.jpg` - When cat leaves
- `cat_sample_YYYYMMDD_HHMMSS.XXX.jpg` - Periodic samples while cat present

## Hysteresis Tracking

To avoid false triggers from brief detections or momentary losses:

- **Entry**: Requires `enter_threshold` consecutive detections (default: 3)
- **Exit**: Requires `exit_threshold` consecutive non-detections (default: 5)

This means a cat must be visible for ~1.5 seconds before triggering entry, and must be gone for ~2.5 seconds before triggering exit.

## Troubleshooting

### Camera not found

Check your camera is connected and accessible:

```bash
ls -la /dev/video*
v4l2-ctl --list-devices
```

### Model not loading

Ensure the ONNX model file exists and is readable:

```bash
ls -la models/yolov8n.onnx
```

### Signal notifications not working

Test signal-cli directly:

```bash
signal-cli -u +YOURNUM send -m "Test" +RECIPIENTNUM
```

## Development

### Running Tests

```bash
cargo test
```

### Running with Debug Logging

```bash
RUST_LOG=debug cargo run -- run --config config.toml
```

## License

MIT
