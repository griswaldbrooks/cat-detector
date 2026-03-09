# Cat Detector Deployment Guide

Complete guide to deploying cat-detector to a new machine. Covers building, system dependencies, configuration, and lessons learned from the catbox deployment.

## Target Machine Requirements

- **OS**: Linux (Ubuntu 22.04+ tested)
- **Architecture**: x86_64
- **Hardware**: USB webcam, modest CPU (runs on i3-6100T)
- **Network**: SSH access from build machine, port 8080 open for web dashboard

## System Dependencies

Install on the **target machine**:

| Dependency | Required | Install | Purpose |
|-----------|----------|---------|---------|
| FFmpeg | Yes | `sudo apt install ffmpeg` | Video recording (H.264 MP4) |
| V4L2 | Yes | Usually pre-installed | Camera access |
| signal-cli | Optional | See [Signal Setup](#signal-cli-setup) | Signal notifications |
| avahi-daemon | Optional | `sudo apt install avahi-daemon` | mDNS (`.local` hostname) |

## Build

On the **build machine** (or target if building locally):

```bash
# Build dependency for V4L2 camera support
sudo apt install libv4l-dev

cargo build --release --features real-camera,web
```

Features:
- `real-camera` — V4L2 webcam support (required for real cameras)
- `web` — Axum web dashboard with MJPEG stream

## ONNX Runtime

The ONNX Runtime library (v1.23.2) is loaded dynamically at runtime via `ORT_DYLIB_PATH`. It's not a system package — it ships with the deployment.

Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) if you need a fresh copy. The deployment directory expects:

```
onnxruntime/
├── lib/
│   ├── libonnxruntime.so -> libonnxruntime.so.1
│   ├── libonnxruntime.so.1 -> libonnxruntime.so.1.23.2
│   └── libonnxruntime.so.1.23.2  (22 MB)
```

## Models

Default model is CLIP ViT-B/32 (recommended for overhead camera angles). Models are large and gitignored — they must be deployed separately.

| Model | Size | Input | Use Case |
|-------|------|-------|----------|
| `clip_vitb32_image.onnx` | 352 MB | 224x224 | Zero-shot cat/room/person classification |
| `clip_text_embeddings.bin` | 6 KB | — | Pre-computed text embeddings (tracked in repo) |

Models go in `models/` relative to the working directory.

## Deploy Script

```bash
./scripts/deploy.sh [TARGET_HOST]  # default: catbox
```

This rsync's the binary, config template, models, and ONNX runtime to `~/cat-detector/` on the target. It does NOT install the systemd service or system dependencies.

## Configuration

Copy `config.example.toml` to `config.toml` on the target and edit:

```toml
[camera]
device_path = "auto"       # auto-detects /dev/video*
frame_width = 640
frame_height = 480
fps = 30

[detector]
model_path = "models/clip_vitb32_image.onnx"
confidence_threshold = 0.5
model_format = "clip"

[storage]
output_dir = "captures"

[notification]
enabled = true
signal_cli_path = "/home/user/bin/signal-cli"
recipient = "+1234567890"        # who receives notifications
account = "+1234567890"          # signal-cli account (required for linked devices)
notify_on_enter = true
notify_on_exit = true
send_video = true                # attach video to exit notifications
attachment_timeout_secs = 120    # longer timeout for video uploads

[tracking]
enter_threshold = 3
exit_threshold = 5
detection_interval_ms = 500

[web]
enabled = true
bind_address = "0.0.0.0"
port = 8080
stream_fps = 5
```

`config.toml` is gitignored — it contains user-specific settings and should not be committed.

## Systemd Service

### Install

```bash
# On target machine:
cd ~/cat-detector
sudo ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so ./cat-detector install-service --config config.toml
```

Or use the script: `./scripts/install-service.sh [TARGET_HOST]`

### Service File

Generated at `/etc/systemd/system/cat-detector.service`. You'll likely need to add `User=` to run as a non-root user (see [Permissions](#permissions) below).

### Management

```bash
sudo systemctl start cat-detector
sudo systemctl stop cat-detector
sudo systemctl restart cat-detector
sudo systemctl status cat-detector
journalctl -u cat-detector -f    # follow logs
```

### Passwordless sudo (recommended)

Add to `/etc/sudoers.d/cat-detector`:

```
username ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart cat-detector, \
    /usr/bin/systemctl stop cat-detector, \
    /usr/bin/systemctl start cat-detector, \
    /usr/bin/systemctl status cat-detector, \
    /usr/bin/systemctl daemon-reload
```

## Permissions

**Important lessons from catbox deployment:**

### 1. Run service as your user, not root

The generated service file has no `User=` directive, so it runs as root by default. This causes two problems:

- **Camera access**: Root can access `/dev/video*`, but if you later switch to a user, it fails
- **signal-cli data**: signal-cli stores account data in `~/.local/share/signal-cli/`, which is user-specific

**Fix**: Add `User=yourusername` to the `[Service]` section:

```bash
sudo sed -i '/^ExecStart=/i User=yourusername' /etc/systemd/system/cat-detector.service
sudo systemctl daemon-reload
sudo systemctl restart cat-detector
```

### 2. Video group membership

The service user must be in the `video` group to access the webcam:

```bash
sudo usermod -aG video yourusername
# Restart service for group change to take effect
sudo systemctl restart cat-detector
```

### 3. File ownership

If the service previously ran as root, captured files will be owned by root. Fix:

```bash
sudo chown -R yourusername:yourusername ~/cat-detector/captures/
```

## Signal-CLI Setup

Signal notifications are optional. If enabled, signal-cli must be installed and linked to a Signal account.

### Install (native binary, no Java required)

```bash
VERSION=0.14.1
curl -L -O "https://github.com/AsamK/signal-cli/releases/download/v${VERSION}/signal-cli-${VERSION}-Linux-native.tar.gz"
mkdir -p ~/opt ~/bin
tar xf signal-cli-${VERSION}-Linux-native.tar.gz -C ~/opt
cp ~/opt/signal-cli ~/bin/signal-cli
chmod +x ~/bin/signal-cli
~/bin/signal-cli --version
```

### Link to existing Signal account (recommended)

This makes the server a linked device on your existing Signal account — no second phone number needed.

```bash
# On the server (outputs a URI):
~/bin/signal-cli link -n "catbox"

# Convert URI to QR code (need qrencode or Python qrcode):
# Option A: qrencode (if available)
~/bin/signal-cli link -n "catbox" | xargs -L 1 qrencode -t ANSI

# Option B: Python
pip3 install --user qrcode
# Then generate QR from the URI output

# On your phone: Signal > Settings > Linked Devices > scan QR code
```

### Configure

In `config.toml`:

```toml
[notification]
enabled = true
signal_cli_path = "/home/yourusername/bin/signal-cli"
recipient = "+1234567890"
account = "+1234567890"     # REQUIRED for linked devices
```

The `account` field passes `-a +number` to signal-cli, which is required when running as a linked device.

### Note: "Note to Self" behavior

If `account` and `recipient` are the same phone number, Signal treats messages as "Note to Self" — they arrive silently with no push notification. To get audible alerts, use a different number for the signal-cli account (requires registering a separate number) or configure your phone's Signal notification settings for the "Note to Self" conversation.

### Test

```bash
cd ~/cat-detector
ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so ./cat-detector test-notification
```

### signal-cli arg ordering quirk

The native signal-cli binary requires the recipient **before** `--attachment` in command args. The code handles this, but if you're debugging manually:

```bash
# CORRECT:
signal-cli -a +ACCOUNT send -m "msg" +RECIPIENT --attachment /path/to/video.mp4

# WRONG (fails with "No recipients given"):
signal-cli -a +ACCOUNT send -m "msg" --attachment /path/to/video.mp4 +RECIPIENT
```

## Runtime Directory Structure

After deployment and first run:

```
~/cat-detector/
├── cat-detector                  # binary
├── config.toml                   # user config (not in git)
├── config.example.toml           # template
├── models/
│   ├── clip_vitb32_image.onnx    # default model
│   └── clip_text_embeddings.bin  # CLIP text embeddings
├── onnxruntime/
│   └── lib/
│       └── libonnxruntime.so*    # ONNX Runtime
└── captures/                     # created at runtime
    ├── cat_entry_*.jpg           # entry snapshots
    ├── cat_exit_*.jpg            # exit snapshots
    ├── cat_sample_*.jpg          # periodic samples during visit
    ├── cat_video_*.mp4           # visit recordings
    └── sessions/
        └── session_*.json        # session metadata
```

## CLI Commands

```bash
# Must set ORT_DYLIB_PATH for all commands that load the model
export ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so

./cat-detector run --config config.toml           # run daemon
./cat-detector test-notification --config config.toml  # test Signal setup
./cat-detector test-image IMAGE --model MODEL      # test detection on image
./cat-detector test-camera --device /dev/video0    # test camera capture
./cat-detector install-service --config config.toml    # install systemd service (root)
./cat-detector uninstall-service                       # remove systemd service (root)
./cat-detector status                                  # show service status
./cat-detector web --port 8080                         # run web dashboard only
```

## Web Dashboard

After starting, available at `http://TARGET:8080`:

- `/` — Live MJPEG stream with detection overlay and system info
- `/sessions` — Browse cat visit sessions with images and video
- `/captures/:filename` — Direct image/video access
- `/api/system-info` — JSON system info (model, threshold, resolution)

## Verification Checklist

After deploying to a new machine:

```bash
# 1. Service running?
sudo systemctl status cat-detector

# 2. Camera detected?
journalctl -u cat-detector -n 20 | grep -i camera

# 3. Model loaded?
journalctl -u cat-detector -n 20 | grep -i "loaded.*model"

# 4. Web dashboard?
curl -s http://localhost:8080 | head -1

# 5. Signal notifications? (if configured)
cd ~/cat-detector && ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so ./cat-detector test-notification

# 6. Trigger a detection — wave a cat photo in front of the camera, check:
journalctl -u cat-detector -f   # watch for "Cat entered" / "Cat exited"
```

## Python Scripts (optional, development only)

For regenerating text embeddings or testing the CLIP pipeline. Not needed on the deployment target.

Requires [pixi](https://pixi.sh):

```bash
pixi run test-clip                         # test CLIP on test images
pixi run -e clip generate-embeddings       # regenerate text embeddings
```
