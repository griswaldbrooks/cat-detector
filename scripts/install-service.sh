#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-catbox}"
REMOTE_DIR="cat-detector"

echo "Installing cat-detector as systemd service on $TARGET..."
ssh "$TARGET" "cd ~/$REMOTE_DIR && \
    sudo ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so \
    ./cat-detector install-service --config ~/$REMOTE_DIR/config.toml"

echo "Starting service..."
ssh "$TARGET" "sudo -n systemctl start cat-detector"

echo "Service status:"
ssh "$TARGET" "systemctl status cat-detector --no-pager"
