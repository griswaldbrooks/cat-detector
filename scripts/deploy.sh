#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-catbox}"
REMOTE_DIR="cat-detector"
BINARY="target/release/cat-detector"

echo "Building cat-detector..."
cargo build --release --features real-camera,web

echo "Ensuring remote directory exists on $TARGET..."
ssh "$TARGET" "mkdir -p ~/$REMOTE_DIR"

echo "Deploying to $TARGET:~/$REMOTE_DIR..."
rsync -avz --progress \
    "$BINARY" \
    config.example.toml \
    models \
    onnxruntime \
    "$TARGET:$REMOTE_DIR/"

# Copy config if it doesn't already exist on remote
ssh "$TARGET" "test -f ~/$REMOTE_DIR/config.toml || cp ~/$REMOTE_DIR/config.example.toml ~/$REMOTE_DIR/config.toml"

echo "Deployed to $TARGET:~/$REMOTE_DIR"
