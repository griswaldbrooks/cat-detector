#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-catbox}"
REMOTE_DIR="/opt/cat-detector"

ssh "$TARGET" "cd $REMOTE_DIR && \
    ORT_DYLIB_PATH=./onnxruntime/lib/libonnxruntime.so \
    RUST_LOG=info \
    ./cat-detector run --config config.toml"
