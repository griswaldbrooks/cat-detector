#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-catbox}"

ssh "$TARGET" "pkill -f 'cat-detector run' || true"
echo "Stopped cat-detector on $TARGET"
