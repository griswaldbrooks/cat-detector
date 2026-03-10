#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract version from Cargo.toml
VERSION=$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "Building cat-detector ${VERSION} .deb packages..."

# Check prerequisites
if ! command -v fpm &>/dev/null; then
    echo "Error: fpm is not installed."
    echo "Install it with: gem install fpm"
    echo "See: https://fpm.readthedocs.io/en/latest/installing.html"
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    echo "Error: cargo is not installed."
    exit 1
fi

# Check that model files exist
ONNX_MODEL="$PROJECT_DIR/models/clip_vitb32_image.onnx"
TEXT_EMBEDDINGS="$PROJECT_DIR/models/clip_text_embeddings.bin"
ORT_LIB="$PROJECT_DIR/onnxruntime/lib/libonnxruntime.so"

for f in "$ONNX_MODEL" "$TEXT_EMBEDDINGS" "$ORT_LIB"; do
    if [ ! -f "$f" ]; then
        echo "Error: Required file not found: $f"
        exit 1
    fi
done

# Build release binary
echo "Building release binary..."
cd "$PROJECT_DIR"
cargo build --release --features real-camera,web

BINARY="$PROJECT_DIR/target/release/cat-detector"
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

# Create staging directory (cleaned up on exit)
STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

echo "Staging files..."

# --- Main package staging ---
MAIN_STAGE="$STAGING/main"
mkdir -p "$MAIN_STAGE/usr/local/bin"
mkdir -p "$MAIN_STAGE/opt/cat-detector/lib"
mkdir -p "$MAIN_STAGE/etc/cat-detector"

cp "$BINARY" "$MAIN_STAGE/usr/local/bin/cat-detector"
cp "$ORT_LIB" "$MAIN_STAGE/opt/cat-detector/lib/libonnxruntime.so"

# Copy any versioned symlinks for the ONNX runtime
for f in "$PROJECT_DIR/onnxruntime/lib/libonnxruntime.so".*; do
    [ -e "$f" ] && cp -a "$f" "$MAIN_STAGE/opt/cat-detector/lib/"
done

cp "$PROJECT_DIR/config.example.toml" "$MAIN_STAGE/etc/cat-detector/config.example.toml"

# --- Models package staging ---
MODELS_STAGE="$STAGING/models"
mkdir -p "$MODELS_STAGE/opt/cat-detector/models"

cp "$ONNX_MODEL" "$MODELS_STAGE/opt/cat-detector/models/clip_vitb32_image.onnx"
cp "$TEXT_EMBEDDINGS" "$MODELS_STAGE/opt/cat-detector/models/clip_text_embeddings.bin"

# --- Build cat-detector-models package ---
echo "Building cat-detector-models_${VERSION}_all.deb..."
fpm \
    -s dir \
    -t deb \
    -n cat-detector-models \
    -v "$VERSION" \
    --architecture all \
    --description "Cat Detector - CLIP model files" \
    --maintainer "griswald" \
    --url "https://github.com/griswaldbrooks/cat-detector" \
    -C "$MODELS_STAGE" \
    -p "$PROJECT_DIR/cat-detector-models_${VERSION}_all.deb" \
    .

# --- Build cat-detector main package ---
echo "Building cat-detector_${VERSION}_amd64.deb..."
fpm \
    -s dir \
    -t deb \
    -n cat-detector \
    -v "$VERSION" \
    --architecture amd64 \
    --description "Webcam-based cat detection with CLIP, Signal notifications, and web dashboard" \
    --maintainer "griswald" \
    --url "https://github.com/griswaldbrooks/cat-detector" \
    --depends ffmpeg \
    --depends libv4l-0 \
    --depends "cat-detector-models = ${VERSION}" \
    --config-files /etc/cat-detector/config.example.toml \
    --deb-systemd "$PROJECT_DIR/scripts/packaging/cat-detector.service" \
    --after-install "$PROJECT_DIR/scripts/packaging/postinst.sh" \
    --before-remove "$PROJECT_DIR/scripts/packaging/prerm.sh" \
    -C "$MAIN_STAGE" \
    -p "$PROJECT_DIR/cat-detector_${VERSION}_amd64.deb" \
    .

echo ""
echo "Packages built successfully:"
ls -lh "$PROJECT_DIR"/cat-detector*_${VERSION}_*.deb
echo ""
echo "Install with:"
echo "  sudo dpkg -i cat-detector-models_${VERSION}_all.deb cat-detector_${VERSION}_amd64.deb"
echo "  sudo apt-get install -f  # resolve dependencies"
