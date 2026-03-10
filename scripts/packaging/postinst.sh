#!/usr/bin/env bash
set -e

CONFIG_DIR="/etc/cat-detector"
CONFIG_TEMPLATE="${CONFIG_DIR}/config.example.toml"
CONFIG_FILE="${CONFIG_DIR}/config.toml"
LDCONFIG_FILE="/etc/ld.so.conf.d/cat-detector.conf"
ORT_LIB_DIR="/opt/cat-detector/lib"

# Create config.toml from template if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_TEMPLATE" ]; then
        cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"
        # Update model path to installed location
        sed -i 's|model_path = "models/clip_vitb32_image.onnx"|model_path = "/opt/cat-detector/models/clip_vitb32_image.onnx"|' "$CONFIG_FILE"
        sed -i 's|# text_embeddings_path = "models/clip_text_embeddings.bin"|text_embeddings_path = "/opt/cat-detector/models/clip_text_embeddings.bin"|' "$CONFIG_FILE"
        echo "Created config file at ${CONFIG_FILE}"
    fi
fi

# Ensure ONNX Runtime library is discoverable via ldconfig
if [ -d "$ORT_LIB_DIR" ]; then
    echo "$ORT_LIB_DIR" > "$LDCONFIG_FILE"
    ldconfig
fi

echo ""
echo "============================================"
echo "  cat-detector installed successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit /etc/cat-detector/config.toml"
echo "  2. Ensure user 'griswald' is in the 'video' group:"
echo "       sudo usermod -aG video griswald"
echo "  3. Start the service:"
echo "       sudo systemctl start cat-detector"
echo "  4. Enable on boot:"
echo "       sudo systemctl enable cat-detector"
echo "  5. View logs:"
echo "       journalctl -u cat-detector -f"
echo ""
