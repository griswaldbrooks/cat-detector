#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-catbox}"

echo "=== Cat Detector .deb Deploy ==="
echo "Target: $TARGET"
echo ""

# Step 1: Build .deb packages
echo "Building .deb packages..."
./scripts/build-deb.sh

# Find the built packages
MODELS_DEB=$(ls cat-detector-models_*_all.deb 2>/dev/null | head -1)
MAIN_DEB=$(ls cat-detector_*_amd64.deb 2>/dev/null | head -1)

if [ -z "$MODELS_DEB" ] || [ -z "$MAIN_DEB" ]; then
    echo "Error: .deb packages not found after build"
    exit 1
fi

echo ""
echo "Packages:"
echo "  $MODELS_DEB"
echo "  $MAIN_DEB"

# Step 2: Copy to target
echo ""
echo "Copying packages to $TARGET:/tmp/..."
scp "$MODELS_DEB" "$MAIN_DEB" "$TARGET:/tmp/"

# Step 3: Install
echo ""
echo "Installing on $TARGET..."
ssh "$TARGET" "sudo -n dpkg -i /tmp/$(basename "$MODELS_DEB") /tmp/$(basename "$MAIN_DEB")"

# Step 4: Verify
echo ""
echo "Verifying service..."
ssh "$TARGET" "sudo -n systemctl status cat-detector --no-pager -l" || true

# Step 5: Quick health check
echo ""
echo "Checking dashboard..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${TARGET}.local:8080/" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "Dashboard: OK (HTTP 200)"
else
    echo "Dashboard: HTTP $HTTP_CODE (may need a moment to start)"
fi

echo ""
echo "Deploy complete!"
