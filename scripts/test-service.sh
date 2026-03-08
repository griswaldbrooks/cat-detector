#!/usr/bin/env bash
set -euo pipefail

# Integration test for systemd service lifecycle.
# Uses an isolated test instance (cat-detector-test) on port 8081.
# Temporarily stops the real service to free the camera, restarts it after.

TARGET="${1:-catbox}"
REMOTE_DIR="cat-detector"
SERVICE_NAME="cat-detector-test"
TEST_PORT=8081
PASS=0
FAIL=0
ERRORS=""
REAL_SERVICE_WAS_RUNNING=false

pass() {
    PASS=$((PASS + 1))
    echo "  PASS: $1"
}

fail() {
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}\n  FAIL: $1"
    echo "  FAIL: $1"
}

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    ssh "$TARGET" "sudo -n systemctl stop $SERVICE_NAME 2>/dev/null || true"
    ssh "$TARGET" "sudo -n systemctl disable $SERVICE_NAME 2>/dev/null || true"
    ssh "$TARGET" "sudo -n rm -f /etc/systemd/system/${SERVICE_NAME}.service"
    ssh "$TARGET" "sudo -n systemctl daemon-reload"
    ssh "$TARGET" "rm -f ~/$REMOTE_DIR/config-test.toml"

    # Restart real service if it was running before
    if [ "$REAL_SERVICE_WAS_RUNNING" = true ]; then
        echo "Restarting real cat-detector service..."
        ssh "$TARGET" "sudo -n systemctl start cat-detector"
        sleep 2
        if ssh "$TARGET" "sudo -n systemctl is-active cat-detector" 2>/dev/null | grep -q "^active"; then
            echo "Real service restored."
        else
            echo "WARNING: Failed to restart real service!"
        fi
    fi

    echo "Cleanup complete."
}

# Always clean up, even on failure
trap cleanup EXIT

echo "=== Systemd Service Integration Test ==="
echo "Target: $TARGET"
echo "Service: $SERVICE_NAME (port $TEST_PORT)"
echo ""

# --- Setup ---
echo "=== Setup ==="

# Stop real service to free the camera
if ssh "$TARGET" "sudo -n systemctl is-active cat-detector" 2>/dev/null | grep -q "^active"; then
    REAL_SERVICE_WAS_RUNNING=true
    echo "Stopping real cat-detector service to free camera..."
    ssh "$TARGET" "sudo -n systemctl stop cat-detector"
    sleep 1
fi

# Create test config with a different port
ssh "$TARGET" "sed 's/port = 8080/port = $TEST_PORT/' ~/$REMOTE_DIR/config.toml > ~/$REMOTE_DIR/config-test.toml"
echo "Created test config on $TARGET (port $TEST_PORT)"

# Get paths
BINARY="/home/griswald/$REMOTE_DIR/cat-detector"
CONFIG="/home/griswald/$REMOTE_DIR/config-test.toml"
ORT_PATH="/home/griswald/$REMOTE_DIR/onnxruntime/lib/libonnxruntime.so"

# Create the test service file directly (bypasses hardcoded service name)
ssh "$TARGET" "sudo -n tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null" <<EOF
[Unit]
Description=Cat Detector Test Instance
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/griswald/$REMOTE_DIR
ExecStart=$BINARY run --config $CONFIG
Restart=on-failure
RestartSec=2
StandardOutput=journal
StandardError=journal
Environment=RUST_LOG=info
Environment=ORT_DYLIB_PATH=$ORT_PATH

[Install]
WantedBy=multi-user.target
EOF

ssh "$TARGET" "sudo -n systemctl daemon-reload"
echo "Created and loaded $SERVICE_NAME service"
echo ""

# --- Test 1: Service starts ---
echo "=== Test 1: Service starts ==="
ssh "$TARGET" "sudo -n systemctl start $SERVICE_NAME"
sleep 3

if ssh "$TARGET" "sudo -n systemctl is-active $SERVICE_NAME" 2>/dev/null | grep -q "^active"; then
    pass "Service started and is active"
else
    fail "Service did not start"
    # Show logs for debugging
    ssh "$TARGET" "journalctl -u $SERVICE_NAME --no-pager -n 10" 2>/dev/null || true
fi

# --- Test 2: Web dashboard responds ---
echo "=== Test 2: Web dashboard responds ==="
HTTP_CODE=$(ssh "$TARGET" "curl -s -o /dev/null -w '%{http_code}' http://localhost:$TEST_PORT/" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    pass "Web dashboard returned HTTP 200 on port $TEST_PORT"
else
    fail "Web dashboard returned HTTP $HTTP_CODE (expected 200)"
fi

# --- Test 3: API endpoint works ---
echo "=== Test 3: API status endpoint ==="
API_RESPONSE=$(ssh "$TARGET" "curl -s http://localhost:$TEST_PORT/api/status" 2>/dev/null || echo "")

if echo "$API_RESPONSE" | grep -q "detecting"; then
    pass "API /api/status returned valid JSON"
else
    fail "API /api/status did not return expected response: $API_RESPONSE"
fi

# --- Test 4: Restart on failure ---
echo "=== Test 4: Restart on failure ==="
PID_BEFORE=$(ssh "$TARGET" "systemctl show -p MainPID --value $SERVICE_NAME" | tr -d '[:space:]')
ssh "$TARGET" "sudo -n kill -9 $PID_BEFORE" 2>/dev/null || true

# Wait for systemd to restart (RestartSec=2 + startup time)
sleep 5

if ssh "$TARGET" "sudo -n systemctl is-active $SERVICE_NAME" 2>/dev/null | grep -q "^active"; then
    PID_AFTER=$(ssh "$TARGET" "systemctl show -p MainPID --value $SERVICE_NAME" | tr -d '[:space:]')
    if [ "$PID_BEFORE" != "$PID_AFTER" ] && [ "$PID_AFTER" != "0" ]; then
        pass "Service restarted after kill (PID $PID_BEFORE -> $PID_AFTER)"
    else
        fail "Service PID did not change after kill (PID=$PID_AFTER)"
    fi
else
    fail "Service did not restart after being killed"
fi

# --- Test 5: Service stops cleanly ---
echo "=== Test 5: Service stops cleanly ==="
ssh "$TARGET" "sudo -n systemctl stop $SERVICE_NAME"
sleep 1

STATUS=$(ssh "$TARGET" "sudo -n systemctl is-active $SERVICE_NAME 2>/dev/null || true" | tr -d '[:space:]')
if [ "$STATUS" = "inactive" ]; then
    pass "Service stopped cleanly"
else
    fail "Service status is '$STATUS' after stop (expected inactive)"
fi

# --- Test 6: Service enable/disable ---
echo "=== Test 6: Enable and disable ==="
ssh "$TARGET" "sudo -n systemctl enable $SERVICE_NAME" 2>/dev/null

if ssh "$TARGET" "sudo -n systemctl is-enabled $SERVICE_NAME" 2>/dev/null | grep -q "^enabled"; then
    pass "Service enabled successfully"
else
    fail "Service was not enabled"
fi

ssh "$TARGET" "sudo -n systemctl disable $SERVICE_NAME" 2>/dev/null

STATUS=$(ssh "$TARGET" "sudo -n systemctl is-enabled $SERVICE_NAME 2>/dev/null || true" | tr -d '[:space:]')
if [ "$STATUS" = "disabled" ]; then
    pass "Service disabled successfully"
else
    fail "Service is-enabled returned '$STATUS' (expected disabled)"
fi

# --- Summary ---
echo ""
echo "==============================="
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo -e "Failures:$ERRORS"
    echo "==============================="
    exit 1
else
    echo "All tests passed!"
    echo "==============================="
    exit 0
fi
