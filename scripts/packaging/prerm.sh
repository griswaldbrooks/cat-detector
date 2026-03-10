#!/usr/bin/env bash
set -e

# Stop the service before removal
if systemctl is-active --quiet cat-detector 2>/dev/null; then
    echo "Stopping cat-detector service..."
    systemctl stop cat-detector
fi
