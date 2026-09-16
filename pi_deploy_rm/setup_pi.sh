#!/bin/bash
# setup_pi.sh — Raspberry Pi 4 Environment Setup Script for Dracarys KWS

set -e

echo "=== Dracarys KWS: Raspberry Pi 4 Deployment Setup ==="

# 1. OS Version & Config Path Detection (Bookworm vs Bullseye)
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $NAME $VERSION"
fi

if [ -f /boot/firmware/config.txt ]; then
    CONFIG_FILE="/boot/firmware/config.txt"
elif [ -f /boot/config.txt ]; then
    CONFIG_FILE="/boot/config.txt"
else
    CONFIG_FILE="/boot/config.txt"
fi

echo "Target Raspberry Pi config file: $CONFIG_FILE"

# 2. Configure INMP441 I2S Microphone Overlay
echo "Checking I2S overlay configuration in $CONFIG_FILE..."
if grep -q "dtoverlay=googlevoicehat-soundcard" "$CONFIG_FILE"; then
    echo "  I2S overlay (googlevoicehat-soundcard) already enabled."
else
    echo "  Adding I2S microphone overlay to $CONFIG_FILE..."
    echo "" | sudo tee -a "$CONFIG_FILE"
    echo "# INMP441 I2S Microphone Overlay for Dracarys KWS" | sudo tee -a "$CONFIG_FILE"
    echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a "$CONFIG_FILE"
    echo "  I2S overlay added. Note: Reboot required if not previously enabled."
fi

# 3. Install Python Dependencies
echo ""
echo "Installing Python edge dependencies..."
pip install --upgrade pip
pip install tflite-runtime numpy sounddevice websocket-client psutil

# 4. Verify I2S Capture Devices
echo ""
echo "=== Audio Capture Device Verification ==="
if command -v arecord &> /dev/null; then
    arecord -l || echo "Warning: No capture devices listed yet. Ensure INMP441 wiring and reboot if required."
else
    echo "alsa-utils not found. Install via: sudo apt-get install -y alsa-utils"
fi

echo ""
echo "=== Deployment Setup Complete ==="
echo "To run live keyword spotting: python3 pi_deploy/live_kws.py"
