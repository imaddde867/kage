#!/bin/bash
# Install the NeuroCache Cortex server as a macOS LaunchAgent.
# Runs automatically on login and restarts if it crashes.
#
# Usage:
#   bash scripts/install_service.sh        # install and start
#   bash scripts/install_service.sh stop   # stop and unload
#   bash scripts/install_service.sh status # show service status
set -euo pipefail

PLIST_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/com.imad.neurocache.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.imad.neurocache.plist"
LABEL="com.imad.neurocache"
LOG_DIR="$HOME/Library/Logs/neurocache"

case "${1:-install}" in
  install)
    mkdir -p "$LOG_DIR"
    cp "$PLIST_SRC" "$PLIST_DEST"
    # Unload first in case it is already registered (idempotent)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"
    echo "✓ NeuroCache service installed and started"
    echo "  Logs: $LOG_DIR/"
    ;;
  stop)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "✓ Stopped" || echo "  Not running"
    ;;
  status)
    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null || echo "  Not installed / not running"
    ;;
  *)
    echo "Usage: $0 [install|stop|status]"
    exit 1
    ;;
esac
