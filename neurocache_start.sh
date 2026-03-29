#!/bin/bash
# neurocache_start.sh — single-command NeuroCache stack launch
# Starts second-brain API, syncs vault, pre-warms Qwen3.5-9B, then launches Kage.
set -euo pipefail

KAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECOND_BRAIN_DIR="/Users/imadeddine/code/second-brain"
SECOND_BRAIN_VENV="$SECOND_BRAIN_DIR/.venv"
KAGE_PYTHON="${HOME}/.local/share/mamba/envs/kage/bin/python"

echo "🧠 Starting NeuroCache..."

# ── 1. second-brain API ───────────────────────────────────────────────────────
SECOND_BRAIN_PID=""
if launchctl print "gui/$(id -u)/com.imad.neurocache" &>/dev/null; then
    echo "  ✓ second-brain API already running (launchd)"
else
    cd "$SECOND_BRAIN_DIR"
    "$SECOND_BRAIN_VENV/bin/uvicorn" api.server:app \
        --port 8765 --host 127.0.0.1 \
        --log-level warning &
    SECOND_BRAIN_PID=$!
    echo "  ✓ second-brain API on :8765 (pid $SECOND_BRAIN_PID)"
    # Give the server a moment to bind
    sleep 1
fi

# ── 2. Vault sync (one-shot index) ────────────────────────────────────────────
echo "  ⟳ Syncing vault..."
"$SECOND_BRAIN_VENV/bin/brain" index ~/Documents/imad-brain \
    2>&1 | grep -E "Done\.|error" || true
echo "  ✓ Vault index up-to-date"

# ── 3. Pre-warm Qwen3.5-9B (removes first-query cold-start delay) ─────────────
cd "$KAGE_DIR"
echo "  ⟳ Loading Qwen3.5-9B..."
# Only pre-warm if NEUROCACHE_ENABLED=true in .env
if grep -q "NEUROCACHE_ENABLED=true" "$KAGE_DIR/.env" 2>/dev/null; then
    # Run in a subshell so import errors don't abort the script
    PYTHONPATH="$KAGE_DIR" "$KAGE_PYTHON" - <<'PYEOF' 2>&1 | tail -2 || echo "  ! pre-warm skipped (model not cached yet — will load on first use)"
from inference.local_llm import get_local_llm
llm = get_local_llm()
llm.load()
print(f"  ✓ Qwen3.5-9B loaded ({llm.ram_used_gb():.1f} GB)")
PYEOF
else
    echo "  - NEUROCACHE_ENABLED not set, skipping pre-warm"
fi

# ── 4. Launch Kage ────────────────────────────────────────────────────────────
cd "$KAGE_DIR"
echo "  ✓ Launching Kage..."
echo ""

# Kill second-brain when Kage exits (only if we launched it manually)
trap '[ -n "$SECOND_BRAIN_PID" ] && { echo ""; echo "  ⏹ Stopping second-brain (pid $SECOND_BRAIN_PID)..."; kill "$SECOND_BRAIN_PID" 2>/dev/null; wait "$SECOND_BRAIN_PID" 2>/dev/null; echo "  Done."; }' EXIT

exec "$KAGE_PYTHON" main.py "$@"
