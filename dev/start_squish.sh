#!/bin/bash
# Start squish server with thinking disabled and log to file
# Usage: bash dev/start_squish.sh

set -e

REPO=/Users/wscholl/squish
PYTHON=$REPO/.venv/bin/python3.14
MODEL_DIR=$REPO/models/Qwen3-8B-bf16
COMPRESSED_DIR=$REPO/models/Qwen3-8B-bf16-compressed
LOG=/tmp/squish_server.log
PORT=11435

# Kill any existing server on this port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
sleep 1

echo "Starting squish server (thinking-budget 0)..." | tee $LOG
# Use caffeinate -i to prevent macOS App Nap from deprioritising the server
# (App Nap applies nice=5 to background CLI processes, killing inference speed)
exec caffeinate -i $PYTHON -m squish.server \
  --model-dir "$MODEL_DIR" \
  --compressed-dir "$COMPRESSED_DIR" \
  --port $PORT \
  --host 127.0.0.1 \
  --thinking-budget 0 \
  --verbose \
  >> $LOG 2>&1
