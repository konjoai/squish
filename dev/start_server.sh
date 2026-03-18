#!/bin/zsh
cd /Users/wscholl/squish

# Kill any existing server
pkill -9 -f squish.server 2>/dev/null
sleep 2

# Start server
PYTHONUNBUFFERED=1 /Users/wscholl/squish/.venv/bin/python3.14 -m squish.server \
  --model-dir models/Qwen3-8B-bf16 \
  --compressed-dir models/Qwen3-8B-bf16-compressed \
  --port 11435 --host 127.0.0.1 \
  --verbose --thinking-budget 512 \
  > /tmp/squish.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait until ready
for i in $(seq 1 45); do
  sleep 2
  HEALTH=$(curl -s http://127.0.0.1:11435/health 2>/dev/null)
  if echo "$HEALTH" | grep -q '"loaded":true'; then
    echo "Server READY after $((i*2))s"
    echo "$HEALTH"
    break
  fi
  echo "  [${i}] waiting... (last log: $(tail -1 /tmp/squish.log 2>/dev/null))"
done

echo "=== STARTUP LOG ==="
cat /tmp/squish.log
