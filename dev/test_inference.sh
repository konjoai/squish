#!/bin/zsh
# Test all endpoint variants and measure latencies
PORT=11435
MODEL="Qwen3-8B-bf16-compressed"

echo "=== Health ==="
curl -s "http://127.0.0.1:${PORT}/health"
echo ""

echo ""
echo "=== Non-streaming inference (30 tokens) ==="
time curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 7 * 8?\"}],\"max_tokens\":30,\"stream\":false}" \
  --max-time 120 | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:200]); print('tokens:', r['usage'])"

echo ""
echo "=== Streaming inference (50 tokens) ==="
time curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 7 * 8?\"}],\"max_tokens\":50,\"stream\":true}" \
  --max-time 120 --no-buffer 2>&1 | head -20

echo ""
echo "=== Stats after tests ==="
curl -s "http://127.0.0.1:${PORT}/health" | python3 -c "
import sys, json
h = json.load(sys.stdin)
print(f\"  loader:      {h['loader']}\")
print(f\"  requests:    {h['requests']}\")
print(f\"  tokens_gen:  {h['tokens_gen']}\")
print(f\"  avg_tps:     {h['avg_tps']:.1f}\")
print(f\"  avg_ttft:    {h['avg_ttft_s']:.3f}s\")
"
