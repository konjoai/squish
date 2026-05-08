#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_PORT=5177
BACKEND_PORT=11435
DEMO_PORT=8001

echo "🚀 squish — Quantization Comparison Dashboard"
echo "================================================"

# Kill any existing processes on our ports
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "python.*demo" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# Start FastAPI backend
echo "📦 Starting FastAPI backend on :$BACKEND_PORT..."
cd "$REPO_DIR"
python -m konjoai.api.app --port "$BACKEND_PORT" > /tmp/squish_backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "⏳ Waiting for backend..."
for i in {1..30}; do
  if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
    echo "✅ Backend ready"
    break
  fi
  sleep 0.5
  if [ $i -eq 30 ]; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
  fi
done

# Start demo server (optional, for comparison feature)
echo "📦 Starting demo server on :$DEMO_PORT..."
python demo/server.py --port "$DEMO_PORT" > /tmp/squish_demo.log 2>&1 &
DEMO_PID=$!
sleep 1

# Start frontend dev server
echo "🎨 Starting frontend on :$FRONTEND_PORT..."
cd "$REPO_DIR/dashboard"
npm run dev > /tmp/squish_frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo "⏳ Waiting for frontend..."
for i in {1..60}; do
  if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
    echo "✅ Frontend ready"
    sleep 1
    break
  fi
  sleep 0.5
  if [ $i -eq 60 ]; then
    echo "⚠️  Frontend timeout, but continuing..."
    break
  fi
done

# Open browser
echo "🌐 Opening browser..."
sleep 1
if command -v open &> /dev/null; then
  open "http://localhost:$FRONTEND_PORT"
elif command -v xdg-open &> /dev/null; then
  xdg-open "http://localhost:$FRONTEND_PORT"
fi

echo ""
echo "✨ All systems ready!"
echo "📍 Frontend:  http://localhost:$FRONTEND_PORT"
echo "📍 Backend:   http://localhost:$BACKEND_PORT"
echo "📍 Demo:      http://localhost:$DEMO_PORT"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Trap Ctrl+C to kill child processes
trap "echo ''; echo 'Stopping all services...'; kill $FRONTEND_PID $BACKEND_PID $DEMO_PID 2>/dev/null || true; exit 0" INT

wait
