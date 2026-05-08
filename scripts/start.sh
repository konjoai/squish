#!/bin/bash

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_PORT=5177
BACKEND_PORT=11435
DEMO_PORT=8001

echo "🚀 squish — Quantization Comparison Dashboard"
echo "================================================"

# Kill any existing processes on our ports
pkill -f "python3.*app.py" 2>/dev/null || true
pkill -f "python3.*demo" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# Try to start FastAPI backend (optional)
BACKEND_READY=0
echo "📦 Trying FastAPI backend on :$BACKEND_PORT..."
cd "$REPO_DIR"
if python3 -m konjoai.api.app --port "$BACKEND_PORT" > /tmp/squish_backend.log 2>&1 &
then
  BACKEND_PID=$!

  # Wait for backend to be ready
  echo "⏳ Waiting for backend..."
  for i in {1..30}; do
    if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
      echo "✅ Backend ready"
      BACKEND_READY=1
      break
    fi
    sleep 0.5
  done

  if [ $BACKEND_READY -eq 0 ]; then
    echo "⚠️  Backend not ready, continuing with demo server only..."
    kill $BACKEND_PID 2>/dev/null || true
    BACKEND_PID=""
  fi
else
  echo "⚠️  FastAPI backend not available (konjoai module not installed)"
  echo "    Using demo server only for comparison feature"
  BACKEND_PID=""
fi

# Start demo server
echo "📦 Starting demo server on :$DEMO_PORT..."
python3 demo/server.py --port "$DEMO_PORT" > /tmp/squish_demo.log 2>&1 &
DEMO_PID=$!

# Wait for demo server
echo "⏳ Waiting for demo server..."
for i in {1..30}; do
  if curl -s http://localhost:$DEMO_PORT/api/health > /dev/null 2>&1; then
    echo "✅ Demo server ready"
    break
  fi
  sleep 0.5
  if [ $i -eq 30 ]; then
    echo "❌ Demo server failed to start"
    kill $DEMO_PID 2>/dev/null || true
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null || true
    exit 1
  fi
done

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
if [ $BACKEND_READY -eq 1 ]; then
  echo "📍 Backend:   http://localhost:$BACKEND_PORT"
fi
echo "📍 Demo:      http://localhost:$DEMO_PORT"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Trap Ctrl+C to kill child processes
cleanup() {
  echo ''
  echo 'Stopping all services...'
  kill $FRONTEND_PID 2>/dev/null || true
  [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null || true
  [ -n "$DEMO_PID" ] && kill $DEMO_PID 2>/dev/null || true
  exit 0
}

trap cleanup INT

wait
