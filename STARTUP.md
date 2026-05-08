# KonjoAI Dashboard Startup Scripts

One-command startup for each repo's frontend + backend + browser.

## Quick Start

Each repo has a startup script that automatically launches the backend, frontend, and opens your browser:

```bash
# Start any individual repo
cd squish && ./scripts/start.sh
cd kyro && ./scripts/start.sh
cd kairu && ./scripts/start.sh
cd vectro && ./scripts/start.sh
cd kohaku && ./scripts/start.sh
cd toki && ./scripts/start.sh
```

Or use the master script from the repo root:

```bash
# Start individual repos
./scripts/start-konjo.sh vectro
./scripts/start-konjo.sh kairu kohaku

# Start all repos
./scripts/start-konjo.sh all
```

## What Each Script Does

1. **Kills existing processes** on the target ports (prevents port conflicts)
2. **Starts the backend** (FastAPI or demo server)
3. **Waits for backend to be ready** (polls `/health` endpoint)
4. **Starts the frontend dev server** (Vite on 517X)
5. **Waits for frontend to be ready** (polls `http://localhost:517X`)
6. **Opens your browser** automatically
7. **Shows summary** with all running ports
8. **Manages cleanup** on Ctrl+C

## Port Allocations

| Repo | Backend | Frontend | Notes |
|------|---------|----------|-------|
| **squish** | 11435 (FastAPI) + 8001 (demo) | 5177 | Both servers for comparison feature |
| **kyro** | 8000 (FastAPI) + 8766 (demo) | 5178 | RAG pipeline visualization |
| **kairu** | 7777 (demo) | 5176 | Speculative decoding |
| **vectro** | 8765 (demo) | 5179 | Embedding compression |
| **kohaku** | 8000 (demo) | 5180 | Hypervector memory |
| **toki** | 8765 (demo) | 5181 | Adversarial hardening |

**Note:** kohaku and kyro both use port 8000 (can't run simultaneously), and vectro/toki both use 8765 (can't run simultaneously).

## Viewing Logs

Each script logs backend and frontend output to `/tmp/`:

```bash
# View logs while running
tail -f /tmp/vectro_backend.log
tail -f /tmp/vectro_demo.log
tail -f /tmp/vectro_frontend.log
```

## Troubleshooting

### "Port already in use"
Another instance is running. Kill it:
```bash
pkill -f "python.*demo"
pkill -f "vite"
```

### "Frontend timeout"
The Vite dev server takes longer to start on first run. Wait another 30 seconds, or check logs:
```bash
tail -f /tmp/{repo}_frontend.log
```

### "Backend failed to start"
Check if dependencies are installed or backend code has issues:
```bash
# For FastAPI repos (squish, kyro)
python -m pip install -e .
python -m konjoai.api.app --port 8000

# For demo servers
cd {repo}/demo && python server.py --port 8765
```

### macOS: "permission denied"
Make sure scripts are executable:
```bash
chmod +x squish/scripts/start.sh
chmod +x ./scripts/start-konjo.sh
```

## Manual Startup (if needed)

If the scripts fail, start manually in separate terminals:

```bash
# Terminal 1: Backend
cd vectro
python demo/server.py --port 8765

# Terminal 2: Frontend
cd vectro/dashboard
npm run dev

# Terminal 3: Browser
open http://localhost:5179
```

## Customization

Edit the port numbers in each `scripts/start.sh` if you need different ports:

```bash
FRONTEND_PORT=5179
DEMO_PORT=8765
```

Or temporarily override with environment variables (modify script to support this).

---

**Quick reminder:** These scripts assume Python 3.10+, Node 18+, and npm/Node dependencies are installed.
