#!/usr/bin/env bash
#
# VisualRef — Full-stack startup script
# Starts the backend (FastAPI + SigLIP + SAM3 + Ollama check) and the
# frontend (Next.js), then runs a health checklist to confirm readiness.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$ROOT_DIR/server"
CLIENT_DIR="$ROOT_DIR/client-next"
SERVER_PORT="${SERVER_PORT:-8001}"
CLIENT_PORT="${CLIENT_PORT:-3000}"
LOG_DIR="$ROOT_DIR/.logs"
mkdir -p "$LOG_DIR"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { printf "${GREEN}  ✓ %s${NC}\n" "$1"; }
fail() { printf "${RED}  ✗ %s${NC}\n" "$1"; }
info() { printf "${CYAN}  ℹ %s${NC}\n" "$1"; }
hdr()  { printf "\n${BOLD}%s${NC}\n" "$1"; }

# ── Cleanup on exit ────────────────────────────────────────────────────────
SERVER_PID="" CLIENT_PID=""
cleanup() {
    hdr "Shutting down…"
    [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null && info "Frontend stopped"
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && info "Server stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Kill anything already on our ports ──────────────────────────────────────
kill_port() {
    local pids
    pids=$(lsof -ti:"$1" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        info "Freed port $1"
    fi
}

hdr "═══════════════════════════════════════"
hdr "       VisualRef  —  Startup"
hdr "═══════════════════════════════════════"

# ── Pre-flight checks ──────────────────────────────────────────────────────
hdr "[1/5] Pre-flight checks"

if [ ! -d "$SERVER_DIR/venv" ]; then
    fail "Python venv not found at $SERVER_DIR/venv"
    echo "       Run:  cd server && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
ok "Python venv found"

if ! command -v node &>/dev/null; then
    fail "Node.js not found — install it first"
    exit 1
fi
ok "Node.js $(node -v)"

if [ ! -d "$CLIENT_DIR/node_modules" ]; then
    info "Installing frontend dependencies…"
    (cd "$CLIENT_DIR" && npm install --silent)
fi
ok "Frontend dependencies ready"

# ── Free ports ──────────────────────────────────────────────────────────────
hdr "[2/5] Freeing ports"
kill_port "$SERVER_PORT"
kill_port "$CLIENT_PORT"
ok "Ports $SERVER_PORT and $CLIENT_PORT are free"

# ── Start backend ───────────────────────────────────────────────────────────
hdr "[3/5] Starting backend  (port $SERVER_PORT)"
info "Loading SigLIP + SAM3 — this may take 2–5 min on first launch…"
(
    cd "$SERVER_DIR"
    source venv/bin/activate
    python -m uvicorn src.retrieval_server_visual:app \
        --host 0.0.0.0 --port "$SERVER_PORT" \
        2>&1 | tee "$LOG_DIR/server.log"
) &
SERVER_PID=$!

# Wait for the server to become healthy
MAX_WAIT=360
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    # Show progress every 15 seconds
    if [ $((WAITED % 15)) -eq 0 ]; then
        info "Still loading models… (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    fail "Server did not become healthy within ${MAX_WAIT}s — check $LOG_DIR/server.log"
    exit 1
fi
ok "Backend is up  (http://localhost:$SERVER_PORT)"

# ── Start frontend ──────────────────────────────────────────────────────────
hdr "[4/5] Starting frontend  (port $CLIENT_PORT)"
(
    cd "$CLIENT_DIR"
    npm run dev -- --port "$CLIENT_PORT" 2>&1 | tee "$LOG_DIR/client.log"
) &
CLIENT_PID=$!

# Wait for Next.js to respond
WAITED=0
while [ $WAITED -lt 30 ]; do
    if curl -sf "http://localhost:$CLIENT_PORT" >/dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done
ok "Frontend is up  (http://localhost:$CLIENT_PORT)"

# ── Health checklist ────────────────────────────────────────────────────────
hdr "[5/5] System checklist"

HEALTH=$(curl -sf "http://localhost:$SERVER_PORT/health" 2>/dev/null || echo "{}")
SAM_STATUS=$(curl -sf "http://localhost:$SERVER_PORT/sam_status" 2>/dev/null || echo "{}")
OLLAMA_STATUS=$(curl -sf "http://localhost:$SERVER_PORT/ollama_status" 2>/dev/null || echo "{}")

# Active dataset (from .env)
ACTIVE_CONFIG=$(grep -E "^CONFIG_PATH=" "$SERVER_DIR/.env" 2>/dev/null | head -1 | sed 's/.*\///' | sed 's/_siglip.yaml//' | sed 's/_clip.*//')
ok "Dataset: ${ACTIVE_CONFIG:-unknown}"

# SigLIP / FAISS index
MODEL_LOADED=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
if [ "$MODEL_LOADED" = "healthy" ]; then
    METRICS=$(curl -sf "http://localhost:$SERVER_PORT/metrics" 2>/dev/null || echo "{}")
    INDEX_SIZE=$(echo "$METRICS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model',{}).get('index_size',0))" 2>/dev/null || echo "0")
    ok "SigLIP encoder + FAISS index loaded ($INDEX_SIZE images)"
else
    fail "SigLIP / FAISS not loaded"
fi

# SAM
SAM_TYPE=$(echo "$SAM_STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_type','none'))" 2>/dev/null || echo "none")
SAM_LOADED=$(echo "$SAM_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('loaded',False))" 2>/dev/null || echo "False")
if [ "$SAM_LOADED" = "True" ]; then
    ok "SAM segmenter: $SAM_TYPE"
else
    fail "SAM segmenter not loaded"
fi

# Ollama
OLLAMA_AVAIL=$(echo "$OLLAMA_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('available',False))" 2>/dev/null || echo "False")
OLLAMA_MODEL=$(echo "$OLLAMA_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','?'))" 2>/dev/null || echo "?")
if [ "$OLLAMA_AVAIL" = "True" ]; then
    ok "Ollama Vision: $OLLAMA_MODEL"
else
    printf "${YELLOW}  ⚠ Ollama Vision: not available (feedback uses image-only embeddings)${NC}\n"
    info "To enable: ollama pull llama3.2-vision && ollama serve"
fi

# GPU
GPU_AVAIL=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('gpu_available',False))" 2>/dev/null || echo "False")
if [ "$GPU_AVAIL" = "True" ]; then
    ok "GPU (CUDA) available"
else
    ok "Running on MPS / CPU"
fi

hdr "═══════════════════════════════════════"
printf "${GREEN}${BOLD}  Ready!  Open http://localhost:$CLIENT_PORT${NC}\n"
hdr "═══════════════════════════════════════"
printf "\n  Press ${BOLD}Ctrl+C${NC} to stop both servers.\n\n"

# ── Keep script alive until Ctrl+C ─────────────────────────────────────────
wait
