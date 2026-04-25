#!/usr/bin/env bash
#
# VisualRef — Full-stack startup script
# Starts the backend (FastAPI + SigLIP + SAM3) and the frontend (Next.js),
# then runs a health checklist to confirm readiness.
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
warn() { printf "${YELLOW}  ⚠ %s${NC}\n" "$1"; }
fail() { printf "${RED}  ✗ %s${NC}\n" "$1"; }
info() { printf "${CYAN}  ℹ %s${NC}\n" "$1"; }
hdr()  { printf "\n${BOLD}%s${NC}\n" "$1"; }

# ── Cleanup on exit ────────────────────────────────────────────────────────
SERVER_PID="" CLIENT_PID=""

# Kill a process and all its descendants, then confirm the port is free.
kill_tree() {
    local pid="$1"
    [ -z "$pid" ] && return
    # Recursively kill children first
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_tree "$child"
    done
    kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
    hdr "Shutting down…"
    if [ -n "$CLIENT_PID" ]; then
        kill_tree "$CLIENT_PID"
        info "Frontend stopped"
    fi
    if [ -n "$SERVER_PID" ]; then
        kill_tree "$SERVER_PID"
        info "Backend stopped"
    fi
    sleep 1
    kill_port "$SERVER_PORT"
    kill_port "$CLIENT_PORT"
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

# ── Read a value from server/.env ──────────────────────────────────────────
read_env() {
    local key="$1" default="${2:-}"
    local val
    val=$(grep -E "^${key}=" "$SERVER_DIR/.env" 2>/dev/null | head -1 | sed "s/^${key}=//" | tr -d "\"'") || true
    echo "${val:-$default}"
}

hdr "═══════════════════════════════════════"
hdr "       VisualRef  —  Startup"
hdr "═══════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────────────
hdr "[1/5] Pre-flight checks"
# ────────────────────────────────────────────────────────────────────────────

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

# ── Environment files (bootstrap from examples) ───────────────────────────
if [ ! -f "$SERVER_DIR/.env" ]; then
    if [ -f "$SERVER_DIR/.env.example" ]; then
        info "Creating server/.env from .env.example (Visual Genome defaults)"
        cp "$SERVER_DIR/.env.example" "$SERVER_DIR/.env"
        ok "server/.env created — edit CONFIG_PATH/INDEX_PATH if you use another corpus"
    else
        fail "server/.env missing — create it (see server/.env.example)"
        exit 1
    fi
fi

if [ ! -f "$CLIENT_DIR/.env.local" ]; then
    if [ -f "$CLIENT_DIR/.env.example" ]; then
        info "Creating client-next/.env.local from .env.example"
        cp "$CLIENT_DIR/.env.example" "$CLIENT_DIR/.env.local"
        ok "client-next/.env.local created"
    else
        warn "client-next/.env.local missing — set NEXT_PUBLIC_SERVER_URL (see README)"
    fi
fi

mkdir -p "$ROOT_DIR/logs"

# ── FAISS index must exist before backend starts ──────────────────────────
INDEX_ABS=""
INDEX_ABS="$(cd "$SERVER_DIR" && "$SERVER_DIR/venv/bin/python" -c "
from pathlib import Path
import re
raw = Path('.env').read_text(encoding='utf-8', errors='replace')
for line in raw.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    m = re.match(r'^INDEX_PATH=(.*)$', line)
    if m:
        val = m.group(1).strip().strip('\"').strip(\"'\")
        print(Path(val).resolve())
        break
" 2>/dev/null)" || true

if [ -z "$INDEX_ABS" ] || [ ! -f "$INDEX_ABS" ]; then
    fail "FAISS index not found${INDEX_ABS:+ at: $INDEX_ABS}"
    echo ""
    echo "       Build it first (Visual Genome):"
    echo "         bash scripts/build_index.sh vg"
    echo "       Then ensure server/.env INDEX_PATH matches the output path."
    exit 1
fi

PATHS_TXT="$(dirname "$INDEX_ABS")/image_paths.txt"
if [ ! -f "$PATHS_TXT" ]; then
    fail "image_paths.txt missing next to index: $PATHS_TXT"
    echo "       Re-run:  bash scripts/build_index.sh vg"
    exit 1
fi
ok "FAISS index ready ($(wc -l < "$PATHS_TXT" | tr -d ' ') paths)"

# ────────────────────────────────────────────────────────────────────────────
hdr "[2/5] Freeing ports"
# ────────────────────────────────────────────────────────────────────────────
kill_port "$SERVER_PORT"
kill_port "$CLIENT_PORT"
ok "Ports $SERVER_PORT and $CLIENT_PORT are free"

# ────────────────────────────────────────────────────────────────────────────
hdr "[3/5] Starting backend  (port $SERVER_PORT)"
# ────────────────────────────────────────────────────────────────────────────
info "Loading SigLIP + SAM3 — this may take 2–5 min on first launch…"
(
    cd "$SERVER_DIR"
    source venv/bin/activate
    MallocStackLogging=NO python -m uvicorn src.retrieval_server_visual:app \
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
    # Bail early if the server process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "Backend process exited unexpectedly — check $LOG_DIR/server.log"
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    if [ $((WAITED % 15)) -eq 0 ]; then
        info "Still loading models… (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    fail "Server did not become healthy within ${MAX_WAIT}s — check $LOG_DIR/server.log"
    exit 1
fi
ok "Backend is up  (http://localhost:$SERVER_PORT)"

# ────────────────────────────────────────────────────────────────────────────
hdr "[4/5] Starting frontend  (port $CLIENT_PORT)"
# ────────────────────────────────────────────────────────────────────────────
(
    cd "$CLIENT_DIR"
    npm run dev -- --port "$CLIENT_PORT" 2>&1 | tee "$LOG_DIR/client.log"
) &
CLIENT_PID=$!

WAITED=0
while [ $WAITED -lt 30 ]; do
    if curl -sf "http://localhost:$CLIENT_PORT" >/dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done
if curl -sf "http://localhost:$CLIENT_PORT" >/dev/null 2>&1; then
    ok "Frontend is up  (http://localhost:$CLIENT_PORT)"
else
    warn "Frontend did not respond within 30s — check $LOG_DIR/client.log"
fi

# ────────────────────────────────────────────────────────────────────────────
hdr "[5/5] System checklist"
# ────────────────────────────────────────────────────────────────────────────

HEALTH=$(curl -sf "http://localhost:$SERVER_PORT/health" 2>/dev/null || echo "{}")
SAM_STATUS=$(curl -sf "http://localhost:$SERVER_PORT/sam_status" 2>/dev/null || echo "{}")
METRICS=$(curl -sf "http://localhost:$SERVER_PORT/metrics" 2>/dev/null || echo "{}")

# Active dataset
ACTIVE_CONFIG=$(grep -E "^CONFIG_PATH=" "$SERVER_DIR/.env" 2>/dev/null | head -1 \
    | sed 's/.*\///' | sed 's/_siglip\.yaml//')
ok "Dataset: ${ACTIVE_CONFIG:-unknown}"

# SigLIP / FAISS
MODEL_STATUS=$(echo "$HEALTH" | "$SERVER_DIR/venv/bin/python" -c \
    "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
if [ "$MODEL_STATUS" = "healthy" ]; then
    INDEX_SIZE=$(echo "$METRICS" | "$SERVER_DIR/venv/bin/python" -c \
        "import sys,json; print(json.load(sys.stdin).get('model',{}).get('index_size',0))" 2>/dev/null || echo "0")
    ok "SigLIP encoder + FAISS index  ($INDEX_SIZE images)"
else
    fail "SigLIP / FAISS not loaded"
fi

# SAM
SAM_TYPE=$(echo "$SAM_STATUS" | "$SERVER_DIR/venv/bin/python" -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('model_type','none'))" 2>/dev/null || echo "none")
SAM_LOADED=$(echo "$SAM_STATUS" | "$SERVER_DIR/venv/bin/python" -c \
    "import sys,json; print(json.load(sys.stdin).get('loaded',False))" 2>/dev/null || echo "False")
if [ "$SAM_LOADED" = "True" ]; then
    ok "SAM segmenter: $SAM_TYPE"
else
    fail "SAM segmenter not loaded"
fi

# VG region phrases
VG_LOADED=$(echo "$HEALTH" | "$SERVER_DIR/venv/bin/python" -c \
    "import sys,json; print(json.load(sys.stdin).get('vg_index_loaded', False))" 2>/dev/null || echo "False")
if [ "$VG_LOADED" = "True" ]; then
    ok "Visual Genome region index loaded"
else
    info "VG region index not loaded (region_descriptions.json optional)"
fi

# GPU
GPU_AVAIL=$(echo "$HEALTH" | "$SERVER_DIR/venv/bin/python" -c \
    "import sys,json; print(json.load(sys.stdin).get('gpu_available',False))" 2>/dev/null || echo "False")
GPU_NAME=$(echo "$METRICS" | "$SERVER_DIR/venv/bin/python" -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('gpu',{}).get('backend','cpu'))" 2>/dev/null || echo "cpu")
if [ "$GPU_AVAIL" = "True" ]; then
    ok "GPU available  (backend: $GPU_NAME)"
else
    ok "Running on MPS / CPU  (backend: $GPU_NAME)"
fi

hdr "═══════════════════════════════════════"
printf "${GREEN}${BOLD}  Ready!  Open http://localhost:$CLIENT_PORT${NC}\n"
hdr "═══════════════════════════════════════"
printf "\n  Press ${BOLD}Ctrl+C${NC} to stop all services.\n\n"

# ── Keep script alive until Ctrl+C ─────────────────────────────────────────
wait
