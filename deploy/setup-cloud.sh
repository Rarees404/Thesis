#!/usr/bin/env bash
#
# Run this on the cloud GPU instance after cloning the repo.
# It sets up the Python env, installs dependencies, and starts the server.
#
set -euo pipefail

GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { printf "${GREEN}  ✓ %s${NC}\n" "$1"; }
info() { printf "${CYAN}  ℹ %s${NC}\n" "$1"; }
hdr()  { printf "\n${BOLD}%s${NC}\n" "$1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$ROOT_DIR/server"

hdr "═══ VisualRef Cloud Setup ═══"

# ── 1. Python env ──
hdr "[1/6] Python environment"
cd "$SERVER_DIR"
if [ ! -d venv ]; then
    python3 -m venv venv
    ok "Created venv"
fi
source venv/bin/activate
pip install --upgrade pip -q
ok "Activated venv"

# ── 2. Install deps ──
hdr "[2/6] Installing Python dependencies"
pip install -r requirements.txt -q
pip install requests -q

# Install SAM 2 (public — no HuggingFace login needed)
if python -c "import sam2" 2>/dev/null; then
    ok "sam2 already installed"
else
    pip install "git+https://github.com/facebookresearch/sam2.git" -q
    ok "SAM 2 installed"
fi
ok "Dependencies installed"

# ── 3. HuggingFace (SAM 2 weights are public, no auth needed) ──
hdr "[3/6] HuggingFace cache"
info "SAM 2.1 weights auto-download from HuggingFace Hub on first run (public model)."

# ── 4. Use cloud .env ──
hdr "[4/6] Configuring for cloud (SAM 2 + CUDA)"
cp "$SCRIPT_DIR/env.cloud" "$SERVER_DIR/.env"
ok "Copied cloud .env (SAM_BACKEND=sam2)"

# ── 5. Check data ──
hdr "[5/6] Checking data files"
FAISS_PATH="$ROOT_DIR/faiss/visual_genome/google/siglip-large-patch16-256/image_index.faiss"
DATA_PATH="$ROOT_DIR/data/visual_genome"

if [ -f "$FAISS_PATH" ]; then
    ok "FAISS index found ($(du -sh "$FAISS_PATH" | cut -f1))"
else
    echo "  ✗ FAISS index missing at: $FAISS_PATH"
    echo "    Build on the instance or upload faiss/visual_genome/ (see README)."
fi

if [ -d "$DATA_PATH" ]; then
    ok "Visual Genome images directory present: $DATA_PATH"
else
    echo "  ✗ Visual Genome images missing at: $DATA_PATH"
    echo "    Run scripts/download_visual_genome.sh on the instance or rsync data/visual_genome/"
fi

# ── 6. Install & start Ollama ──
hdr "[6/6] Ollama Vision (optional)"
if command -v ollama &>/dev/null; then
    ok "Ollama is installed"
    if ollama list 2>/dev/null | grep -q "llama3.2-vision"; then
        ok "llama3.2-vision model available"
    else
        info "Pull the model: ollama pull llama3.2-vision"
    fi
else
    info "Ollama not installed. To enable VLM captions:"
    echo "       curl -fsSL https://ollama.com/install.sh | sh"
    echo "       ollama pull llama3.2-vision"
    echo "       (Optional — system works without it)"
fi

hdr "═══ Setup Complete ═══"
echo ""
echo "  To start the server:"
echo "    cd server && source venv/bin/activate"
echo "    python -m uvicorn src.retrieval_server_visual:app --host 0.0.0.0 --port 8001"
echo ""
echo "  Then on your Mac, update client-next/.env.local:"
echo "    NEXT_PUBLIC_SERVER_URL=http://<cloud-ip>:8001"
echo ""
