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

# Install SAM3
if [ -d "$SERVER_DIR/sam3" ]; then
    cd "$SERVER_DIR/sam3"
    pip install -e . -q
    cd "$SERVER_DIR"
    ok "SAM3 installed"
else
    info "sam3/ not found — clone it: git clone https://github.com/facebookresearch/sam3.git server/sam3"
fi
ok "Dependencies installed"

# ── 3. HuggingFace login (for SAM3 gated model) ──
hdr "[3/6] HuggingFace authentication"
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    ok "Already logged in to HuggingFace"
else
    info "SAM3 requires HuggingFace access. Run:"
    echo "       huggingface-cli login"
    echo "       (Use a token from https://huggingface.co/settings/tokens)"
fi

# ── 4. Use cloud .env ──
hdr "[4/6] Configuring for cloud (SAM3 + CUDA)"
cp "$SCRIPT_DIR/env.cloud" "$SERVER_DIR/.env"
ok "Copied cloud .env (SAM_BACKEND=sam3)"

# ── 5. Check data ──
hdr "[5/6] Checking data files"
FAISS_PATH="$ROOT_DIR/faiss/coco/google/siglip-large-patch16-256/image_index.faiss"
DATA_PATH="$ROOT_DIR/data/coco/val2014"

if [ -f "$FAISS_PATH" ]; then
    ok "FAISS index found ($(du -sh "$FAISS_PATH" | cut -f1))"
else
    echo "  ✗ FAISS index missing at: $FAISS_PATH"
    echo "    Upload it: scp -r faiss/ user@cloud-ip:/path/to/visualref/"
fi

if [ -d "$DATA_PATH" ]; then
    COUNT=$(ls "$DATA_PATH" | wc -l)
    ok "COCO val2014 found ($COUNT images)"
else
    echo "  ✗ COCO images missing at: $DATA_PATH"
    echo "    Upload: scp -r data/ user@cloud-ip:/path/to/visualref/"
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
