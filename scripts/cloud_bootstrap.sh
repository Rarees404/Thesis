#!/usr/bin/env bash
# cloud_bootstrap.sh — Full one-shot setup for VisualRef on a cloud GPU instance.
#
# Usage:
#   bash cloud_bootstrap.sh /path/to/github_deploy_key
#
# The deploy key must have read access to git@github.com:Rarees404/Thesis.git
# Generate on your local machine:
#   ssh-keygen -t ed25519 -C "visualref-deploy" -f ~/.ssh/visualref_deploy -N ""
#   # Then add ~/.ssh/visualref_deploy.pub as a deploy key on the GitHub repo.
#   # Then scp the private key to the server:
#   # scp ~/.ssh/visualref_deploy user@SERVER_IP:/tmp/visualref_deploy
#   # Then run this script:
#   # bash cloud_bootstrap.sh /tmp/visualref_deploy

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { printf "${GREEN}  ✓ %s${NC}\n" "$1"; }
warn() { printf "${YELLOW}  ⚠ %s${NC}\n" "$1"; }
fail() { printf "${RED}  ✗ %s${NC}\n" "$1"; exit 1; }
info() { printf "${CYAN}  ℹ %s${NC}\n" "$1"; }
hdr()  { printf "\n${BOLD}══════════════════════════════════════${NC}\n${BOLD}  %s${NC}\n${BOLD}══════════════════════════════════════${NC}\n" "$1"; }

LOG_FILE="/tmp/visualref_bootstrap.log"
exec > >(tee -a "$LOG_FILE") 2>&1

hdr "VisualRef — Cloud Bootstrap"
info "Full log: $LOG_FILE"
echo ""

# ── Args ─────────────────────────────────────────────────────────────────────
DEPLOY_KEY="${1:-}"
if [ -z "$DEPLOY_KEY" ]; then
    fail "Usage: bash cloud_bootstrap.sh /path/to/github_deploy_key"
fi
if [ ! -f "$DEPLOY_KEY" ]; then
    fail "Deploy key not found: $DEPLOY_KEY"
fi

REPO_BRANCH="Protoype3.0"
REPO_SSH="git@github.com:Rarees404/Thesis.git"
INSTALL_DIR="$HOME/visualref"

# ─────────────────────────────────────────────────────────────────────────────
hdr "[1/9] System dependencies"
# ─────────────────────────────────────────────────────────────────────────────

info "Updating apt and installing base packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl wget unzip build-essential \
    python3 python3-pip python3-venv python3-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg libssl-dev ca-certificates gnupg lsb-release
ok "System packages installed"

# Node.js 20 LTS
if ! command -v node &>/dev/null || [[ "$(node --version | tr -d 'v' | cut -d. -f1)" -lt 18 ]]; then
    info "Installing Node.js 20 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    ok "Node.js $(node --version) installed"
else
    ok "Node.js $(node --version) already installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "[2/9] GitHub SSH deploy key"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p ~/.ssh
chmod 700 ~/.ssh

INSTALL_KEY="$HOME/.ssh/visualref_deploy"
cp "$DEPLOY_KEY" "$INSTALL_KEY"
chmod 600 "$INSTALL_KEY"

# Write SSH config (idempotent: remove old block if present)
SSH_CONFIG="$HOME/.ssh/config"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

if ! grep -q "Host github-visualref" "$SSH_CONFIG" 2>/dev/null; then
    cat >> "$SSH_CONFIG" << EOF

# VisualRef deploy key
Host github-visualref
  HostName github.com
  User git
  IdentityFile $INSTALL_KEY
  StrictHostKeyChecking no
  IdentitiesOnly yes
EOF
    ok "SSH config entry added"
else
    ok "SSH config entry already present"
fi

# Test the key
info "Testing GitHub SSH access..."
if ssh -T git@github.com -i "$INSTALL_KEY" -o StrictHostKeyChecking=no 2>&1 | grep -q "successfully authenticated"; then
    ok "GitHub SSH key is valid"
else
    warn "GitHub SSH test output did not show 'successfully authenticated' — proceeding anyway"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "[3/9] Clone repository"
# ─────────────────────────────────────────────────────────────────────────────

# Rewrite SSH URL to use our named host entry
REPO_SSH_NAMED="${REPO_SSH/github.com/github-visualref}"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Repo already cloned — pulling latest on branch $REPO_BRANCH..."
    git -C "$INSTALL_DIR" fetch origin
    git -C "$INSTALL_DIR" checkout "$REPO_BRANCH"
    git -C "$INSTALL_DIR" pull origin "$REPO_BRANCH"
    ok "Repository updated"
else
    info "Cloning $REPO_SSH → $INSTALL_DIR (branch: $REPO_BRANCH)..."
    GIT_SSH_COMMAND="ssh -i $INSTALL_KEY -o StrictHostKeyChecking=no" \
        git clone --branch "$REPO_BRANCH" "$REPO_SSH" "$INSTALL_DIR"
    ok "Repository cloned"
fi

cd "$INSTALL_DIR"
SERVER_DIR="$INSTALL_DIR/server"
CLIENT_DIR="$INSTALL_DIR/client-next"

# ─────────────────────────────────────────────────────────────────────────────
hdr "[4/9] Python virtual environment"
# ─────────────────────────────────────────────────────────────────────────────

VENV_DIR="$SERVER_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    ok "venv created"
else
    ok "venv already exists"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── Detect CUDA and install the right torch variant ─────────────────────────
CUDA_VER=""
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1 || echo "")
fi
if [ -z "$CUDA_VER" ] && command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || echo "")
fi

MAJOR="${CUDA_VER%%.*}"

if [ -n "$MAJOR" ] && [ "$MAJOR" -ge 11 ]; then
    # cu118 wheels cover CUDA 11.x and 12.x and include torch==2.1.2
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    info "CUDA $CUDA_VER detected → installing torch+cu118"
else
    TORCH_INDEX=""
    warn "No CUDA detected — installing CPU-only torch (inference will be slow)"
fi

# Install torch first (heavy, separate step so failures are obvious)
info "Installing torch==2.1.2..."
if [ -n "$TORCH_INDEX" ]; then
    pip install "torch==2.1.2" --extra-index-url "$TORCH_INDEX" --quiet
else
    pip install "torch==2.1.2" --quiet
fi
ok "torch installed"

# Install remaining requirements (skip torch lines, already installed)
info "Installing server requirements..."
grep -v "^torch==" "$SERVER_DIR/requirements.txt" | grep -v "^#" | grep -v "^$" \
    | pip install -r /dev/stdin --quiet
ok "Server requirements installed"

# ─────────────────────────────────────────────────────────────────────────────
hdr "[5/9] SAM 3 (facebook/sam3)"
# ─────────────────────────────────────────────────────────────────────────────

SAM3_DIR="$SERVER_DIR/sam3"

if [ -d "$SAM3_DIR" ] && [ -f "$SAM3_DIR/pyproject.toml" ]; then
    info "SAM 3 source already present — reinstalling editable package..."
else
    info "Cloning facebook/sam3..."
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
    ok "SAM 3 cloned"
fi

info "Installing SAM 3 (editable)..."
pip install -e "$SAM3_DIR" --quiet
ok "SAM 3 installed — weights will download from HuggingFace on first use"

deactivate

# ─────────────────────────────────────────────────────────────────────────────
hdr "[6/9] Environment files"
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -f "$SERVER_DIR/.env" ]; then
    cp "$SERVER_DIR/.env.example" "$SERVER_DIR/.env"
    ok "server/.env created from example"
else
    ok "server/.env already exists"
fi

if [ ! -f "$CLIENT_DIR/.env.local" ]; then
    cp "$CLIENT_DIR/.env.example" "$CLIENT_DIR/.env.local"
    ok "client-next/.env.local created from example"
else
    ok "client-next/.env.local already exists"
fi

# Patch server URL in client env to use 0.0.0.0 so it's reachable externally
SERVER_PORT=$(grep "^SERVER_PORT=" "$SERVER_DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' || echo "8001")
SERVER_PORT="${SERVER_PORT:-8001}"
if grep -q "NEXT_PUBLIC_SERVER_URL" "$CLIENT_DIR/.env.local"; then
    sed -i "s|NEXT_PUBLIC_SERVER_URL=.*|NEXT_PUBLIC_SERVER_URL=http://0.0.0.0:${SERVER_PORT}|" "$CLIENT_DIR/.env.local"
fi
ok "Client .env.local points to backend port $SERVER_PORT"

# ─────────────────────────────────────────────────────────────────────────────
hdr "[7/9] Ollama + llama3.2-vision"
# ─────────────────────────────────────────────────────────────────────────────

if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    ok "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
fi

# Start ollama serve in background if not running
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    info "Starting ollama serve..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "$OLLAMA_PID" > /tmp/ollama_bootstrap.pid

    # Wait up to 30s
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        warn "Ollama did not start — skipping model pull. Check /tmp/ollama.log"
    else
        ok "Ollama service is up"
    fi
else
    ok "Ollama service already running"
fi

if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    MODEL_PRESENT=$(curl -sf http://127.0.0.1:11434/api/tags 2>/dev/null \
        | "$VENV_DIR/bin/python" -c "
import sys, json
data = json.load(sys.stdin)
names = [m.get('name','') for m in data.get('models', [])]
found = any(n == 'llama3.2-vision' or n.startswith('llama3.2-vision:') for n in names)
print('yes' if found else 'no')
" 2>/dev/null || echo "no")

    if [ "$MODEL_PRESENT" = "yes" ]; then
        ok "llama3.2-vision already pulled"
    else
        info "Pulling llama3.2-vision (~5 GB, this takes a while)..."
        ollama pull llama3.2-vision
        ok "llama3.2-vision pulled"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "[8/9] Visual Genome dataset + FAISS index"
# ─────────────────────────────────────────────────────────────────────────────

VG_DIR="$INSTALL_DIR/data/visual_genome"
mkdir -p "$VG_DIR"

download_file() {
    local url="$1" dest="$2" label="$3"
    if [ -f "$dest" ]; then
        info "$label already downloaded — skipping"
        return
    fi
    info "Downloading $label..."
    wget -q --show-progress -c -O "$dest" "$url" || curl -L -# -C - -o "$dest" "$url"
}

# VG_100K images (~10 GB)
if [ -d "$VG_DIR/VG_100K" ] && [ "$(find "$VG_DIR/VG_100K" -name '*.jpg' | wc -l)" -gt 100 ]; then
    ok "VG_100K images already present"
else
    download_file \
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
        "$VG_DIR/images.zip" \
        "VG_100K (~10 GB)"
    info "Extracting VG_100K..."
    unzip -q -o "$VG_DIR/images.zip" -d "$VG_DIR"
    rm -f "$VG_DIR/images.zip"
    ok "VG_100K extracted"
fi

# VG_100K_2 images (~5 GB)
if [ -d "$VG_DIR/VG_100K_2" ] && [ "$(find "$VG_DIR/VG_100K_2" -name '*.jpg' | wc -l)" -gt 100 ]; then
    ok "VG_100K_2 images already present"
else
    download_file \
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
        "$VG_DIR/images2.zip" \
        "VG_100K_2 (~5 GB)"
    info "Extracting VG_100K_2..."
    unzip -q -o "$VG_DIR/images2.zip" -d "$VG_DIR"
    rm -f "$VG_DIR/images2.zip"
    ok "VG_100K_2 extracted"
fi

# Region descriptions (~36 MB)
if [ ! -f "$VG_DIR/region_descriptions.json" ]; then
    download_file \
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip" \
        "$VG_DIR/region_descriptions.json.zip" \
        "region_descriptions.json (~36 MB)"
    unzip -q -o "$VG_DIR/region_descriptions.json.zip" -d "$VG_DIR"
    rm -f "$VG_DIR/region_descriptions.json.zip"
    ok "region_descriptions.json extracted"
else
    ok "region_descriptions.json already present"
fi

# Image metadata (~3 MB)
if [ ! -f "$VG_DIR/image_data.json" ]; then
    download_file \
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip" \
        "$VG_DIR/image_data.json.zip" \
        "image_data.json (~3 MB)"
    unzip -q -o "$VG_DIR/image_data.json.zip" -d "$VG_DIR"
    rm -f "$VG_DIR/image_data.json.zip"
    ok "image_data.json extracted"
else
    ok "image_data.json already present"
fi

# Build FAISS index
FAISS_INDEX="$INSTALL_DIR/faiss/visual_genome/google/siglip-large-patch16-256/image_index.faiss"
if [ -f "$FAISS_INDEX" ]; then
    ok "FAISS index already built"
else
    info "Building FAISS index (this takes 30-90 min on CPU, ~10 min on GPU)..."
    bash "$INSTALL_DIR/scripts/build_index.sh" vg
    ok "FAISS index built"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "[9/9] Frontend dependencies"
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -d "$CLIENT_DIR/node_modules" ]; then
    info "Installing frontend npm packages..."
    (cd "$CLIENT_DIR" && npm install --silent)
    ok "Frontend packages installed"
else
    ok "Frontend packages already installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
hdr "Bootstrap complete!"
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_IMAGES=$(find "$VG_DIR" -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' 2>/dev/null | wc -l | tr -d ' ')

echo ""
ok "Repo:          $INSTALL_DIR"
ok "Images:        $TOTAL_IMAGES Visual Genome images"
ok "FAISS index:   $FAISS_INDEX"
ok "venv:          $VENV_DIR"
echo ""
info "To start the application:"
printf "  ${BOLD}cd $INSTALL_DIR && bash start.sh${NC}\n"
echo ""
info "To expose the frontend publicly (if on a cloud VM):"
printf "  Open port 3000 in your firewall/security group,\n"
printf "  then set NEXT_PUBLIC_SERVER_URL=http://YOUR_SERVER_IP:8001 in client-next/.env.local\n"
echo ""
info "Full log saved to: $LOG_FILE"
echo ""
