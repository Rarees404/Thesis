#!/usr/bin/env bash
set -euo pipefail

# =========================================================================
# setup_models.sh — Download and install SAM 2 + Ollama 3.2 Vision
# Run from the repository root: bash scripts/setup_models.sh
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_DIR="$REPO_ROOT/server"

# Prefer the server venv's pip/python if present, so the install lands in the
# same environment the backend will run under.
if [ -x "$SERVER_DIR/venv/bin/python" ]; then
    PY="$SERVER_DIR/venv/bin/python"
    PIP="$SERVER_DIR/venv/bin/pip"
else
    PY="$(command -v python3 || command -v python)"
    PIP="$(command -v pip3 || command -v pip)"
fi

echo "========================================"
echo "  VisualReF — Model Setup"
echo "========================================"
echo ""

# ----- 1. SAM 2 (facebook/sam2 — public, no HF auth required) -----
echo "[1/3] Setting up SAM 2..."
if "$PY" -c "import sam2" 2>/dev/null; then
    echo "  sam2 Python package already installed"
else
    echo "  Installing sam2 from GitHub (facebookresearch/sam2)..."
    "$PIP" install "git+https://github.com/facebookresearch/sam2.git" 2>&1 | tail -3
fi
echo "  SAM 2 ready. Weights (~162 MB for base_plus) download from HuggingFace on first run."
echo ""

# ----- 2. Ollama -----
echo "[2/3] Setting up Ollama..."
if command -v ollama &> /dev/null; then
    echo "  Ollama is already installed: $(ollama --version 2>&1 || echo 'unknown version')"
else
    echo "  Ollama not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS, install Ollama from: https://ollama.com/download/mac"
        echo "  Or via Homebrew: brew install ollama"
        echo ""
        echo "  Attempting brew install..."
        if command -v brew &> /dev/null; then
            brew install ollama || echo "  Homebrew install failed — download from https://ollama.com"
        else
            echo "  Homebrew not found. Please install Ollama manually from https://ollama.com"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  Installing Ollama via official script..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "  Unsupported OS. Download Ollama from https://ollama.com"
    fi
fi
echo ""

# ----- 3. Pull Llama 3.2 Vision model -----
echo "[3/3] Pulling llama3.2-vision model..."
if command -v ollama &> /dev/null; then
    echo "  Starting Ollama server (if not already running)..."
    ollama serve &>/dev/null &
    sleep 2

    echo "  Pulling llama3.2-vision (this may take a while on first download)..."
    ollama pull llama3.2-vision || echo "  Pull failed — make sure Ollama is running: ollama serve"
    echo ""
    echo "  Verifying model is available..."
    ollama list | grep -i "llama3.2-vision" && echo "  llama3.2-vision is ready!" || echo "  Model not found in list — check ollama pull output above"
else
    echo "  Skipping model pull — Ollama is not installed"
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  To start the application:"
echo "    ./start.sh"
echo ""
echo "  Make sure Ollama is running:"
echo "    ollama serve"
echo "========================================"
