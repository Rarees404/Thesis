#!/usr/bin/env bash
# download_visual_genome.sh — Download Visual Genome images + region descriptions.
#
# Visual Genome contains ~108k images split across two zip files,
# plus structured metadata (region descriptions, scene graphs, etc.).
#
# Total download: ~15 GB images + ~36 MB metadata.
#
# Run from the repository root:
#   bash scripts/download_visual_genome.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VG_DIR="$ROOT_DIR/data/visual_genome"

mkdir -p "$VG_DIR"

echo "========================================"
echo "  Visual Genome — Dataset Download"
echo "========================================"
echo ""
echo "Target: $VG_DIR"
echo ""

download() {
  local url="$1" dest="$2"
  if command -v wget &>/dev/null; then
    wget -c -O "$dest" "$url"
  elif command -v curl &>/dev/null; then
    curl -L -C - -o "$dest" "$url"
  else
    echo "Error: neither wget nor curl found. Install one and retry."
    exit 1
  fi
}

# ── Part 1: VG_100K ──
PART1_URL="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
PART1_ZIP="$VG_DIR/images.zip"

if [ -d "$VG_DIR/VG_100K" ] && [ "$(ls "$VG_DIR/VG_100K" 2>/dev/null | head -5 | wc -l)" -gt 0 ]; then
  echo "[1/4] VG_100K already present — skipping download."
else
  echo "[1/4] Downloading VG_100K (~10 GB)..."
  download "$PART1_URL" "$PART1_ZIP"
  echo "  Extracting..."
  unzip -q -o "$PART1_ZIP" -d "$VG_DIR"
  rm -f "$PART1_ZIP"
  echo "  Done."
fi

# ── Part 2: VG_100K_2 ──
PART2_URL="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
PART2_ZIP="$VG_DIR/images2.zip"

if [ -d "$VG_DIR/VG_100K_2" ] && [ "$(ls "$VG_DIR/VG_100K_2" 2>/dev/null | head -5 | wc -l)" -gt 0 ]; then
  echo "[2/4] VG_100K_2 already present — skipping download."
else
  echo "[2/4] Downloading VG_100K_2 (~5 GB)..."
  download "$PART2_URL" "$PART2_ZIP"
  echo "  Extracting..."
  unzip -q -o "$PART2_ZIP" -d "$VG_DIR"
  rm -f "$PART2_ZIP"
  echo "  Done."
fi

# ── Part 3: Region descriptions ──
REGIONS_URL="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
REGIONS_ZIP="$VG_DIR/region_descriptions.json.zip"
REGIONS_JSON="$VG_DIR/region_descriptions.json"

if [ -f "$REGIONS_JSON" ]; then
  echo "[3/4] region_descriptions.json already present — skipping download."
else
  echo "[3/4] Downloading region_descriptions.json (~36 MB)..."
  download "$REGIONS_URL" "$REGIONS_ZIP"
  echo "  Extracting..."
  unzip -q -o "$REGIONS_ZIP" -d "$VG_DIR"
  rm -f "$REGIONS_ZIP"
  echo "  Done."
fi

# ── Part 4: Image metadata ──
IMAGE_META_URL="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
IMAGE_META_ZIP="$VG_DIR/image_data.json.zip"
IMAGE_META_JSON="$VG_DIR/image_data.json"

if [ -f "$IMAGE_META_JSON" ]; then
  echo "[4/4] image_data.json already present — skipping download."
else
  echo "[4/4] Downloading image_data.json (~3 MB)..."
  download "$IMAGE_META_URL" "$IMAGE_META_ZIP"
  echo "  Extracting..."
  unzip -q -o "$IMAGE_META_ZIP" -d "$VG_DIR"
  rm -f "$IMAGE_META_ZIP"
  echo "  Done."
fi

TOTAL=$(find "$VG_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
echo ""
echo "========================================"
echo "  Visual Genome download complete!"
echo "  Images: $TOTAL"
echo "  Metadata: region_descriptions.json, image_data.json"
echo "  Location: $VG_DIR"
echo ""
echo "  Next steps:"
echo "    1. Build the FAISS index:"
echo "       bash scripts/build_index.sh vg"
echo ""
echo "    2. Switch server/.env to Visual Genome:"
echo "       CONFIG_PATH=../configs/demo/vg_siglip.yaml"
echo "       INDEX_PATH=../faiss/visual_genome/google/siglip-large-patch16-256/image_index.faiss"
echo ""
echo "    3. Restart: ./start.sh"
echo "========================================"
