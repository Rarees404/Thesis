#!/usr/bin/env bash
# build_index.sh — Build the Visual Genome FAISS index.
#
# Run from the repository root:
#   bash scripts/build_index.sh [model]
#
# Example:
#   bash scripts/build_index.sh           # Visual Genome (data/visual_genome/)
#
# Requires: server venv (or python with torch). Images under <repo>/data/visual_genome/
# (see README). If region_descriptions.json is present, a hybrid region index is also built.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$ROOT_DIR/data"

MODEL_FAMILY="siglip"
MODEL_ID="google/siglip-large-patch16-256"
# Images encoded per forward pass. Raise for more throughput on a big GPU,
# lower if you hit out-of-memory on MPS/CPU.
BATCH_SIZE=16

# Detect device (torch must be importable). Prefer the server venv's python.
if "$ROOT_DIR/server/venv/bin/python" -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  DEVICE="mps"
elif "$ROOT_DIR/server/venv/bin/python" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  DEVICE="mps"
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi

PYTHON="${ROOT_DIR}/server/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

DATA_PATHS="$DATA_ROOT/visual_genome"
OUTPUT_DIR="$ROOT_DIR/faiss/visual_genome"

if [ ! -d "$DATA_PATHS" ] || [ -z "$(ls -A "$DATA_PATHS" 2>/dev/null)" ]; then
  echo "Error: data directory missing or empty: $DATA_PATHS"
  echo "Prepare images under $DATA_ROOT (see README)."
  exit 1
fi

echo ""
echo "Building Visual Genome FAISS index"
echo "  Model:   $MODEL_ID"
echo "  Data:    $DATA_PATHS"
echo "  Output:  $OUTPUT_DIR"
echo "  Device:  $DEVICE"
echo ""

# Hybrid mode: when VG region phrase annotations are present, also build the
# region index used for hard-filter kNN and region-phrase lookups.
VG_REGIONS_ARG=""
VG_REGIONS="$DATA_ROOT/visual_genome/region_descriptions.json"
if [ -f "$VG_REGIONS" ]; then
  VG_REGIONS_ARG="--vg_regions $VG_REGIONS"
  echo "  VG hybrid: region_descriptions.json found → building hybrid index"
fi

cd "$ROOT_DIR/server"
"$PYTHON" -m src.utils.write_faiss_index \
  --data "$DATA_PATHS" \
  --output "$OUTPUT_DIR" \
  --model_family "$MODEL_FAMILY" \
  --model_id "$MODEL_ID" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  $VG_REGIONS_ARG

echo ""
echo "Done! Index: $OUTPUT_DIR/$MODEL_ID/image_index.faiss"
echo "Paths file: $OUTPUT_DIR/$MODEL_ID/image_paths.txt"
echo ""
echo "Update server/.env (paths relative to server/):"
echo "  CONFIG_PATH=../configs/demo/vg_siglip.yaml"
echo "  INDEX_PATH=../faiss/visual_genome/${MODEL_ID}/image_index.faiss"
echo ""
