#!/usr/bin/env bash
# build_index.sh — Build FAISS indexes for retrieval.
#
# Run from the repository root:
#   bash scripts/build_index.sh <dataset> [model]
#
# Datasets: coco | vg | retail | combined
# Models:   siglip (default) | clip
#
# Requires: server venv active OR use system python with torch installed.
# Data must live under <repo>/data/ (see README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$ROOT_DIR/data"

DATASET="${1:-coco}"
MODEL="${2:-siglip}"

if [ "$MODEL" = "siglip" ]; then
  MODEL_FAMILY="siglip"
  MODEL_ID="google/siglip-large-patch16-256"
  BATCH_SIZE=16
elif [ "$MODEL" = "clip" ]; then
  MODEL_FAMILY="clip"
  MODEL_ID="openai/clip-vit-large-patch14"
  BATCH_SIZE=32
else
  echo "Unknown model: $MODEL. Use 'siglip' or 'clip'."
  exit 1
fi

# Detect device (torch must be importable)
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

case "$DATASET" in
  coco)
    DATA_PATHS="$DATA_ROOT/coco/val2014"
    OUTPUT_DIR="$ROOT_DIR/faiss/coco"
    ;;
  vg)
    DATA_PATHS="$DATA_ROOT/visual_genome"
    OUTPUT_DIR="$ROOT_DIR/faiss/visual_genome"
    ;;
  retail)
    DATA_PATHS="$DATA_ROOT/retail"
    OUTPUT_DIR="$ROOT_DIR/faiss/retail"
    ;;
  combined)
    COMBINED_TMP="$(mktemp -d)"
    trap 'rm -rf "$COMBINED_TMP"' EXIT
    echo "Creating combined dataset from available sources..."
    for src in "$DATA_ROOT/coco/val2014" "$DATA_ROOT/visual_genome" "$DATA_ROOT/retail"; do
      if [ -d "$src" ]; then
        BASENAME="$(basename "$src")"
        PARENT="$(basename "$(dirname "$src")")"
        ln -sf "$(cd "$src" && pwd)" "$COMBINED_TMP/${PARENT}_${BASENAME}"
        echo "  + $src"
      else
        echo "  - $src (not found, skipping)"
      fi
    done
    DATA_PATHS="$COMBINED_TMP"
    OUTPUT_DIR="$ROOT_DIR/faiss/combined"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Options: coco, vg, retail, combined"
    exit 1
    ;;
esac

if [ ! -d "$DATA_PATHS" ] || [ -z "$(ls -A "$DATA_PATHS" 2>/dev/null)" ]; then
  echo "Error: data directory missing or empty: $DATA_PATHS"
  echo "Prepare images under $DATA_ROOT (see README)."
  exit 1
fi

echo ""
echo "Building FAISS index"
echo "  Dataset: $DATASET"
echo "  Model:   $MODEL_ID"
echo "  Data:    $DATA_PATHS"
echo "  Output:  $OUTPUT_DIR"
echo "  Device:  $DEVICE"
echo ""

cd "$ROOT_DIR/server"
"$PYTHON" -m src.utils.write_faiss_index \
  --data "$DATA_PATHS" \
  --output "$OUTPUT_DIR" \
  --model_family "$MODEL_FAMILY" \
  --model_id "$MODEL_ID" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"

case "$DATASET" in
  coco)     REL_OUT="faiss/coco" ;;
  vg)       REL_OUT="faiss/visual_genome" ;;
  retail)   REL_OUT="faiss/retail" ;;
  combined) REL_OUT="faiss/combined" ;;
esac

if [ "$MODEL" = "siglip" ]; then
  CFG_CASE="$DATASET"
  case "$CFG_CASE" in
    coco)     CFG="coco_siglip" ;;
    vg)       CFG="vg_siglip" ;;
    retail)   CFG="retail_siglip" ;;
    combined) CFG="combined_siglip" ;;
  esac
else
  CFG="(use a configs/demo/*_clip*.yaml matching your dataset — CLIP paths differ)"
fi

echo ""
echo "Done! Index: $OUTPUT_DIR/$MODEL_ID/image_index.faiss"
echo "Paths file: $OUTPUT_DIR/$MODEL_ID/image_paths.txt"
echo ""
echo "Update server/.env (paths relative to server/):"
if [ "$MODEL" = "siglip" ]; then
  echo "  CONFIG_PATH=../configs/demo/${CFG}.yaml"
else
  echo "  CONFIG_PATH=../configs/demo/<matching_clip_config>.yaml"
fi
echo "  INDEX_PATH=../${REL_OUT}/${MODEL_ID}/image_index.faiss"
echo ""

