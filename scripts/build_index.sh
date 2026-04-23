#!/usr/bin/env bash
# build_index.sh — Build ONE FAISS index per invocation (does not touch other corpora).
#
# Run from the repository root:
#   bash scripts/build_index.sh <dataset> [model]
#
# Datasets: vg (default) | coco | combined
#
# Examples:
#   bash scripts/build_index.sh           # Visual Genome (default)
#   bash scripts/build_index.sh coco      # COCO val2014 (requires data/coco/val2014)
#   bash scripts/build_index.sh combined  # merges coco+vg dirs when both exist
#
# For a full VG rebuild only:  bash scripts/build_all_indexes.sh
#
# Requires: server venv (or python with torch). Data under <repo>/data/ (see README).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$ROOT_DIR/data"

DATASET="${1:-vg}"

MODEL_FAMILY="siglip"
MODEL_ID="google/siglip-large-patch16-256"
BATCH_SIZE=16

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
  combined)
    COMBINED_TMP="$(mktemp -d)"
    trap 'rm -rf "$COMBINED_TMP"' EXIT
    echo "Creating combined dataset from available sources..."
    for src in "$DATA_ROOT/coco/val2014" "$DATA_ROOT/visual_genome"; do
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
    echo "Options: coco, vg, combined"
    exit 1
    ;;
esac

if [ ! -d "$DATA_PATHS" ] || [ -z "$(ls -A "$DATA_PATHS" 2>/dev/null)" ]; then
  echo "Error: data directory missing or empty: $DATA_PATHS"
  echo "Prepare images under $DATA_ROOT (see README)."
  exit 1
fi

echo ""
echo "Building FAISS index (only this corpus — other faiss/* trees are untouched)"
echo "  Dataset: $DATASET"
echo "  Model:   $MODEL_ID"
echo "  Data:    $DATA_PATHS"
echo "  Output:  $OUTPUT_DIR"
echo "  Device:  $DEVICE"
echo ""

VG_REGIONS_ARG=""
if [ "$DATASET" = "vg" ] || [ "$DATASET" = "combined" ]; then
  VG_REGIONS="$DATA_ROOT/visual_genome/region_descriptions.json"
  if [ -f "$VG_REGIONS" ]; then
    VG_REGIONS_ARG="--vg_regions $VG_REGIONS"
    echo "  VG hybrid: region_descriptions.json found → building hybrid index"
  fi
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

case "$DATASET" in
  coco)     REL_OUT="faiss/coco" ;;
  vg)       REL_OUT="faiss/visual_genome" ;;
  combined) REL_OUT="faiss/combined" ;;
esac

case "$DATASET" in
  coco)     CFG="coco_siglip" ;;
  vg)       CFG="vg_siglip" ;;
  combined) CFG="combined_siglip" ;;
esac

echo ""
echo "Done! Index: $OUTPUT_DIR/$MODEL_ID/image_index.faiss"
echo "Paths file: $OUTPUT_DIR/$MODEL_ID/image_paths.txt"
echo ""
echo "Update server/.env (paths relative to server/):"
echo "  CONFIG_PATH=../configs/demo/${CFG}.yaml"
echo "  INDEX_PATH=../${REL_OUT}/${MODEL_ID}/image_index.faiss"
echo ""

