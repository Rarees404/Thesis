#!/bin/bash
# build_index.sh — Build FAISS indexes for retrieval.
#
# Usage:
#   cd visualref/
#   bash scripts/build_index.sh <dataset> [model]
#
# Datasets:
#   coco         — COCO val2014 (~40K images)
#   vg           — Visual Genome (~108K images)
#   retail       — Retail product images
#   combined     — All datasets merged into one index
#
# Models:
#   siglip  (default) — google/siglip-large-patch16-256
#   clip              — openai/clip-vit-large-patch14
#
# Examples:
#   bash scripts/build_index.sh coco
#   bash scripts/build_index.sh vg
#   bash scripts/build_index.sh retail
#   bash scripts/build_index.sh combined
#   bash scripts/build_index.sh coco clip

set -e

DATASET=${1:-coco}
MODEL=${2:-siglip}

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

# Detect device
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  DEVICE="mps"
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi

DATA_ROOT="../../data"

case "$DATASET" in
  coco)
    DATA_PATHS="$DATA_ROOT/coco/val2014"
    OUTPUT_DIR="../faiss/coco"
    ;;
  vg)
    DATA_PATHS="$DATA_ROOT/visual_genome"
    OUTPUT_DIR="../faiss/visual_genome"
    ;;
  retail)
    DATA_PATHS="$DATA_ROOT/retail"
    OUTPUT_DIR="../faiss/retail"
    ;;
  combined)
    # For combined, we pass a temp directory with symlinks to all datasets
    COMBINED_TMP=$(mktemp -d)
    trap "rm -rf $COMBINED_TMP" EXIT

    echo "Creating combined dataset from all available sources..."
    for src in "$DATA_ROOT/coco/val2014" "$DATA_ROOT/visual_genome" "$DATA_ROOT/retail"; do
      if [ -d "$src" ]; then
        BASENAME=$(basename "$src")
        PARENT=$(basename "$(dirname "$src")")
        LINK_NAME="${PARENT}_${BASENAME}"
        ln -sf "$(cd "$src" && pwd)" "$COMBINED_TMP/$LINK_NAME"
        echo "  + $src"
      else
        echo "  - $src (not found, skipping)"
      fi
    done

    DATA_PATHS="$COMBINED_TMP"
    OUTPUT_DIR="../faiss/combined"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Options: coco, vg, retail, combined"
    exit 1
    ;;
esac

echo ""
echo "Building FAISS index"
echo "  Dataset: $DATASET"
echo "  Model:   $MODEL_ID"
echo "  Data:    $DATA_PATHS"
echo "  Output:  $OUTPUT_DIR"
echo "  Device:  $DEVICE"
echo "  Batches: $BATCH_SIZE"
echo ""

cd server

python -m src.utils.write_faiss_index \
  --data "$DATA_PATHS" \
  --output "$OUTPUT_DIR" \
  --model_family "$MODEL_FAMILY" \
  --model_id "$MODEL_ID" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"

echo ""
echo "Done! Index saved to $OUTPUT_DIR/$MODEL_ID/"
echo ""

case "$DATASET" in
  coco)     CFG="coco_siglip" ;;
  vg)       CFG="vg_siglip" ;;
  retail)   CFG="retail_siglip" ;;
  combined) CFG="combined_siglip" ;;
esac

echo "To use this index, update server/.env:"
echo ""
echo "  CONFIG_PATH=../configs/demo/${CFG}.yaml"
echo "  INDEX_PATH=../$OUTPUT_DIR/$MODEL_ID/image_index.faiss"
echo ""
