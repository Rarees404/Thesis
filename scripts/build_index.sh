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
#   retail       — data/retail (RP2K, etc.)
#   retail786k   — data/retail786k (Retail-786k 256px)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$REPO_ROOT/.." && pwd)"
# Images live next to the repo: /workspace/data/... when repo is /workspace/visualref
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/data}"

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

# Detect device (override: DEVICE=cpu bash ... for GPUs PyTorch does not support yet, e.g. sm_120)
if [ -n "${DEVICE:-}" ]; then
  :
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  DEVICE="mps"
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi

case "$DATASET" in
  coco)
    DATA_PATHS="$DATA_ROOT/coco/val2014"
    OUTPUT_DIR="$REPO_ROOT/faiss/coco"
    ;;
  vg)
    DATA_PATHS="$DATA_ROOT/visual_genome"
    OUTPUT_DIR="$REPO_ROOT/faiss/visual_genome"
    ;;
  retail)
    DATA_PATHS="$DATA_ROOT/retail"
    OUTPUT_DIR="$REPO_ROOT/faiss/retail"
    ;;
  retail786k)
    DATA_PATHS="$DATA_ROOT/retail786k"
    OUTPUT_DIR="$REPO_ROOT/faiss/retail786k"
    ;;
  combined)
    # For combined, we pass a temp directory with symlinks to all datasets
    COMBINED_TMP=$(mktemp -d)
    trap "rm -rf $COMBINED_TMP" EXIT

    echo "Creating combined dataset from all available sources..."
    for src in "$DATA_ROOT/coco/val2014" "$DATA_ROOT/visual_genome" "$DATA_ROOT/retail" "$DATA_ROOT/retail786k"; do
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
    OUTPUT_DIR="$REPO_ROOT/faiss/combined"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Options: coco, vg, retail, retail786k, combined"
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

cd "$REPO_ROOT/server"

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
  coco)        CFG="coco_siglip"; FAISS_SUB="coco" ;;
  vg)          CFG="vg_siglip"; FAISS_SUB="visual_genome" ;;
  retail)      CFG="retail_siglip"; FAISS_SUB="retail" ;;
  retail786k)  CFG="retail786k_siglip"; FAISS_SUB="retail786k" ;;
  combined)    CFG="combined_siglip"; FAISS_SUB="combined" ;;
esac

echo "To use this index, set in server/.env (paths relative to server/):"
echo ""
echo "  CONFIG_PATH=../configs/demo/${CFG}.yaml"
echo "  INDEX_PATH=../faiss/${FAISS_SUB}/${MODEL_ID}/image_index.faiss"
echo ""
