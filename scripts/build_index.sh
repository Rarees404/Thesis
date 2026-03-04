#!/bin/bash
# build_index.sh — Rebuild the FAISS index locally after cloning.
# The index files are NOT stored in git (too large). Run this script once
# after cloning, or whenever you change the model or dataset.
#
# Usage:
#   cd visualref/
#   bash scripts/build_index.sh [model] [dataset_path]
#
# Examples:
#   bash scripts/build_index.sh siglip ../../data/coco/val2014
#   bash scripts/build_index.sh clip   ../../data/coco/val2014

set -e

MODEL=${1:-siglip}
DATA=${2:-../../data/coco/val2014}

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
if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  DEVICE="mps"
elif python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
else
  DEVICE="cpu"
fi

echo "Building FAISS index"
echo "  Model:   $MODEL_ID"
echo "  Data:    $DATA"
echo "  Device:  $DEVICE"
echo "  Batches: $BATCH_SIZE"
echo ""

cd server

python -m src.utils.write_faiss_index \
  --data "$DATA" \
  --output ../faiss/coco \
  --model_family "$MODEL_FAMILY" \
  --model_id "$MODEL_ID" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"

echo ""
echo "Done. Start the server with:"
echo ""
echo "  CONFIG_PATH=../configs/demo/coco_${MODEL}.yaml \\"
echo "  INDEX_PATH=../faiss/coco/${MODEL_ID}/image_index.faiss \\"
echo "  python -m uvicorn src.retrieval_server_visual:app --host 0.0.0.0 --port 8001"
