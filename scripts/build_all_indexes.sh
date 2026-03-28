#!/usr/bin/env bash
# Build FAISS indexes for every dataset present under data/, then a combined index.
# Run from repo root:  bash scripts/build_all_indexes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$ROOT_DIR/data"

echo "Building FAISS indexes (SigLIP) for available datasets under $DATA_ROOT"
echo ""

if [ -d "$DATA_ROOT/coco/val2014" ]; then
  echo "━━━ COCO val2014 ━━━"
  bash "$SCRIPT_DIR/build_index.sh" coco
  echo ""
else
  echo "⚠ Skip COCO — not found at $DATA_ROOT/coco/val2014"
fi

if [ -d "$DATA_ROOT/visual_genome" ]; then
  echo "━━━ Visual Genome ━━━"
  bash "$SCRIPT_DIR/build_index.sh" vg
  echo ""
else
  echo "⚠ Skip Visual Genome — not found at $DATA_ROOT/visual_genome"
fi

if [ -d "$DATA_ROOT/retail" ]; then
  echo "━━━ Retail ━━━"
  bash "$SCRIPT_DIR/build_index.sh" retail
  echo ""
else
  echo "⚠ Skip Retail — not found at $DATA_ROOT/retail"
fi

echo "━━━ Combined ━━━"
bash "$SCRIPT_DIR/build_index.sh" combined

echo ""
echo "Done."
