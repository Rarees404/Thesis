#!/bin/bash
# build_all_indexes.sh — Build FAISS indexes for all datasets present under <workspace>/data.
#
# Usage:
#   cd /workspace/visualref
#   bash scripts/build_all_indexes.sh
#
# Same DATA_ROOT as build_index.sh (parent of repo + /data).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$REPO_ROOT/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/data}"

echo "═══ Building all FAISS indexes ═══"
echo "DATA_ROOT=$DATA_ROOT"
echo ""

if [ -d "$DATA_ROOT/coco/val2014" ]; then
  echo "━━━ [1] COCO val2014 ━━━"
  bash "$SCRIPT_DIR/build_index.sh" coco
  echo ""
else
  echo "⚠ COCO val2014 not found at $DATA_ROOT/coco/val2014 — run: bash scripts/download_datasets.sh"
fi

if [ -d "$DATA_ROOT/visual_genome" ]; then
  echo "━━━ [2] Visual Genome ━━━"
  bash "$SCRIPT_DIR/build_index.sh" vg
  echo ""
else
  echo "⚠ Visual Genome not found — optional: bash scripts/download_datasets.sh --with-vg"
fi

if [ -d "$DATA_ROOT/retail" ]; then
  echo "━━━ [3] Retail (data/retail) ━━━"
  bash "$SCRIPT_DIR/build_index.sh" retail
  echo ""
elif [ -d "$DATA_ROOT/retail786k" ]; then
  echo "━━━ [3] Retail-786k ━━━"
  bash "$SCRIPT_DIR/build_index.sh" retail786k
  echo ""
else
  echo "⚠ No $DATA_ROOT/retail — skipping retail index"
  echo ""
fi

echo "━━━ [4] Combined index ━━━"
bash "$SCRIPT_DIR/build_index.sh" combined

echo ""
echo "═══ All indexes built ═══"
echo ""
echo "Default single-dataset .env (COCO):"
echo "  CONFIG_PATH=../configs/demo/coco_siglip.yaml"
echo "  INDEX_PATH=../faiss/coco/google/siglip-large-patch16-256/image_index.faiss"
echo ""
echo "Combined:"
echo "  CONFIG_PATH=../configs/demo/combined_siglip.yaml"
echo "  INDEX_PATH=../faiss/combined/google/siglip-large-patch16-256/image_index.faiss"
