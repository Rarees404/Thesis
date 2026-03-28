#!/bin/bash
# build_all_indexes.sh — Build FAISS indexes for all available datasets,
# then build a combined index with everything.
#
# Usage:
#   cd visualref/
#   bash scripts/build_all_indexes.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="$(cd "$SCRIPT_DIR/../../data" 2>/dev/null && pwd || echo "")"

echo "═══ Building all FAISS indexes ═══"
echo ""

if [ -d "$DATA_ROOT/coco/val2014" ]; then
  echo "━━━ [1] COCO val2014 ━━━"
  bash "$SCRIPT_DIR/build_index.sh" coco
  echo ""
else
  echo "⚠ COCO val2014 not found at $DATA_ROOT/coco/val2014 — skipping"
fi

if [ -d "$DATA_ROOT/visual_genome" ]; then
  echo "━━━ [2] Visual Genome ━━━"
  bash "$SCRIPT_DIR/build_index.sh" vg
  echo ""
else
  echo "⚠ Visual Genome not found at $DATA_ROOT/visual_genome — skipping"
fi

if [ -d "$DATA_ROOT/retail" ]; then
  echo "━━━ [3] Retail ━━━"
  bash "$SCRIPT_DIR/build_index.sh" retail
  echo ""
else
  echo "⚠ Retail not found at $DATA_ROOT/retail — skipping"
fi

echo "━━━ [4] Combined index ━━━"
bash "$SCRIPT_DIR/build_index.sh" combined

echo ""
echo "═══ All indexes built ═══"
echo ""
echo "Available configs:"
[ -d "$DATA_ROOT/coco/val2014" ]   && echo "  COCO:     configs/demo/coco_siglip.yaml"
[ -d "$DATA_ROOT/visual_genome" ]  && echo "  VG:       configs/demo/vg_siglip.yaml"
[ -d "$DATA_ROOT/retail" ]         && echo "  Retail:   configs/demo/retail_siglip.yaml"
echo "  Combined: configs/demo/combined_siglip.yaml"
