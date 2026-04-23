#!/usr/bin/env bash
# Rebuild the Visual Genome FAISS index (SigLIP + optional VG hybrid metadata).
#
# Run from repo root:  bash scripts/build_all_indexes.sh
#
# For MS-COCO (optional), place images under data/coco/val2014/ and run:
#   bash scripts/build_index.sh coco

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$ROOT_DIR/data"

echo "Building Visual Genome index under $DATA_ROOT"
echo ""

if [ ! -d "$DATA_ROOT/visual_genome" ]; then
  echo "Error: Visual Genome data not found at $DATA_ROOT/visual_genome"
  echo "Run:  bash scripts/download_visual_genome.sh"
  exit 1
fi

bash "$SCRIPT_DIR/build_index.sh" vg

echo ""
echo "Done."
