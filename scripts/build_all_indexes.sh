#!/usr/bin/env bash
# Rebuild the Visual Genome FAISS index (SigLIP + optional VG hybrid metadata).
#
# Thin wrapper around build_index.sh that first checks the VG data is present.
# Run from repo root:  bash scripts/build_all_indexes.sh

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

bash "$SCRIPT_DIR/build_index.sh"

echo ""
echo "Done."
