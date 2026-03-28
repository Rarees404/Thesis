#!/usr/bin/env bash
# download_datasets.sh — Download image corpora under <workspace>/data/ (sibling of visualref).
#
# Usage (from repo root):
#   bash scripts/download_datasets.sh              # COCO val2014 only (~6.3 GB)
#   bash scripts/download_datasets.sh --with-vg    # + Visual Genome (~15 GB)
#   bash scripts/download_datasets.sh --with-retail786k-256  # + Retail-786k 256px (large)
#
# Override data root:
#   DATA_ROOT=/data bash scripts/download_datasets.sh
#
set -euo pipefail

WITH_VG=0
WITH_RETAIL=0

for arg in "$@"; do
  case "$arg" in
    --with-vg) WITH_VG=1 ;;
    --with-retail786k-256) WITH_RETAIL=1 ;;
    -h|--help)
      echo "Usage: $0 [--with-vg] [--with-retail786k-256]"
      exit 0
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$REPO_ROOT/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/data}"

echo "Data root: $DATA_ROOT"
mkdir -p "$DATA_ROOT"

download_coco_val2014() {
  local dest="$DATA_ROOT/coco"
  mkdir -p "$dest"
  if [ -d "$dest/val2014" ] && [ "$(find "$dest/val2014" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "[COCO] val2014 already present ($(find "$dest/val2014" -type f 2>/dev/null | wc -l) files), skipping."
    return 0
  fi
  echo "[COCO] Downloading val2014.zip (~1 GB) from cocodataset.org …"
  cd "$dest"
  curl -L --fail --retry 3 -o val2014.zip "http://images.cocodataset.org/zips/val2014.zip"
  echo "[COCO] Unzipping (this takes a few minutes) …"
  unzip -q val2014.zip
  rm -f val2014.zip
  echo "[COCO] Done: $dest/val2014"
}

download_visual_genome() {
  local dest="$DATA_ROOT/visual_genome"
  mkdir -p "$dest"
  cd "$dest"
  if [ -d VG_100K ] && [ "$(find VG_100K -type f 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "[VG] Already looks populated, skipping."
    return 0
  fi
  echo "[VG] Downloading images.zip (~9 GB) …"
  curl -L --fail --retry 3 -O "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
  echo "[VG] Downloading images2.zip (~5.5 GB) …"
  curl -L --fail --retry 3 -O "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
  echo "[VG] Unzipping …"
  unzip -q images.zip && unzip -q images2.zip
  rm -f images.zip images2.zip
  echo "[VG] Done: $dest"
}

download_retail786k_256() {
  local dest="$DATA_ROOT/retail786k"
  mkdir -p "$dest"
  cd "$dest"
  if [ -f .extracted_ok ]; then
    echo "[Retail-786k] Already extracted, skipping."
    return 0
  fi
  echo "[Retail-786k] Downloading retail-786k_256.zip from Zenodo (very large, be patient) …"
  curl -L --fail --retry 3 -o retail-786k_256.zip \
    "https://zenodo.org/record/7970567/files/retail-786k_256.zip?download=1"
  echo "[Retail-786k] Unzipping …"
  unzip -q retail-786k_256.zip
  rm -f retail-786k_256.zip
  touch .extracted_ok
  echo "[Retail-786k] Done: $dest"
}

download_coco_val2014

if [ "$WITH_VG" -eq 1 ]; then
  download_visual_genome
fi

if [ "$WITH_RETAIL" -eq 1 ]; then
  download_retail786k_256
fi

echo ""
echo "═══ Download finished ═══"
echo "Next (from $REPO_ROOT):"
echo "  bash scripts/build_index.sh coco"
echo "  # optional: bash scripts/build_index.sh vg"
echo "  # optional: bash scripts/build_index.sh combined"
echo ""
