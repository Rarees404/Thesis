"""
Offline builder for the per-region SigLIP embedding index.

Each Visual Genome image has ~30 human-annotated bounding boxes, each with a
short phrase ("yellow taxi", "child running", ...). We crop every box, encode
with SigLIP, and persist:

    region_index.faiss        IndexFlatIP over L2-normalized embeddings
    region_meta.jsonl         row-by-row metadata (image_path, phrase, bbox, ...)
    region_index.progress.txt last image index processed (for crash-resume)

At runtime, services/region_index.py loads these and uses them for hard
filtering during relevance feedback: kNN on a user-marked region's embedding
returns the corpus images containing similar objects, which can then be
boosted (positive) or blacklisted (negative) for the rest of the session.

Why VG boxes instead of SAM3 "everything" mode:
- VG annotations are already loaded by vg_regions.py — zero extra cost.
- Each box carries an English phrase, so the runtime can show a human-readable
  reason for a filter ("removed 47 images containing 'chair'").
- ~10x faster to build than running a SAM3 grid-of-points pass over the corpus.
- The crops are looser than SAM masks, but for kNN matching the embedding still
  captures the dominant object — sufficient for hard filtering.

Usage:

    cd server
    ./venv/bin/python -m src.precompute.build_region_index \\
        --image-paths ../faiss/visual_genome/google/siglip-large-patch16-256/image_paths.txt \\
        --vg-dir ../data/visual_genome \\
        --output-dir ../faiss/visual_genome/google/siglip-large-patch16-256 \\
        --limit 100        # smoke test on first 100 images

Re-running picks up where it left off via region_index.progress.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

# Make `src.*` imports work whether invoked as a module or a file.
_THIS_FILE = Path(__file__).resolve()
_SERVER_DIR = _THIS_FILE.parents[2]  # .../server/
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from src.utils.vg_regions import VGRegionIndex  # noqa: E402

logger = logging.getLogger("build_region_index")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_siglip(model_id: str, device: str) -> Tuple[SiglipModel, AutoProcessor]:
    logger.info("Loading SigLIP %s on %s", model_id, device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = SiglipModel.from_pretrained(model_id).to(device).eval()
    return model, processor


@torch.no_grad()
def _encode_images(
    model: SiglipModel,
    processor: AutoProcessor,
    images: List[Image.Image],
    device: str,
) -> np.ndarray:
    """Returns (N, dim) float32 numpy array, L2-normalized along dim=-1."""
    pixel = processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
    pixel = pixel.to(device)
    feats = model.get_image_features(pixel_values=pixel)
    feats = F.normalize(feats.float(), p=2, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def _iter_corpus_paths(image_paths_file: Path) -> Iterable[str]:
    with open(image_paths_file) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def _crop_region(
    image: Image.Image,
    region: dict,
    min_area: int,
) -> Optional[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """Crop a VG region from the image, returning (crop, (x, y, w, h)) or None."""
    iw, ih = image.size
    x = max(0, int(region.get("x", 0)))
    y = max(0, int(region.get("y", 0)))
    w = int(region.get("width", 0))
    h = int(region.get("height", 0))
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)
    if x2 <= x or y2 <= y:
        return None
    if (x2 - x) * (y2 - y) < min_area:
        return None
    return image.crop((x, y, x2, y2)), (x, y, x2 - x, y2 - y)


def _resolve_corpus_path(raw: str, repo_root: Path) -> str:
    """image_paths.txt may contain absolute paths from a different host. If the
    file doesn't exist as-is, try re-rooting under data/visual_genome/."""
    if os.path.isfile(raw):
        return raw
    base = os.path.basename(raw)
    for sub in ("VG_100K", "VG_100K_2"):
        cand = repo_root / "data" / "visual_genome" / sub / base
        if cand.is_file():
            return str(cand)
    return raw  # caller will skip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-paths", required=True, help="image_paths.txt for the existing image-level FAISS index")
    ap.add_argument("--vg-dir", required=True, help="Directory containing region_descriptions.json")
    ap.add_argument("--output-dir", required=True, help="Where to write region_index.faiss + region_meta.jsonl")
    ap.add_argument("--model-id", default="google/siglip-large-patch16-256")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-area", type=int, default=1024, help="Skip regions smaller than this many pixels (32x32 default)")
    ap.add_argument("--max-regions-per-image", type=int, default=30)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--checkpoint-every", type=int, default=100, help="Persist FAISS + progress every N images")
    args = ap.parse_args()

    _setup_logging()
    device = _pick_device(args.device)
    logger.info("Device: %s", device)

    # Locate the repo root from the VG dir for path re-rooting
    vg_dir = Path(args.vg_dir).resolve()
    repo_root = vg_dir.parent.parent  # .../<repo>/data/visual_genome -> repo

    # Load VG region descriptions
    vg = VGRegionIndex.load(str(vg_dir))
    if vg is None:
        logger.error("Could not load VG regions from %s", vg_dir)
        return 1

    # Output paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "region_index.faiss"
    meta_path = out_dir / "region_meta.jsonl"
    progress_path = out_dir / "region_index.progress.txt"

    # Resume support: skip images we've already processed.
    start_index = 0
    if progress_path.exists():
        try:
            start_index = int(progress_path.read_text().strip()) + 1
            logger.info("Resuming from image index %d", start_index)
        except ValueError:
            pass

    # Load or create FAISS index
    index: Optional[faiss.Index] = None
    if faiss_path.exists() and start_index > 0:
        index = faiss.read_index(str(faiss_path))
        logger.info("Loaded existing index: %d vectors, dim=%d", index.ntotal, index.d)

    # Open metadata file in append mode (line-buffered so a crash doesn't lose the last batch)
    meta_file = open(meta_path, "a", buffering=1)

    model, processor = _load_siglip(args.model_id, device)

    paths = list(_iter_corpus_paths(Path(args.image_paths)))
    if args.limit > 0:
        paths = paths[: args.limit]
    total = len(paths)
    logger.info("Corpus size: %d images (limit=%d)", total, args.limit or 0)

    added = skipped = no_regions = missing = 0
    t0 = time.time()

    for i, raw_path in enumerate(paths):
        if i < start_index:
            continue

        img_path = _resolve_corpus_path(raw_path, repo_root)
        if not os.path.isfile(img_path):
            missing += 1
            progress_path.write_text(str(i))
            continue

        regions = vg.get_regions(img_path)
        if not regions:
            no_regions += 1
            progress_path.write_text(str(i))
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path, exc)
            skipped += 1
            progress_path.write_text(str(i))
            continue

        crops: List[Image.Image] = []
        kept: List[Tuple[int, dict, Tuple[int, int, int, int]]] = []
        for region_idx, region in enumerate(regions[: args.max_regions_per_image]):
            res = _crop_region(image, region, args.min_area)
            if res is None:
                continue
            crop, bbox = res
            crops.append(crop)
            kept.append((region_idx, region, bbox))

        if not crops:
            skipped += 1
            progress_path.write_text(str(i))
            continue

        embeds_chunks: List[np.ndarray] = []
        for j in range(0, len(crops), args.batch_size):
            batch = crops[j : j + args.batch_size]
            embeds_chunks.append(_encode_images(model, processor, batch, device))
        embeds = np.concatenate(embeds_chunks, axis=0)

        if index is None:
            index = faiss.IndexFlatIP(embeds.shape[1])
            logger.info("Initialized FAISS IndexFlatIP, dim=%d", embeds.shape[1])

        index.add(embeds)

        for region_idx, region, bbox in kept:
            row = {
                "image_path": img_path,
                "region_idx": region_idx,
                "phrase": region.get("phrase", ""),
                "bbox": list(bbox),
                "source": "vg",
            }
            meta_file.write(json.dumps(row) + "\n")
        added += len(kept)

        # Periodic checkpoint
        if (i + 1) % args.checkpoint_every == 0 or i == total - 1:
            faiss.write_index(index, str(faiss_path))
            meta_file.flush()
            progress_path.write_text(str(i))
            elapsed = time.time() - t0
            rate = (i + 1 - start_index) / max(elapsed, 1e-3)
            logger.info(
                "Progress %d/%d  added=%d  skipped=%d  no_regions=%d  missing=%d  rate=%.1f img/s",
                i + 1, total, added, skipped, no_regions, missing, rate,
            )

    # Final flush
    if index is not None:
        faiss.write_index(index, str(faiss_path))
    meta_file.close()
    if total > 0:
        progress_path.write_text(str(total - 1))

    logger.info(
        "Done. Wrote %d region embeddings to %s. Metadata: %s",
        index.ntotal if index else 0, faiss_path, meta_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
