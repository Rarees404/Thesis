import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import faiss
from src.models.configs import get_model_config


def _log(msg: str) -> None:
    """Line-buffered timestamp log for long-running index jobs."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset with images')
    parser.add_argument('--output', type=str, required=True, help='Path to output FAISS index')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_family', type=str, required=True, help='VLM model family')
    parser.add_argument('--model_id', type=str, required=True, help='HF model id')
    parser.add_argument('--index_type', type=str, default='flat_ip', help='Index type')
    parser.add_argument(
        '--m',
        type=int,
        default=32,
        help='Number of connections per layer only for `hnsw` index type.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device cuda/cpu for generating embeddings',
    )
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--vg_regions', type=str, default=None,
                        help='Path to VG region_descriptions.json for hybrid text+image indexing')
    parser.add_argument('--vg_fusion_weight', type=float, default=0.3,
                        help='Weight for text embedding in hybrid fusion (0=image only, 1=text only)')
    return parser.parse_args()


def _extract_image_id(path: str):
    import re
    basename = os.path.basename(path)
    m = re.match(r"(\d+)\.\w+$", basename)
    return int(m.group(1)) if m else None


def load_vg_regions(regions_path: str):
    import json
    _log(f"[VG] Loading JSON: {regions_path}")
    t0 = time.perf_counter()
    with open(regions_path, "r") as f:
        raw = json.load(f)
    regions_by_id = {}
    for entry in raw:
        image_id = entry.get("id")
        if image_id is None:
            continue
        phrases = [r.get("phrase", "").strip() for r in entry.get("regions", []) if r.get("phrase", "").strip()]
        if phrases:
            regions_by_id[image_id] = phrases
    elapsed = time.perf_counter() - t0
    _log(f"[VG] Parsed region_descriptions: {len(regions_by_id)} images with phrases ({elapsed:.1f}s)")
    return regions_by_id


def encode_vg_text_per_image(
    vlm_wrapper,
    image_paths,
    regions_by_id,
    batch_size=128,
    device="cuda",
):
    """Encode VG region phrases for each image → mean text embedding per image."""
    _log(f"[VG] Phase 2a — collect region phrases for {len(image_paths):,} indexed images…")
    t_collect = time.perf_counter()
    all_texts = []
    all_indices = []
    for idx, path in tqdm(
        enumerate(image_paths),
        total=len(image_paths),
        desc="VG: collect phrases",
        unit="img",
        file=sys.stdout,
        mininterval=0.5,
    ):
        image_id = _extract_image_id(path)
        if image_id and image_id in regions_by_id:
            phrases = regions_by_id[image_id][:10]
            for phrase in phrases:
                all_texts.append(phrase)
                all_indices.append(idx)

    if not all_texts:
        _log("[VG] No phrases matched indexed images — skipping hybrid text")
        return None

    n_batches = (len(all_texts) + batch_size - 1) // batch_size
    unique_imgs = len(set(all_indices))
    _log(
        f"[VG] Phase 2a done ({time.perf_counter() - t_collect:.1f}s): "
        f"{len(all_texts):,} phrase rows → {n_batches:,} batches @ batch_size={batch_size} "
        f"(covers {unique_imgs:,} images)"
    )

    _log("[VG] Phase 2b — SigLIP text encode (this can take a long time)…")
    t_enc = time.perf_counter()
    text_embs = []
    vlm_wrapper.model.eval()
    with torch.no_grad():
        for i in tqdm(
            range(0, len(all_texts), batch_size),
            desc="VG: encode text batches",
            unit="batch",
            file=sys.stdout,
            mininterval=0.5,
        ):
            batch = all_texts[i : i + batch_size]
            processed = vlm_wrapper.process_inputs(text=batch)
            emb = vlm_wrapper.get_text_embeddings(processed).to("cpu").numpy()
            text_embs.append(emb)
    text_embs = np.concatenate(text_embs, axis=0)
    dim = text_embs.shape[1]
    _log(f"[VG] Phase 2b done ({time.perf_counter() - t_enc:.1f}s); embedding dim={dim}")

    _log("[VG] Phase 2c — aggregate phrase embeddings per image (mean)…")
    t_agg = time.perf_counter()
    per_image = {}
    for emb_idx, img_idx in enumerate(all_indices):
        if img_idx not in per_image:
            per_image[img_idx] = []
        per_image[img_idx].append(text_embs[emb_idx])

    result = np.zeros((len(image_paths), dim), dtype=np.float32)
    hit = 0
    for img_idx, embs in tqdm(
        per_image.items(),
        desc="VG: per-image mean",
        unit="img",
        file=sys.stdout,
        mininterval=0.5,
    ):
        result[img_idx] = np.mean(embs, axis=0)
        hit += 1
    _log(
        f"[VG] Phase 2c done ({time.perf_counter() - t_agg:.1f}s): "
        f"{hit:,}/{len(image_paths):,} images have fused VG phrase embeddings"
    )
    return result


def get_image_paths(data_dir):
    extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, f'**/*.{ext}'), recursive=True))
    # Filter out non-files and tiny/truncated files (< 1 KB is almost certainly corrupt)
    valid = [p for p in image_paths if os.path.isfile(p) and os.path.getsize(p) > 1024]
    removed = len(image_paths) - len(valid)
    if removed:
        print(f"[warn] Filtered out {removed} invalid/tiny files (<1KB)")
    return valid


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return img, self.paths[idx]
        except Exception:
            placeholder = Image.new('RGB', (256, 256), (0, 0, 0))
            return placeholder, None


def collate(batch):
    imgs, paths = zip(*batch)
    valid = [(img, p) for img, p in zip(imgs, paths) if p is not None]
    if not valid:
        return [], []
    imgs_out, paths_out = zip(*valid)
    return list(imgs_out), list(paths_out)


def encode_images_fast(vlm_wrapper, image_paths, batch_size=64, num_workers=8, device="cuda"):
    import sys
    # MPS + multiprocessing workers causes a segfault on macOS during teardown
    if device == "mps" or sys.platform == "darwin":
        num_workers = 0

    ds = ImagePathDataset(image_paths)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "collate_fn": collate,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **dataloader_kwargs)
    features, paths_out = [], []
    skipped = 0
    vlm_wrapper.model.eval()
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Encoding images"):
            if not imgs:
                skipped += len(paths) if paths else 0
                continue
            processed = vlm_wrapper.process_inputs(images=imgs)
            emb = vlm_wrapper.get_image_embeddings(processed).to("cpu").numpy()
            features.append(emb)
            paths_out.extend(paths)
    if skipped:
        print(f"[warn] Skipped {skipped} corrupted/unreadable images")
    return np.concatenate(features, axis=0), paths_out


def create_faiss_index(features, feature_dim, index_type='flat_ip', m=32):
    features = np.ascontiguousarray(features.astype(np.float32))
    # Avoid faiss.normalize_L2: it uses OpenMP which conflicts with PyTorch's
    # bundled libomp on Apple Silicon, causing a segfault.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    features /= norms

    if index_type == 'flat_ip':
        index = faiss.IndexFlatIP(feature_dim)
    elif index_type == 'hnsw':
        index = faiss.IndexHNSWFlat(feature_dim, m)
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    index.add(features)

    return index


def main():
    args = parse_args()
    _log("=== FAISS index build started ===")
    _log(f"data={args.data} output={args.output} model={args.model_id} device={args.device}")

    # Get all image paths
    image_paths = get_image_paths(args.data)
    _log(f"Phase 1 — collect paths: {len(image_paths):,} images")

    model_config = get_model_config(args.model_family, args.model_id)

    _log("Loading SigLIP model + processor…")
    t_load = time.perf_counter()
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
    model = model_config["model_class"].from_pretrained(model_config["model_id"])
    wrapper = model_config["wrapper_class"](model=model, processor=processor)

    model.to(args.device)
    model.eval()
    _log(f"SigLIP loaded ({time.perf_counter() - t_load:.1f}s)")

    _log("Phase 1 — encode images (tqdm below)")
    t_img = time.perf_counter()
    features, paths = encode_images_fast(
        wrapper,
        image_paths,
        args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    _log(
        f"Phase 1 done ({time.perf_counter() - t_img:.1f}s): "
        f"matrix shape {features.shape} for {len(paths):,} paths"
    )

    # --- Optional: VG hybrid text+image fusion ---
    if args.vg_regions and os.path.isfile(args.vg_regions):
        _log("--- Phase 2 — Visual Genome hybrid (region phrase text embeddings) ---")
        regions_by_id = load_vg_regions(args.vg_regions)
        text_features = encode_vg_text_per_image(
            wrapper,
            paths,
            regions_by_id,
            batch_size=max(32, min(args.batch_size * 4, 256)),
            device=args.device,
        )
        if text_features is not None:
            _log(f"[VG] Phase 2d — normalize + fuse (vg_fusion_weight={args.vg_fusion_weight})…")
            t_fuse = time.perf_counter()
            img_norms = np.linalg.norm(features, axis=1, keepdims=True)
            img_norms = np.where(img_norms == 0, 1.0, img_norms)
            features_norm = features / img_norms

            txt_norms = np.linalg.norm(text_features, axis=1, keepdims=True)
            txt_norms = np.where(txt_norms == 0, 1.0, txt_norms)
            text_features_norm = text_features / txt_norms

            w = args.vg_fusion_weight
            has_text = np.linalg.norm(text_features, axis=1) > 0
            features_norm[has_text] = (
                (1 - w) * features_norm[has_text] + w * text_features_norm[has_text]
            )
            features = features_norm
            _log(f"[VG] Fusion done ({time.perf_counter() - t_fuse:.1f}s)")
    elif args.vg_regions:
        _log(f"[VG] Warning: {args.vg_regions} not found, skipping hybrid indexing")

    # Explicitly release model from MPS/GPU memory before FAISS operations.
    _log("Releasing SigLIP from memory before FAISS build…")
    del model, wrapper, processor
    import gc
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    feature_dim = features.shape[1]

    _log(f"Phase 3 — build FAISS IndexFlatIP (dim={feature_dim}, vectors={features.shape[0]:,})…")
    t_faiss = time.perf_counter()
    index = create_faiss_index(features, feature_dim)
    _log(f"Phase 3 done ({time.perf_counter() - t_faiss:.1f}s)")

    # Save index and paths
    output_dir = os.path.join(args.output, args.model_id)
    os.makedirs(output_dir, exist_ok=True)
    faiss_path = os.path.join(output_dir, "image_index.faiss")
    paths_path = os.path.join(output_dir, "image_paths.txt")

    _log(f"Writing {faiss_path}…")
    faiss.write_index(index, faiss_path)
    _log(f"Writing {paths_path} ({len(paths):,} lines)…")
    with open(paths_path, "w") as f:
        for path in paths:
            f.write(f"{path}\n")

    _log(f"=== Index build finished: {len(paths):,} images → {output_dir} ===")

if __name__ == "__main__":
    main()


