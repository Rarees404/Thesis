"""
Segmentation — SAM 2.1 (facebook/sam2.1-hiera-base-plus via HuggingFace Hub).
Point-prompt interactive masks for click-to-segment relevant/irrelevant regions.
"""

import base64
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _compute_prompt_box(
    coords: np.ndarray,
    labels: np.ndarray,
    image_size: Tuple[int, int],
    padding_ratio_fg: float = 0.20,
    padding_ratio_neg_only: float = 0.13,
) -> Optional[np.ndarray]:
    """
    Box prompt in pixel XYXY to localize SAM.

    - With at least one positive (relevant) click: box around positive points.
    - With only negative (irrelevant) clicks: box around *all* clicked points
      with tighter padding.  Without this, SAM sees the whole image and often
      returns huge rectangular masks for negative-only prompts.
    """
    img_h, img_w = image_size
    if img_h == 0 or img_w == 0 or len(coords) == 0:
        return None

    if np.any(labels == 1):
        pts = coords[labels == 1]
        pr = padding_ratio_fg
    else:
        pts = coords
        pr = padding_ratio_neg_only

    pad_x = max(24.0, pr * img_w)
    pad_y = max(24.0, pr * img_h)

    x1 = max(0.0, float(pts[:, 0].min()) - pad_x)
    y1 = max(0.0, float(pts[:, 1].min()) - pad_y)
    x2 = min(float(img_w), float(pts[:, 0].max()) + pad_x)
    y2 = min(float(img_h), float(pts[:, 1].max()) + pad_y)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    max_area_fraction: float = 0.35,
) -> int:
    """
    Among multimask candidates prefer the most precise one.
    Rejects masks > max_area_fraction of image; picks highest-scored survivor.
    Falls back to global argmax when all are large.
    masks may be logit floats — threshold at 0 (= sigmoid > 0.5).
    """
    binary   = masks > 0
    total_px = binary.shape[1] * binary.shape[2]
    max_px   = int(max_area_fraction * total_px)
    areas    = binary.sum(axis=(1, 2))
    valid    = np.where(areas <= max_px)[0]

    if len(valid) == 0:
        return int(np.argmax(scores))
    return int(valid[np.argmax(scores[valid])])


# ---------------------------------------------------------------------------
# SAM 2.1  (point prompts — facebook/sam2.1-hiera-base-plus on HuggingFace)
# ---------------------------------------------------------------------------

class SAM2Segmenter:
    def __init__(
        self,
        device: str = "cpu",
        model_id: str = "facebook/sam2.1-hiera-base-plus",
    ):
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print(f"[SAM2] Loading {model_id} (downloading from HuggingFace Hub on first run)...")
        try:
            model = build_sam2_hf(model_id, device=device)
        except Exception as e:
            print(f"[SAM2] Could not load on {device} ({e}), falling back to CPU")
            device = "cpu"
            model = build_sam2_hf(model_id, device="cpu")

        try:
            actual_device = str(next(model.parameters()).device)
        except StopIteration:
            actual_device = device

        self.predictor = SAM2ImagePredictor(model)
        self.device = actual_device
        self._current_path: Optional[str] = None
        self._current_image_size: Tuple[int, int] = (0, 0)
        self._logit_cache: Dict[str, np.ndarray] = {}
        self._LOGIT_CACHE_MAX = 50  # prevent unbounded memory growth on large corpora
        print(f"[SAM2] Ready on {actual_device}")

    def set_image(self, image: Image.Image, path: Optional[str] = None):
        # Skip re-encoding only when the path matches AND the predictor's
        # internal _is_image_set flag confirms the image is still loaded.
        already_set = (
            path is not None
            and path == self._current_path
            and getattr(self.predictor, "_is_image_set", False)
        )
        if already_set:
            return
        with torch.no_grad():
            self.predictor.set_image(np.array(image.convert("RGB")))
        self._current_path = path
        self._current_image_size = (image.height, image.width)

    def segment_points(self, points: List[Dict], image_path: Optional[str] = None) -> Dict:
        """Point-based interactive segmentation via SAM 2.1."""
        if not getattr(self.predictor, "_is_image_set", False):
            raise RuntimeError("Image not set — call set_image() first")

        img_h, img_w = self._current_image_size
        coords = np.array(
            [
                [
                    max(0.0, min(float(p["x"]), img_w - 1 if img_w > 0 else float(p["x"]))),
                    max(0.0, min(float(p["y"]), img_h - 1 if img_h > 0 else float(p["y"]))),
                ]
                for p in points
            ],
            dtype=np.float32,
        )
        labels = np.array([p["label"] for p in points], dtype=np.int32)
        use_multimask = (len(points) == 1)

        box = _compute_prompt_box(coords, labels, self._current_image_size)
        cache_key = image_path or ""
        has_negative = bool(np.any(labels == 0))
        prev_logits = None if has_negative else self._logit_cache.get(cache_key)

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                box=box,
                mask_input=prev_logits,
                multimask_output=use_multimask,
                return_logits=True,
            )

        if use_multimask and masks.shape[0] > 1:
            if np.all(labels == 1):
                best = int(np.argmax(scores))
            else:
                best = _select_best_mask(masks, scores, max_area_fraction=0.28)
        else:
            best = int(np.argmax(scores))

        if has_negative or not bool(np.all(labels == 1)):
            self._logit_cache.pop(cache_key, None)
        else:
            # Evict oldest entry when cache is full (insertion-order eviction)
            if cache_key not in self._logit_cache and len(self._logit_cache) >= self._LOGIT_CACHE_MAX:
                oldest = next(iter(self._logit_cache))
                del self._logit_cache[oldest]
            self._logit_cache[cache_key] = logits[best: best + 1]
        return {"mask": masks[best] > 0, "score": float(scores[best])}

    def clear_logit_cache(self, image_path: Optional[str] = None):
        if image_path and image_path in self._logit_cache:
            del self._logit_cache[image_path]
        elif image_path is None:
            self._logit_cache.clear()


def build_segmenter(device: str, checkpoints_dir: str, backend: str = "sam2"):
    """
    Returns (segmenter, model_type) where model_type is 'sam2' or 'none'.
    SAM 2.1 weights are downloaded automatically from HuggingFace Hub.
    """
    backend = backend.lower().strip()
    if backend not in ("auto", "sam2"):
        print(f"[SAM] Unknown backend '{backend}', defaulting to sam2")
        backend = "sam2"

    try:
        seg = SAM2Segmenter(device=device)
        return seg, "sam2"
    except Exception as e:
        print(f"[SAM] SAM 2 failed to load: {e}")
        print("[SAM] Make sure sam2 is installed: pip install git+https://github.com/facebookresearch/sam2.git")
        print("[SAM] First-run also requires outbound network access to huggingface.co")
        return None, "none"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def apply_mask(image: Image.Image, mask: np.ndarray, fill_value: int = 128) -> Image.Image:
    """
    Crop the image to the bounding box of `mask`, with non-masked pixels set
    to neutral gray (128) rather than black. Neutral gray keeps SigLIP patch
    embeddings from being biased toward dark/empty regions.
    """
    img_np = np.array(image.convert("RGB"))
    if mask.shape != (img_np.shape[0], img_np.shape[1]):
        m = Image.fromarray(mask.astype(np.uint8) * 255)
        m = m.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        mask = np.array(m) > 127

    bg = np.full_like(img_np, fill_value)
    composite = np.where(mask[:, :, np.newaxis], img_np, bg)

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return image
    cropped = composite[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return Image.fromarray(cropped.astype(np.uint8))


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask as RLE in COCO convention:
      counts[0]  = number of background pixels before the first foreground pixel (may be 0)
      counts[1]  = length of first foreground run
      counts[2]  = length of next background run
      ...
    """
    flat = mask.flatten().astype(np.uint8)
    n = len(flat)
    if n == 0:
        return {"counts": [], "size": list(mask.shape)}

    diffs = np.diff(flat.astype(np.int8))
    change_positions = np.where(diffs != 0)[0] + 1

    boundaries = np.concatenate([[0], change_positions, [n]])
    counts = np.diff(boundaries).tolist()

    if flat[0] == 1:
        counts = [0] + counts

    return {"counts": counts, "size": list(mask.shape)}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode a COCO-convention RLE back to a binary mask."""
    h, w = rle["size"]
    n = h * w
    flat = np.zeros(n, dtype=np.uint8)
    pos, val = 0, 0
    for length in rle.get("counts", []):
        if pos >= n:
            break
        end = min(pos + length, n)
        flat[pos:end] = val
        pos += length
        val = 1 - val
    return flat.reshape(h, w).astype(bool)


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
