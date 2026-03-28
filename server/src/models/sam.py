"""
Segmentation — SAM 3 primary, SAM 2 fallback.
Point-prompt interactive masks only (click on image). Text queries use the
retrieval model; optional VLM captions are separate from this module.
"""

import base64
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Shared accuracy helpers (used by both SAM2 and SAM3)
# ---------------------------------------------------------------------------

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
# SAM 2  (point prompts — always available)
# ---------------------------------------------------------------------------

class SAM2Segmenter:
    def __init__(self, checkpoint: str, config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml", device: str = "cpu"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.device = device
        self._current_path: Optional[str] = None
        self._current_image_size: Tuple[int, int] = (0, 0)
        self._logit_cache: Dict[str, np.ndarray] = {}
        print(f"[SAM2] Loaded on {device}")

    def set_image(self, image: Image.Image, path: Optional[str] = None):
        if path and path == self._current_path:
            return
        self.predictor.set_image(np.array(image.convert("RGB")))
        self._current_path = path
        self._current_image_size = (image.height, image.width)

    def segment_points(self, points: List[Dict], image_path: Optional[str] = None) -> Dict:
        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.array([p["label"] for p in points], dtype=np.int32)
        use_multimask = (len(points) == 1)

        box = _compute_prompt_box(coords, labels, self._current_image_size)
        cache_key = image_path or ""
        has_negative = bool(np.any(labels == 0))
        # Cached logits from a positive-only step conflict with negative prompts.
        prev_logits = None if has_negative else self._logit_cache.get(cache_key)

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
            self._logit_cache[cache_key] = logits[best: best + 1]
        # `> 0` threshold — DO NOT use .astype(bool) on logits (negatives → True).
        return {"mask": masks[best] > 0, "score": float(scores[best])}

    def clear_logit_cache(self, image_path: Optional[str] = None):
        if image_path and image_path in self._logit_cache:
            del self._logit_cache[image_path]
        elif image_path is None:
            self._logit_cache.clear()


# ---------------------------------------------------------------------------
# SAM 3  (point prompts — facebook/sam3 on HuggingFace)
# ---------------------------------------------------------------------------

class SAM3Segmenter:
    def __init__(self, device: str = "cpu"):
        import sam3 as _sam3_pkg
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # pkg_resources.resource_filename fails when sam3 is path-imported (no __file__).
        # Resolve the BPE vocab path from the package's directory on disk.
        bpe_path = os.path.join(_sam3_pkg.__path__[0], "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

        print(f"[SAM3] Loading model (downloading from HuggingFace Hub → facebook/sam3)...")
        model = build_sam3_image_model(device=device, enable_inst_interactivity=True, bpe_path=bpe_path)

        # SAM3's builder only moves to CUDA; explicitly move for MPS/CPU.
        try:
            model = model.to(device)
        except Exception as e:
            print(f"[SAM3] Could not move model to {device} ({e}), falling back to CPU")
            device = "cpu"
            model = model.to("cpu")

        # Detect the actual device from model parameters (in case of fallback above)
        try:
            actual_device = str(next(model.parameters()).device)
        except StopIteration:
            actual_device = device

        self.processor = Sam3Processor(model, device=actual_device)
        self._interactive = model.inst_interactive_predictor
        self.device = actual_device
        self._current_path: Optional[str] = None
        self._current_image_size: Tuple[int, int] = (0, 0)
        self._state: Optional[dict] = None
        self._interactive_set = False
        # Logit cache for iterative refinement (same mechanism as SAM2)
        self._logit_cache: Dict[str, np.ndarray] = {}
        print(f"[SAM3] Ready on {actual_device}")

    def set_image(self, image: Image.Image, path: Optional[str] = None):
        if path and path == self._current_path and self._state is not None:
            return
        self._state = self.processor.set_image(image)
        if self._interactive is not None:
            self._interactive.set_image(np.array(image.convert("RGB")))
            self._interactive_set = True
        self._current_path = path
        self._current_image_size = (image.height, image.width)

    def segment_points(self, points: List[Dict], image_path: Optional[str] = None) -> Dict:
        """Point-based interactive segmentation — same interface as SAM2."""
        if self._interactive is None or not self._interactive_set:
            raise RuntimeError("Interactive predictor not available")

        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.array([p["label"] for p in points], dtype=np.int32)
        use_multimask = (len(points) == 1)

        box = _compute_prompt_box(coords, labels, self._current_image_size)
        cache_key = image_path or ""
        has_negative = bool(np.any(labels == 0))
        prev_logits = None if has_negative else self._logit_cache.get(cache_key)

        masks, scores, logits = self._interactive.predict(
            point_coords=coords,
            point_labels=labels,
            box=box,
            mask_input=prev_logits,
            multimask_output=use_multimask,
            return_logits=True,
        )

        # Single-click relevant: trust top IoU mask (often the full object).
        # Negative-only or mixed prompts: prefer a tighter mask among multimask
        # candidates to avoid half-image rectangles.
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
            self._logit_cache[cache_key] = logits[best: best + 1]
        return {"mask": masks[best] > 0, "score": float(scores[best])}

    def clear_logit_cache(self, image_path: Optional[str] = None):
        if image_path and image_path in self._logit_cache:
            del self._logit_cache[image_path]
        elif image_path is None:
            self._logit_cache.clear()


# ---------------------------------------------------------------------------
# Unified factory — tries SAM 3, falls back to SAM 2
# ---------------------------------------------------------------------------

def build_segmenter(device: str, checkpoints_dir: str, backend: str = "auto"):
    """
    Returns (segmenter, model_type) where model_type is 'sam3' or 'sam2'.
    backend: "auto" (try sam3→sam2), "sam3" (sam3 only), "sam2" (sam2 only).
    """
    checkpoints_dir = os.path.normpath(checkpoints_dir)
    backend = backend.lower().strip()

    if backend in ("auto", "sam3"):
        try:
            seg = SAM3Segmenter(device=device)
            return seg, "sam3"
        except Exception as e:
            if backend == "sam3":
                print(f"[SAM] SAM 3 failed to load: {e}")
                return None, "none"
            print(f"[SAM] SAM 3 unavailable ({e}), falling back to SAM 2")

    if backend in ("auto", "sam2"):
        sam2_ckpt = os.path.join(checkpoints_dir, "sam2_hiera_base_plus.pt")
        print(f"[SAM] Looking for SAM 2 at: {sam2_ckpt} (exists: {os.path.exists(sam2_ckpt)})")
        if os.path.exists(sam2_ckpt):
            try:
                seg = SAM2Segmenter(checkpoint=sam2_ckpt, device=device)
                return seg, "sam2"
            except Exception as e:
                print(f"[SAM] SAM 2 failed to load: {e}")

    print(f"[SAM] No SAM backend available (requested: {backend})")
    return None, "none"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def apply_mask(image: Image.Image, mask: np.ndarray, fill_value: int = 128) -> Image.Image:
    """
    Crop the image to the bounding box of `mask`, with non-masked pixels set
    to neutral gray (128) rather than black. Neutral gray keeps SigLIP patch
    embeddings from being biased toward dark/empty regions — SigLIP was trained
    on natural images, not black-background crops.
    """
    img_np = np.array(image.convert("RGB"))
    if mask.shape != (img_np.shape[0], img_np.shape[1]):
        m = Image.fromarray(mask.astype(np.uint8) * 255)
        m = m.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        mask = np.array(m) > 127

    # Fill background with neutral gray instead of black zeros
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
    Both this function and rle_to_mask (and the frontend decoder) must use the
    same convention.  The previous implementation based on np.diff(prepend=0)
    dropped the initial background run entirely, causing every decoded mask to
    appear at a wrong position in the image.
    """
    flat = mask.flatten().astype(np.uint8)
    n = len(flat)
    if n == 0:
        return {"counts": [], "size": list(mask.shape)}

    # Find indices where the value changes (transition boundaries)
    diffs = np.diff(flat.astype(np.int8))
    change_positions = np.where(diffs != 0)[0] + 1   # +1: change takes effect at next pixel

    # Boundaries: start of image, every transition, end of image
    boundaries = np.concatenate([[0], change_positions, [n]])
    counts = np.diff(boundaries).tolist()

    # COCO convention: first count is always a background run.
    # If the mask starts with foreground, prepend a zero-length background run.
    if flat[0] == 1:
        counts = [0] + counts

    return {"counts": counts, "size": list(mask.shape)}


def rle_to_mask(rle: dict) -> np.ndarray:
    """
    Decode a COCO-convention RLE back to a binary mask.
    counts[0] is always a background run (may be 0), then alternates fg/bg.
    """
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0  # start with background (val=0)
    for length in rle["counts"]:
        flat[pos:pos + length] = val
        pos += length
        val = 1 - val
    return flat.reshape(h, w).astype(bool)


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
