"""
Segmentation wrapper with automatic fallback:
  - SAM 3: text + point prompts (requires HuggingFace gated access)
  - SAM 2: point prompts only (publicly available fallback)
"""

import base64
import os
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


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
        # Cache last logit mask per image path for iterative refinement.
        # Shape stored: (1, 256, 256) — can be passed back as mask_input.
        self._logit_cache: Dict[str, np.ndarray] = {}
        print(f"[SAM2] Loaded on {device}")

    def set_image(self, image: Image.Image, path: Optional[str] = None):
        if path and path == self._current_path:
            return
        self.predictor.set_image(np.array(image.convert("RGB")))
        self._current_path = path

    def segment_points(self, points: List[Dict], image_path: Optional[str] = None) -> Dict:
        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.array([p["label"] for p in points], dtype=np.int32)
        n_points = len(points)

        # SAM2 accuracy rules:
        #   - 1 click  → multimask_output=True  (pick best of 3 candidates)
        #   - N clicks → multimask_output=False (model commits to one unified mask)
        use_multimask = (n_points == 1)

        # Retrieve cached logits from prior call on this image for iterative
        # refinement. This is SAM2's primary interactive mechanism.
        cache_key = image_path or ""
        prev_logits = self._logit_cache.get(cache_key)

        masks, scores, logits = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            mask_input=prev_logits,
            multimask_output=use_multimask,
            return_logits=True,
        )

        best = int(np.argmax(scores))

        # Cache the best logit for the next incremental click on this image.
        # logits shape: (N_masks, 256, 256) — store the best one as (1, 256, 256).
        self._logit_cache[cache_key] = logits[best: best + 1]

        return {"mask": masks[best].astype(bool), "score": float(scores[best])}

    def clear_logit_cache(self, image_path: Optional[str] = None):
        """Call when the user resets annotations for an image."""
        if image_path and image_path in self._logit_cache:
            del self._logit_cache[image_path]
        elif image_path is None:
            self._logit_cache.clear()

    def segment_text(self, text_prompt: str) -> List[Dict]:
        raise NotImplementedError("SAM 2 does not support text prompts — upgrade to SAM 3")


# ---------------------------------------------------------------------------
# SAM 3  (text + point prompts — requires gated HuggingFace access)
# ---------------------------------------------------------------------------

class SAM3Segmenter:
    def __init__(self, device: str = "cpu"):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print(f"[SAM3] Loading model on {device} (downloading from HuggingFace Hub)...")
        model = build_sam3_image_model(device=device, enable_inst_interactivity=True)
        self.processor = Sam3Processor(model, device=device)
        self._interactive = model.inst_interactive_predictor
        self.device = device
        self._current_path: Optional[str] = None
        self._state: Optional[dict] = None
        self._interactive_set = False
        print(f"[SAM3] Ready on {device}")

    def set_image(self, image: Image.Image, path: Optional[str] = None):
        if path and path == self._current_path and self._state is not None:
            return
        self._state = self.processor.set_image(image)
        if self._interactive is not None:
            self._interactive.set_image(np.array(image.convert("RGB")))
            self._interactive_set = True
        self._current_path = path

    def segment_text(self, text_prompt: str) -> List[Dict]:
        if self._state is None:
            raise RuntimeError("Call set_image() first")
        state = self.processor.set_text_prompt(text_prompt, self._state)
        masks, scores, boxes = state["masks"], state["scores"], state["boxes"]
        return [
            {
                "mask": masks[i, 0].cpu().numpy().astype(bool),
                "score": float(scores[i].cpu()),
                "bbox": boxes[i].cpu().tolist(),
            }
            for i in range(masks.shape[0])
        ]

    def segment_points(self, points: List[Dict], image_path: Optional[str] = None) -> Dict:
        if self._interactive is None or not self._interactive_set:
            raise RuntimeError("Interactive predictor not available")
        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        labels = np.array([p["label"] for p in points], dtype=np.int32)
        n_points = len(points)
        use_multimask = (n_points == 1)
        masks, scores, _ = self._interactive.predict(
            point_coords=coords, point_labels=labels, multimask_output=use_multimask,
        )
        best = int(np.argmax(scores))
        return {"mask": masks[best].astype(bool), "score": float(scores[best])}


# ---------------------------------------------------------------------------
# Unified factory — tries SAM 3, falls back to SAM 2
# ---------------------------------------------------------------------------

def build_segmenter(device: str, checkpoints_dir: str):
    """
    Returns (segmenter, model_type) where model_type is 'sam3' or 'sam2'.
    """
    checkpoints_dir = os.path.normpath(checkpoints_dir)

    try:
        seg = SAM3Segmenter(device=device)
        return seg, "sam3"
    except Exception as e:
        print(f"[SAM] SAM 3 unavailable ({e}), falling back to SAM 2")

    sam2_ckpt = os.path.join(checkpoints_dir, "sam2_hiera_base_plus.pt")
    print(f"[SAM] Looking for SAM 2 at: {sam2_ckpt} (exists: {os.path.exists(sam2_ckpt)})")
    if os.path.exists(sam2_ckpt):
        try:
            seg = SAM2Segmenter(checkpoint=sam2_ckpt, device=device)
            return seg, "sam2"
        except Exception as e:
            print(f"[SAM] SAM 2 failed to load: {e}")

    print(f"[SAM] No SAM checkpoint found in {checkpoints_dir}")
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
