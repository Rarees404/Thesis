"""
Ollama 3.2 Vision client for query-aware auto-captioning of SAM crops.

Key improvement over baseline: when a context image is provided alongside the
isolated crop, both are sent to Ollama so it can describe the object both in
isolation and in its surrounding scene.  This produces dramatically richer
captions (e.g. "a red bicycle parked outside a café" vs. "a bicycle").
"""

import base64
import logging
import time
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RELEVANT_PROMPT = (
    "The user is searching for images of: '{query}'. "
    "This image region is something the user WANTS more of. "
    "Write one descriptive sentence (under 20 words) naming the specific objects, "
    "colors, textures, or scene elements visible here that are relevant to the search."
)

_RELEVANT_PROMPT_WITH_HINT = (
    "The user is searching for: '{query}'. "
    "The user described what they want: '{user_hint}'. "
    "This image region shows something the user WANTS. "
    "Write one descriptive sentence (under 20 words) naming the specific visual "
    "elements here that match both the search and the user's description."
)

_IRRELEVANT_PROMPT = (
    "The user is searching for images of: '{query}'. "
    "This image region is something the user does NOT want. "
    "Write one descriptive sentence (under 20 words) naming the specific objects, "
    "colors, or scene elements visible here that make it unwanted."
)

_IRRELEVANT_PROMPT_WITH_HINT = (
    "The user is searching for: '{query}'. "
    "The user wants to avoid: '{user_hint}'. "
    "This image region shows something the user does NOT want. "
    "Write one descriptive sentence (under 20 words) naming the specific visual "
    "elements here that match what the user wants to avoid."
)

# Context-aware prompt: used when both crop and full-context image are available.
_RELEVANT_CONTEXT_PROMPT = (
    "The user is searching for: '{query}'. "
    "Image 1 shows an isolated region the user WANTS more of. "
    "Image 2 shows where that region appears in the full scene. "
    "Using both images, write one sentence (under 25 words) that describes the "
    "specific object, its visual properties, AND its scene context."
)

_RELEVANT_CONTEXT_WITH_HINT = (
    "The user is searching for: '{query}'. "
    "The user described what they want: '{user_hint}'. "
    "Image 1 is an isolated region; Image 2 shows its scene context. "
    "Write one sentence (under 25 words) naming the visual elements that match "
    "both the search query and the user's description."
)

_IRRELEVANT_CONTEXT_PROMPT = (
    "The user is searching for: '{query}'. "
    "Image 1 shows an isolated region the user does NOT want. "
    "Image 2 shows where that region appears in the full scene. "
    "Write one sentence (under 25 words) describing the specific object and "
    "context that makes it unwanted for this search."
)

_IRRELEVANT_CONTEXT_WITH_HINT = (
    "The user is searching for: '{query}'. "
    "The user wants to avoid: '{user_hint}'. "
    "Image 1 is an isolated region; Image 2 shows its scene context. "
    "Write one sentence (under 25 words) naming the visual elements that match "
    "what the user wants to avoid."
)


def _image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _resize_for_ollama(img: Image.Image, max_side: int = 384) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _draw_highlight_box(
    image: Image.Image,
    bbox: Optional[tuple],
    color: str = "#00FF88",
    width: int = 4,
) -> Image.Image:
    """Draw a coloured rectangle on a copy of `image` to highlight a region."""
    img = image.copy().convert("RGB")
    if bbox is None:
        return img
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return img


def check_ollama(url: str, model: str) -> bool:
    """Return True if Ollama is reachable and `model` is pulled."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False


def caption_crop(
    image: Image.Image,
    query: str,
    label: str,
    url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    timeout: float = 60.0,
    user_hint: Optional[str] = None,
    context_image: Optional[Image.Image] = None,
    bbox: Optional[tuple] = None,
) -> Optional[str]:
    """
    Send a SAM crop to Ollama Vision and return a short caption.

    When `context_image` is provided (the full scene image or a highlighted
    thumbnail), both the isolated crop and the context are sent together.
    This produces richer captions that describe the object AND its setting.

    When `user_hint` is provided, the prompt steers Ollama toward what the
    user cares about rather than generic object description.

    Returns None on any error so callers can gracefully skip.
    """
    has_hint = bool(user_hint and user_hint.strip())
    hint = user_hint.strip() if has_hint else ""
    has_context = context_image is not None

    # Choose prompt
    if has_context:
        if label == "Relevant":
            prompt = (_RELEVANT_CONTEXT_WITH_HINT if has_hint else _RELEVANT_CONTEXT_PROMPT).format(
                query=query, user_hint=hint
            )
        else:
            prompt = (_IRRELEVANT_CONTEXT_WITH_HINT if has_hint else _IRRELEVANT_CONTEXT_PROMPT).format(
                query=query, user_hint=hint
            )
    else:
        if label == "Relevant":
            prompt = (_RELEVANT_PROMPT_WITH_HINT if has_hint else _RELEVANT_PROMPT).format(
                query=query, user_hint=hint
            )
        else:
            prompt = (_IRRELEVANT_PROMPT_WITH_HINT if has_hint else _IRRELEVANT_PROMPT).format(
                query=query, user_hint=hint
            )

    crop_small = _resize_for_ollama(image, max_side=384)
    images_b64 = [_image_to_b64(crop_small)]

    if has_context:
        ctx = _draw_highlight_box(context_image, bbox, color="#00FF88", width=3)
        ctx_small = _resize_for_ollama(ctx, max_side=384)
        images_b64.append(_image_to_b64(ctx_small))

    payload = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 60,
        },
    }

    try:
        logger.info(
            "[Ollama] Captioning %s crop%s (%dx%d)...",
            label,
            " + context" if has_context else "",
            image.size[0],
            image.size[1],
        )
        t0 = time.time()
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        elapsed = time.time() - t0
        if text:
            logger.info("[Ollama] %s caption (%.1fs): %s", label, elapsed, text)
        return text or None
    except Exception as e:
        logger.warning("[Ollama] caption_crop failed: %s", e)
        return None


# Maximum crops per label per feedback round to keep latency bounded.
# On Apple Silicon MPS, llama3.2-vision takes ~20-40 s per crop.
MAX_CROPS_PER_LABEL = 1


def batch_caption(
    crops: List[Image.Image],
    query: str,
    labels: List[str],
    url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    timeout: float = 60.0,
    user_hint: Optional[str] = None,
    context_images: Optional[List[Optional[Image.Image]]] = None,
    bboxes: Optional[List[Optional[tuple]]] = None,
) -> List[Optional[str]]:
    """
    Caption a list of SAM crops sequentially.
    Limits to MAX_CROPS_PER_LABEL to keep latency reasonable.

    When `context_images` is provided, each crop is captioned with its
    corresponding full-scene context image for richer descriptions.
    """
    results: List[Optional[str]] = []
    count = 0
    for i, (crop, label) in enumerate(zip(crops, labels)):
        if count >= MAX_CROPS_PER_LABEL:
            logger.info("[Ollama] Skipping remaining crops (limit=%d reached)", MAX_CROPS_PER_LABEL)
            results.append(None)
            continue
        ctx = context_images[i] if context_images and i < len(context_images) else None
        box = bboxes[i] if bboxes and i < len(bboxes) else None
        cap = caption_crop(
            crop, query, label,
            url=url, model=model, timeout=timeout,
            user_hint=user_hint,
            context_image=ctx,
            bbox=box,
        )
        results.append(cap)
        count += 1
    return results
