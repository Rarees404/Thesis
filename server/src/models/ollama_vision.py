"""
Ollama 3.2 Vision client for query-aware auto-captioning of SAM crops.

Wraps Ollama's /api/generate endpoint. Each SAM-segmented crop is sent with
a prompt that references the user's search query, producing a short description
that SigLIP can encode into a rich text embedding for Rocchio feedback.
"""

import base64
import logging
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)

_RELEVANT_PROMPT = (
    "The user is searching for: '{query}'. "
    "Describe what is visible in this image region in under 15 words, "
    "focusing on aspects that match the search."
)

_IRRELEVANT_PROMPT = (
    "The user is searching for: '{query}'. "
    "Describe what is visible in this image region in under 15 words, "
    "focusing on what makes it different from the search."
)


def _image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def check_ollama(url: str, model: str) -> bool:
    """Return True if Ollama is reachable and `model` is pulled."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        # Ollama tag format: "llama3.2-vision:latest" — match base name
        return any(model in m for m in models)
    except Exception:
        return False


def caption_crop(
    image: Image.Image,
    query: str,
    label: str,
    url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    timeout: float = 30.0,
) -> Optional[str]:
    """
    Send a single SAM crop to Ollama Vision and return a short caption.

    Returns None on any error so the caller can gracefully skip.
    """
    template = _RELEVANT_PROMPT if label == "Relevant" else _IRRELEVANT_PROMPT
    prompt = template.format(query=query)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_image_to_b64(image)],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 40,
        },
    }

    try:
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            logger.info("[Ollama] %s caption: %s", label, text)
        return text or None
    except Exception as e:
        logger.warning("[Ollama] caption_crop failed: %s", e)
        return None


def batch_caption(
    crops: List[Image.Image],
    query: str,
    labels: List[str],
    url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    timeout: float = 30.0,
) -> List[Optional[str]]:
    """
    Caption a list of SAM crops sequentially.
    Returns a list of strings (or None for failures) parallel to the input.
    """
    results: List[Optional[str]] = []
    for crop, label in zip(crops, labels):
        cap = caption_crop(crop, query, label, url=url, model=model, timeout=timeout)
        results.append(cap)
    return results
