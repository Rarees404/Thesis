from typing import Any, Dict, Optional

from transformers import AutoProcessor, SiglipModel

from src.models.siglip import SigLipWrapper

CONFIGS = {
    "siglip": {
        "model_id": "google/siglip-base-patch16-256",
        "model_class": SiglipModel,
        "processor_class": AutoProcessor,
        "wrapper_class": SigLipWrapper,
    },
}


def get_model_config(
        model_family: str,
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    base = CONFIGS.get(model_family)
    if not base:
        raise ValueError(f"Model config is not parsed. Please use model_family from {list(CONFIGS.keys())}")
    # Return a shallow copy so callers can mutate without affecting the global registry
    config = dict(base)
    if model_id is not None:
        config["model_id"] = model_id
    return config
