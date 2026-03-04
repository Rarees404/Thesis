from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, SiglipModel

from src.models.vlm_wrapper import VLMWrapperRetrieval


@dataclass
class SigLipWrapper(VLMWrapperRetrieval):
    model: Any = field(default=None)
    processor: Any = field(default=None)

    def process_inputs(self, images=None, text=None) -> Dict[str, Any]:
        assert images is not None or text is not None
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        # Move each tensor to the model's device individually to avoid
        # issues with MPS and mixed tensor types
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def get_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        outputs = self.model(**inputs)
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
            "logits_per_image": outputs.logits_per_image,
            "logits_per_text": outputs.logits_per_text,
            "vision_model_output": outputs.vision_model_output.last_hidden_state,
            "text_model_output": outputs.text_model_output.last_hidden_state,
        }

    def get_text_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats.float(), p=2, dim=-1)

    def get_image_embeddings(self, inputs: Dict[str, Any], **kwargs) -> Any:
        feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        return F.normalize(feats.float(), p=2, dim=-1)

