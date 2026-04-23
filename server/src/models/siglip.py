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
        if images is None and text is None:
            raise ValueError("process_inputs: at least one of images or text must be provided")

        # Process image-only and text-only paths separately to avoid SigLIP
        # processor failures when one modality is None (behaviour varies across
        # transformers versions and processor implementations).
        result: Dict[str, Any] = {}

        if images is not None:
            try:
                img_inputs = self.processor.image_processor(
                    images=images, return_tensors="pt"
                )
            except AttributeError:
                # Fallback: some processor versions expose the image sub-processor differently
                img_inputs = self.processor(images=images, text=None, return_tensors="pt")
                img_inputs = {k: v for k, v in img_inputs.items() if "pixel" in k}
            result.update(img_inputs)

        if text is not None:
            try:
                tok = self.processor.tokenizer
                txt_inputs = tok(
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )
            except AttributeError:
                # Fallback: processor itself is a tokenizer-like object
                txt_inputs = self.processor(
                    images=None, text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )
                txt_inputs = {k: v for k, v in txt_inputs.items() if "pixel" not in k}
            result.update(txt_inputs)

        # Move each tensor to the model's device individually to avoid
        # issues with MPS and mixed tensor types.
        return {k: v.to(self.model.device) for k, v in result.items()}

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

