import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.vlm_wrapper import VLMWrapperRetrieval

logger = logging.getLogger(__name__)


class RocchioUpdate:
    """Classic Rocchio relevance-feedback update over (normalized) embeddings.

        upd_q = alpha * q  +  beta * mean(relevant)  -  gamma * mean(irrelevant)

    The three coefficients trade off how much the refined query keeps vs. moves:
      - alpha (query inertia): how much of the current query is retained. High → stable.
      - beta  (positive pull): how strongly the query moves toward relevant feedback.
      - gamma (negative push): how strongly it moves away from irrelevant feedback.
        Kept smaller than beta so a single "not this" click can't dominate.

    Defaults here are conservative (0.8 / 0.1 / 0.1). The live retrieval service
    instantiates this with 0.8 / 0.5 / 0.15 (see RetrievalServiceVisual.__init__),
    and the dedicated user-text pass overrides them per-call (see retrieval_service).
    Output is L2-normalized so it stays comparable under FAISS inner-product search.
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.1, gamma: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        norm_output: bool = True
    ):
        return self.rocchio_update(
            query_embeddings,
            positive_embeddings,
            negative_embeddings,
            self.alpha,
            self.beta,
            self.gamma,
            norm_output
        )


    def rocchio_update(
        self,
        query_embeddings: torch.Tensor,
        avg_relevance_vector: Optional[torch.Tensor] = None,
        avg_non_relevance_vector: Optional[torch.Tensor] = None,
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        norm_output: bool = True
    ):
        """
        Update the query embeddings using Rocchio's algorithm
            upd_q = alpha * q + beta * positive_feedback - gamma * negative_feedback

        Args:
            query_embedddings: initial query embeddings
            avg_relevance_vector: average relevance (positive feedback) vector
            avg_non_relevance_vector: average non-relevance (negative feedback) vector
            alpha: coefficient for initial query embeddings
            beta: coefficient for positive feedback
            gamma: coefficient for negative feedback
            norm_output: whether to normalize the output

        If both avg_relevance_vector and avg_non_relevance_vector are None or beta and gamma are 0,
        the query embeddings are returned unchanged.
        """
        if avg_non_relevance_vector is None:
            avg_non_relevance_vector = torch.zeros_like(query_embeddings)
            gamma = 0.0
        if avg_relevance_vector is None:
            avg_relevance_vector = torch.zeros_like(query_embeddings)
            beta = 0.0
        updated_query_embeddings = (
            alpha * query_embeddings + \
            beta * avg_relevance_vector - \
            gamma * avg_non_relevance_vector
        )
        if norm_output:
            updated_query_embeddings = F.normalize(updated_query_embeddings, p=2, dim=-1)
        return updated_query_embeddings


class RelevanceFeedback(ABC):
    """
    Abstract class for relevance feedback models.

    Instances are callable and require at least a query.
    """

    @abstractmethod
    def __call__(self, query: str, *args, **kwargs):
        pass


class ImageBasedVLMRelevanceFeedback(RelevanceFeedback):
    def __init__(
        self,
        vlm_wrapper_retrieval: VLMWrapperRetrieval,
        img_size: int = 224,
    ):
        self.vlm_wrapper_retrieval = vlm_wrapper_retrieval
        self.img_size = img_size

    def __call__(
        self,
        query: str,
        relevant_image_paths: List[str],
        annotator_json_boxes_list: Optional[List[Any]] = None,
        sam_annotations: Optional[List[Optional[Dict]]] = None,
        image_labels: Optional[List[Optional[str]]] = None,
        top_k_feedback: int = 5,
    ):
        if not relevant_image_paths:
            return {"relevant_segments": [], "irrelevant_segments": []}

        images = []
        for image_path in relevant_image_paths:
            image = Image.open(image_path).convert("RGB")
            images.append(image)

        relevant_segments: List[Image.Image] = []
        irrelevant_segments: List[Image.Image] = []

        # 1) SAM masks (per-image; only items with masks contribute)
        has_any_sam = bool(
            sam_annotations and any(a is not None for a in sam_annotations)
        )
        if has_any_sam:
            sam_result = self._extract_sam_segments(
                images=images, sam_annotations=sam_annotations,
            )
            relevant_segments.extend(sam_result["relevant_segments"])
            irrelevant_segments.extend(sam_result["irrelevant_segments"])

        # 2) Bounding boxes (legacy annotator path; only when no SAM was given)
        has_any_box = bool(
            annotator_json_boxes_list
            and any(b is not None and len(b) > 0 for b in annotator_json_boxes_list)
        )
        if has_any_box and not has_any_sam:
            box_result = self._extract_image_segments(
                images=images,
                annotator_json_boxes_list=annotator_json_boxes_list,
            )
            relevant_segments.extend(box_result["relevant_segments"])
            irrelevant_segments.extend(box_result["irrelevant_segments"])

        # 3) Full-image labels — only for images with no SAM mask / no box on that index
        if image_labels:
            for i, lbl in enumerate(image_labels):
                if lbl not in ("Relevant", "Irrelevant"):
                    continue
                has_sam_i = (
                    sam_annotations
                    and i < len(sam_annotations)
                    and sam_annotations[i] is not None
                    and sam_annotations[i].get("mask_rle")
                )
                has_box_i = (
                    annotator_json_boxes_list
                    and i < len(annotator_json_boxes_list)
                    and annotator_json_boxes_list[i]
                )
                if has_sam_i or has_box_i:
                    continue
                full = images[i].resize(
                    (self.img_size, self.img_size), Image.BICUBIC,
                )
                if lbl == "Relevant":
                    relevant_segments.append(full)
                else:
                    irrelevant_segments.append(full)

        return {
            "relevant_segments": relevant_segments,
            "irrelevant_segments": irrelevant_segments,
        }

    def _extract_sam_segments(
        self,
        images: List[Image.Image],
        sam_annotations: List[Optional[Dict]],
    ) -> Dict[str, List[Image.Image]]:
        """Extract segments from SAM mask regions sent by the frontend."""
        from src.models.sam import rle_to_mask

        relevant_segments = []
        irrelevant_segments = []

        for i, annot in enumerate(sam_annotations):
            if annot is None:
                continue

            mask_rle = annot.get("mask_rle")
            label = annot.get("label", "Relevant")

            if mask_rle:
                mask = rle_to_mask(mask_rle)
                img_np = np.array(images[i].convert("RGB"))

                if mask.shape != (img_np.shape[0], img_np.shape[1]):
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize(
                        (img_np.shape[1], img_np.shape[0]),
                        PILImage.NEAREST,
                    )
                    mask = np.array(mask_img) > 127

                gray_bg = np.full_like(img_np, 128)
                masked = np.where(mask[:, :, np.newaxis], img_np, gray_bg)
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    continue
                cropped = masked[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
                segment = Image.fromarray(cropped.astype(np.uint8)).resize(
                    (self.img_size, self.img_size), Image.BICUBIC,
                )

                if label == "Relevant":
                    relevant_segments.append(segment)
                elif label == "Irrelevant":
                    irrelevant_segments.append(segment)

        return {
            "relevant_segments": relevant_segments,
            "irrelevant_segments": irrelevant_segments,
        }

    def _extract_image_segments(
        self,
        images: List[Image.Image],
        annotator_json_boxes_list: List[Optional[List[Dict[str, Any]]]]
    ) -> Dict[str, List[Image.Image]]:
        irrelevant_segments = []
        relevant_segments = []
        for i, boxes in enumerate(annotator_json_boxes_list):
            if boxes is not None:
                img = images[i]
                orig_w, orig_h = img.size
                for annot in boxes:
                    # Boxes from frontend are in the IMG_SIZE coordinate space (e.g. 224x224 or 256x256)
                    # We need to scale them back to the original image dimensions to crop with full resolution.
                    xmin = int(max(0, annot["xmin"] * orig_w / self.img_size))
                    ymin = int(max(0, annot["ymin"] * orig_h / self.img_size))
                    xmax = int(min(orig_w, annot["xmax"] * orig_w / self.img_size))
                    ymax = int(min(orig_h, annot["ymax"] * orig_h / self.img_size))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    # Crop from original image then resize
                    segment = img.crop((xmin, ymin, xmax, ymax)).resize(
                        (self.img_size, self.img_size), Image.Resampling.BICUBIC
                    )

                    if annot["label"] == "Relevant":
                        relevant_segments.append(segment)
                    elif annot["label"] == "Irrelevant":
                        irrelevant_segments.append(segment)
                    else:
                        logger.warning("Unknown annotation label '%s' — skipping box", annot["label"])
        return {
            "relevant_segments": relevant_segments,
            "irrelevant_segments": irrelevant_segments
        }
