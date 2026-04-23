import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

import faiss
from src.models.configs import get_model_config
from src.config import resolve_repo
from src.models.ollama_vision import batch_caption, caption_crop
from src.models.relevance_feedback import (
    ImageBasedVLMRelevanceFeedback,
    RocchioUpdate,
)
from functools import lru_cache
from src.utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        config: Dict[str, Any],
        faiss_index: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
    ):
        self.config = config
        self.faiss_index = faiss_index
        self.accumulated_query_embeddings = {"query_embedding": None}
        self.retrieval_round = 1
        self.experiment_id = 0
        self.device = device

        self._init_backbone()
        self._init_rocchio_update(alpha=alpha, beta=beta, gamma=gamma)
        self._init_faiss_index()
        self._init_b64_cache()

    def _init_backbone(self):
        self.backbone_config = get_model_config(
            self.config["VLM_MODEL_FAMILY"],
            self.config["VLM_MODEL_NAME"]
        )
        self.backbone = self.backbone_config["model_class"].from_pretrained(
            self.config["VLM_MODEL_NAME"]
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        self.backbone_processor = (
            self.backbone_config["processor_class"]
            .from_pretrained(self.config["VLM_MODEL_NAME"])
        )
        self.wrapper = self.backbone_config["wrapper_class"](
            model=self.backbone,
            processor=self.backbone_processor,
        )
        logger.info("Loaded backbone %s on %s", self.config["VLM_MODEL_NAME"], self.device)

    def _init_rocchio_update(self, alpha, beta, gamma):
        self.rocchio_update = RocchioUpdate(alpha=alpha, beta=beta, gamma=gamma)

    def _init_b64_cache(self):
        @lru_cache(maxsize=2000)
        def _get_cached_b64(path: str, size: int) -> str:
            img = Image.open(path).convert("RGB")
            img = img.resize((size, size), Image.Resampling.BICUBIC)
            return image_to_base64(img)
        self._b64_cache = _get_cached_b64

    def _init_faiss_index(self):
        index_path_str = str(self.faiss_index)
        if not os.path.exists(index_path_str):
            raise ValueError(
                f"FAISS index file not found at '{index_path_str}'. "
                "Verify APP_INDEX_PATH/INDEX_PATH and volume mounts."
            )
        try:
            self.index = faiss.read_index(index_path_str)
        except RuntimeError as e:
            raise ValueError(f"Failed to read FAISS index: {e}")
        try:
            with open(
                os.path.join(os.path.dirname(index_path_str), "image_paths.txt"), "r"
            ) as f:
                self.candidate_image_paths = [line.strip() for line in f.readlines()]
            self.candidate_image_paths = [resolve_repo(p) for p in self.candidate_image_paths]
        except FileNotFoundError as e:
            raise ValueError(f"Failed to read image paths: {e}")

    @staticmethod
    def _to_faiss(tensor: torch.Tensor) -> np.ndarray:
        """Move a tensor to CPU float32 numpy — required by FAISS on all devices."""
        arr = tensor.detach().cpu().float().numpy()
        # FAISS search requires shape (n_queries, dim)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

    def search_images(self, query: str, top_k: int = 5):
        self.experiment_id += 1

        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        self.accumulated_query_embeddings["query_embedding"] = query_embedding

        scores, img_ids = self.index.search(self._to_faiss(query_embedding), top_k)
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()

        if isinstance(img_ids, (int, np.integer)):
            img_ids = [int(img_ids)]
            scores = [scores]

        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        img_size = self.config.get("IMG_SIZE", 224)
        images_b64 = [self._b64_cache(p, img_size) for p in retrieved_image_paths]

        return images_b64, scores, retrieved_image_paths


class RetrievalServiceVisual(RetrievalService):
    def __init__(
        self,
        config: Dict[str, Any],
        faiss_index: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.8,
        beta: float = 0.5,
        gamma: float = 0.15,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2-vision",
    ):
        super().__init__(
            config=config,
            faiss_index=faiss_index,
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self._init_image_based_relevance_feedback()

    def _init_image_based_relevance_feedback(self):
        self.image_based_relevance_feedback = ImageBasedVLMRelevanceFeedback(
            vlm_wrapper_retrieval=self.wrapper,
        )

    # ------------------------------------------------------------------
    # Ollama captioning helpers
    # ------------------------------------------------------------------

    def _ollama_caption_segments(
        self,
        segments: List[Image.Image],
        query: str,
        label: str,
        ollama_available: bool,
        user_hint: Optional[str] = None,
        context_images: Optional[List[Optional[Image.Image]]] = None,
        bboxes: Optional[List[Optional[tuple]]] = None,
        caption_cache: Optional[Dict[str, str]] = None,
        cache_image_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Auto-caption SAM crops via Ollama.

        1. Check the pre-computed caption cache first (from background /segment task).
        2. For any crop not in cache, call Ollama with both the isolated crop and
           the full-scene context image for richer descriptions.
        """
        if not ollama_available or not segments:
            return []

        # Resolve captions from the pre-computed cache where possible
        captions_from_cache: List[Optional[str]] = []
        uncached_indices: List[int] = []
        hint = (user_hint or "").strip()

        for i, seg in enumerate(segments):
            img_path = (cache_image_paths or [])[i] if cache_image_paths else None
            if caption_cache is not None and img_path:
                from src.retrieval_server_visual import _caption_cache_key
                ck = _caption_cache_key(img_path, label, query, hint)
                if ck in caption_cache and caption_cache[ck]:
                    captions_from_cache.append(caption_cache[ck])
                    continue
            captions_from_cache.append(None)
            uncached_indices.append(i)

        # Any already-cached captions are immediately available
        final_captions = list(captions_from_cache)

        # For uncached crops, call Ollama with context
        if uncached_indices:
            crops_to_caption = [segments[i] for i in uncached_indices]
            labels_list = [label] * len(crops_to_caption)
            ctx_images = (
                [context_images[i] if context_images else None for i in uncached_indices]
                if context_images else None
            )
            bboxes_list = (
                [bboxes[i] if bboxes else None for i in uncached_indices]
                if bboxes else None
            )

            new_captions = batch_caption(
                crops=crops_to_caption,
                query=query,
                labels=labels_list,
                url=self.ollama_url,
                model=self.ollama_model,
                user_hint=user_hint,
                context_images=ctx_images,
                bboxes=bboxes_list,
            )
            for j, orig_i in enumerate(uncached_indices):
                if j < len(new_captions) and new_captions[j]:
                    final_captions[orig_i] = new_captions[j]

        return [c for c in final_captions if c]

    # ------------------------------------------------------------------
    # Core feedback method
    # ------------------------------------------------------------------

    def process_and_apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_image_paths: List[str],
        relevant_captions: Optional[str] = None,
        irrelevant_captions: Optional[str] = None,
        annotator_json_boxes_list: Optional[List[Any]] = None,
        sam_annotations: Optional[List[Any]] = None,
        fuse_initial_query: bool = False,
        ollama_available: bool = False,
        vg_region_index=None,
        caption_cache: Optional[Dict[str, str]] = None,
    ):
        # ------------------------------------------------------------------
        # Step 1: Extract SAM segments (pixel crops from masks)
        # ------------------------------------------------------------------
        relevance_results = self.image_based_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            annotator_json_boxes_list=annotator_json_boxes_list,
            sam_annotations=sam_annotations,
            top_k_feedback=top_k,
        )
        relevant_segments: List[Image.Image] = relevance_results["relevant_segments"]
        irrelevant_segments: List[Image.Image] = relevance_results["irrelevant_segments"]

        # Build per-segment context images (full scene) and bounding boxes for
        # context-aware Ollama captioning.
        rel_context_imgs: List[Optional[Image.Image]] = []
        rel_bboxes: List[Optional[tuple]] = []
        irr_context_imgs: List[Optional[Image.Image]] = []
        irr_bboxes: List[Optional[tuple]] = []
        rel_img_paths: List[str] = []
        irr_img_paths: List[str] = []

        if sam_annotations:
            for ann in sam_annotations:
                if ann is None:
                    continue
                label_str = ann.get("label", "Relevant")
                img_path = ann.get("image_path", "")
                mask_rle = ann.get("mask_rle")

                ctx_img: Optional[Image.Image] = None
                bbox: Optional[tuple] = None
                if img_path:
                    try:
                        ctx_img = Image.open(img_path).convert("RGB")
                        if mask_rle:
                            from src.models.sam import rle_to_mask
                            m = rle_to_mask(mask_rle)
                            rows, cols = np.where(m)
                            if len(rows) > 0:
                                # Scale bbox from preview space to full image space
                                orig_w, orig_h = ctx_img.size
                                ph, pw = m.shape
                                x1 = int(cols.min() * orig_w / pw)
                                y1 = int(rows.min() * orig_h / ph)
                                x2 = int(cols.max() * orig_w / pw)
                                y2 = int(rows.max() * orig_h / ph)
                                bbox = (x1, y1, x2, y2)
                    except Exception:
                        ctx_img = None
                        bbox = None

                if label_str == "Relevant":
                    rel_context_imgs.append(ctx_img)
                    rel_bboxes.append(bbox)
                    rel_img_paths.append(img_path)
                else:
                    irr_context_imgs.append(ctx_img)
                    irr_bboxes.append(bbox)
                    irr_img_paths.append(img_path)

        # ------------------------------------------------------------------
        # Step 2: VG region phrases (fast, no Ollama needed)
        # ------------------------------------------------------------------
        vg_pos_phrases: List[str] = []
        vg_neg_phrases: List[str] = []
        if sam_annotations:
            for ann in sam_annotations:
                if ann is None:
                    continue
                label_str = ann.get("label", "Relevant")
                is_relevant = (label_str == "Relevant")
                target = vg_pos_phrases if is_relevant else vg_neg_phrases

                precomputed: List[str] = ann.get("vg_phrases") or []
                if precomputed:
                    target.extend(precomputed[:5])
                    continue

                if vg_region_index is None:
                    continue

                image_path = ann.get("image_path", "")
                mask_rle = ann.get("mask_rle")
                if image_path and mask_rle and vg_region_index.has_image(image_path):
                    try:
                        from src.models.sam import rle_to_mask
                        mask = rle_to_mask(mask_rle)
                        with Image.open(image_path) as _img:
                            orig_w, orig_h = _img.size
                        if mask.shape != (orig_h, orig_w):
                            m = Image.fromarray(mask.astype(np.uint8) * 255)
                            m = m.resize((orig_w, orig_h), Image.NEAREST)
                            mask = np.array(m) > 127
                        phrases = vg_region_index.top_phrases_for_mask(
                            image_path, mask, top_k=5
                        )
                        if phrases:
                            target.extend(phrases)
                            continue
                    except Exception as exc:
                        logger.warning("[VG] mask-IoU lookup failed for %s: %s", image_path, exc)

                if image_path:
                    fallback = vg_region_index.get_all_phrases(image_path)
                    target.extend(fallback[:3])

        if vg_pos_phrases or vg_neg_phrases:
            logger.info(
                "[VG] Region phrases: %d positive, %d negative",
                len(vg_pos_phrases), len(vg_neg_phrases),
            )

        # ------------------------------------------------------------------
        # Step 3: Ollama context-aware captioning of SAM crops
        # ------------------------------------------------------------------
        pos_hint = (relevant_captions or "").strip() or None
        neg_hint = (irrelevant_captions or "").strip() or None

        if vg_pos_phrases and not pos_hint:
            vlm_pos_captions: List[str] = []
            logger.info("[VG] Skipping Ollama for positive — VG phrases available, no pos hint")
        else:
            vlm_pos_captions = self._ollama_caption_segments(
                segments=relevant_segments,
                query=query,
                label="Relevant",
                ollama_available=ollama_available,
                user_hint=pos_hint,
                context_images=rel_context_imgs or None,
                bboxes=rel_bboxes or None,
                caption_cache=caption_cache,
                cache_image_paths=rel_img_paths or None,
            )

        if vg_neg_phrases and not neg_hint:
            vlm_neg_captions: List[str] = []
            logger.info("[VG] Skipping Ollama for negative — VG phrases available, no neg hint")
        else:
            vlm_neg_captions = self._ollama_caption_segments(
                segments=irrelevant_segments,
                query=query,
                label="Irrelevant",
                ollama_available=ollama_available,
                user_hint=neg_hint,
                context_images=irr_context_imgs or None,
                bboxes=irr_bboxes or None,
                caption_cache=caption_cache,
                cache_image_paths=irr_img_paths or None,
            )

        # Visual grounding via Ollama when user typed a hint but there are no SAM
        # segments for that label — caption the top-k images directly.
        if pos_hint and not relevant_segments and ollama_available and relevant_image_paths:
            for path in relevant_image_paths[:2]:
                try:
                    img = Image.open(path).convert("RGB")
                    img.thumbnail((384, 384))
                    cap = caption_crop(
                        img, query, "Relevant",
                        url=self.ollama_url, model=self.ollama_model,
                        user_hint=pos_hint,
                    )
                    if cap:
                        vlm_pos_captions.append(cap)
                except Exception as exc:
                    logger.warning("[Ollama] Visual-grounding Relevant failed for %s: %s", path, exc)

        if neg_hint and not irrelevant_segments and ollama_available and relevant_image_paths:
            for path in relevant_image_paths[:2]:
                try:
                    img = Image.open(path).convert("RGB")
                    img.thumbnail((384, 384))
                    cap = caption_crop(
                        img, query, "Irrelevant",
                        url=self.ollama_url, model=self.ollama_model,
                        user_hint=neg_hint,
                    )
                    if cap:
                        vlm_neg_captions.append(cap)
                except Exception as exc:
                    logger.warning("[Ollama] Visual-grounding Irrelevant failed for %s: %s", path, exc)

        has_vlm = bool(vlm_pos_captions or vlm_neg_captions)
        if has_vlm:
            logger.info(
                "[Ollama] Auto-captions: %d positive, %d negative",
                len(vlm_pos_captions), len(vlm_neg_captions),
            )

        # ------------------------------------------------------------------
        # Step 4: Compute embeddings and Rocchio update
        # ------------------------------------------------------------------
        with torch.no_grad():
            positive_image_embeddings = self._safe_image_embeddings(relevant_segments)
            negative_image_embeddings = self._safe_image_embeddings(irrelevant_segments)

            all_pos_texts: List[str] = []
            if pos_hint:
                all_pos_texts.append(pos_hint)
            all_pos_texts.extend(vg_pos_phrases)
            all_pos_texts.extend(vlm_pos_captions)

            all_neg_texts: List[str] = []
            if neg_hint:
                all_neg_texts.append(neg_hint)
            all_neg_texts.extend(vg_neg_phrases)
            all_neg_texts.extend(vlm_neg_captions)

            positive_text_embeddings = self._safe_text_embeddings(all_pos_texts)
            negative_text_embeddings = self._safe_text_embeddings(all_neg_texts)

            # Weighted fusion: text carries more weight when VLM captions are present
            img_w = 0.4 if has_vlm else 0.5
            txt_w = 0.6 if has_vlm else 0.5

            positive_embeddings = self._fuse(positive_image_embeddings, positive_text_embeddings, img_w, txt_w)
            negative_embeddings = self._fuse(negative_image_embeddings, negative_text_embeddings, img_w, txt_w)

            processed_query = self.wrapper.process_inputs(text=query)
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

            accumulated = self.accumulated_query_embeddings["query_embedding"]
            if accumulated is None:
                logger.warning("[Rocchio] No accumulated query — bootstrapping from fresh text embedding")
                accumulated = query_embedding

            rocchio_query = (accumulated + query_embedding) / 2 if fuse_initial_query else accumulated

            self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
                query_embeddings=rocchio_query,
                positive_embeddings=positive_embeddings,
                negative_embeddings=negative_embeddings,
            )

        # ------------------------------------------------------------------
        # Step 5: Search
        # ------------------------------------------------------------------
        scores, img_ids = self.index.search(
            self._to_faiss(self.accumulated_query_embeddings["query_embedding"]),
            top_k,
        )
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()

        if isinstance(img_ids, (int, np.integer)):
            img_ids = [int(img_ids)]
            scores = [scores]

        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        img_size = self.config.get("IMG_SIZE", 224)
        images_b64 = [self._b64_cache(p, img_size) for p in retrieved_image_paths]
        self.retrieval_round += 1

        return images_b64, scores, retrieved_image_paths

    # ------------------------------------------------------------------
    # Private embedding helpers (safe, no-crash)
    # ------------------------------------------------------------------

    def _safe_image_embeddings(
        self, segments: List[Image.Image]
    ) -> Optional[torch.Tensor]:
        if not segments:
            return None
        try:
            inputs = self.wrapper.process_inputs(images=segments)
            embs = self.wrapper.get_image_embeddings(inputs)
            return embs.mean(dim=0)
        except Exception as exc:
            logger.warning("[Embed] Image embedding failed: %s", exc)
            return None

    def _safe_text_embeddings(
        self, texts: List[str]
    ) -> Optional[torch.Tensor]:
        if not texts:
            return None
        # Deduplicate and cap to avoid runaway batch sizes
        unique = list(dict.fromkeys(t for t in texts if t and t.strip()))[:20]
        if not unique:
            return None
        try:
            inputs = self.wrapper.process_inputs(text=unique)
            embs = self.wrapper.get_text_embeddings(inputs)
            return embs.mean(dim=0)
        except Exception as exc:
            logger.warning("[Embed] Text embedding failed: %s", exc)
            return None

    @staticmethod
    def _fuse(
        img_emb: Optional[torch.Tensor],
        txt_emb: Optional[torch.Tensor],
        img_w: float,
        txt_w: float,
    ) -> Optional[torch.Tensor]:
        if img_emb is not None and txt_emb is not None:
            return img_w * img_emb + txt_w * txt_emb
        return img_emb if img_emb is not None else txt_emb
