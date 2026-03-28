import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

import faiss
from src.models.configs import get_model_config
from src.config import resolve_repo
from src.models.llava import init_llava
from src.models.ollama_vision import batch_caption
from src.models.relevance_feedback import (
    CaptionVLMRelevanceFeedback,
    ImageBasedVLMRelevanceFeedback,
    RocchioUpdate,
)
from src.utils.image_utils import resize_images

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        config: Dict[str, Any],
        faiss_index: str,
        captioning_model_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
    ):
        self.config = config
        self.captioning_model_config = captioning_model_config if captioning_model_config is not None else None
        self.faiss_index = faiss_index
        self.accumulated_query_embeddings = {"query_embedding": None}
        self.retrieval_round = 1
        self.experiment_id = 0
        self.device = device
        
        self._init_backbone()
        if self.captioning_model_config is not None:
            self._init_captioning_model()
            self._init_captioning_relevance_feedback()
        self._init_rocchio_update(alpha=alpha, beta=beta, gamma=gamma)
        self._init_faiss_index()

    def _init_backbone(self):
        self.backbone_config = get_model_config(
            self.config["VLM_MODEL_FAMILY"],
            self.config["VLM_MODEL_NAME"]
        )
        self.backbone = self.backbone_config["model_class"].from_pretrained(self.config["VLM_MODEL_NAME"])
        self.backbone.to(self.device)
        self.backbone.eval()
        self.backbone_processor = (
            self.backbone_config["processor_class"]
            .from_pretrained(self.config["VLM_MODEL_NAME"])
        )

        self.wrapper = self.backbone_config["wrapper_class"](
            model=self.backbone,
            processor=self.backbone_processor
        )
        print(f"[backbone] Loaded {self.config['VLM_MODEL_NAME']} on {self.device}")
    
    def _init_captioning_model(self):
        model_config = get_model_config(
            self.captioning_model_config["MODEL_FAMILY"], 
            self.captioning_model_config["MODEL_ID"]
        )
        if self.captioning_model_config["MODEL_FAMILY"] == "llava":
            self.captioning_model = init_llava(
                model_config=model_config,
                device=self.device,
                use_8bit=self.captioning_model_config["USE_8BIT"]
            )
        else:
            raise ValueError(
                f"Captioning model family {self.captioning_model_config['model_family']} not supported"
            )

    def _init_captioning_relevance_feedback(self):
        self.captioning_relevance_feedback = CaptionVLMRelevanceFeedback(
            vlm_wrapper_retrieval=self.wrapper,
            vlm_wrapper_captioning=self.captioning_model,
        )

    def _init_rocchio_update(
        self,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        multiple: bool = False,
    ):
        self.rocchio_update = RocchioUpdate(alpha=alpha, beta=beta, gamma=gamma)

    def _init_faiss_index(self):
        # Ensure FAISS receives a plain string path and validate existence
        index_path_str = str(self.faiss_index)
        if not os.path.exists(index_path_str):
            raise ValueError(
                f"FAISS index file not found at '{index_path_str}'. "
                f"Verify APP_INDEX_PATH/INDEX_PATH and volume mounts."
            )
        try:
            self.index = faiss.read_index(index_path_str)
        except RuntimeError as e:
            raise ValueError(f"Failed to read FAISS index: {e}. Check if the index file exists.")
        try:
            with open(
                os.path.join(os.path.dirname(index_path_str),
                "image_paths.txt"),
                "r"
            ) as f:
                self.candidate_image_paths = [line.strip() for line in f.readlines()]
            # Normalize candidate paths to absolute repo-root-based paths
            self.candidate_image_paths = [resolve_repo(p) for p in self.candidate_image_paths]
        except FileNotFoundError as e:
            raise ValueError(f"Failed to read image paths: {e}. Check if the image paths file exists.")

    @staticmethod
    def _to_faiss(tensor: torch.Tensor):
        """Move a tensor to CPU float32 numpy — required by FAISS on all devices."""
        return tensor.detach().cpu().float().numpy()

    def search_images(self, query: str, top_k: int = 5):
        """Extract image_search function logic"""
        self.experiment_id += 1

        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        self.accumulated_query_embeddings["query_embedding"] = query_embedding

        scores, img_ids = self.index.search(self._to_faiss(query_embedding), top_k)
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        print(retrieved_image_paths)
        for path in retrieved_image_paths:
            assert os.path.exists(path), f"Image path {path} does not exist"
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        print(retrieved_images)
        retrieved_images = resize_images(
            retrieved_images,
            (self.config.get("IMG_SIZE", 224), self.config.get("IMG_SIZE", 224))
        )

        return retrieved_images, scores, retrieved_image_paths

    def process_feedback(
        self,
        query: str,
        relevant_image_paths: List[str],
        user_prompt: Optional[str] = None,
        annotator_json_boxes_list: Optional[List[Any]] = None,
        visualization: bool = False,
        top_k_feedback: int = 5,
        prompt_based_on_query: bool = False,
        relevant_captions: Optional[Union[List[str], str]] = None,
        irrelevant_captions: Optional[Union[List[str], str]] = None,
        prompt: Optional[str] = None
    ):
        relevance_feedback_results = self.captioning_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            user_prompt=user_prompt,
            visualization=visualization,
            top_k_feedback=top_k_feedback,
            annotator_json_boxes_list=annotator_json_boxes_list,
            prompt_based_on_query=prompt_based_on_query,
            relevant_captions=relevant_captions,
            irrelevant_captions=irrelevant_captions,
            prompt=prompt
        )

        return {
            "positive": relevance_feedback_results["positive"].tolist() if relevance_feedback_results["positive"] is not None else None,
            "negative": relevance_feedback_results["negative"].tolist() if relevance_feedback_results["negative"] is not None else None,
            "relevant_captions": relevance_feedback_results["relevant_captions"],
            "irrelevant_captions": relevance_feedback_results["irrelevant_captions"],
            "explanation": relevance_feedback_results["explanation"]
        }

    def apply_feedback(
        self,
        query: str,
        top_k: int,
        relevant_captions: Optional[Union[List[str], torch.Tensor]] = None,
        irrelevant_captions: Optional[Union[List[str], torch.Tensor]] = None,
        fuse_initial_query: bool = False
    ):
        """Extract feedback_loop function logic"""
        processed_query = self.wrapper.process_inputs(text=query)
        with torch.no_grad():
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

        rocchio_query_embedding = (self.accumulated_query_embeddings["query_embedding"] + query_embedding) / 2 if (
            fuse_initial_query
        ) else self.accumulated_query_embeddings["query_embedding"]

        relevant_captions = [cap for cap in relevant_captions if cap != ""]
        irrelevant_captions = [cap for cap in irrelevant_captions if cap != ""]

        print(relevant_captions, irrelevant_captions)

        with torch.no_grad():
            if relevant_captions is not None and relevant_captions:
                positive_embeddings = self.wrapper.get_text_embeddings(
                        self.wrapper.process_inputs(text=relevant_captions)
                    )
                positive_embeddings = positive_embeddings.mean(dim=0)
            else:
                positive_embeddings = None
            if irrelevant_captions is not None and irrelevant_captions:
                negative_embeddings = self.wrapper.get_text_embeddings(
                self.wrapper.process_inputs(text=irrelevant_captions)
            )
                negative_embeddings = negative_embeddings.mean(dim=0)
            else:
                negative_embeddings = None

        self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
            query_embeddings=rocchio_query_embedding,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings
        )

        scores, img_ids = self.index.search(
            self._to_faiss(self.accumulated_query_embeddings["query_embedding"]), top_k
        )
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(
            retrieved_images,
            (self.config.get("IMG_SIZE", 224), self.config.get("IMG_SIZE", 224))
        )

        self.retrieval_round += 1
        return retrieved_images, scores, retrieved_image_paths


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

    def _ollama_caption_segments(
        self,
        segments: List[Image.Image],
        query: str,
        label: str,
        ollama_available: bool,
    ) -> List[str]:
        """Auto-caption a list of SAM crops via Ollama. Returns only non-empty captions."""
        if not ollama_available or not segments:
            return []
        captions = batch_caption(
            crops=segments,
            query=query,
            labels=[label] * len(segments),
            url=self.ollama_url,
            model=self.ollama_model,
        )
        return [c for c in captions if c]

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
    ):
        relevance_feedback_results = self.image_based_relevance_feedback(
            query=query,
            relevant_image_paths=relevant_image_paths,
            annotator_json_boxes_list=annotator_json_boxes_list,
            sam_annotations=sam_annotations,
            top_k_feedback=top_k
        )

        relevant_segments = relevance_feedback_results["relevant_segments"]
        irrelevant_segments = relevance_feedback_results["irrelevant_segments"]

        # --- Ollama auto-captioning of SAM crops ---
        vlm_pos_captions = self._ollama_caption_segments(
            relevant_segments, query, "Relevant", ollama_available,
        )
        vlm_neg_captions = self._ollama_caption_segments(
            irrelevant_segments, query, "Irrelevant", ollama_available,
        )
        has_vlm = bool(vlm_pos_captions or vlm_neg_captions)
        if has_vlm:
            logger.info(
                "[Ollama] Auto-captions: %d positive, %d negative",
                len(vlm_pos_captions), len(vlm_neg_captions),
            )

        with torch.no_grad():
            # --- Image embeddings from SAM crops ---
            if relevant_segments is not None and relevant_segments:
                positive_image_embeddings = self.wrapper.get_image_embeddings(
                    self.wrapper.process_inputs(images=relevant_segments)
                )
                positive_image_embeddings = positive_image_embeddings.mean(dim=0)
            else:
                positive_image_embeddings = None

            if irrelevant_segments is not None and irrelevant_segments:
                negative_image_embeddings = self.wrapper.get_image_embeddings(
                    self.wrapper.process_inputs(images=irrelevant_segments)
                )
                negative_image_embeddings = negative_image_embeddings.mean(dim=0)
            else:
                negative_image_embeddings = None

            # --- Text embeddings: merge user-typed text + Ollama auto-captions ---
            all_pos_texts: List[str] = []
            if relevant_captions and relevant_captions.strip():
                all_pos_texts.append(relevant_captions.strip())
            all_pos_texts.extend(vlm_pos_captions)

            all_neg_texts: List[str] = []
            if irrelevant_captions and irrelevant_captions.strip():
                all_neg_texts.append(irrelevant_captions.strip())
            all_neg_texts.extend(vlm_neg_captions)

            if all_pos_texts:
                pos_text_embs = []
                for txt in all_pos_texts:
                    emb = self.wrapper.get_text_embeddings(
                        self.wrapper.process_inputs(text=txt)
                    )
                    pos_text_embs.append(emb)
                positive_text_embeddings = torch.stack(pos_text_embs).mean(dim=0)
            else:
                positive_text_embeddings = None

            if all_neg_texts:
                neg_text_embs = []
                for txt in all_neg_texts:
                    emb = self.wrapper.get_text_embeddings(
                        self.wrapper.process_inputs(text=txt)
                    )
                    neg_text_embs.append(emb)
                negative_text_embeddings = torch.stack(neg_text_embs).mean(dim=0)
            else:
                negative_text_embeddings = None

            # --- Weighted combination of image + text embeddings ---
            # With VLM captions the text signal is richer → weight it more heavily
            img_w = 0.4 if has_vlm else 0.5
            txt_w = 0.6 if has_vlm else 0.5

            if positive_image_embeddings is not None and positive_text_embeddings is not None:
                positive_embeddings = img_w * positive_image_embeddings + txt_w * positive_text_embeddings
            elif positive_image_embeddings is not None:
                positive_embeddings = positive_image_embeddings
            elif positive_text_embeddings is not None:
                positive_embeddings = positive_text_embeddings
            else:
                positive_embeddings = None

            if negative_image_embeddings is not None and negative_text_embeddings is not None:
                negative_embeddings = img_w * negative_image_embeddings + txt_w * negative_text_embeddings
            elif negative_image_embeddings is not None:
                negative_embeddings = negative_image_embeddings
            elif negative_text_embeddings is not None:
                negative_embeddings = negative_text_embeddings
            else:
                negative_embeddings = None

            processed_query = self.wrapper.process_inputs(text=query)
            query_embedding = self.wrapper.get_text_embeddings(processed_query)

            if fuse_initial_query:
                print("[Rocchio] Fusing fresh text embedding with accumulated query")
                rocchio_query_embedding = (self.accumulated_query_embeddings["query_embedding"] + query_embedding) / 2
            else:
                rocchio_query_embedding = self.accumulated_query_embeddings["query_embedding"]

            self.accumulated_query_embeddings["query_embedding"] = self.rocchio_update(
                query_embeddings=rocchio_query_embedding,
                positive_embeddings=positive_embeddings,
                negative_embeddings=negative_embeddings,
            )

        scores, img_ids = self.index.search(
            self._to_faiss(self.accumulated_query_embeddings["query_embedding"]),
            top_k
        )
        scores = scores.squeeze().tolist()
        img_ids = img_ids.squeeze().tolist()
        retrieved_image_paths = [self.candidate_image_paths[i] for i in img_ids]
        retrieved_images = [Image.open(path) for path in retrieved_image_paths]
        retrieved_images = resize_images(
            retrieved_images,
            (self.config.get("IMG_SIZE", 224), self.config.get("IMG_SIZE", 224))
        )

        self.retrieval_round += 1

        return retrieved_images, scores, retrieved_image_paths
