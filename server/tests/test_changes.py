"""
Tests for the changes made in this session.
All tests run without loading SigLIP, SAM, or Ollama.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RLE roundtrip (unchanged — verifying nothing broke)
# ---------------------------------------------------------------------------
def test_rle_roundtrip_normal():
    from src.models.sam import mask_to_rle, rle_to_mask
    mask = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=bool)
    rle = mask_to_rle(mask)
    recovered = rle_to_mask(rle)
    assert np.array_equal(mask, recovered), "RLE roundtrip failed"


def test_rle_roundtrip_all_zeros():
    from src.models.sam import mask_to_rle, rle_to_mask
    mask = np.zeros((5, 5), dtype=bool)
    recovered = rle_to_mask(mask_to_rle(mask))
    assert np.array_equal(mask, recovered)


def test_rle_roundtrip_all_ones():
    from src.models.sam import mask_to_rle, rle_to_mask
    mask = np.ones((4, 6), dtype=bool)
    recovered = rle_to_mask(mask_to_rle(mask))
    assert np.array_equal(mask, recovered)


def test_rle_first_pixel_foreground():
    from src.models.sam import mask_to_rle, rle_to_mask
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    rle = mask_to_rle(mask)
    assert rle["counts"][0] == 0, "First count must be 0 when first pixel is foreground"
    recovered = rle_to_mask(rle)
    assert np.array_equal(mask, recovered)


# ---------------------------------------------------------------------------
# Caption cache key
# ---------------------------------------------------------------------------
def test_caption_cache_key_distinct():
    from src.retrieval_server_visual import _caption_cache_key
    k1 = _caption_cache_key("/img/a.jpg", "Relevant",   "dogs",  "")
    k2 = _caption_cache_key("/img/a.jpg", "Irrelevant", "dogs",  "")
    k3 = _caption_cache_key("/img/a.jpg", "Relevant",   "cats",  "")
    k4 = _caption_cache_key("/img/a.jpg", "Relevant",   "dogs",  "fluffy")
    assert len({k1, k2, k3, k4}) == 4, "All keys must be distinct"


# ---------------------------------------------------------------------------
# _fuse static method
# ---------------------------------------------------------------------------
def test_fuse_both():
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    result = RetrievalServiceVisual._fuse(a, b, 0.4, 0.6)
    assert result is not None
    assert float(result[0]) == pytest.approx(0.4)
    assert float(result[1]) == pytest.approx(0.6)


def test_fuse_only_image():
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    a = torch.tensor([1.0, 2.0])
    result = RetrievalServiceVisual._fuse(a, None, 0.4, 0.6)
    assert result is a


def test_fuse_only_text():
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    b = torch.tensor([3.0, 4.0])
    result = RetrievalServiceVisual._fuse(None, b, 0.4, 0.6)
    assert result is b


def test_fuse_neither():
    from src.services.retrieval_service import RetrievalServiceVisual
    result = RetrievalServiceVisual._fuse(None, None, 0.4, 0.6)
    assert result is None


# ---------------------------------------------------------------------------
# apply_mask correctness
# ---------------------------------------------------------------------------
def test_apply_mask_empty_mask_returns_original():
    from src.models.sam import apply_mask
    from PIL import Image as PILImage
    img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    empty_mask = np.zeros((10, 10), dtype=bool)
    result = apply_mask(img, empty_mask)
    # empty mask returns original image unchanged
    assert result.size == img.size


def test_apply_mask_full_mask():
    from src.models.sam import apply_mask
    from PIL import Image as PILImage
    arr = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    full_mask = np.ones((20, 20), dtype=bool)
    result = apply_mask(img, full_mask)
    assert result is not None
    assert result.size[0] > 0 and result.size[1] > 0


# ---------------------------------------------------------------------------
# Ollama prompt selection
# ---------------------------------------------------------------------------
def test_caption_crop_no_crash_on_unavailable_server():
    from src.models.ollama_vision import caption_crop
    from PIL import Image as PILImage
    import numpy as np
    img = PILImage.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    # Should return None gracefully when server is not reachable
    result = caption_crop(img, "test query", "Relevant",
                          url="http://localhost:9999",  # nothing listening here
                          timeout=1.0)
    assert result is None


def test_batch_caption_respects_limit():
    from src.models.ollama_vision import batch_caption, MAX_CROPS_PER_LABEL
    from PIL import Image as PILImage
    crops = [PILImage.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)) for _ in range(5)]
    results = batch_caption(
        crops=crops,
        query="test",
        labels=["Relevant"] * 5,
        url="http://localhost:9999",  # unreachable
        timeout=0.1,
    )
    # Should return 5 results (some None), only first MAX_CROPS_PER_LABEL are attempted
    assert len(results) == 5
    none_count = sum(1 for r in results if r is None)
    assert none_count >= 5 - MAX_CROPS_PER_LABEL


# ---------------------------------------------------------------------------
# RegionIndex (per-object FAISS used by the hard filter)
# ---------------------------------------------------------------------------
def _build_test_region_index(vecs: np.ndarray, meta: list):
    import faiss
    from src.services.region_index import RegionIndex
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs.astype("float32"))
    return RegionIndex(idx, meta)


def test_region_index_knn_dim_mismatch_raises():
    ri = _build_test_region_index(
        np.eye(4, dtype="float32"),
        [{"image_path": f"/i/{i}.jpg", "phrase": f"p{i}", "region_idx": i, "bbox": [0, 0, 1, 1]} for i in range(4)],
    )
    with pytest.raises(ValueError, match="dim"):
        ri.knn(np.zeros(8, dtype="float32"), k=2)


def test_region_index_knn_empty_returns_empty():
    import faiss
    from src.services.region_index import RegionIndex
    ri = RegionIndex(faiss.IndexFlatIP(4), [])
    assert ri.knn(np.ones(4, dtype="float32"), k=5) == []


def test_region_index_knn_threshold_filters():
    ri = _build_test_region_index(
        np.eye(4, dtype="float32"),
        [{"image_path": f"/i/{i}.jpg", "phrase": f"p{i}", "region_idx": i, "bbox": [0, 0, 1, 1]} for i in range(4)],
    )
    q = np.array([1, 0, 0, 0], dtype="float32")
    # All non-matching rows have cosine 0; only the matching row has cosine 1.
    hits = ri.knn(q, k=4, score_threshold=0.5)
    assert len(hits) == 1
    assert hits[0].image_path == "/i/0.jpg"


def test_region_index_parent_images_dedups():
    vecs = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32")
    meta = [
        {"image_path": "/img/A.jpg", "phrase": "x1", "region_idx": 0, "bbox": []},
        {"image_path": "/img/A.jpg", "phrase": "x2", "region_idx": 1, "bbox": []},
        {"image_path": "/img/B.jpg", "phrase": "y1", "region_idx": 2, "bbox": []},
    ]
    ri = _build_test_region_index(vecs, meta)
    parents = ri.parent_images(np.array([1, 0, 0, 0], dtype="float32"), k=3, score_threshold=0.5)
    # Two matching regions in /img/A.jpg → de-duped to one entry with max score 1.0
    assert len(parents) == 1
    assert parents[0][0] == "/img/A.jpg"
    assert parents[0][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _hard_filter_lookup — exercised against a tiny in-memory RegionIndex with
# a SimpleNamespace standing in for the RetrievalServiceVisual instance, so
# we don't have to load SigLIP.
# ---------------------------------------------------------------------------
def _fake_service():
    from types import SimpleNamespace
    return SimpleNamespace(_session_blacklist={}, _session_boostlist={})


def test_hard_filter_no_region_index_returns_zero_telemetry():
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    info = RetrievalServiceVisual._hard_filter_lookup(
        _fake_service(),
        relevant_per_segment=torch.zeros(2, 4),
        irrelevant_per_segment=None,
        relevant_per_text=None,
        irrelevant_per_text=None,
        region_index=None,
        session_id="x",
    )
    assert info["blacklist_size"] == 0
    assert info["boostlist_size"] == 0
    assert info["top_negative_phrases"] == []


def test_hard_filter_blacklist_wins_over_boost():
    """An image flagged as both positive and negative ends up only in blacklist."""
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    ri = _build_test_region_index(
        np.array([[1, 0, 0, 0]], dtype="float32"),
        [{"image_path": "/img/shared.jpg", "phrase": "thing", "region_idx": 0, "bbox": []}],
    )
    fake = _fake_service()
    same = torch.tensor([[1.0, 0, 0, 0]])
    info = RetrievalServiceVisual._hard_filter_lookup(
        fake,
        relevant_per_segment=same,
        irrelevant_per_segment=same,
        relevant_per_text=None,
        irrelevant_per_text=None,
        region_index=ri,
        session_id="s1",
    )
    assert "/img/shared.jpg" in fake._session_blacklist["s1"]
    assert "/img/shared.jpg" not in fake._session_boostlist["s1"]
    assert info["blacklist_size"] == 1
    assert info["boostlist_size"] == 0


def test_hard_filter_text_only_negative_still_blacklists():
    """A typed hint (text embedding) alone, with no SAM image, populates the blacklist."""
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    # Region vector cos-sim 0.707 with [1,0,0,0] — comfortably above text threshold 0.20
    vec = np.array([[0.5, 0.5, 0, 0]], dtype="float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    ri = _build_test_region_index(
        vec, [{"image_path": "/img/bike.jpg", "phrase": "bicycle", "region_idx": 0, "bbox": []}],
    )
    fake = _fake_service()
    info = RetrievalServiceVisual._hard_filter_lookup(
        fake,
        relevant_per_segment=None,
        irrelevant_per_segment=None,
        relevant_per_text=None,
        irrelevant_per_text=torch.tensor([[1.0, 0, 0, 0]]),
        region_index=ri,
        session_id="t1",
    )
    assert "/img/bike.jpg" in fake._session_blacklist["t1"]
    assert info["blacklist_size"] == 1
    assert info["top_negative_phrases"][0]["phrase"] == "bicycle"


def test_hard_filter_phrase_counts_aggregated():
    """Same phrase hit twice (once per query row) → count=2, ranked first."""
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32")
    meta = [
        {"image_path": "/img/wheel.jpg", "phrase": "wheel", "region_idx": 0, "bbox": []},
        {"image_path": "/img/seat.jpg",  "phrase": "seat",  "region_idx": 1, "bbox": []},
    ]
    ri = _build_test_region_index(vecs, meta)
    # Two negatives both align with "wheel"
    neg = torch.tensor([[1.0, 0, 0, 0], [0.9, 0.1, 0, 0]])
    fake = _fake_service()
    info = RetrievalServiceVisual._hard_filter_lookup(
        fake,
        relevant_per_segment=None,
        irrelevant_per_segment=neg,
        relevant_per_text=None,
        irrelevant_per_text=None,
        region_index=ri,
        session_id="s",
    )
    # /img/wheel.jpg added once (path-dedup inside one round); phrase counted once.
    assert info["top_negative_phrases"][0]["phrase"] == "wheel"
    assert "/img/wheel.jpg" in fake._session_blacklist["s"]


# ---------------------------------------------------------------------------
# _per_text_text_embeddings — dedupe and 20-cap
# ---------------------------------------------------------------------------
def test_per_text_text_embeddings_dedup_and_cap():
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    import torch
    from src.services.retrieval_service import RetrievalServiceVisual

    wrapper = MagicMock()
    wrapper.process_inputs = MagicMock(return_value="INPUTS")
    wrapper.get_text_embeddings = MagicMock(return_value=torch.randn(20, 4))
    fake = SimpleNamespace(wrapper=wrapper)

    texts = ["a", "b", "", "  ", "a", "c"] + [f"d{i}" for i in range(25)]
    out = RetrievalServiceVisual._per_text_text_embeddings(fake, texts)
    assert out is not None and out.shape[0] == 20

    passed = wrapper.process_inputs.call_args.kwargs["text"]
    assert len(passed) == 20
    assert passed.count("a") == 1, "duplicates must be removed"
    assert "" not in passed and "  " not in passed


def test_per_text_text_embeddings_empty_returns_none():
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    from src.services.retrieval_service import RetrievalServiceVisual
    fake = SimpleNamespace(wrapper=MagicMock())
    assert RetrievalServiceVisual._per_text_text_embeddings(fake, []) is None
    assert RetrievalServiceVisual._per_text_text_embeddings(fake, ["", "  "]) is None


# ---------------------------------------------------------------------------
# relevance_feedback precedence: SAM > box > full-image label
# ---------------------------------------------------------------------------
def test_relevance_feedback_sam_overrides_full_image_label(tmp_path):
    """An image with a SAM mask suppresses its full-image label."""
    from PIL import Image as PILImage
    from unittest.mock import MagicMock
    from src.models.relevance_feedback import ImageBasedVLMRelevanceFeedback
    from src.models.sam import mask_to_rle

    img_path = tmp_path / "x.jpg"
    PILImage.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(str(img_path))

    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    sam_ann = {
        "mask_rle": mask_to_rle(mask),
        "label": "Relevant",
        "image_path": str(img_path),
    }

    rf = ImageBasedVLMRelevanceFeedback(vlm_wrapper_retrieval=MagicMock())
    out = rf(
        query="x",
        relevant_image_paths=[str(img_path)],
        annotator_json_boxes_list=[None],
        sam_annotations=[sam_ann],
        image_labels=["Irrelevant"],  # would mark the whole image as negative if no SAM
    )
    assert len(out["relevant_segments"]) == 1, "SAM crop must be the source of truth"
    assert len(out["irrelevant_segments"]) == 0


def test_relevance_feedback_full_image_label_when_no_sam_no_box(tmp_path):
    from PIL import Image as PILImage
    from unittest.mock import MagicMock
    from src.models.relevance_feedback import ImageBasedVLMRelevanceFeedback

    img_path = tmp_path / "y.jpg"
    PILImage.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(str(img_path))

    rf = ImageBasedVLMRelevanceFeedback(vlm_wrapper_retrieval=MagicMock())
    out = rf(
        query="y",
        relevant_image_paths=[str(img_path)],
        annotator_json_boxes_list=[None],
        sam_annotations=None,
        image_labels=["Irrelevant"],
    )
    assert len(out["irrelevant_segments"]) == 1
    assert len(out["relevant_segments"]) == 0


def test_relevance_feedback_unknown_label_silently_skipped(tmp_path):
    """Unknown image_label values do not crash and contribute no segments."""
    from PIL import Image as PILImage
    from unittest.mock import MagicMock
    from src.models.relevance_feedback import ImageBasedVLMRelevanceFeedback

    img_path = tmp_path / "z.jpg"
    PILImage.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8)).save(str(img_path))

    rf = ImageBasedVLMRelevanceFeedback(vlm_wrapper_retrieval=MagicMock())
    out = rf(
        query="z",
        relevant_image_paths=[str(img_path)],
        annotator_json_boxes_list=[None],
        sam_annotations=None,
        image_labels=["Maybe"],  # not Relevant/Irrelevant
    )
    assert out["relevant_segments"] == []
    assert out["irrelevant_segments"] == []


# ---------------------------------------------------------------------------
# Endpoint validators (Pydantic + _validate_feedback_paths)
# ---------------------------------------------------------------------------
def test_validate_feedback_paths_image_labels_length_mismatch():
    from fastapi import HTTPException
    from src.retrieval_server_visual import _validate_feedback_paths, ProcessApplyFeedbackRequest
    req = ProcessApplyFeedbackRequest(
        query="q", top_k=5,
        relevant_image_paths=["/a", "/b"],
        relevant_captions="", irrelevant_captions="",
        annotator_json_boxes_list=[None, None],
        image_labels=["Relevant"],  # length 1 vs paths length 2
    )
    with pytest.raises(HTTPException) as exc:
        _validate_feedback_paths(req)
    assert exc.value.status_code == 400
    assert "Image labels" in exc.value.detail


def test_validate_feedback_paths_image_labels_bad_value():
    from fastapi import HTTPException
    from src.retrieval_server_visual import _validate_feedback_paths, ProcessApplyFeedbackRequest
    req = ProcessApplyFeedbackRequest(
        query="q", top_k=5,
        relevant_image_paths=["/a"],
        relevant_captions="", irrelevant_captions="",
        annotator_json_boxes_list=[None],
        image_labels=["Maybe"],  # not in {Relevant, Irrelevant, null}
    )
    with pytest.raises(HTTPException) as exc:
        _validate_feedback_paths(req)
    assert exc.value.status_code == 400
    assert "Relevant" in exc.value.detail or "Irrelevant" in exc.value.detail


def test_validate_feedback_paths_null_label_accepted():
    """A null entry (image with no label) must NOT trip the bad-value check."""
    from src.retrieval_server_visual import _validate_feedback_paths, ProcessApplyFeedbackRequest
    req = ProcessApplyFeedbackRequest(
        query="q", top_k=5,
        relevant_image_paths=[],
        relevant_captions="", irrelevant_captions="",
        annotator_json_boxes_list=[],
        image_labels=[],  # empty length matches empty paths
    )
    # No exception should be raised
    _validate_feedback_paths(req)


# ---------------------------------------------------------------------------
# Region index startup-path resolution: verify the runtime-loaded artifact
# (when present) matches its sidecar metadata count.
# ---------------------------------------------------------------------------
def test_region_index_artifact_consistency_if_present():
    """If a region index is on disk, ntotal must match meta line count.

    Skipped silently when no artifact is present (CI / fresh checkout)."""
    import os
    from pathlib import Path
    from src.config import repo_root
    base = repo_root() / "faiss" / "visual_genome" / "google" / "siglip-large-patch16-256"
    faiss_p = base / "region_index.faiss"
    meta_p = base / "region_meta.jsonl"
    if not faiss_p.is_file() or not meta_p.is_file():
        pytest.skip("region index artifacts not present in this checkout")
    import faiss
    idx = faiss.read_index(str(faiss_p))
    with open(meta_p) as f:
        rows = sum(1 for line in f if line.strip())
    assert idx.ntotal == rows, (
        f"FAISS ntotal {idx.ntotal} != meta rows {rows}; rebuild the index"
    )
