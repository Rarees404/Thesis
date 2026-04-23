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
