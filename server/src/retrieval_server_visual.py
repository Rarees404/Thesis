
import asyncio
import os
import time
import platform
import subprocess
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image as PILImage

from src.config import settings, resolve_repo
from src.services.retrieval_service import RetrievalServiceVisual
from src.models.sam import build_segmenter, apply_mask, mask_to_rle, image_to_b64 as sam_img_to_b64
from src.models.ollama_vision import check_ollama, caption_crop

from src.utils.utils import load_yaml
from contextlib import asynccontextmanager

_start_time = time.time()

# Short TTL cache for /metrics — powermetrics and cpu_percent sampling are costly if polled often.
_metrics_cache: dict | None = None
_metrics_cache_expiry: float = 0.0
_METRICS_CACHE_TTL_SEC = 3.0

# ---------------------------------------------------------------------------
# Async caption cache — Ollama captions are pre-computed right after /segment
# so they're available instantly when /apply_feedback is called.
# Key: "{image_path}::{label}::{query}" → caption string (or empty sentinel)
# ---------------------------------------------------------------------------
_caption_cache: Dict[str, str] = {}
_caption_in_flight: set = set()
_CAPTION_CACHE_MAX = 200  # evict oldest when full

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield


app = FastAPI(title="Retrieval Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str
    # Dimensions of each base64 preview (same as IMG_SIZE in config)
    preview_width: int
    preview_height: int


class ProcessApplyFeedbackRequest(BaseModel):
    query: str
    top_k: int
    relevant_image_paths: List[str]
    relevant_captions: str
    irrelevant_captions: str
    annotator_json_boxes_list: List[Any]
    sam_annotations: Optional[List[Any]] = None
    fuse_initial_query: bool = False


class ProcessApplyFeedbackResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str
    preview_width: int
    preview_height: int


class SegmentPoint(BaseModel):
    x: float
    y: float
    label: int  # 1 = foreground/relevant, 0 = background/irrelevant


class SegmentRequest(BaseModel):
    image_path: str
    points: List[SegmentPoint]
    # Pixel space of click coordinates (browser naturalWidth / naturalHeight of the preview)
    coord_width: Optional[int] = None
    coord_height: Optional[int] = None
    # Current search query — used for background Ollama captioning pre-fetch
    query: Optional[str] = None
    # User's hint text, if already typed — enriches the background caption
    user_hint: Optional[str] = None
    # Expected label for background captioning ("Relevant" | "Irrelevant")
    label: Optional[str] = "Relevant"


class SegmentResponse(BaseModel):
    mask_rle: dict
    region_b64: str
    score: float
    width: int
    height: int
    vg_phrases: Optional[List[str]] = None
    # Pre-computed Ollama caption (from cache, or None if not yet ready)
    cached_caption: Optional[str] = None


class CaptionRequest(BaseModel):
    image_b64: str
    query: str
    label: str = "Relevant"
    user_hint: Optional[str] = None


class CaptionResponse(BaseModel):
    caption: Optional[str]
    model: str
    latency_ms: int


retrieval_service: Optional[RetrievalServiceVisual] = None
sam_segmenter = None
sam_model_type: str = "none"
ollama_available: bool = False

vg_index = None  # Optional[VGRegionIndex]

# Serialize all SAM calls — the model has shared mutable state that is NOT thread-safe.
_sam_lock: asyncio.Lock = asyncio.Lock()


def _preview_side() -> Tuple[int, int]:
    if retrieval_service is None:
        return 224, 224
    s = int(retrieval_service.config.get("IMG_SIZE", 224))
    return s, s


def _mask_to_preview_space(mask: np.ndarray, pw: int, ph: int) -> np.ndarray:
    """Downsample a full-res boolean mask to the UI preview size."""
    h, w = int(mask.shape[0]), int(mask.shape[1])
    if w == pw and h == ph:
        return mask.astype(bool)
    pil_m = PILImage.fromarray(mask.astype(np.uint8) * 255)
    pil_m = pil_m.resize((pw, ph), PILImage.Resampling.NEAREST)
    return np.array(pil_m) > 127


def _preview_like_search(full: PILImage.Image, pw: int, ph: int) -> PILImage.Image:
    return full.resize((pw, ph), PILImage.Resampling.BICUBIC)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) bounding box of the True region, or None."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))


def _evict_caption_cache():
    """Keep the cache bounded by removing the oldest half when full."""
    global _caption_cache
    if len(_caption_cache) >= _CAPTION_CACHE_MAX:
        keys = list(_caption_cache.keys())
        for k in keys[: len(keys) // 2]:
            _caption_cache.pop(k, None)


def _caption_cache_key(image_path: str, label: str, query: str, hint: str = "") -> str:
    return f"{image_path}::{label}::{query}::{hint}"


async def _background_caption(
    image_path: str,
    label: str,
    query: str,
    region_b64: str,
    bbox: Optional[Tuple[int, int, int, int]],
    user_hint: str = "",
):
    """
    Fire-and-forget Ollama caption called after /segment.
    Result stored in _caption_cache so /apply_feedback can reuse it instantly.
    """
    global _caption_cache, _caption_in_flight
    cache_key = _caption_cache_key(image_path, label, query, user_hint)
    if cache_key in _caption_cache or cache_key in _caption_in_flight:
        return
    _caption_in_flight.add(cache_key)
    try:
        import base64 as b64mod
        img_bytes = b64mod.b64decode(region_b64)
        crop = PILImage.open(BytesIO(img_bytes)).convert("RGB")

        # Also load the full-scene image for context captioning
        context_img: Optional[PILImage.Image] = None
        try:
            context_img = PILImage.open(image_path).convert("RGB")
        except Exception:
            pass

        def _run():
            return caption_crop(
                image=crop,
                query=query,
                label=label,
                url=settings.ollama_url,
                model=settings.ollama_model,
                timeout=90.0,
                user_hint=user_hint or None,
                context_image=context_img,
                bbox=bbox,
            )

        caption = await asyncio.to_thread(_run)
        _evict_caption_cache()
        # Store even if None (empty sentinel prevents re-triggering)
        _caption_cache[cache_key] = caption or ""
        if caption:
            import logging; logging.getLogger(__name__).info(
                "[BG-Caption] %s | %s: %s", label, image_path[-40:], caption
            )
    except Exception as exc:
        import logging; logging.getLogger(__name__).warning(
            "[BG-Caption] Failed for %s: %s", image_path[-40:], exc
        )
    finally:
        _caption_in_flight.discard(cache_key)


async def startup_event():
    global retrieval_service, sam_segmenter, sam_model_type, vg_index

    config = load_yaml(settings.config_path)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[startup] Using device: {device}")

    retrieval_service = RetrievalServiceVisual(
        config=config,
        faiss_index=settings.index_path,
        device=device,
        ollama_url=settings.ollama_url,
        ollama_model=settings.ollama_model,
    )
    print(
        f"[startup] FAISS index: {settings.index_path} "
        f"({getattr(retrieval_service.index, 'ntotal', '?')} vectors)"
    )

    checkpoints_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
    )
    sam_segmenter, sam_model_type = build_segmenter(
        device=device, checkpoints_dir=checkpoints_dir, backend=settings.sam_backend,
    )
    print(f"[startup] SAM backend: {sam_model_type} (requested: {settings.sam_backend})")

    global ollama_available
    if settings.ollama_enabled:
        ollama_available = check_ollama(settings.ollama_url, settings.ollama_model)
        if ollama_available:
            print(f"[startup] Ollama vision: available ({settings.ollama_model})")
        else:
            print("[startup] Ollama vision: not available (feedback will use image-only embeddings)")
    else:
        print("[startup] Ollama vision: disabled via config")

    corpus_path = (config.get("IMAGE_CORPUS_PATH") or "").strip()
    if corpus_path and "visual_genome" in corpus_path:
        try:
            from src.utils.vg_regions import VGRegionIndex
            vg_dir = resolve_repo(corpus_path.rstrip("/"))
            vg_index = VGRegionIndex.load(vg_dir)
            if vg_index:
                print(f"[startup] VG region index loaded ({vg_index.image_count} images)")
            else:
                print("[startup] VG region_descriptions.json not found — VG phrases disabled")
        except Exception as e:
            print(f"[startup] VG region loading failed: {e}")
            vg_index = None


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not ready — models still loading")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if request.top_k < 1 or request.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    try:
        def _run():
            return retrieval_service.search_images(request.query, request.top_k)
        images_b64, scores, image_paths = await asyncio.to_thread(_run)
        pw, ph = _preview_side()
        return SearchResponse(
            images=images_b64,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Search completed successfully",
            preview_width=pw,
            preview_height=ph,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply_feedback", response_model=ProcessApplyFeedbackResponse)
async def apply_feedback(request: ProcessApplyFeedbackRequest):
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not ready — models still loading")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        def _run():
            return retrieval_service.process_and_apply_feedback(
                query=request.query,
                top_k=request.top_k,
                relevant_image_paths=request.relevant_image_paths,
                relevant_captions=request.relevant_captions,
                irrelevant_captions=request.irrelevant_captions,
                annotator_json_boxes_list=request.annotator_json_boxes_list,
                sam_annotations=request.sam_annotations,
                fuse_initial_query=request.fuse_initial_query,
                ollama_available=ollama_available,
                vg_region_index=vg_index,
                caption_cache=_caption_cache,
            )
        images_b64, scores, image_paths = await asyncio.to_thread(_run)
        pw, ph = _preview_side()
        return ProcessApplyFeedbackResponse(
            images=images_b64,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Feedback applied successfully",
            preview_width=pw,
            preview_height=ph,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment", response_model=SegmentResponse)
async def segment_image(request: SegmentRequest):
    if sam_segmenter is None:
        raise HTTPException(status_code=503, detail="No SAM model loaded")
    if retrieval_service is not None:
        allowed = set(retrieval_service.candidate_image_paths)
        if request.image_path not in allowed:
            raise HTTPException(status_code=403, detail="Image path not in search corpus")
    if not request.points:
        raise HTTPException(status_code=400, detail="At least one point is required")
    try:
        def _run_segment():
            image = PILImage.open(request.image_path).convert("RGB")
            W, H = image.size

            cw = request.coord_width if request.coord_width and request.coord_width > 0 else W
            ch = request.coord_height if request.coord_height and request.coord_height > 0 else H
            sx = W / float(cw)
            sy = H / float(ch)
            scaled_points = [
                {"x": p.x * sx, "y": p.y * sy, "label": p.label} for p in request.points
            ]

            sam_segmenter.set_image(image, path=request.image_path)
            result = sam_segmenter.segment_points(scaled_points, image_path=request.image_path)

            mask_full = result["mask"].astype(bool)
            mask = _mask_to_preview_space(mask_full, cw, ch)
            preview = _preview_like_search(image, cw, ch)
            region = apply_mask(preview, mask)
            rle = mask_to_rle(mask)
            region_b64 = sam_img_to_b64(region)

            # Compute bounding box in original image space for context captioning
            bbox_orig = _mask_bbox(mask_full)

            phrases = None
            if vg_index is not None:
                try:
                    phrases = vg_index.top_phrases_for_mask(request.image_path, mask_full, top_k=3)
                except Exception as vg_err:
                    print(f"[SAM] VG phrase lookup failed (non-fatal): {vg_err}")

            return SegmentResponse(
                mask_rle=rle,
                region_b64=region_b64,
                score=result["score"],
                width=cw,
                height=ch,
                vg_phrases=phrases if phrases else None,
            ), region_b64, bbox_orig

        async with _sam_lock:
            try:
                seg_resp, region_b64, bbox_orig = await asyncio.wait_for(
                    asyncio.to_thread(_run_segment),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Segmentation timed out (>60 s)")

        # Fire background Ollama captioning immediately after segmentation completes —
        # by the time the user clicks "Apply Feedback" the caption is likely ready.
        if ollama_available and request.query and request.query.strip():
            label = request.label or "Relevant"
            hint = (request.user_hint or "").strip()
            cache_key = _caption_cache_key(request.image_path, label, request.query.strip(), hint)
            if cache_key not in _caption_cache and cache_key not in _caption_in_flight:
                asyncio.create_task(
                    _background_caption(
                        image_path=request.image_path,
                        label=label,
                        query=request.query.strip(),
                        region_b64=region_b64,
                        bbox=bbox_orig,
                        user_hint=hint,
                    )
                )

        # Return cached caption if instantly available (from a previous round)
        cache_key_ret = _caption_cache_key(
            request.image_path,
            request.label or "Relevant",
            (request.query or "").strip(),
            (request.user_hint or "").strip(),
        )
        seg_resp.cached_caption = _caption_cache.get(cache_key_ret) or None
        return seg_resp

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/caption", response_model=CaptionResponse)
async def caption_image(request: CaptionRequest):
    if not ollama_available:
        raise HTTPException(status_code=503, detail="Ollama Vision is not available")
    try:
        import base64 as b64mod
        img_bytes = b64mod.b64decode(request.image_b64)
        img = PILImage.open(BytesIO(img_bytes)).convert("RGB")

        start = time.time()
        caption = caption_crop(
            image=img,
            query=request.query,
            label=request.label,
            url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=90.0,
            user_hint=request.user_hint,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        return CaptionResponse(
            caption=caption,
            model=settings.ollama_model,
            latency_ms=elapsed_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sam_status")
async def sam_status():
    return {
        "loaded": sam_segmenter is not None,
        "model_type": sam_model_type,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "ollama_available": ollama_available,
        "vg_index_loaded": vg_index is not None,
    }


@app.get("/ollama_status")
async def ollama_status():
    return {
        "available": ollama_available,
        "model": settings.ollama_model,
        "url": settings.ollama_url,
    }


@app.get("/caption_cache_status")
async def caption_cache_status():
    """Debug endpoint: shows how many captions are pre-computed."""
    return {
        "cached": len(_caption_cache),
        "in_flight": len(_caption_in_flight),
    }


@app.get("/caption_lookup")
async def caption_lookup(
    image_path: str,
    query: str,
    label: str = "Relevant",
    user_hint: str = "",
):
    """
    Return a cached Ollama caption if one has finished computing for this
    (image, label, query, hint) tuple. Used by the frontend to poll for the
    background caption started by /segment — closes the UX loop so the user
    sees what Ollama described without having to click again.
    """
    key = _caption_cache_key(image_path, label, query.strip(), user_hint.strip())
    caption = _caption_cache.get(key)
    in_flight = key in _caption_in_flight
    return {
        "caption": caption or None,
        "ready": bool(caption),
        "in_flight": in_flight,
    }


def _mps_gpu_util_pct() -> float:
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics", "--samplers", "gpu_power",
             "-n", "1", "-i", "200", "--format", "plist"],
            timeout=3, stderr=subprocess.DEVNULL,
        )
        import plistlib
        data = plistlib.loads(out)
        gpu_data = data.get("gpu", {})
        ratio = gpu_data.get("gpu_active_ratio") or gpu_data.get("active_ratio")
        if ratio is not None:
            return round(float(ratio) * 100, 1)
    except Exception:
        pass
    try:
        per_cpu = psutil.cpu_percent(percpu=True)
        half = max(1, len(per_cpu) // 2)
        return round(sum(per_cpu[:half]) / half, 1)
    except Exception:
        return 0.0


def _gpu_metrics() -> dict:
    info: dict = {
        "available": False, "name": None,
        "memory_used_mb": 0, "memory_total_mb": 0,
        "utilization_pct": 0, "backend": "none",
    }
    if torch.cuda.is_available():
        info.update({
            "available": True, "backend": "cuda",
            "name": torch.cuda.get_device_name(0),
            "memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
            "memory_total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1),
        })
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                timeout=2,
            )
            info["utilization_pct"] = int(out.decode().strip().split("\n")[0])
        except Exception:
            info["utilization_pct"] = round(
                info["memory_used_mb"] / max(info["memory_total_mb"], 1) * 100, 1)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        allocated = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
        driver = getattr(torch.mps, "driver_allocated_memory", lambda: 0)()
        used = driver if driver > 0 else allocated
        mem = psutil.virtual_memory()
        info.update({
            "available": True, "backend": "mps",
            "name": "Apple M-series GPU (MPS)",
            "memory_used_mb": round(used / 1024 / 1024, 1),
            "memory_total_mb": round(mem.total / 1024 / 1024, 1),
            "utilization_pct": _mps_gpu_util_pct(),
        })
    return info


@app.get("/metrics")
async def metrics():
    global _metrics_cache, _metrics_cache_expiry
    now = time.time()
    if _metrics_cache is not None and now < _metrics_cache_expiry:
        return {**_metrics_cache, "timestamp": now}

    cpu_pct = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()
    gpu = _gpu_metrics()

    payload = {
        "timestamp": now,
        "uptime_seconds": round(now - _start_time, 1),
        "system": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
        "cpu": {
            "percent": cpu_pct,
            "count": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "freq_mhz": round(cpu_freq.current, 0) if cpu_freq else 0,
        },
        "memory": {
            "total_mb": round(mem.total / 1024 / 1024, 1),
            "used_mb": round(mem.used / 1024 / 1024, 1),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 1),
            "used_gb": round(disk.used / 1024 / 1024 / 1024, 1),
            "percent": round(disk.percent, 1),
        },
        "network": {
            "bytes_sent_mb": round(net.bytes_sent / 1024 / 1024, 1),
            "bytes_recv_mb": round(net.bytes_recv / 1024 / 1024, 1),
        },
        "gpu": gpu,
        "model": {
            "loaded": retrieval_service is not None,
            "index_size": (
                retrieval_service.index.ntotal
                if retrieval_service and hasattr(retrieval_service, "index") else 0
            ),
        },
    }
    _metrics_cache = payload
    _metrics_cache_expiry = now + _METRICS_CACHE_TTL_SEC
    return payload


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")
