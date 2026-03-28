
import os
import time
import platform
import subprocess
from typing import Any, List, Optional, Tuple

import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image as PILImage

from src.config import settings
from src.services.retrieval_service import RetrievalServiceVisual
from src.models.sam import build_segmenter, apply_mask, mask_to_rle, image_to_b64 as sam_img_to_b64
from src.models.ollama_vision import check_ollama
from src.utils.image_utils import image_to_base64
from src.utils.utils import load_yaml

_start_time = time.time()

app = FastAPI(title="Retrieval Server")

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
    # Dimensions of each base64 preview (same as IMG_SIZE in config — clicks are in this space)
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
    # Pixel space of click coordinates (browser naturalWidth / naturalHeight of the preview image)
    coord_width: Optional[int] = None
    coord_height: Optional[int] = None


class SegmentResponse(BaseModel):
    mask_rle: dict
    region_b64: str
    score: float
    width: int
    height: int


retrieval_service: Optional[RetrievalServiceVisual] = None
sam_segmenter = None
sam_model_type: str = "none"
ollama_available: bool = False


def _preview_side() -> Tuple[int, int]:
    """Square preview size for thumbnails — must match click coordinate space from the UI."""
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
    """Match `resize_images()` in retrieval_service (BICUBIC square)."""
    return full.resize((pw, ph), PILImage.Resampling.BICUBIC)


@app.on_event("startup")
async def startup_event():
    global retrieval_service, sam_segmenter, sam_model_type

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
            print(f"[startup] Ollama vision: not available (feedback will use image-only embeddings)")
    else:
        print("[startup] Ollama vision: disabled via config")


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    try:
        images, scores, image_paths = retrieval_service.search_images(request.query, request.top_k)
        images = [image_to_base64(img) for img in images]
        pw, ph = _preview_side()
        return SearchResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Search completed successfully",
            preview_width=pw,
            preview_height=ph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply_feedback", response_model=ProcessApplyFeedbackResponse)
async def apply_feedback(request: ProcessApplyFeedbackRequest):
    try:
        images, scores, image_paths = retrieval_service.process_and_apply_feedback(
            query=request.query,
            top_k=request.top_k,
            relevant_image_paths=request.relevant_image_paths,
            relevant_captions=request.relevant_captions,
            irrelevant_captions=request.irrelevant_captions,
            annotator_json_boxes_list=request.annotator_json_boxes_list,
            sam_annotations=request.sam_annotations,
            fuse_initial_query=request.fuse_initial_query,
            ollama_available=ollama_available,
        )
        images = [image_to_base64(img) for img in images]
        pw, ph = _preview_side()
        return ProcessApplyFeedbackResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Feedback applied successfully",
            preview_width=pw,
            preview_height=ph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment", response_model=SegmentResponse)
async def segment_image(request: SegmentRequest):
    if sam_segmenter is None:
        raise HTTPException(status_code=503, detail="No SAM model loaded")
    try:
        image = PILImage.open(request.image_path).convert("RGB")
        W, H = image.size

        # Clicks are on the search preview (e.g. 224×224), but this file is full resolution.
        # Map preview-space coordinates → original pixels, run SAM at full res, then
        # downsample the mask back to preview space so the overlay aligns with the UI.
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

        return SegmentResponse(
            mask_rle=rle,
            region_b64=sam_img_to_b64(region),
            score=result["score"],
            width=cw,
            height=ch,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
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
    }


@app.get("/ollama_status")
async def ollama_status():
    return {
        "available": ollama_available,
        "model": settings.ollama_model,
        "url": settings.ollama_url,
    }


def _mps_gpu_util_pct() -> float:
    """
    Read Apple GPU active residency via powermetrics.
    Requires passwordless sudo for powermetrics, or falls back to
    a CPU-load proxy (P-core %) which correlates with ML workloads on M-series.
    Returns a float 0–100.
    """
    try:
        # Sample for 200 ms — short enough to not block the endpoint
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics",
             "--samplers", "gpu_power",
             "-n", "1", "-i", "200",
             "--format", "plist"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        import plistlib
        data = plistlib.loads(out)
        gpu_data = data.get("gpu", {})
        # Key is "gpu_active_ratio" (0.0–1.0) on macOS 13+
        ratio = gpu_data.get("gpu_active_ratio", None)
        if ratio is not None:
            return round(float(ratio) * 100, 1)
        # Older key name
        ratio = gpu_data.get("active_ratio", None)
        if ratio is not None:
            return round(float(ratio) * 100, 1)
    except Exception:
        pass

    # Fallback: use P-core (performance core) CPU utilisation as a proxy.
    # On Apple Silicon the GPU and Neural Engine share the same die as the CPU;
    # heavy ML workloads show up here when powermetrics is unavailable.
    try:
        per_cpu = psutil.cpu_percent(percpu=True)
        # P-cores are typically the first N/2 logical cores on M-series
        half = max(1, len(per_cpu) // 2)
        return round(sum(per_cpu[:half]) / half, 1)
    except Exception:
        return 0.0


def _gpu_metrics() -> dict:
    """Collect GPU metrics: CUDA via nvidia-smi, MPS via powermetrics/proxy."""
    info: dict = {
        "available": False,
        "name": None,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "utilization_pct": 0,
        "backend": "none",
    }

    if torch.cuda.is_available():
        info["available"] = True
        info["backend"] = "cuda"
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_used_mb"] = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1)
        info["memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1
        )
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            info["utilization_pct"] = int(out.decode().strip().split("\n")[0])
        except Exception:
            info["utilization_pct"] = round(
                info["memory_used_mb"] / max(info["memory_total_mb"], 1) * 100, 1
            )

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["available"] = True
        info["backend"] = "mps"
        info["name"] = "Apple M-series GPU (MPS)"

        # PyTorch-allocated MPS memory
        allocated = 0
        driver_allocated = 0
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory()
        if hasattr(torch.mps, "driver_allocated_memory"):
            driver_allocated = torch.mps.driver_allocated_memory()

        # Use driver_allocated (includes framework overhead) when available
        used = driver_allocated if driver_allocated > 0 else allocated
        mem = psutil.virtual_memory()

        info["memory_used_mb"] = round(used / 1024 / 1024, 1)
        # On Apple Silicon, GPU and CPU share unified memory.
        # Report total system RAM as the "total" VRAM.
        info["memory_total_mb"] = round(mem.total / 1024 / 1024, 1)
        info["utilization_pct"] = _mps_gpu_util_pct()

    return info


@app.get("/metrics")
async def metrics():
    cpu_pct = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()
    gpu = _gpu_metrics()

    return {
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - _start_time, 1),
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
            "index_size": retrieval_service.index.ntotal if retrieval_service and hasattr(retrieval_service, "index") else 0,
        },
    }


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    port = args.port
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")
