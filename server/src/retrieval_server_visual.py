
import os
import time
import platform
import subprocess
from typing import Any, List, Optional

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.services.retrieval_service import RetrievalServiceVisual
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


class ProcessApplyFeedbackRequest(BaseModel):
    query: str
    top_k: int
    relevant_image_paths: List[str]
    relevant_captions: str
    irrelevant_captions: str
    annotator_json_boxes_list: List[Any]
    fuse_initial_query: bool = False


class ProcessApplyFeedbackResponse(BaseModel):
    images: List[str]
    image_paths: List[str]
    scores: List[float]
    success: bool
    message: str


retrieval_service: Optional[RetrievalServiceVisual] = None


@app.on_event("startup")
async def startup_event():
    global retrieval_service

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
    )


@app.post("/search", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    try:
        images, scores, image_paths = retrieval_service.search_images(request.query, request.top_k)
        images = [image_to_base64(img) for img in images]
        return SearchResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Search completed successfully"
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
            fuse_initial_query=request.fuse_initial_query
        )
        images = [image_to_base64(img) for img in images]
        return ProcessApplyFeedbackResponse(
            images=images,
            image_paths=image_paths,
            scores=scores,
            success=True,
            message="Feedback applied successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}


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
