"""
Visual Genome region descriptions — load, index, and look up at runtime.

Builds a fast lookup: image_filename → list of (phrase, x, y, w, h)
so the server can find region descriptions for any VG image without Ollama.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

Region = Dict


def _extract_image_id(path: str) -> Optional[int]:
    basename = os.path.basename(path)
    m = re.match(r"(\d+)\.\w+$", basename)
    return int(m.group(1)) if m else None


class VGRegionIndex:
    """
    In-memory index of Visual Genome region descriptions.

    Usage:
        vg = VGRegionIndex.load(vg_dir="/path/to/data/visual_genome")
        regions = vg.get_regions_for_image(".../VG_100K/42.jpg")
        phrase = vg.best_region_for_point(regions, click_x, click_y, img_w, img_h)
    """

    def __init__(self, regions_by_id: Dict[int, List[Region]]):
        self._regions = regions_by_id
        logger.info("[VG] Loaded region descriptions for %d images", len(self._regions))

    @classmethod
    def load(cls, vg_dir: str) -> Optional["VGRegionIndex"]:
        regions_path = os.path.join(vg_dir, "region_descriptions.json")
        if not os.path.isfile(regions_path):
            logger.warning("[VG] region_descriptions.json not found at %s", regions_path)
            return None

        logger.info("[VG] Loading region_descriptions.json...")
        with open(regions_path, "r") as f:
            raw = json.load(f)

        regions_by_id: Dict[int, List[Region]] = {}
        total_regions = 0
        for entry in raw:
            image_id = entry.get("id")
            if image_id is None:
                continue
            regions = []
            for r in entry.get("regions", []):
                phrase = r.get("phrase", "").strip()
                if not phrase:
                    continue
                regions.append({
                    "phrase": phrase,
                    "x": r.get("x", 0),
                    "y": r.get("y", 0),
                    "width": r.get("width", 0),
                    "height": r.get("height", 0),
                })
            if regions:
                regions_by_id[image_id] = regions
                total_regions += len(regions)

        logger.info("[VG] Indexed %d regions across %d images", total_regions, len(regions_by_id))
        return cls(regions_by_id)

    def get_regions(self, image_path: str) -> List[Region]:
        image_id = _extract_image_id(image_path)
        if image_id is None:
            return []
        return self._regions.get(image_id, [])

    def get_all_phrases(self, image_path: str) -> List[str]:
        return [r["phrase"] for r in self.get_regions(image_path)]

    def get_phrases_by_id(self, image_id: int) -> List[str]:
        return [r["phrase"] for r in self._regions.get(image_id, [])]

    def best_region_for_point(
        self, image_path: str, click_x: float, click_y: float,
        img_w: int, img_h: int,
    ) -> Optional[str]:
        regions = self.get_regions(image_path)
        if not regions:
            return None
        best_phrase, best_dist = None, float("inf")
        for r in regions:
            cx = r["x"] + r["width"] / 2.0
            cy = r["y"] + r["height"] / 2.0
            dist = (click_x - cx) ** 2 + (click_y - cy) ** 2
            if r["x"] <= click_x <= r["x"] + r["width"] and r["y"] <= click_y <= r["y"] + r["height"]:
                dist *= 0.01
            if dist < best_dist:
                best_dist = dist
                best_phrase = r["phrase"]
        return best_phrase

    def top_phrases_for_mask(
        self, image_path: str, mask: np.ndarray,
        top_k: int = 3, min_iou: float = 0.05,
    ) -> List[str]:
        regions = self.get_regions(image_path)
        if not regions or mask.sum() == 0:
            return []
        h, w = mask.shape[:2]
        scored = []
        for r in regions:
            rx = max(0, min(r["x"], w - 1))
            ry = max(0, min(r["y"], h - 1))
            rw = min(r["width"], w - rx)
            rh = min(r["height"], h - ry)
            if rw <= 0 or rh <= 0:
                continue
            region_mask = np.zeros((h, w), dtype=bool)
            region_mask[ry:ry + rh, rx:rx + rw] = True
            intersection = (mask & region_mask).sum()
            union = (mask | region_mask).sum()
            if union == 0:
                continue
            iou = float(intersection) / float(union)
            if iou >= min_iou:
                scored.append((iou, r["phrase"]))
        scored.sort(key=lambda x: -x[0])
        return [phrase for _, phrase in scored[:top_k]]

    @property
    def image_count(self) -> int:
        return len(self._regions)

    def has_image(self, image_path: str) -> bool:
        image_id = _extract_image_id(image_path)
        return image_id is not None and image_id in self._regions
