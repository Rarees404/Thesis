"""
Runtime wrapper around the per-region SigLIP FAISS index produced by
src.precompute.build_region_index.

Loaded once at server startup. Used during /apply_feedback to do hard filtering:

    region_idx.knn(user_region_embedding, k=50, score_threshold=0.7)
        -> list of (image_path, score, phrase) for the k nearest VG regions.

The retrieval service then:
  - Collects parent image paths from the negatives -> session blacklist.
  - Collects parent image paths from the positives -> session boost set.
  - Over-fetches from the image-level FAISS, drops blacklisted, re-ranks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegionHit:
    image_path: str
    score: float
    phrase: str
    region_idx: int
    bbox: List[int]


class RegionIndex:
    """In-memory FAISS index of VG region embeddings + sidecar metadata."""

    def __init__(self, index: faiss.Index, meta: List[dict]):
        if index.ntotal != len(meta):
            raise ValueError(
                f"FAISS ntotal ({index.ntotal}) does not match metadata rows ({len(meta)})"
            )
        self._index = index
        self._meta = meta
        logger.info(
            "[RegionIndex] Loaded %d region embeddings (dim=%d)",
            index.ntotal, index.d,
        )

    # -- factory ---------------------------------------------------------------

    @classmethod
    def load(
        cls,
        faiss_path: str,
        meta_path: str,
    ) -> Optional["RegionIndex"]:
        if not os.path.isfile(faiss_path) or not os.path.isfile(meta_path):
            logger.warning(
                "[RegionIndex] Artifacts missing — skipping load. faiss=%s meta=%s",
                faiss_path, meta_path,
            )
            return None
        try:
            index = faiss.read_index(str(faiss_path))
            meta: List[dict] = []
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        meta.append(json.loads(line))
            return cls(index, meta)
        except Exception as exc:
            logger.exception("[RegionIndex] Failed to load: %s", exc)
            return None

    @classmethod
    def from_default_paths(cls, repo_root: Path) -> Optional["RegionIndex"]:
        base = repo_root / "faiss" / "visual_genome" / "google" / "siglip-large-patch16-256"
        return cls.load(str(base / "region_index.faiss"), str(base / "region_meta.jsonl"))

    # -- properties ------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    @property
    def dim(self) -> int:
        return self._index.d

    # -- query -----------------------------------------------------------------

    def knn(
        self,
        query: np.ndarray,
        k: int = 50,
        score_threshold: float = 0.0,
    ) -> List[RegionHit]:
        """Return up to k nearest VG regions to `query`, filtered by score.

        `query` must be (1, dim) or (dim,) and L2-normalized (since we use IP).
        Score is cosine similarity in [-1, 1] for normalized vectors.
        """
        if self._index.ntotal == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self._index.d:
            raise ValueError(
                f"Query dim {q.shape[1]} does not match index dim {self._index.d}"
            )
        D, I = self._index.search(q, min(k, self._index.ntotal))
        hits: List[RegionHit] = []
        for row, score in zip(I[0], D[0]):
            if row < 0:
                continue
            if score < score_threshold:
                continue
            m = self._meta[row]
            hits.append(
                RegionHit(
                    image_path=m["image_path"],
                    score=float(score),
                    phrase=m.get("phrase", ""),
                    region_idx=int(m.get("region_idx", -1)),
                    bbox=list(m.get("bbox", [])),
                )
            )
        return hits

    def parent_images(
        self,
        query: np.ndarray,
        k: int = 50,
        score_threshold: float = 0.0,
    ) -> List[tuple[str, float]]:
        """Like knn() but de-duped to one entry per parent image, max-score."""
        best: dict[str, float] = {}
        for hit in self.knn(query, k=k, score_threshold=score_threshold):
            if hit.image_path not in best or hit.score > best[hit.image_path]:
                best[hit.image_path] = hit.score
        return sorted(best.items(), key=lambda x: -x[1])
