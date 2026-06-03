"""Ranking metrics for the offline retrieval evaluation.

All functions take a ranked list of image identifiers (most-relevant first)
and a set of identifiers judged relevant by the VG scene-graph ground truth.
Relevance is binary, so nDCG uses gain in {0, 1}.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    if not topk:
        return 0.0
    hits = sum(1 for p in topk if p in relevant)
    return hits / len(topk)


def dcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(ranked[:k]):
        if p in relevant:
            # rank position i is 0-based; standard DCG discount is 1/log2(rank+2)
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(ranked, relevant, k) / idcg


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    for i, p in enumerate(ranked):
        if p in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(
    ranked: Sequence[str],
    relevant: Set[str],
    ks: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    """All metrics for a single ranked result list."""
    out: Dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"nDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
    out["MRR"] = reciprocal_rank(ranked, relevant)
    return out


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each metric across queries. Missing keys treated as 0."""
    if not per_query:
        return {}
    keys = sorted({k for d in per_query for k in d})
    n = len(per_query)
    return {k: sum(d.get(k, 0.0) for d in per_query) / n for k in keys}
