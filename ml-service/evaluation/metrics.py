"""
PriceIQ ML Service — Recommendation Evaluation Metrics

Pure functions implementing standard ranking metrics:
  - Hit@K: 1 if target item appears in top-K recommendations, else 0
  - NDCG@K: Normalized Discounted Cumulative Gain at rank K (for binary relevance)
  - MRR: Mean Reciprocal Rank (1/rank if found, else 0)
  - Catalog Coverage: Proportion of total catalog recommended across evaluation set
"""
import math
from typing import List, Any, Set


def hit_at_k(recommended: List[Any], target: Any, k: int = 10) -> float:
    """
    Compute Hit@K for a single prediction.

    Args:
        recommended: List of recommended item IDs (ranked best first).
        target: True next item ID.
        k: Cutoff rank.

    Returns:
        1.0 if target is in recommended[:k], else 0.0.
    """
    if not recommended or target is None:
        return 0.0
    top_k = recommended[:k]
    return 1.0 if str(target) in [str(item) for item in top_k] else 0.0


def ndcg_at_k(recommended: List[Any], target: Any, k: int = 10) -> float:
    """
    Compute NDCG@K for a single prediction with binary relevance (target is relevant, others not).

    Formula:
        DCG@K = 1 / log2(rank + 1)   where rank is 1-indexed (rank in [1, K])
        IDCG@K = 1 / log2(1 + 1) = 1.0
        NDCG@K = DCG@K / IDCG@K = 1 / log2(rank + 1)

    Args:
        recommended: List of recommended item IDs (ranked best first).
        target: True next item ID.
        k: Cutoff rank.

    Returns:
        NDCG@K in [0.0, 1.0].
    """
    if not recommended or target is None:
        return 0.0

    target_str = str(target)
    top_k = [str(item) for item in recommended[:k]]

    try:
        rank = top_k.index(target_str) + 1  # 1-indexed
        return 1.0 / math.log2(rank + 1)
    except ValueError:
        return 0.0


def mrr_at_k(recommended: List[Any], target: Any, k: int = None) -> float:
    """
    Compute Reciprocal Rank (RR) for a single prediction.
    Averaged across queries, this produces Mean Reciprocal Rank (MRR).

    Formula:
        RR = 1 / rank   where rank is 1-indexed position of target
        RR = 0.0        if target is not in recommendations (or rank > k)

    Args:
        recommended: List of recommended item IDs (ranked best first).
        target: True next item ID.
        k: Optional cutoff rank. If None, considers all recommendations.

    Returns:
        Reciprocal rank in [0.0, 1.0].
    """
    if not recommended or target is None:
        return 0.0

    target_str = str(target)
    items = [str(item) for item in (recommended[:k] if k is not None else recommended)]

    try:
        rank = items.index(target_str) + 1  # 1-indexed
        return 1.0 / rank
    except ValueError:
        return 0.0


def compute_catalog_coverage(all_recommended_sets: List[List[Any]], catalog_items: Set[Any]) -> float:
    """
    Compute catalog coverage: fraction of catalog items recommended at least once.

    Args:
        all_recommended_sets: List of recommendation lists across queries.
        catalog_items: Set of all valid catalog item IDs.

    Returns:
        Coverage ratio in [0.0, 1.0].
    """
    if not catalog_items:
        return 0.0

    catalog_str = {str(item) for item in catalog_items}
    unique_recommended = set()
    for rec_list in all_recommended_sets:
        unique_recommended.update(str(item) for item in rec_list)

    recommended_in_catalog = unique_recommended.intersection(catalog_str)
    return len(recommended_in_catalog) / len(catalog_str)


def precision_recall_f1_at_k(recommended: List[Any], target: Any, k: int = 10):
    """
    Compute Precision@K, Recall@K, and F1@K for next-item prediction (single ground truth).
    """
    if not recommended or target is None:
        return 0.0, 0.0, 0.0

    hit = 1.0 if str(target) in [str(x) for x in recommended[:k]] else 0.0
    precision = hit / float(k)
    recall = hit / 1.0  # single ground truth target
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_per_category_metrics(predictions_with_meta: List[dict], catalog: dict):
    """
    Compute per-category accuracy, precision@10, recall@10, and F1@10.
    predictions_with_meta: List of {"target": pid, "recommended": [pids], "hit@10": 1/0}
    """
    category_stats = {}
    for p in predictions_with_meta:
        target_pid = p["target"]
        prod_info = catalog.get(target_pid) or catalog.get(str(target_pid)) or {}
        cat = prod_info.get("category", "Unknown")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits@1": 0, "hits@5": 0, "hits@10": 0, "f1_sum": 0.0}

        stats = category_stats[cat]
        stats["total"] += 1
        rec = p["recommended"]
        t_str = str(target_pid)
        rec_strs = [str(x) for x in rec]

        if t_str in rec_strs[:1]:
            stats["hits@1"] += 1
        if t_str in rec_strs[:5]:
            stats["hits@5"] += 1
        if t_str in rec_strs[:10]:
            stats["hits@10"] += 1
            p10, r10, f1_10 = precision_recall_f1_at_k(rec, target_pid, k=10)
            stats["f1_sum"] += f1_10

    results = {}
    for cat, stats in category_stats.items():
        total = stats["total"]
        if total > 0:
            results[cat] = {
                "sample_count": total,
                "hit@1": round(stats["hits@1"] / total, 4),
                "hit@5": round(stats["hits@5"] / total, 4),
                "hit@10": round(stats["hits@10"] / total, 4),
                "avg_f1@10": round(stats["f1_sum"] / total, 4),
            }
    return results
