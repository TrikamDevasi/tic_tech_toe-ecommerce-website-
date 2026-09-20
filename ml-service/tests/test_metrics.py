"""
Unit Tests for Ranking Metrics (Hit@K, NDCG@K, MRR, Coverage)
"""
import math
import pytest
from evaluation.metrics import hit_at_k, ndcg_at_k, mrr_at_k, compute_catalog_coverage


def test_hit_at_k():
    recs = ["p1", "p2", "p3", "p4", "p5"]

    assert hit_at_k(recs, target="p1", k=5) == 1.0
    assert hit_at_k(recs, target="p5", k=5) == 1.0
    assert hit_at_k(recs, target="p5", k=3) == 0.0
    assert hit_at_k(recs, target="unknown", k=5) == 0.0
    assert hit_at_k([], target="p1", k=5) == 0.0


def test_ndcg_at_k():
    recs = ["p1", "p2", "p3", "p4", "p5"]

    # Rank 1: 1 / log2(1 + 1) = 1.0
    assert ndcg_at_k(recs, target="p1", k=5) == 1.0

    # Rank 2: 1 / log2(2 + 1) = 1 / 1.58496 = 0.6309
    expected_rank2 = 1.0 / math.log2(3)
    assert pytest.approx(ndcg_at_k(recs, target="p2", k=5), 0.0001) == expected_rank2

    # Target outside top-k
    assert ndcg_at_k(recs, target="p5", k=3) == 0.0
    assert ndcg_at_k(recs, target="not_present", k=5) == 0.0


def test_mrr():
    recs = ["p1", "p2", "p3", "p4"]

    assert mrr_at_k(recs, target="p1") == 1.0
    assert mrr_at_k(recs, target="p2") == 0.5
    assert pytest.approx(mrr_at_k(recs, target="p3"), 0.001) == 1.0 / 3.0
    assert mrr_at_k(recs, target="missing") == 0.0


def test_catalog_coverage():
    all_recs = [
        ["p1", "p2"],
        ["p2", "p3"],
    ]
    catalog = {"p1", "p2", "p3", "p4", "p5"}

    # 3 out of 5 items recommended -> 0.6
    coverage = compute_catalog_coverage(all_recs, catalog)
    assert coverage == 0.6


def test_precision_recall_f1():
    from evaluation.metrics import precision_recall_f1_at_k
    recs = ["p1", "p2", "p3", "p4", "p5"]

    # Target in top 5, k=5 -> precision=1/5=0.2, recall=1.0, f1=2*0.2*1.0/1.2=0.3333
    p, r, f1 = precision_recall_f1_at_k(recs, target="p1", k=5)
    assert p == 0.2
    assert r == 1.0
    assert pytest.approx(f1, 0.001) == 1.0 / 3.0

    # Miss
    p_miss, r_miss, f1_miss = precision_recall_f1_at_k(recs, target="p99", k=5)
    assert p_miss == 0.0
    assert r_miss == 0.0
    assert f1_miss == 0.0


def test_per_category_metrics():
    from evaluation.metrics import compute_per_category_metrics
    preds = [
        {"target": 1, "recommended": [1, 2, 3], "hit@10": 1.0},
        {"target": 2, "recommended": [9, 8, 2], "hit@10": 1.0},
        {"target": 3, "recommended": [9, 8, 7], "hit@10": 0.0},
    ]
    catalog = {
        1: {"category": "Electronics"},
        2: {"category": "Electronics"},
        3: {"category": "Fashion"},
    }
    cat_metrics = compute_per_category_metrics(preds, catalog)
    assert "Electronics" in cat_metrics
    assert "Fashion" in cat_metrics
    assert cat_metrics["Electronics"]["hit@10"] == 1.0
    assert cat_metrics["Fashion"]["hit@10"] == 0.0
