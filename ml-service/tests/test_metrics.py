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
