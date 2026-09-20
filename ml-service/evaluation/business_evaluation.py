"""
PriceIQ ML Service — Business & Operational Evaluation
Evaluates production-readiness business metrics:
  - Catalog Coverage (fraction of 215 items recommended across all test sessions)
  - Intra-list Diversity (fraction of distinct items and categories per recommendation list)
  - Duplicate Recommendation Rate (intra-list duplicates)
  - Category Diversity (average distinct categories per top-10 list)
  - Price Disparity / Consistency (price delta between session history and recommendations)
  - Inference Latency (mean, p50, p95, p99 ms)
  - Fallback Rate (cold start and confidence fallbacks)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np

import config
from inference.recommender import HybridRecommender


def evaluate_business_metrics(test_sequences, recommender=None, top_k=10):
    if recommender is None:
        recommender = HybridRecommender()
        recommender.load()

    total_sessions = len(test_sequences)
    all_recommended_items = set()
    catalog_items = set(recommender.catalog.keys())
    total_catalog_size = len(catalog_items) if catalog_items else 215

    intra_list_diversities = []
    category_diversities = []
    duplicate_rates = []
    price_deltas = []
    latencies = []
    fallback_count = 0
    strategy_counts = {}

    for seq in test_sequences:
        pids = seq.get("product_ids", [])
        clean_history = [p for p in pids[:-1] if p is not None] if len(pids) >= 2 else pids
        
        t0 = time.perf_counter()
        rec_res = recommender.recommend_session(clean_history, top_k=top_k, exclude_history=True)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        strat = rec_res.get("strategy", "unknown")
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        if rec_res.get("is_fallback", False) or "fallback" in strat or "cold_start" in strat:
            fallback_count += 1

        recs = rec_res.get("recommendations", [])
        rec_pids = [r["productId"] for r in recs]
        all_recommended_items.update(rec_pids)

        # Duplicate rate within list
        if rec_pids:
            unique_in_list = len(set(rec_pids))
            dup_rate = 1.0 - (unique_in_list / len(rec_pids))
            duplicate_rates.append(dup_rate)
        else:
            duplicate_rates.append(0.0)

        # Intra-list category diversity
        rec_cats = [r.get("category", "Unknown") for r in recs]
        unique_cats = len(set(rec_cats))
        cat_diversity = unique_cats / len(rec_cats) if rec_cats else 0.0
        intra_list_diversities.append(cat_diversity)
        category_diversities.append(unique_cats)

        # Price disparity: avg price of history vs avg price of recs
        hist_prices = [
            float(recommender.catalog.get(pid, {}).get("price", 0))
            for pid in clean_history
            if pid in recommender.catalog
        ]
        rec_prices = [float(r.get("price", 0)) for r in recs]

        if hist_prices and rec_prices:
            avg_hist_price = float(np.mean(hist_prices))
            avg_rec_price = float(np.mean(rec_prices))
            price_deltas.append(abs(avg_rec_price - avg_hist_price))

    # Aggregate metrics
    catalog_coverage_pct = round((len(all_recommended_items) / total_catalog_size) * 100, 2)
    mean_category_diversity = round(float(np.mean(category_diversities)), 2)
    mean_intra_list_diversity = round(float(np.mean(intra_list_diversities)) * 100, 2)
    mean_duplicate_rate = round(float(np.mean(duplicate_rates)) * 100, 2)
    mean_price_delta = round(float(np.mean(price_deltas)), 2) if price_deltas else 0.0
    fallback_rate = round((fallback_count / total_sessions) * 100, 2) if total_sessions else 0.0

    latency_p50 = round(float(np.percentile(latencies, 50)), 2)
    latency_p95 = round(float(np.percentile(latencies, 95)), 2)
    latency_p99 = round(float(np.percentile(latencies, 99)), 2)
    latency_mean = round(float(np.mean(latencies)), 2)

    results = {
        "total_test_sessions_evaluated": total_sessions,
        "total_catalog_size": total_catalog_size,
        "unique_items_recommended": len(all_recommended_items),
        "catalog_coverage_pct": catalog_coverage_pct,
        "mean_distinct_categories_per_list": mean_category_diversity,
        "intra_list_category_diversity_pct": mean_intra_list_diversity,
        "duplicate_recommendation_rate_pct": mean_duplicate_rate,
        "average_price_delta_inr": mean_price_delta,
        "fallback_rate_pct": fallback_rate,
        "strategy_distribution": strategy_counts,
        "latency_metrics_ms": {
            "mean": latency_mean,
            "p50": latency_p50,
            "p95": latency_p95,
            "p99": latency_p99,
        },
    }

    out_path = os.path.join(config.ARTIFACTS_DIR, "business_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("  PRICEIQ BUSINESS & OPERATIONAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Catalog Coverage:             {catalog_coverage_pct}% ({len(all_recommended_items)}/{total_catalog_size} items)")
    print(f"  Distinct Categories / Top-10: {mean_category_diversity} categories")
    print(f"  Intra-list Diversity:         {mean_intra_list_diversity}%")
    print(f"  Duplicate Recommendation Rate:{mean_duplicate_rate}%")
    print(f"  Average Price Disparity:      ₹{mean_price_delta:,.2f}")
    print(f"  Fallback Rate:                {fallback_rate}%")
    print(f"  Latency (Mean / p50 / p95):   {latency_mean}ms / {latency_p50}ms / {latency_p95}ms")
    print("=" * 65)

    return results


if __name__ == "__main__":
    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    with open(os.path.join(data_dir, "test_sequences.json"), "r") as f:
        test_seqs = json.load(f)
    evaluate_business_metrics(test_seqs)
