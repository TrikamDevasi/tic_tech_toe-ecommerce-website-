"""
PriceIQ ML Service — Production Cold-Start Behavior Test

Exercises the REAL production recommender (inference.recommender.HybridRecommender)
exactly as the API does, for: empty session, 1 item, 2 items, and a normal
(5+) session. Verifies strategy, catalog consistency vs MongoDB, and calibration.

Requires MongoDB (product catalog) to be reachable — same as production.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from inference.recommender import HybridRecommender
from database import get_db


def main():
    db = get_db()
    products = list(db["products"].find({}, {"_id": 0}).limit(12))
    pids = sorted(p["id"] for p in products)
    print(f"MongoDB: {db['products'].count_documents({})} products, sample pids: {pids[:5]}...")

    rec = HybridRecommender()
    rec.load()
    print(f"Loaded catalog size={len(rec.catalog)} vocab={len(rec.vocab)} ready={rec.is_ready}")

    test_cases = {
        "empty_session": [],
        "one_item": [pids[0]],
        "two_items": [pids[0], pids[3]],
        "three_items": [pids[0], pids[3], pids[7]],
        "normal_session": [pids[0], pids[3], pids[7], pids[2], pids[9]],
    }

    results = {}
    catalog_pids = set(rec.catalog.keys())
    # Verify against MongoDB directly (canonical product IDs, not mock)
    mongo_pid_set = set(p["id"] for p in db["products"].find({}, {"_id": 0, "id": 1}))

    for name, hist in test_cases.items():
        out = rec.recommend_session(session_history=hist, top_k=5)
        results[name] = out
        rec_pids = [it["productId"] for it in out["recommendations"]]
        in_catalog = [pid for pid in rec_pids if pid in catalog_pids]
        in_mongo = [pid for pid in rec_pids if pid in mongo_pid_set]
        unique = len(set(rec_pids)) == len(rec_pids)
        print(f"\n[{name}] session_len={out['session_length']} strategy={out['strategy']} "
              f"latency={out['latency_ms']}ms conf_pct={out['confidence_pct']} low_conf={out['is_low_confidence']}")
        print(f"  msg: {out['confidence_message']}")
        print(f"  recs={rec_pids}")
        print(f"  checks: unique={unique} all_in_catalog={len(in_catalog)==len(rec_pids)} "
              f"all_in_mongo={len(in_mongo)==len(rec_pids)}")

    # Aggregate assertions
    assert results["empty_session"]["strategy"] == "popularity_cold_start"
    assert results["one_item"]["strategy"] == "content_tfidf"
    assert results["two_items"]["strategy"] == "content_tfidf"
    assert results["three_items"]["strategy"] == "gru4rec_neural"
    assert results["normal_session"]["strategy"] == "gru4rec_neural"
    for name, out in results.items():
        assert len(out["recommendations"]) == 5, name
        assert len({it["productId"] for it in out["recommendations"]}) == 5, name
        assert all(it["productId"] in catalog_pids for it in out["recommendations"]), name
        assert all(it["productId"] in mongo_pid_set for it in out["recommendations"]), name
        assert 0.0 <= out["confidence"] <= 1.0, name

    # Save for the report
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "artifacts", "coldstart_prod_test.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✅ Cold-start production test PASSED — all strategies, catalog/Mongo ID consistency and "
          "uniqueness verified.")


if __name__ == "__main__":
    main()