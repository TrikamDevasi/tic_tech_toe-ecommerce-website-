"""
PriceIQ ML Service — Model Evaluation Pipeline

Evaluates GRU4Rec and three baseline recommenders (Popularity, Recently Viewed,
Content/TF-IDF) on the same held-out test split.

Metrics computed:
  - Hit@5, Hit@10, Hit@20
  - NDCG@5, NDCG@10, NDCG@20
  - MRR (Mean Reciprocal Rank)
  - Catalog Coverage
  - Inference Latency (ms/query)

Usage:
    python -m evaluation.evaluate
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import torch

import config
from database import get_db
from data.dataset import ItemVocabulary
from data.preprocessing import load_product_catalog
from models.gru4rec import GRU4Rec
from models.baselines import (
    PopularityRecommender,
    RecentlyViewedRecommender,
    ContentRecommender,
)
from evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    mrr_at_k,
    compute_catalog_coverage,
)


def predict_gru4rec(model, prefix_pids, vocab, device, top_k=20, exclude_history=True):
    """
    Run GRU4Rec inference for a single session prefix.
    Returns list of recommended product IDs (canonical ints/strings).
    """
    indices = [vocab.to_idx(pid) for pid in prefix_pids]
    # Truncate or pad to MAX_SEQ_LEN
    if len(indices) > config.MAX_SEQ_LEN:
        indices = indices[-config.MAX_SEQ_LEN:]
    pad_len = config.MAX_SEQ_LEN - len(indices)
    padded = [config.PAD_IDX] * pad_len + indices

    input_tensor = torch.tensor([padded], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)[0]  # shape: (vocab_size,)

        # Mask special tokens
        logits[config.PAD_IDX] = -float("inf")
        logits[config.UNK_IDX] = -float("inf")

        # Mask history items if exclude_history is True
        if exclude_history:
            for pid in prefix_pids:
                idx = vocab.to_idx(pid)
                if idx >= config.SPECIAL_TOKENS:
                    logits[idx] = -float("inf")

        k = min(top_k, vocab.num_items)
        top_indices = torch.topk(logits, k=k).indices.cpu().tolist()

    recommended_pids = [vocab.to_id(idx) for idx in top_indices if idx >= config.SPECIAL_TOKENS]
    return recommended_pids


def evaluate_models(
    test_sequences=None,
    train_sequences=None,
    vocab=None,
    model=None,
    catalog=None,
    device=None,
    output_path=None,
):
    """
    Evaluate GRU4Rec and all baselines on test_sequences.
    Returns dict with benchmark results and saves to experiment_results.json.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load artifacts if not provided in memory
    sequences_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    if test_sequences is None:
        test_path = os.path.join(sequences_dir, "test_sequences.json")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test sequences not found at {test_path}. Run training or build_dataset first."
            )
        with open(test_path, "r") as f:
            test_sequences = json.load(f)

    if train_sequences is None:
        train_path = os.path.join(sequences_dir, "train_sequences.json")
        if os.path.exists(train_path):
            with open(train_path, "r") as f:
                train_sequences = json.load(f)
        else:
            train_sequences = []

    if vocab is None:
        vocab = ItemVocabulary()
        vocab_path = os.path.join(config.MAPPINGS_DIR, "item_vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab not found at {vocab_path}.")
        vocab.load(vocab_path)

    if catalog is None:
        db = get_db()
        catalog = load_product_catalog(db)

    # Load GRU4Rec checkpoint if not provided
    if model is None:
        model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}.")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = GRU4Rec(vocab_size=len(vocab)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Initialize and fit Baselines
    print("\n📦 Fitting baseline models on training data...")
    popularity_model = PopularityRecommender()
    if train_sequences:
        popularity_model.train(train_sequences)

    recently_viewed_model = RecentlyViewedRecommender()

    content_model = ContentRecommender()
    if catalog:
        content_model.train_from_catalog(catalog)

    all_catalog_pids = set(catalog.keys()) if catalog else set(vocab.all_product_ids())

    # 3. Benchmark Models Definition
    models_dict = {
        "GRU4Rec": {
            "fn": lambda hist: predict_gru4rec(model, hist, vocab, device, top_k=20),
            "type": "Neural (Session-based RNN)",
        },
        "Popularity": {
            "fn": lambda hist: [pid for pid, _ in popularity_model.recommend(hist, top_k=20)],
            "type": "Non-personalized Baseline",
        },
        "RecentlyViewed": {
            "fn": lambda hist: [pid for pid, _ in recently_viewed_model.recommend(hist, top_k=20)],
            "type": "Heuristic Baseline",
        },
        "Content_TFIDF": {
            "fn": lambda hist: [pid for pid, _ in content_model.recommend(hist, top_k=20)],
            "type": "Content-based Cosine Baseline",
        },
    }

    print(f"\n🚀 Evaluating {len(models_dict)} models on {len(test_sequences)} test sessions...")

    # 4. Evaluation Loop
    results = {
        name: {
            "type": info["type"],
            "hit@5": [],
            "hit@10": [],
            "hit@20": [],
            "ndcg@5": [],
            "ndcg@10": [],
            "ndcg@20": [],
            "mrr": [],
            "latencies_ms": [],
            "all_recommendations": [],
        }
        for name, info in models_dict.items()
    }

    evaluated_count = 0
    for seq in test_sequences:
        pids = seq.get("product_ids", [])
        if len(pids) < 2:
            continue

        prefix = pids[:-1]
        target = pids[-1]
        evaluated_count += 1

        for model_name, info in models_dict.items():
            t0 = time.perf_counter()
            try:
                recommended = info["fn"](prefix)
            except Exception as e:
                recommended = []
            latency_ms = (time.perf_counter() - t0) * 1000.0

            m_dict = results[model_name]
            m_dict["latencies_ms"].append(latency_ms)
            m_dict["all_recommendations"].append(recommended)

            # Metrics
            m_dict["hit@5"].append(hit_at_k(recommended, target, k=5))
            m_dict["hit@10"].append(hit_at_k(recommended, target, k=10))
            m_dict["hit@20"].append(hit_at_k(recommended, target, k=20))
            m_dict["ndcg@5"].append(ndcg_at_k(recommended, target, k=5))
            m_dict["ndcg@10"].append(ndcg_at_k(recommended, target, k=10))
            m_dict["ndcg@20"].append(ndcg_at_k(recommended, target, k=20))
            m_dict["mrr"].append(mrr_at_k(recommended, target))

    # 5. Aggregate metrics
    summary = {
        "test_samples_count": evaluated_count,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }

    for model_name, m_dict in results.items():
        n = max(len(m_dict["hit@10"]), 1)
        coverage = compute_catalog_coverage(m_dict["all_recommendations"], all_catalog_pids)
        summary["models"][model_name] = {
            "type": m_dict["type"],
            "hit@5": round(float(np.mean(m_dict["hit@5"])), 4),
            "hit@10": round(float(np.mean(m_dict["hit@10"])), 4),
            "hit@20": round(float(np.mean(m_dict["hit@20"])), 4),
            "ndcg@5": round(float(np.mean(m_dict["ndcg@5"])), 4),
            "ndcg@10": round(float(np.mean(m_dict["ndcg@10"])), 4),
            "ndcg@20": round(float(np.mean(m_dict["ndcg@20"])), 4),
            "mrr": round(float(np.mean(m_dict["mrr"])), 4),
            "catalog_coverage": round(coverage, 4),
            "avg_latency_ms": round(float(np.mean(m_dict["latencies_ms"])), 2),
            "p95_latency_ms": round(float(np.percentile(m_dict["latencies_ms"], 95)), 2),
        }

    # 6. Display Formatted Benchmark Table
    header = f"{'Model':<16} | {'Type':<28} | {'Hit@5':<7} | {'Hit@10':<7} | {'NDCG@10':<8} | {'MRR':<7} | {'Coverage':<9} | {'Latency':<8}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(f"🏆 BENCHMARK RESULTS (Evaluated on {evaluated_count} held-out test sequences)")
    print(divider)
    print(header)
    print(divider)

    for name, stats in summary["models"].items():
        print(
            f"{name:<16} | {stats['type']:<28} | {stats['hit@5']:<7.4f} | {stats['hit@10']:<7.4f} | "
            f"{stats['ndcg@10']:<8.4f} | {stats['mrr']:<7.4f} | {stats['catalog_coverage']:<9.4f} | {stats['avg_latency_ms']:<5.1f}ms"
        )
    print(divider + "\n")

    # 7. Save to experiment_results.json
    if output_path is None:
        output_path = os.path.join(config.ARTIFACTS_DIR, "experiment_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Benchmark results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    evaluate_models()
