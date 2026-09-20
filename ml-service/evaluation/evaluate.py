"""
PriceIQ ML Service — Model Evaluation Pipeline

Evaluates GRU4Rec and three baseline recommenders (Popularity, Recently Viewed,
Content/TF-IDF) on the same held-out test split.

Metrics computed:
  - Hit@1, Hit@5, Hit@10, Hit@20
  - NDCG@5, NDCG@10, NDCG@20
  - MRR (Mean Reciprocal Rank)
  - Precision@10, Recall@10, F1@10
  - Catalog Coverage
  - Per-Category Performance Breakdown (weakest vs strongest classes)
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
    precision_recall_f1_at_k,
    compute_per_category_metrics,
)


def predict_gru4rec(model, prefix_pids, vocab, device, top_k=20, exclude_history=True):
    """
    Run GRU4Rec inference for a single session prefix.
    Returns list of recommended product IDs (canonical ints/strings).
    """
    indices = [vocab.to_idx(pid) for pid in prefix_pids]
    if len(indices) > config.MAX_SEQ_LEN:
        indices = indices[-config.MAX_SEQ_LEN:]
    pad_len = config.MAX_SEQ_LEN - len(indices)
    padded = [config.PAD_IDX] * pad_len + indices

    input_tensor = torch.tensor([padded], dtype=torch.long, device=device)

    exclude_set = set(indices) if exclude_history else set()

    conf_res = model.predict_with_confidence(
        input_tensor,
        top_k=top_k,
        exclude_indices=exclude_set,
        temperature=1.0,
    )
    top_indices = conf_res["top_indices"]
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
                f"Test sequences not found at {test_path}. Run python -m data.build_dataset first."
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
        vocab = ItemVocabulary.from_file()

    if catalog is None:
        catalog = load_product_catalog()

    all_catalog_pids = set(catalog.keys()) if catalog else set(vocab.all_product_ids())

    # 2. Load model if not passed
    if model is None:
        model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. Run python -m training.train first."
            )
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = GRU4Rec(
            vocab_size=len(vocab),
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_GRU_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Train Baselines on Training Sequences
    print("\n📦 Initializing Baselines for Comparative Benchmark...")
    pop_model = PopularityRecommender()
    pop_model.train(train_sequences)

    recent_model = RecentlyViewedRecommender()

    content_model = ContentRecommender()
    if catalog:
        content_model.train_from_catalog(catalog)

    models_dict = {
        "GRU4Rec": {
            "fn": lambda prefix: predict_gru4rec(model, prefix, vocab, device, top_k=20),
            "type": "Neural (Session-based RNN)",
        },
        "Popularity": {
            "fn": lambda prefix: [pid for pid, _ in pop_model.recommend(prefix, top_k=20)],
            "type": "Non-personalized Baseline",
        },
        "RecentlyViewed": {
            "fn": lambda prefix: [pid for pid, _ in recent_model.recommend(prefix, top_k=20)],
            "type": "Heuristic Baseline",
        },
        "Content_TFIDF": {
            "fn": lambda prefix: [pid for pid, _ in content_model.recommend(prefix, top_k=20)],
            "type": "Content-based Cosine Baseline",
        },
    }

    # 4. Evaluation Loop
    print(f"\n🧪 Evaluating on {len(test_sequences)} test sessions...")

    results = {
        name: {
            "type": info["type"],
            "hit@1": [],
            "hit@5": [],
            "hit@10": [],
            "hit@20": [],
            "ndcg@5": [],
            "ndcg@10": [],
            "ndcg@20": [],
            "mrr": [],
            "precision@10": [],
            "recall@10": [],
            "f1@10": [],
            "latencies_ms": [],
            "all_recommendations": [],
            "pred_meta": [],
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
            except Exception:
                recommended = []
            latency_ms = (time.perf_counter() - t0) * 1000.0

            m_dict = results[model_name]
            m_dict["latencies_ms"].append(latency_ms)
            m_dict["all_recommendations"].append(recommended)

            # Ranking Metrics
            h1 = hit_at_k(recommended, target, k=1)
            h5 = hit_at_k(recommended, target, k=5)
            h10 = hit_at_k(recommended, target, k=10)
            h20 = hit_at_k(recommended, target, k=20)
            n5 = ndcg_at_k(recommended, target, k=5)
            n10 = ndcg_at_k(recommended, target, k=10)
            n20 = ndcg_at_k(recommended, target, k=20)
            mrr = mrr_at_k(recommended, target)
            p10, r10, f10 = precision_recall_f1_at_k(recommended, target, k=10)

            m_dict["hit@1"].append(h1)
            m_dict["hit@5"].append(h5)
            m_dict["hit@10"].append(h10)
            m_dict["hit@20"].append(h20)
            m_dict["ndcg@5"].append(n5)
            m_dict["ndcg@10"].append(n10)
            m_dict["ndcg@20"].append(n20)
            m_dict["mrr"].append(mrr)
            m_dict["precision@10"].append(p10)
            m_dict["recall@10"].append(r10)
            m_dict["f1@10"].append(f10)
            m_dict["pred_meta"].append({"target": target, "recommended": recommended, "hit@10": h10})

    # 5. Aggregate metrics
    summary = {
        "test_samples_count": evaluated_count,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }

    for model_name, m_dict in results.items():
        coverage = compute_catalog_coverage(m_dict["all_recommendations"], all_catalog_pids)
        summary["models"][model_name] = {
            "type": m_dict["type"],
            "hit@1": round(float(np.mean(m_dict["hit@1"])), 4),
            "hit@5": round(float(np.mean(m_dict["hit@5"])), 4),
            "hit@10": round(float(np.mean(m_dict["hit@10"])), 4),
            "hit@20": round(float(np.mean(m_dict["hit@20"])), 4),
            "ndcg@5": round(float(np.mean(m_dict["ndcg@5"])), 4),
            "ndcg@10": round(float(np.mean(m_dict["ndcg@10"])), 4),
            "ndcg@20": round(float(np.mean(m_dict["ndcg@20"])), 4),
            "mrr": round(float(np.mean(m_dict["mrr"])), 4),
            "precision@10": round(float(np.mean(m_dict["precision@10"])), 4),
            "recall@10": round(float(np.mean(m_dict["recall@10"])), 4),
            "f1@10": round(float(np.mean(m_dict["f1@10"])), 4),
            "catalog_coverage": round(coverage, 4),
            "avg_latency_ms": round(float(np.mean(m_dict["latencies_ms"])), 2),
            "p95_latency_ms": round(float(np.percentile(m_dict["latencies_ms"], 95)), 2),
        }

    # Per-category evaluation for GRU4Rec
    gru_category_metrics = compute_per_category_metrics(results["GRU4Rec"]["pred_meta"], catalog)
    summary["category_breakdown"] = gru_category_metrics

    # Sort categories by F1 to find strongest and weakest
    sorted_cats = sorted(gru_category_metrics.items(), key=lambda x: x[1]["avg_f1@10"], reverse=True)
    summary["strongest_classes"] = [c[0] for c in sorted_cats[:3]]
    summary["weakest_classes"] = [c[0] for c in sorted_cats[-3:]]

    # Top-level keys for direct backward-compatibility with backend dashboard endpoints
    gru_stats = summary["models"]["GRU4Rec"]
    summary["ndcg_at_10"] = gru_stats["ndcg@10"]
    summary["hit_rate"] = gru_stats["hit@10"]
    summary["hit_rate_at_10"] = gru_stats["hit@10"]
    summary["sample_size"] = evaluated_count
    summary["evaluated_at"] = summary["evaluation_timestamp"]

    # 6. Display Formatted Benchmark Table
    header = f"{'Model':<16} | {'Type':<28} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'NDCG@10':<8} | {'MRR':<7} | {'F1@10':<7} | {'Coverage':<9} | {'Latency':<8}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(f"🏆 BENCHMARK RESULTS (Evaluated on {evaluated_count} held-out test sequences)")
    print(divider)
    print(header)
    print(divider)

    for name, stats in summary["models"].items():
        print(
            f"{name:<16} | {stats['type']:<28} | {stats['hit@1']:<7.4f} | {stats['hit@5']:<7.4f} | {stats['hit@10']:<7.4f} | "
            f"{stats['ndcg@10']:<8.4f} | {stats['mrr']:<7.4f} | {stats['f1@10']:<7.4f} | {stats['catalog_coverage']:<9.4f} | {stats['avg_latency_ms']:<5.1f}ms"
        )
    print(divider)

    print("\n📊 Per-Category Performance (GRU4Rec):")
    for cat, c_metrics in sorted(gru_category_metrics.items(), key=lambda x: -x[1]["hit@10"]):
        print(f"  • {cat:<18}: Hit@10={c_metrics['hit@10']*100:.1f}%, Hit@5={c_metrics['hit@5']*100:.1f}%, F1={c_metrics['avg_f1@10']:.4f} (N={c_metrics['sample_count']})")
    print(f"\n  Strongest Categories: {', '.join(summary['strongest_classes'])}")
    print(f"  Weakest Categories:   {', '.join(summary['weakest_classes'])}\n")

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
