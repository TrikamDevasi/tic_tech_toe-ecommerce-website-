"""
PriceIQ ML Service — Multi-Model Offline Evaluation Pipeline

Evaluates 6 recommendation systems on the same held-out test split (449 sessions):
  1. Popularity (Non-personalized baseline)
  2. Recently Viewed (Recency heuristic)
  3. Content TF-IDF (Cosine similarity on catalog descriptions)
  4. Item Transition (First-order Markov Chain)
  5. GRU4Rec (Trained Recurrent Neural Network)
  6. Hybrid Recommender (GRU4Rec + TF-IDF + Popularity)

Evaluates both operational settings:
  A. Standard Next-Item Prediction (Session completion, allowing repeat examination)
  B. Discovery Mode (Strictly predicting unviewed catalog items)

Computes:
  - Hit@1, Hit@5, Hit@10, Hit@20
  - NDCG@5, NDCG@10, NDCG@20
  - MRR (Mean Reciprocal Rank)
  - Precision@10, Recall@10, F1@10
  - Catalog Coverage
  - Inference Latency (avg, p50, p95)
  - 95% Bootstrap Confidence Intervals (1,000 resamples)
  - Category Breakdown (strongest vs weakest classes)
  - Session Length Slices (Short: L<=3 vs Long: L>3)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import torch

import config
from data.dataset import ItemVocabulary
from data.preprocessing import load_product_catalog
from models.gru4rec import GRU4Rec
from models.baselines import (
    PopularityRecommender,
    RecentlyViewedRecommender,
    ContentRecommender,
    ItemTransitionRecommender,
)
from models.hybrid import HybridRecommender
from evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    mrr_at_k,
    compute_catalog_coverage,
    precision_recall_f1_at_k,
    compute_per_category_metrics,
)


def compute_bootstrap_ci(values, n_bootstraps=1000, alpha=0.05, seed=42):
    """Compute 95% bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    n = len(arr)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    bootstrap_means = np.mean(arr[indices], axis=1)
    lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha / 2.0)))
    mean = float(np.mean(arr))
    return round(mean, 4), round(lower, 4), round(upper, 4)


def evaluate_mode(models_dict, test_sequences, all_catalog_pids, catalog, exclude_history=False):
    """Evaluate a dictionary of models on test_sequences for a specific history exclusion mode."""
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
            "f1@10": [],
            "latencies_ms": [],
            "all_recs": [],
            "pred_meta": [],
            "short_hit10": [],
            "long_hit10": [],
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
        is_short = len(prefix) <= 3
        evaluated_count += 1

        for model_name, info in models_dict.items():
            t0 = time.perf_counter()
            try:
                recommended = info["fn"](prefix, exclude_history)
            except Exception:
                recommended = []
            lat_ms = (time.perf_counter() - t0) * 1000.0

            m = results[model_name]
            m["latencies_ms"].append(lat_ms)
            m["all_recs"].append(recommended)

            h1 = hit_at_k(recommended, target, k=1)
            h5 = hit_at_k(recommended, target, k=5)
            h10 = hit_at_k(recommended, target, k=10)
            h20 = hit_at_k(recommended, target, k=20)
            n5 = ndcg_at_k(recommended, target, k=5)
            n10 = ndcg_at_k(recommended, target, k=10)
            n20 = ndcg_at_k(recommended, target, k=20)
            mrr = mrr_at_k(recommended, target)
            _, _, f10 = precision_recall_f1_at_k(recommended, target, k=10)

            m["hit@1"].append(h1)
            m["hit@5"].append(h5)
            m["hit@10"].append(h10)
            m["hit@20"].append(h20)
            m["ndcg@5"].append(n5)
            m["ndcg@10"].append(n10)
            m["ndcg@20"].append(n20)
            m["mrr"].append(mrr)
            m["f1@10"].append(f10)
            m["pred_meta"].append({"target": target, "recommended": recommended, "hit@10": h10})

            if is_short:
                m["short_hit10"].append(h10)
            else:
                m["long_hit10"].append(h10)

    summary_models = {}
    bootstrap_cis = {}

    for name, m in results.items():
        cov = compute_catalog_coverage(m["all_recs"], all_catalog_pids)
        h10_mean, h10_low, h10_high = compute_bootstrap_ci(m["hit@10"])
        ndcg_mean, ndcg_low, ndcg_high = compute_bootstrap_ci(m["ndcg@10"])

        summary_models[name] = {
            "type": m["type"],
            "hit@1": round(float(np.mean(m["hit@1"])), 4),
            "hit@5": round(float(np.mean(m["hit@5"])), 4),
            "hit@10": round(float(np.mean(m["hit@10"])), 4),
            "hit@20": round(float(np.mean(m["hit@20"])), 4),
            "ndcg@5": round(float(np.mean(m["ndcg@5"])), 4),
            "ndcg@10": round(float(np.mean(m["ndcg@10"])), 4),
            "ndcg@20": round(float(np.mean(m["ndcg@20"])), 4),
            "mrr": round(float(np.mean(m["mrr"])), 4),
            "f1@10": round(float(np.mean(m["f1@10"])), 4),
            "catalog_coverage": round(cov, 4),
            "short_session_hit@10": round(float(np.mean(m["short_hit10"])), 4) if m["short_hit10"] else 0.0,
            "long_session_hit@10": round(float(np.mean(m["long_hit10"])), 4) if m["long_hit10"] else 0.0,
            "avg_latency_ms": round(float(np.mean(m["latencies_ms"])), 2),
            "p50_latency_ms": round(float(np.percentile(m["latencies_ms"], 50)), 2),
            "p95_latency_ms": round(float(np.percentile(m["latencies_ms"], 95)), 2),
        }
        bootstrap_cis[name] = {
            "hit@10": {"mean": h10_mean, "ci_95": [h10_low, h10_high]},
            "ndcg@10": {"mean": ndcg_mean, "ci_95": [ndcg_low, ndcg_high]},
        }

    # Per-category evaluation for GRU4Rec
    category_metrics = compute_per_category_metrics(results["GRU4Rec"]["pred_meta"], catalog)

    return summary_models, bootstrap_cis, category_metrics, evaluated_count


def evaluate_all_models(output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    with open(os.path.join(data_dir, "test_sequences.json"), "r") as f:
        test_sequences = json.load(f)
    with open(os.path.join(data_dir, "train_sequences.json"), "r") as f:
        train_sequences = json.load(f)

    vocab = ItemVocabulary.from_file()
    catalog = load_product_catalog()
    all_catalog_pids = set(catalog.keys()) if catalog else set(vocab.all_product_ids())

    # 1. Load GRU4Rec Best Model
    model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    m_config = checkpoint.get("model_config", {
        "vocab_size": len(vocab),
        "embedding_dim": config.EMBEDDING_DIM,
        "hidden_dim": config.HIDDEN_DIM,
        "num_layers": config.NUM_GRU_LAYERS,
        "dropout": config.DROPOUT,
    })

    gru_model = GRU4Rec(
        vocab_size=m_config["vocab_size"],
        embedding_dim=m_config["embedding_dim"],
        hidden_dim=m_config["hidden_dim"],
        num_layers=m_config["num_layers"],
        dropout=m_config["dropout"],
    ).to(device)
    gru_model.load_state_dict(checkpoint["model_state_dict"])
    gru_model.eval()

    # 2. Baselines
    pop_model = PopularityRecommender()
    pop_model.train(train_sequences)

    recent_model = RecentlyViewedRecommender()

    content_model = ContentRecommender()
    content_model.train_from_catalog(catalog)

    markov_model = ItemTransitionRecommender()
    markov_model.train(train_sequences)

    hybrid_model = HybridRecommender(
        gru_model=gru_model,
        content_model=content_model,
        pop_model=pop_model,
        vocab=vocab,
        alpha=0.60,
        beta=0.30,
        gamma=0.10,
        device=device,
    )

    # Function mappings
    def predict_gru(prefix, exclude_history):
        indices = [vocab.to_idx(pid) for pid in prefix]
        if len(indices) > config.MAX_SEQ_LEN:
            indices = indices[-config.MAX_SEQ_LEN:]
        pad_len = config.MAX_SEQ_LEN - len(indices)
        padded = [config.PAD_IDX] * pad_len + indices
        inp = torch.tensor([padded], dtype=torch.long, device=device)
        excl_set = set(indices) if exclude_history else set()
        res = gru_model.predict_with_confidence(inp, top_k=20, exclude_indices=excl_set)
        return [vocab.to_id(idx) for idx in res["top_indices"] if idx >= config.SPECIAL_TOKENS]

    def predict_hybrid(prefix, exclude_history):
        recs = hybrid_model.recommend(prefix, top_k=20, exclude_history=exclude_history)
        return [pid for pid, _ in recs]

    models_dict = {
        "Popularity": {
            "fn": lambda prefix, excl: [pid for pid, _ in pop_model.recommend(prefix if excl else [], top_k=20)],
            "type": "Non-personalized Baseline",
        },
        "RecentlyViewed": {
            "fn": lambda prefix, excl: [pid for pid, _ in recent_model.recommend(prefix, top_k=20)],
            "type": "Heuristic Recency Baseline",
        },
        "Content_TFIDF": {
            "fn": lambda prefix, excl: [pid for pid, _ in content_model.recommend(prefix, top_k=20)],
            "type": "Content Cosine Similarity",
        },
        "ItemTransition_Markov": {
            "fn": lambda prefix, excl: [pid for pid, _ in markov_model.recommend(prefix, top_k=20)],
            "type": "Markov Chain (Item-to-Item)",
        },
        "GRU4Rec": {
            "fn": predict_gru,
            "type": f"Neural ({m_config['num_layers']}-layer GRU, hid={m_config['hidden_dim']})",
        },
        "Hybrid": {
            "fn": predict_hybrid,
            "type": "Hybrid (GRU4Rec + TFIDF + Pop)",
        },
    }

    # Evaluate Mode A: Standard Next-Item Prediction
    print("\n==============================================================================================================")
    print("  MODE A: STANDARD NEXT-ITEM PREDICTION (Session Completion, Full Catalog)")
    print("==============================================================================================================")
    std_models, std_cis, cat_metrics, n_samples = evaluate_mode(
        models_dict, test_sequences, all_catalog_pids, catalog, exclude_history=False
    )

    print(f"{'Model':<24} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'Hit@20':<7} | {'NDCG@10':<8} | {'MRR':<7} | {'Coverage':<9} | {'Latency':<8}")
    print("-" * 110)
    for name, st in std_models.items():
        print(
            f"{name:<24} | "
            f"{st['hit@1']*100:5.2f}% | "
            f"{st['hit@5']*100:5.2f}% | "
            f"{st['hit@10']*100:5.2f}% | "
            f"{st['hit@20']*100:5.2f}% | "
            f"{st['ndcg@10']:7.4f} | "
            f"{st['mrr']:6.4f} | "
            f"{st['catalog_coverage']*100:7.1f}% | "
            f"{st['avg_latency_ms']:5.2f} ms"
        )

    # Evaluate Mode B: Discovery Mode (Pure Unseen Items)
    print("\n==============================================================================================================")
    print("  MODE B: DISCOVERY MODE (Strictly Predicting Unseen Items in History)")
    print("==============================================================================================================")
    disc_models, disc_cis, _, _ = evaluate_mode(
        models_dict, test_sequences, all_catalog_pids, catalog, exclude_history=True
    )

    print(f"{'Model':<24} | {'Hit@1':<7} | {'Hit@5':<7} | {'Hit@10':<7} | {'Hit@20':<7} | {'NDCG@10':<8} | {'MRR':<7} | {'Coverage':<9} | {'Latency':<8}")
    print("-" * 110)
    for name, st in disc_models.items():
        print(
            f"{name:<24} | "
            f"{st['hit@1']*100:5.2f}% | "
            f"{st['hit@5']*100:5.2f}% | "
            f"{st['hit@10']*100:5.2f}% | "
            f"{st['hit@20']*100:5.2f}% | "
            f"{st['ndcg@10']:7.4f} | "
            f"{st['mrr']:6.4f} | "
            f"{st['catalog_coverage']*100:7.1f}% | "
            f"{st['avg_latency_ms']:5.2f} ms"
        )

    # Category strengths
    sorted_cats = sorted(cat_metrics.items(), key=lambda x: x[1]["avg_f1@10"], reverse=True)

    summary = {
        "test_samples_count": n_samples,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "standard_evaluation": {
            "models": std_models,
            "bootstrap_cis": std_cis,
        },
        "discovery_evaluation": {
            "models": disc_models,
            "bootstrap_cis": disc_cis,
        },
        "models": std_models,  # backward compatibility for dashboard
        "category_breakdown": cat_metrics,
        "strongest_classes": [c[0] for c in sorted_cats[:3]],
        "weakest_classes": [c[0] for c in sorted_cats[-3:]],
        "ndcg_at_10": std_models["GRU4Rec"]["ndcg@10"],
        "hit_rate": std_models["GRU4Rec"]["hit@10"],
        "hit_rate_at_10": std_models["GRU4Rec"]["hit@10"],
        "sample_size": n_samples,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if output_path is None:
        output_path = os.path.join(config.ARTIFACTS_DIR, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Consolidated benchmark saved to {output_path}")
    return summary


def evaluate_models(test_sequences=None, vocab=None, model=None, output_path=None):
    """
    Evaluation entrypoint used by the FastAPI application (/evaluate, /train).

    - When a freshly trained model + vocab + test_sequences are supplied (API /train),
      benchmarks ONLY that model on the provided split without touching artifact baselines.
    - Otherwise runs the full multi-model benchmark on the held-out test split.
      It never writes when passed explicit model objects; evaluate_all_models(output_path=...)
      is used for a persisted full benchmark.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        # Full multi-model benchmark on the held-out test split (API /evaluate).
        return evaluate_all_models(output_path=output_path)

    if test_sequences is None:
        with open(os.path.join(config.ARTIFACTS_DIR, "data", "test_sequences.json"), "r") as f:
            test_sequences = json.load(f)
    if vocab is None:
        vocab = ItemVocabulary.from_file()
    catalog = load_product_catalog()
    all_catalog_pids = set(catalog.keys()) if catalog else set(vocab.all_product_ids())

    def predict_gru(prefix, exclude_history):
        indices = [vocab.to_idx(pid) for pid in prefix]
        if len(indices) > config.MAX_SEQ_LEN:
            indices = indices[-config.MAX_SEQ_LEN:]
        pad_len = config.MAX_SEQ_LEN - len(indices)
        padded = [config.PAD_IDX] * pad_len + indices
        inp = torch.tensor([padded], dtype=torch.long, device=device)
        excl_set = set(indices) if exclude_history else set()
        res = model.predict_with_confidence(inp, top_k=20, exclude_indices=excl_set)
        return [vocab.to_id(idx) for idx in res["top_indices"] if idx >= config.SPECIAL_TOKENS]

    models_dict = {
        "GRU4Rec": {
            "fn": predict_gru,
            "type": f"Neural ({getattr(model, 'num_layers', '?')}-layer GRU, hid={getattr(model, 'hidden_dim', '?')})",
        },
    }

    std_models, std_cis, cat_metrics, n_samples = evaluate_mode(
        models_dict, test_sequences, all_catalog_pids, catalog, exclude_history=True
    )

    summary = {
        "test_samples_count": n_samples,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "standard_evaluation": {"models": std_models, "bootstrap_cis": std_cis},
        "models": std_models,
        "category_breakdown": cat_metrics,
        "ndcg_at_10": std_models["GRU4Rec"]["ndcg@10"],
        "hit_rate": std_models["GRU4Rec"]["hit@10"],
        "hit_rate_at_10": std_models["GRU4Rec"]["hit@10"],
        "sample_size": n_samples,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary


# Backward-compatible alias
evaluate_all = evaluate_all_models


if __name__ == "__main__":
    evaluate_all_models()
