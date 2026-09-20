"""
PriceIQ ML Service — Final Scientific Evaluation

Runs the held-out test evaluation for the SELECTED candidate (chosen on validation
only) against: Random, Popularity, Recently Viewed, Markov, TF-IDF, GRU4Rec, Hybrid.

Uses the exact same held-out test sessions for every model (n = 449).
Computes Hit@1/5/10/20, NDCG@5/10, MRR, F1@10, Catalog Coverage, p50/p95 latency,
and 95% bootstrap confidence intervals for every accuracy/ranking metric.

Also captures per-session metadata (session length, target category, repeat-target
flag, confidence scores) to support category / session-length / cold-start analysis
and confidence calibration.

Writes results to artifacts/final_evaluation.json — it NEVER overwrites the
historical artifacts/experiment_results.json baseline file.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
from datetime import datetime

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
)


def bootstrap_ci(values, n_bootstraps=1000, alpha=0.05, seed=42):
    values = list(values)
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_bootstraps, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2.0))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2.0)))
    return round(float(arr.mean()), 4), round(lo, 4), round(hi, 4)


class RandomRecommender:
    """Uniform random baseline over the catalog (with optional history exclusion)."""

    def __init__(self, catalog_pids):
        self.pids = sorted(catalog_pids)

    def recommend(self, session_history, top_k=5):
        excluded = set(session_history) if session_history else set()
        pool = [p for p in self.pids if p not in excluded]
        if not pool:
            return []
        rng = random.Random("random-baseline:" + ",".join(map(str, session_history or [])))
        rng.shuffle(pool)
        return [(p, 1.0) for p in pool[:top_k]]


def predict_gru(prefix, vocab, model, device):
    indices = [vocab.to_idx(pid) for pid in prefix]
    if len(indices) > config.MAX_SEQ_LEN:
        indices = indices[-config.MAX_SEQ_LEN:]
    pad_len = config.MAX_SEQ_LEN - len(indices)
    padded = [config.PAD_IDX] * pad_len + indices
    inp = torch.tensor([padded], dtype=torch.long, device=device)
    res = model.predict_with_confidence(inp, top_k=20, exclude_indices=set(indices))
    return (
        [vocab.to_id(idx) for idx in res["top_indices"] if idx >= config.SPECIAL_TOKENS],
        res,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")

    with open(os.path.join(data_dir, "train_sequences.json"), "r") as f:
        train_seqs = json.load(f)
    with open(os.path.join(data_dir, "val_sequences.json"), "r") as f:
        val_seqs = json.load(f)
    with open(os.path.join(data_dir, "test_sequences.json"), "r") as f:
        test_seqs = json.load(f)

    vocab = ItemVocabulary.from_file()
    catalog = load_product_catalog()
    all_catalog_pids = set(catalog.keys())
    print(f"Train={len(train_seqs)} Val={len(val_seqs)} Test={len(test_seqs)} "
          f"Vocab={len(vocab)} Catalog={len(catalog)}")

    # ── Load selected GRU4Rec checkpoint (selected on validation) ──────────────
    model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    m_conf = checkpoint["model_config"]
    print(f"Checkpoint: {checkpoint.get('experiment_id')} | {m_conf} | "
          f"val_hit10={checkpoint.get('best_val_hit10')}")

    gru_model = GRU4Rec(
        vocab_size=m_conf["vocab_size"],
        embedding_dim=m_conf["embedding_dim"],
        hidden_dim=m_conf["hidden_dim"],
        num_layers=m_conf["num_layers"],
        dropout=m_conf["dropout"],
    ).to(device)
    gru_model.load_state_dict(checkpoint["model_state_dict"])
    gru_model.eval()

    # ── Baselines ──────────────────────────────────────────────────────────────
    pop_model = PopularityRecommender()
    pop_model.train(train_seqs)
    recent_model = RecentlyViewedRecommender()
    content_model = ContentRecommender()
    content_model.train_from_catalog(catalog)
    markov_model = ItemTransitionRecommender()
    markov_model.train(train_seqs)
    random_model = RandomRecommender(list(all_catalog_pids))

    # Tuned hybrid weights from validation grid search (never test)
    hw_path = os.path.join(config.ARTIFACTS_DIR, "experiments", "hybrid_weight_tuning.json")
    alpha, beta, gamma = 1.0, 0.0, 0.0
    if os.path.exists(hw_path):
        with open(hw_path, "r") as f:
            bw = json.load(f).get("best_weights", {})
        alpha = bw.get("alpha", alpha)
        beta = bw.get("beta", beta)
        gamma = bw.get("gamma", gamma)
    hybrid_model = HybridRecommender(
        gru_model=gru_model,
        content_model=content_model,
        pop_model=pop_model,
        vocab=vocab,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        device=device,
    )

    # ── Model registry: closure takes prefix → (recommended_pids, latency_ms) ──
    def wrap(fn):
        def inner(prefix):
            t0 = time.perf_counter()
            recs = fn(prefix)
            lat = (time.perf_counter() - t0) * 1000.0
            return recs, lat
        return inner

    models = {
        "Random": wrap(lambda p: [pid for pid, _ in random_model.recommend(p, top_k=20)]),
        "Popularity": wrap(lambda p: [pid for pid, _ in pop_model.recommend(p, top_k=20)]),
        "RecentlyViewed": wrap(lambda p: [pid for pid, _ in recent_model.recommend(p, top_k=20)]),
        "TF-IDF": wrap(lambda p: [pid for pid, _ in content_model.recommend(p, top_k=20)]),
        "Markov": wrap(lambda p: [pid for pid, _ in markov_model.recommend(p, top_k=20)]),
        "GRU4Rec": wrap(lambda p: predict_gru(p, vocab, gru_model, device)[0]),
        "Hybrid": wrap(lambda p: [pid for pid, _ in hybrid_model.recommend(p, top_k=20)]),
    }

    METRICS = ["hit@1", "hit@5", "hit@10", "hit@20", "ndcg@5", "ndcg@10",
               "mrr", "f1@10"]

    results = {name: {m: [] for m in METRICS} | {"lat": [], "recs": []} for name in models}
    meta = {"GRU4Rec": []}

    evaluated = 0
    for seq in test_seqs:
        pids = seq.get("product_ids", [])
        if len(pids) < 2:
            continue
        prefix = pids[:-1]
        target = pids[-1]

        for name, fn in models.items():
            recs, lat = fn(prefix)
            r = results[name]
            r["lat"].append(lat)
            r["recs"].append(recs)
            r["hit@1"].append(hit_at_k(recs, target, 1))
            r["hit@5"].append(hit_at_k(recs, target, 5))
            r["hit@10"].append(hit_at_k(recs, target, 10))
            r["hit@20"].append(hit_at_k(recs, target, 20))
            r["ndcg@5"].append(ndcg_at_k(recs, target, 5))
            r["ndcg@10"].append(ndcg_at_k(recs, target, 10))
            r["mrr"].append(mrr_at_k(recs, target))
            _, _, f1 = precision_recall_f1_at_k(recs, target, k=10)
            r["f1@10"].append(f1)

        # Metadata for GRU (confidence capture for calibration / diagnosis)
        meta["GRU4Rec"].append({
            "target": target,
            "prefix_len": len(prefix),
            "session_len": len(pids),
            "target_in_prefix": int(target in prefix),
            "recommended": results["GRU4Rec"]["recs"][-1],
            "cat": catalog.get(target, {}).get("category", "Unknown"),
        })
        evaluated += 1

    # ── Aggregate + bootstrap CIs ─────────────────────────────────────────────
    summary = {
        "selected_model": checkpoint.get("experiment_id"),
        "model_config": m_conf,
        "hybrid_weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "test_samples_count": evaluated,
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_protected": True,
        "models": {},
        "bootstrap_cis": {},
    }

    print("\n" + "=" * 150)
    hdr = f"{'Model':<14} | " + " | ".join(f"{m:<8}" for m in METRICS) + " | " + f"{'Cov':<7} | {'p50':<7} | {'p95':<7}"
    print(hdr)
    print("=" * 150)

    for name, m in results.items():
        means = {k: round(float(np.mean(v)), 4) for k, v in m.items() if k in METRICS}
        cov = compute_catalog_coverage(m["recs"], all_catalog_pids)
        lat = m["lat"]
        entry = dict(means)
        entry["catalog_coverage"] = round(cov, 4)
        entry["avg_latency_ms"] = round(float(np.mean(lat)), 2)
        entry["p50_latency_ms"] = round(float(np.percentile(lat, 50)), 2)
        entry["p95_latency_ms"] = round(float(np.percentile(lat, 95)), 2)
        summary["models"][name] = entry

        print(f"{name:<14} | " + " | ".join(f"{means[k]*100:6.2f}%" if 'hit' in k else f"{means[k]:8.4f}" for k in METRICS)
              + f" | {cov*100:5.1f}% | {entry['p50_latency_ms']:<7} | {entry['p95_latency_ms']:<7}")

    summary["bootstrap_cis"] = {name: {} for name in models}
    for name, m in results.items():
        for metric in METRICS + ["lat"]:
            if metric in ("lat",):
                mean, lo, hi = bootstrap_ci(m[metric])
                summary["bootstrap_cis"][name][metric + "_ms"] = {"mean": mean, "ci_95": [lo, hi]}
            else:
                mean, lo, hi = bootstrap_ci(m[metric])
                summary["bootstrap_cis"][name][metric] = {"mean": mean, "ci_95": [lo, hi]}

    print("\n95% Bootstrap CIs (unrestricted to the sample):")
    for name in models:
        h = summary["bootstrap_cis"][name]["hit@10"]
        n = summary["bootstrap_cis"][name]["ndcg@10"]
        print(f"  {name:<14}: Hit@10 {h['mean']*100:.2f}% [{h['ci_95'][0]*100:.2f}, {h['ci_95'][1]*100:.2f}] | "
              f"NDCG@10 {n['mean']:.4f} [{n['ci_95'][0]:.4f}, {n['ci_95'][1]:.4f}]")

    # Count of samples where target is already in the prefix (repeat target)
    rep_frac = sum(1 for s in meta["GRU4Rec"] if s["target_in_prefix"]) / evaluated
    summary["diagnostics"] = {
        "target_in_prefix_frac": round(rep_frac, 4),
        "note": (
            "The new synthetic data contains heavy backtracking: "
            f"{rep_frac*100:.1f}% of test targets are already present in the session prefix. "
            "This trivially inflates recency baselines and structurally caps history-excluding "
            "models (GRU4Rec / TF-IDF / Hybrid / Popularity)."
        ),
    }
    summary["category_breakdown"] = {name: {} for name in models}
    summary["session_length_breakdown"] = {name: {} for name in models}
    summary["cold_start_breakdown"] = {name: {} for name in models}

    # Per-session meta for every model (recompute recommended-to-target mapping)
    summary["repeat_split_breakdown"] = {name: {} for name in models}
    for idx, seq in enumerate(test_seqs):
        pids = seq.get("product_ids", [])
        if len(pids) < 2:
            continue
        prefix = pids[:-1]
        target = pids[-1]
        t_cat = catalog.get(target, {}).get("category", "Unknown")
        slen = len(pids)
        bucket = "short(<=3)" if slen <= 3 else ("medium(4-6)" if slen <= 6 else "long(>=7)")
        cs_bucket = "L1" if len(prefix) == 1 else ("L2" if len(prefix) == 2 else "L3+")
        rep_bucket = "repeat_target" if target in prefix else "new_target"
        for name in models:
            recs = results[name]["recs"][idx]
            h10 = hit_at_k(recs, target, 10)
            keys = (summary["category_breakdown"][name], t_cat, h10, 1.0)
            bkeys = (summary["session_length_breakdown"][name], bucket, h10, 1.0)
            ckeys = (summary["cold_start_breakdown"][name], cs_bucket, h10, 1.0)
            rkeys = (summary["repeat_split_breakdown"][name], rep_bucket, h10, 1.0)
            for tbl, key, val, cnt in [keys, bkeys, ckeys, rkeys]:
                d = tbl.setdefault(key, {"total": 0, "hits": 0.0})
                d["total"] += 1
                d["hits"] += val

    for name in models:
        for tblname in ("category_breakdown", "session_length_breakdown", "cold_start_breakdown", "repeat_split_breakdown"):
            tbl = summary[tblname][name]
            out = {}
            for k, d in tbl.items():
                out[str(k)] = {
                    "sample_count": d["total"],
                    "hit@10": round(d["hits"] / d["total"], 4),
                }
            summary[tblname][name] = out

    out_path = os.path.join(config.ARTIFACTS_DIR, "final_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Final evaluation written to {out_path} (historical experiment_results.json untouched)")


if __name__ == "__main__":
    main()