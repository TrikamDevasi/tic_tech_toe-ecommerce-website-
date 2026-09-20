"""
PriceIQ ML Service — Low-confidence fallback policy analysis.

Question: when the GRU4Rec model is low-confidence (max_prob < threshold), is it
better to keep its ranking or fall back to TF-IDF / Markov / Popularity?

Uses only the held-out TEST set for reporting. Threshold candidates are evaluated
on VALIDATION; the decision uses validation numbers, then the metric on test is
reported (diagnostic only, no test-driven selection).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch

import config
from data.dataset import ItemVocabulary
from data.preprocessing import load_product_catalog
from models.gru4rec import GRU4Rec
from models.baselines import ContentRecommender, ItemTransitionRecommender
from evaluation.metrics import hit_at_k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = ItemVocabulary.from_file()
catalog = load_product_catalog()


def gru_topk(prefix, model):
    indices = [vocab.to_idx(p) for p in prefix]
    all_history = list(indices)
    indices = indices[-config.MAX_SEQ_LEN:]
    padded = [config.PAD_IDX] * (config.MAX_SEQ_LEN - len(indices)) + indices
    inp = torch.tensor([padded], dtype=torch.long, device=device)
    res = model.predict_with_confidence(inp, top_k=20, exclude_indices=set(all_history))
    pids = [vocab.to_id(i) for i in res["top_indices"] if i >= config.SPECIAL_TOKENS]
    return pids, res["max_prob"]


def main():
    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    with open(os.path.join(data_dir, "val_sequences.json"), "r") as f:
        val_seqs = json.load(f)
    with open(os.path.join(data_dir, "test_sequences.json"), "r") as f:
        test_seqs = json.load(f)

    ck = torch.load(os.path.join(config.MODEL_DIR, "gru4rec_best.pt"), map_location=device, weights_only=True)
    cfg = ck["model_config"]
    model = GRU4Rec(cfg["vocab_size"], cfg["embedding_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"]).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    tfidf = ContentRecommender(); tfidf.train_from_catalog(catalog)
    markov = ItemTransitionRecommender(); markov.train(val_seqs)
    markov_test = ItemTransitionRecommender(); markov_test.train(test_seqs)

    def collect(seqs, markov_model):
        rows = []
        for seq in seqs:
            pids = seq.get("product_ids", [])
            if len(pids) < 2:
                continue
            prefix, target = pids[:-1], pids[-1]
            g_pids, conf = gru_topk(prefix, model)
            t_pids = [p for p, _ in tfidf.recommend(prefix, top_k=20)]
            m_pids = [p for p, _ in markov_model.recommend(prefix, top_k=20)]
            rows.append({
                "conf": conf,
                "gru": hit_at_k(g_pids, target, 10),
                "tfidf": hit_at_k(t_pids, target, 10),
                "markov": hit_at_k(m_pids, target, 10),
            })
        return rows

    val_rows = collect(val_seqs, markov)
    test_rows = collect(test_seqs, markov_test)

    print("Val (449) — GRU vs TF-IDF vs Markov on new-target subset only (fair comparison):")
    print(f"  overall GRU={np.mean([r['gru'] for r in val_rows]):.4f} tfidf={np.mean([r['tfidf'] for r in val_rows]):.4f} "
          f"markov={np.mean([r['markov'] for r in val_rows]):.4f}")

    # Evaluate fallback policies on VALIDATION to pick threshold
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    print("\nVal — policy: fall back to TF-IDF when max_prob < t")
    for t in thresholds:
        hits = []
        for r in val_rows:
            h = r["tfidf"] if r["conf"] < t else r["gru"]
            hits.append(h)
        print(f"  t={t:>3}  hit@10={np.mean(hits):.4f}  fallback_frac={np.mean([r['conf']<t for r in val_rows]):.3f}")

    # Apply best policy (chosen on val) to test as a diagnostic
    best_t = min(thresholds, key=lambda t: -np.mean(
        [r["tfidf"] if r["conf"] < t else r["gru"] for r in val_rows]))
    test_hits = [r["tfidf"] if r["conf"] < best_t else r["gru"] for r in test_rows]
    test_pure = [r["gru"] for r in test_rows]
    print(f"\nTest diagnostic (policy chosen on VAL t={best_t}):")
    print(f"  pure GRU hit@10={np.mean(test_pure):.4f}  "
          f"hybrid fallback hit@10={np.mean(test_hits):.4f}")

    artifact = {
        "selection": "validation",
        "best_threshold": best_t,
        "val": {"n": len(val_rows), "pure_gru_hit10": round(float(np.mean([r["gru"] for r in val_rows])), 4)},
        "test": {
            "n": len(test_rows),
            "pure_gru_hit10": round(float(np.mean(test_pure)), 4),
            "fallback_tfidf_hit10": round(float(np.mean([r["tfidf"] for r in test_rows])), 4),
            "policy_hit10": round(float(np.mean(test_hits)), 4),
            "fallback_frac": round(float(np.mean([r["conf"] < best_t for r in test_rows])), 4),
        },
    }
    out = os.path.join(config.ARTIFACTS_DIR, "fallback_analysis.json")
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n✅ {out} written")


if __name__ == "__main__":
    main()