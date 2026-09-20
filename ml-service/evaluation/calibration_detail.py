"""
PriceIQ ML Service — Detailed Calibration Analysis

Fork-analysis of the calibration methodology:
  1. Top-1 calibration:  confidence = max_predicted_prob,          accuracy = (target == top-1)
  2. Top-10 mass calib.: confidence = sum(top-k probs),            accuracy = (target in top-k)

Temperature is fit on VALIDATION only using two objectives:
  - standard target-token cross-entropy (masked logits, numerically safe)
  - ECE directly (n_bins=10)

Both are then applied to the held-out test set. Nothing on test is ever tuned.
Writes ml-service/artifacts/calibration_detail.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch

import config
from data.dataset import ItemVocabulary
from models.gru4rec import GRU4Rec
from evaluation.calibration import compute_ece


def masked_logits(model, seq, vocab, device):
    """Compute masked logits (as produced at prediction time) for a sequence."""
    pids = seq.get("product_ids", [])
    if len(pids) < 2:
        return None, None, None
    prefix, target = pids[:-1], pids[-1]
    all_history = [vocab.to_idx(pid) for pid in prefix]
    indices = all_history[-config.MAX_SEQ_LEN:]
    padded = [config.PAD_IDX] * (config.MAX_SEQ_LEN - len(indices)) + indices
    inp = torch.tensor([padded], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inp).squeeze(0).float().cpu().numpy()
    mask = np.full(logits.shape, -np.inf)
    mask[config.PAD_IDX] = 0.0
    mask[config.UNK_IDX] = 0.0
    safe = np.where(np.isfinite(logits), logits, 0.0)
    safe[config.PAD_IDX] = -np.inf
    safe[config.UNK_IDX] = -np.inf
    for idx in all_history:
        if idx >= config.SPECIAL_TOKENS and idx < len(safe):
            safe[idx] = -np.inf
    t_idx = vocab.to_idx(target)
    return safe, t_idx, all_history


def apply_temperature(logits, T):
    scaled = logits / max(T, 1e-4)
    scaled = np.where(np.isfinite(scaled), scaled, -np.inf)
    probs = np.exp(scaled - np.max(scaled[scaled > -np.inf]))
    probs = probs / np.sum(probs)
    for i in np.where(np.isneginf(scaled))[0]:
        probs[i] = 0.0
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = probs / np.maximum(np.sum(probs), 1e-9)
    return probs


def collect(logits, target_idx, all_history, vocab, T):
    probs = apply_temperature(logits, T)
    order = np.argsort(-probs)
    top_indices = [int(i) for i in order if i >= config.SPECIAL_TOKENS and i not in all_history][:20]
    top_probs = [float(probs[i]) for i in top_indices]
    top_pids = [vocab.to_id(i) for i in top_indices]
    top1_conf = top_probs[0] if top_probs else 0.0
    top1_hit = int(vocab.to_id(int(target_idx)) == top_pids[0]) if top_pids else 0
    top10_mass = float(min(1.0, sum(top_probs[:10])))
    top10_hit = int(vocab.to_id(int(target_idx)) in top_pids[:10])
    return top1_conf, top1_hit, top10_mass, top10_hit


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = ItemVocabulary.from_file()
    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    with open(os.path.join(data_dir, "val_sequences.json"), "r") as f:
        val_seqs = json.load(f)
    with open(os.path.join(data_dir, "test_sequences.json"), "r") as f:
        test_seqs = json.load(f)

    model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    cfg = checkpoint.get("model_config", {})
    model = GRU4Rec(
        vocab_size=cfg.get("vocab_size", len(vocab)),
        embedding_dim=cfg.get("embedding_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def rows(seqs):
        out = []
        for seq in seqs:
            logits, t_idx, hist = masked_logits(model, seq, vocab, device)
            if logits is None:
                continue
            if t_idx < config.SPECIAL_TOKENS:
                continue
            out.append((logits, t_idx, hist))
        return out

    val_rows = rows(val_seqs)
    test_rows = rows(test_seqs)
    print(f"Val usable={len(val_rows)} Test usable={len(test_rows)}")

    def eval_rows(rows_, T):
        c1, a1, c10, a10 = [], [], [], []
        for logits, t_idx, hist in rows_:
            r = collect(logits, t_idx, hist, vocab, float(T))
            c1.append(r[0]); a1.append(r[1]); c10.append(r[2]); a10.append(r[3])
        return {
            "T": round(float(T), 4),
            "top1_ece": compute_ece(np.array(c1), np.array(a1), n_bins=10)[:2],
            "top1_mean_conf": round(float(np.mean(c1)), 4),
            "top1_acc": round(float(np.mean(a1)), 4),
            "top10_ece": compute_ece(np.array(c10), np.array(a10), n_bins=10)[:2],
            "top10_mean_conf": round(float(np.mean(c10)), 4),
            "top10_acc": round(float(np.mean(a10)), 4),
        }

    # Baseline T=1.0 on val and test
    val_base = eval_rows(val_rows, 1.0)
    test_base = eval_rows(test_rows, 1.0)

    # CE objective over a robust grid (masked logits)
    def ce_loss(T):
        total = 0.0
        for logits, t_idx, _ in val_rows:
            probs = apply_temperature(logits, T)
            p = max(float(probs[t_idx]), 1e-12)
            total += -np.log(p)
        return total / len(val_rows)

    temps = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]
    ce_results = {t: round(ce_loss(t), 6) for t in temps}
    t_ce = min(ce_results, key=ce_results.get)

    # ECE objective over the same grid (top-10 mass)
    ece_results = {}
    for t in temps:
        r = eval_rows(val_rows, t)
        ece_results[t] = r["top10_ece"][0]
    t_ece = min(ece_results, key=ece_results.get)
    t_ece1 = min(temps, key=lambda t: eval_rows(val_rows, t)["top1_ece"][0])

    test_ce = eval_rows(test_rows, t_ce)
    test_ece10 = eval_rows(test_rows, t_ece)
    test_ece1 = eval_rows(test_rows, t_ece1)

    artifact = {
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "uncalibrated_T1": {"val": val_base, "test": test_base},
        "temperature_by_CE_top1": {"T": t_ce, "val": eval_rows(val_rows, t_ce), "test": test_ce},
        "temperature_by_ECE_top10": {"T": t_ece, "val": eval_rows(val_rows, t_ece), "test": test_ece10},
        "temperature_by_ECE_top1": {"T": t_ece1, "val": eval_rows(val_rows, t_ece1), "test": test_ece1},
        "grid_ce_loss": ce_results,
        "grid_ece_top10": ece_results,
    }
    out = os.path.join(config.ARTIFACTS_DIR, "calibration_detail.json")
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    print()
    print(f"T=1.0 (val):  top1 ECE={val_base['top1_ece'][0]}  top10 ECE={val_base['top10_ece'][0]}  "
          f"top10 acc={val_base['top10_acc']:.3f} conf={val_base['top10_mean_conf']:.3f}")
    print(f"T=1.0 (test): top1 ECE={test_base['top1_ece'][0]}  top10 ECE={test_base['top10_ece'][0]}")
    print(f"CE grid min -> T={t_ce}  (test top1 ECE={test_ce['top1_ece'][0]})")
    print(f"ECE10 grid min -> T={t_ece} (test top10 ECE={test_ece10['top10_ece'][0]})")
    print(f"ECE1 grid min -> T={t_ece1}  (test top1 ECE={test_ece1['top1_ece'][0]})")
    print(f"grid CE: {ce_results}")
    print(f"grid ECE10: {ece_results}")
    print(f"✅ calibration_detail.json written")


if __name__ == "__main__":
    main()