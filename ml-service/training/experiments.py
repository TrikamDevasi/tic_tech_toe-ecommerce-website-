"""
PriceIQ ML Service — Systematic Experiment Runner & Hyperparameter Tuning

Executes controlled, reproducible experiments across GRU4Rec architectures and
tunes Hybrid Recommender weights (α, β, γ) STRICTLY on Train and Validation data.
The held-out Test set is NEVER accessed during this phase.

Every experiment is recorded in artifacts/experiments/experiment_xxx.json.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import config
from data.dataset import ItemVocabulary, SessionDataset, collate_fn
from data.preprocessing import load_product_catalog
from models.gru4rec import GRU4Rec
from models.baselines import ContentRecommender, PopularityRecommender, ItemTransitionRecommender
from models.hybrid import HybridRecommender
from evaluation.metrics import hit_at_k, ndcg_at_k, mrr_at_k


def evaluate_on_validation_sessions(model, val_sequences, vocab, device, top_k=10):
    """
    Evaluate GRU4Rec on validation session sequences (predicting final item from prefix).
    Returns dict of ranking metrics on the validation set.
    """
    model.eval()
    hits1, hits5, hits10, ndcgs10, mrrs = [], [], [], [], []

    for seq in val_sequences:
        pids = seq.get("product_ids", [])
        if len(pids) < 2:
            continue
        prefix = pids[:-1]
        target = pids[-1]

        indices = [vocab.to_idx(pid) for pid in prefix]
        if len(indices) > config.MAX_SEQ_LEN:
            indices = indices[-config.MAX_SEQ_LEN:]
        pad_len = config.MAX_SEQ_LEN - len(indices)
        padded = [config.PAD_IDX] * pad_len + indices

        inp_tensor = torch.tensor([padded], dtype=torch.long, device=device)
        exclude_set = set(indices)

        with torch.no_grad():
            res = model.predict_with_confidence(inp_tensor, top_k=top_k, exclude_indices=exclude_set)
        top_indices = res["top_indices"]
        recommended = [vocab.to_id(idx) for idx in top_indices if idx >= config.SPECIAL_TOKENS]

        hits1.append(hit_at_k(recommended, target, k=1))
        hits5.append(hit_at_k(recommended, target, k=5))
        hits10.append(hit_at_k(recommended, target, k=10))
        ndcgs10.append(ndcg_at_k(recommended, target, k=10))
        mrrs.append(mrr_at_k(recommended, target))

    return {
        "hit@1": round(float(np.mean(hits1)), 4) if hits1 else 0.0,
        "hit@5": round(float(np.mean(hits5)), 4) if hits5 else 0.0,
        "hit@10": round(float(np.mean(hits10)), 4) if hits10 else 0.0,
        "ndcg@10": round(float(np.mean(ndcgs10)), 4) if ndcgs10 else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "sample_count": len(hits10),
    }


def run_single_experiment(
    exp_id: str,
    params: dict,
    train_seqs: list,
    val_seqs: list,
    vocab: ItemVocabulary,
    device: torch.device,
    experiments_dir: str,
):
    """
    Train a single GRU4Rec configuration and log metrics.
    Returns (summary_dict, model).
    """
    print(f"\n{'='*70}")
    print(f"🚀 Running {exp_id}: {params.get('description', '')}")
    print(f"   Parameters: {params}")
    print(f"{'='*70}")

    train_ds = SessionDataset(train_seqs, vocab, max_seq_len=params.get("max_seq_len", 20))
    val_ds = SessionDataset(val_seqs, vocab, max_seq_len=params.get("max_seq_len", 20))

    batch_size = params.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GRU4Rec(
        vocab_size=len(vocab),
        embedding_dim=params.get("embedding_dim", 64),
        hidden_dim=params.get("hidden_dim", 128),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.25),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    epochs = params.get("epochs", 25)
    patience = params.get("patience", 5)

    best_val_loss = float("inf")
    best_val_hit10 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_hit10": [],
        "val_ndcg10": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * len(targets)

        avg_train_loss = total_train_loss / len(train_ds)

        # 2. Validation Loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                total_val_loss += loss.item() * len(targets)

        avg_val_loss = total_val_loss / len(val_ds)
        scheduler.step(avg_val_loss)

        # 3. Session-level Validation Ranking Evaluation
        val_metrics = evaluate_on_validation_sessions(model, val_seqs, vocab, device, top_k=10)
        curr_hit10 = val_metrics["hit@10"]
        curr_ndcg10 = val_metrics["ndcg@10"]

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_hit10"].append(round(curr_hit10, 4))
        history["val_ndcg10"].append(round(curr_ndcg10, 4))

        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Hit@10: {curr_hit10*100:.2f}% | "
            f"Val NDCG@10: {curr_ndcg10:.4f}"
        )

        # Checkpointing condition (track best combination of loss and Hit@10)
        if curr_hit10 > best_val_hit10 or (curr_hit10 == best_val_hit10 and avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            best_val_hit10 = curr_hit10
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    training_time = round(time.time() - start_time, 2)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation evaluation with best model
    best_val_metrics = evaluate_on_validation_sessions(model, val_seqs, vocab, device, top_k=20)

    result = {
        "experiment_id": exp_id,
        "description": params.get("description", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params,
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history["train_loss"]),
        "validation_metrics": best_val_metrics,
        "history": history,
    }

    # Save artifact
    exp_path = os.path.join(experiments_dir, f"{exp_id}.json")
    with open(exp_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  ✅ Saved {exp_id} results to {exp_path} (Best Val Hit@10: {best_val_metrics['hit@10']*100:.2f}%)")

    return result, model


def grid_search_hybrid_weights(
    gru_model,
    content_model,
    pop_model,
    val_sequences,
    vocab,
    catalog,
    device,
    experiments_dir,
):
    """
    Grid search Hybrid Recommender weights (α, β, γ) on the VALIDATION set.
    """
    print(f"\n{'='*70}")
    print("🔬 Grid Searching Hybrid Recommender Weights (α, β, γ) on Validation Set")
    print(f"{'='*70}")

    hybrid = HybridRecommender(
        gru_model=gru_model,
        content_model=content_model,
        pop_model=pop_model,
        vocab=vocab,
        device=device,
    )

    weight_candidates = [
        # (alpha: neural, beta: content, gamma: popularity)
        (1.0, 0.0, 0.0),   # Pure GRU4Rec
        (0.0, 1.0, 0.0),   # Pure Content TF-IDF
        (0.0, 0.0, 1.0),   # Pure Popularity
        (0.70, 0.30, 0.00),
        (0.60, 0.40, 0.00),
        (0.50, 0.50, 0.00),
        (0.40, 0.60, 0.00),
        (0.30, 0.70, 0.00),
        (0.50, 0.40, 0.10),
        (0.40, 0.50, 0.10),
        (0.60, 0.30, 0.10),
        (0.30, 0.60, 0.10),
        (0.45, 0.45, 0.10),
    ]

    best_ndcg = -1.0
    best_weights = None
    all_grid_results = []

    for alpha, beta, gamma in weight_candidates:
        hybrid.set_weights(alpha, beta, gamma)
        hits1, hits5, hits10, ndcgs10, mrrs = [], [], [], [], []

        for seq in val_sequences:
            pids = seq.get("product_ids", [])
            if len(pids) < 2:
                continue
            prefix = pids[:-1]
            target = pids[-1]

            recs = hybrid.recommend(prefix, top_k=10)
            recommended = [pid for pid, _ in recs]

            hits1.append(hit_at_k(recommended, target, k=1))
            hits5.append(hit_at_k(recommended, target, k=5))
            hits10.append(hit_at_k(recommended, target, k=10))
            ndcgs10.append(ndcg_at_k(recommended, target, k=10))
            mrrs.append(mrr_at_k(recommended, target))

        res_entry = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "val_hit@1": round(float(np.mean(hits1)), 4),
            "val_hit@5": round(float(np.mean(hits5)), 4),
            "val_hit@10": round(float(np.mean(hits10)), 4),
            "val_ndcg@10": round(float(np.mean(ndcgs10)), 4),
            "val_mrr": round(float(np.mean(mrrs)), 4),
        }
        all_grid_results.append(res_entry)

        print(
            f"  α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f} | "
            f"Val Hit@10: {res_entry['val_hit@10']*100:5.2f}% | "
            f"Val NDCG@10: {res_entry['val_ndcg@10']:.4f} | "
            f"Val MRR: {res_entry['val_mrr']:.4f}"
        )

        if res_entry["val_ndcg@10"] > best_ndcg:
            best_ndcg = res_entry["val_ndcg@10"]
            best_weights = (alpha, beta, gamma)

    tuning_summary = {
        "best_weights": {
            "alpha": best_weights[0],
            "beta": best_weights[1],
            "gamma": best_weights[2],
        },
        "best_val_ndcg@10": best_ndcg,
        "grid_results": all_grid_results,
    }

    hybrid_path = os.path.join(experiments_dir, "hybrid_weight_tuning.json")
    with open(hybrid_path, "w") as f:
        json.dump(tuning_summary, f, indent=2)
    print(f"\n  🏆 Optimal Hybrid Weights on Validation Set: {best_weights} with Val NDCG@10 = {best_ndcg:.4f}")

    return best_weights, tuning_summary


def run_all_experiments():
    """Main experiment suite controller."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments_dir = os.path.join(config.ARTIFACTS_DIR, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. Load Data
    vocab = ItemVocabulary.from_file()
    catalog = load_product_catalog()
    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    with open(os.path.join(data_dir, "train_sequences.json"), "r") as f:
        train_seqs = json.load(f)
    with open(os.path.join(data_dir, "val_sequences.json"), "r") as f:
        val_seqs = json.load(f)

    # 2. Experiment Configurations Matrix
    exp_matrix = [
        {
            "id": "experiment_001",
            "description": "Baseline GRU4Rec (dim=64/128, 2 layers, drop=0.25, lr=1e-3)",
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.25,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_seq_len": 20,
            "epochs": 20,
            "patience": 5,
        },
        {
            "id": "experiment_002",
            "description": "High Capacity Network (dim=128/256, 2 layers, drop=0.30, lr=1e-3)",
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.30,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_seq_len": 20,
            "epochs": 20,
            "patience": 5,
        },
        {
            "id": "experiment_003",
            "description": "Compact Low-Latency Architecture (dim=32/64, 1 layer, drop=0.10, lr=1e-3)",
            "embedding_dim": 32,
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_seq_len": 20,
            "epochs": 20,
            "patience": 5,
        },
        {
            "id": "experiment_004",
            "description": "Lower Learning Rate (dim=64/128, 2 layers, drop=0.20, lr=5e-4)",
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.20,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "max_seq_len": 20,
            "epochs": 20,
            "patience": 5,
        },
        {
            "id": "experiment_005",
            "description": "Single-Layer High Hidden Dim (dim=64/256, 1 layer, drop=0.20, lr=1e-3)",
            "embedding_dim": 64,
            "hidden_dim": 256,
            "num_layers": 1,
            "dropout": 0.20,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_seq_len": 20,
            "epochs": 20,
            "patience": 5,
        },
        {
            "id": "experiment_006",
            "description": "Balanced Tuned Representation (dim=128/128, 2 layers, drop=0.20, lr=1e-3)",
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.20,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_seq_len": 20,
            "epochs": 20,
            "patience": 5,
        },
    ]

    all_summaries = []
    best_val_hit10 = -1.0
    best_model = None
    best_exp_id = None
    best_params = None

    for exp in exp_matrix:
        res, model = run_single_experiment(
            exp["id"],
            exp,
            train_seqs,
            val_seqs,
            vocab,
            device,
            experiments_dir,
        )
        all_summaries.append(res)
        val_hit10 = res["validation_metrics"]["hit@10"]
        if val_hit10 > best_val_hit10:
            best_val_hit10 = val_hit10
            best_model = model
            best_exp_id = exp["id"]
            best_params = exp

    print(f"\n{'='*70}")
    print(f"🏆 Best GRU4Rec Architecture on Validation Set: {best_exp_id}")
    print(f"   Val Hit@10 = {best_val_hit10*100:.2f}% | Parameters: {best_params}")
    print(f"{'='*70}")

    # Save best GRU4Rec checkpoint
    best_ckpt_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
    torch.save({
        "experiment_id": best_exp_id,
        "model_state_dict": best_model.state_dict(),
        "model_config": {
            "vocab_size": len(vocab),
            "embedding_dim": best_params["embedding_dim"],
            "hidden_dim": best_params["hidden_dim"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"],
        },
        "best_val_hit10": best_val_hit10,
    }, best_ckpt_path)
    print(f"  Checkpoint saved to {best_ckpt_path}")

    # Update config defaults if best params differ
    config_update_note = {
        "best_experiment": best_exp_id,
        "embedding_dim": best_params["embedding_dim"],
        "hidden_dim": best_params["hidden_dim"],
        "num_layers": best_params["num_layers"],
        "dropout": best_params["dropout"],
    }
    with open(os.path.join(config.MODEL_DIR, "best_model_config.json"), "w") as f:
        json.dump(config_update_note, f, indent=2)

    # 3. Train Baselines on Train Sequences
    print("\n📦 Training Baselines on Train Sequences for Hybrid Weight Tuning...")
    pop_model = PopularityRecommender()
    pop_model.train(train_seqs)

    content_model = ContentRecommender()
    content_model.train_from_catalog(catalog)

    # 4. Grid Search Hybrid Weights on Validation Data
    best_weights, tuning_summary = grid_search_hybrid_weights(
        best_model,
        content_model,
        pop_model,
        val_seqs,
        vocab,
        catalog,
        device,
        experiments_dir,
    )

    # 5. Save All Validation Experiments Comparison Table
    comparison_table = {
        "experiments": all_summaries,
        "best_gru4rec": best_exp_id,
        "best_hybrid_weights": best_weights,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(experiments_dir, "validation_experiments_summary.json"), "w") as f:
        json.dump(comparison_table, f, indent=2)

    print("\n✅ All validation experiments and hybrid tuning completed successfully!")


if __name__ == "__main__":
    run_all_experiments()
