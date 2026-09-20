"""
PriceIQ ML Service — Feature Ablation Experiment
Tests whether enriching GRU4Rec with side features (Category, Price Tier, Event Type)
improves or harms generalization on held-out validation sequences.

Rule: Only add a feature if an ablation experiment proves it improves validation performance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from data.dataset import ItemVocabulary
from database import get_db

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_catalog_metadata():
    """Load category, brand, and price tier maps for all products."""
    db = get_db()
    products = list(db.products.find({}, {"_id": 0, "id": 1, "category": 1, "livePrice": 1, "basePrice": 1}))
    
    categories = sorted(set(p.get("category", "Unknown") for p in products))
    cat_to_idx = {c: i + 1 for i, c in enumerate(categories)}  # 0 is PAD/UNK
    
    prices = [float(p.get("livePrice") or p.get("basePrice") or 0) for p in products]
    prices = [pr for pr in prices if pr > 0]
    q20, q40, q60, q80 = np.quantile(prices, [0.2, 0.4, 0.6, 0.8])
    
    def get_price_tier(price):
        if price is None:
            return 0
        if price <= q20:
            return 1
        elif price <= q40:
            return 2
        elif price <= q60:
            return 3
        elif price <= q80:
            return 4
        return 5
        
    pid_to_meta = {}
    for p in products:
        pid = p["id"]
        cat = p.get("category", "Unknown")
        price = float(p.get("livePrice") or p.get("basePrice") or 0)
        pid_to_meta[pid] = {
            "cat_idx": cat_to_idx.get(cat, 0),
            "price_tier": get_price_tier(price),
        }
    return pid_to_meta, len(cat_to_idx) + 1, 6


class MultiFeatureSessionDataset(Dataset):
    def __init__(self, sequences, vocab, pid_to_meta, max_len=15):
        self.samples = []
        for seq in sequences:
            pids = seq.get("product_ids", [])
            if len(pids) < 2:
                continue
            for i in range(1, len(pids)):
                prefix = pids[:i]
                target = pids[i]
                t_idx = vocab.to_idx(target)
                if t_idx < config.SPECIAL_TOKENS:
                    continue
                
                # Truncate prefix
                p_items = prefix[-max_len:]
                pad_len = max_len - len(p_items)
                
                item_indices = [vocab.to_idx(p) for p in p_items]
                cat_indices = [pid_to_meta.get(p, {}).get("cat_idx", 0) for p in p_items]
                price_indices = [pid_to_meta.get(p, {}).get("price_tier", 0) for p in p_items]
                
                item_padded = [config.PAD_IDX] * pad_len + item_indices
                cat_padded = [0] * pad_len + cat_indices
                price_padded = [0] * pad_len + price_indices
                
                self.samples.append({
                    "items": torch.tensor(item_padded, dtype=torch.long),
                    "cats": torch.tensor(cat_padded, dtype=torch.long),
                    "prices": torch.tensor(price_padded, dtype=torch.long),
                    "target": t_idx,
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]


class AblationGRUModel(nn.Module):
    def __init__(self, vocab_size, num_cats, num_prices, variant="item_only", total_emb=64, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.variant = variant
        
        if variant == "item_only":
            self.item_emb = nn.Embedding(vocab_size, total_emb, padding_idx=config.PAD_IDX)
            input_dim = total_emb
        elif variant == "item_plus_category":
            item_dim = 48
            cat_dim = 16
            self.item_emb = nn.Embedding(vocab_size, item_dim, padding_idx=config.PAD_IDX)
            self.cat_emb = nn.Embedding(num_cats, cat_dim, padding_idx=0)
            input_dim = item_dim + cat_dim
        elif variant == "item_plus_cat_plus_price":
            item_dim = 44
            cat_dim = 12
            price_dim = 8
            self.item_emb = nn.Embedding(vocab_size, item_dim, padding_idx=config.PAD_IDX)
            self.cat_emb = nn.Embedding(num_cats, cat_dim, padding_idx=0)
            self.price_emb = nn.Embedding(num_prices, price_dim, padding_idx=0)
            input_dim = item_dim + cat_dim + price_dim
        else:
            raise ValueError(f"Unknown variant {variant}")
            
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, items, cats=None, prices=None):
        if self.variant == "item_only":
            x = self.item_emb(items)
        elif self.variant == "item_plus_category":
            x = torch.cat([self.item_emb(items), self.cat_emb(cats)], dim=-1)
        elif self.variant == "item_plus_cat_plus_price":
            x = torch.cat([self.item_emb(items), self.cat_emb(cats), self.price_emb(prices)], dim=-1)
            
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        return self.output_layer(last_hidden)


def evaluate_variant(model, val_loader, device, vocab_size):
    model.eval()
    hits_1, hits_5, hits_10, mrr_list, ndcg_list = [], [], [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            items = batch["items"].to(device)
            cats = batch["cats"].to(device)
            prices = batch["prices"].to(device)
            targets = batch["target"].tolist()
            
            logits = model(items, cats, prices)
            logits[:, config.PAD_IDX] = -1e9
            logits[:, config.UNK_IDX] = -1e9
            
            top10 = torch.topk(logits, 10, dim=1).indices.cpu().tolist()
            
            for pred_top, target in zip(top10, targets):
                hits_1.append(1.0 if target == pred_top[0] else 0.0)
                hits_5.append(1.0 if target in pred_top[:5] else 0.0)
                hits_10.append(1.0 if target in pred_top[:10] else 0.0)
                if target in pred_top[:10]:
                    rank = pred_top.index(target) + 1
                    mrr_list.append(1.0 / rank)
                    ndcg_list.append(1.0 / np.log2(rank + 1))
                else:
                    mrr_list.append(0.0)
                    ndcg_list.append(0.0)
                    
    return {
        "hit@1": round(float(np.mean(hits_1)) * 100, 2),
        "hit@5": round(float(np.mean(hits_5)) * 100, 2),
        "hit@10": round(float(np.mean(hits_10)) * 100, 2),
        "ndcg@10": round(float(np.mean(ndcg_list)), 4),
        "mrr": round(float(np.mean(mrr_list)), 4),
    }


def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = ItemVocabulary.from_file()
    pid_to_meta, num_cats, num_prices = load_catalog_metadata()
    
    data_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    with open(os.path.join(data_dir, "train_sequences.json"), "r") as f:
        train_seqs = json.load(f)
    with open(os.path.join(data_dir, "val_sequences.json"), "r") as f:
        val_seqs = json.load(f)
        
    train_dataset = MultiFeatureSessionDataset(train_seqs, vocab, pid_to_meta)
    val_dataset = MultiFeatureSessionDataset(val_seqs, vocab, pid_to_meta)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    variants = [
        ("item_only", "Baseline (Item ID only)"),
        ("item_plus_category", "Variant A (Item ID + Category)"),
        ("item_plus_cat_plus_price", "Variant B (Item ID + Category + Price Tier)"),
    ]
    
    results = {}
    
    print("\n" + "=" * 70)
    print("  FEATURE ABLATION EXPERIMENT ON VALIDATION DATA")
    print("  (Equal epochs=10, LR=0.001, Seed=42, Hidden=256, TotalEmb=64)")
    print("=" * 70)
    
    criterion = nn.CrossEntropyLoss()
    
    for var_key, var_name in variants:
        print(f"\n--- Training {var_name} ---")
        torch.manual_seed(SEED)
        model = AblationGRUModel(
            vocab_size=len(vocab),
            num_cats=num_cats,
            num_prices=num_prices,
            variant=var_key,
            total_emb=64,
            hidden_dim=256,
            dropout=0.2,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        start_t = time.perf_counter()
        for epoch in range(1, 11):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                items = batch["items"].to(device)
                cats = batch["cats"].to(device)
                prices = batch["prices"].to(device)
                targets = batch["target"].to(device)
                
                optimizer.zero_grad()
                logits = model(items, cats, prices)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            if epoch % 5 == 0 or epoch == 10:
                print(f"  Epoch {epoch:02d}/10 | Train Loss: {avg_loss:.4f}")
                
        dur = time.perf_counter() - start_t
        eval_metrics = evaluate_variant(model, val_loader, device, len(vocab))
        eval_metrics["train_time_sec"] = round(dur, 2)
        eval_metrics["description"] = var_name
        results[var_key] = eval_metrics
        
        print(f"  Result -> Hit@1: {eval_metrics['hit@1']}% | Hit@5: {eval_metrics['hit@5']}% | Hit@10: {eval_metrics['hit@10']}% | NDCG@10: {eval_metrics['ndcg@10']} | MRR: {eval_metrics['mrr']}")

    # Save results artifact
    out_path = os.path.join(config.ARTIFACTS_DIR, "feature_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Ablation results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_ablation()
