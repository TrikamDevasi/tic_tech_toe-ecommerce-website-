"""
PriceIQ ML Service — GRU4Rec Training Pipeline

Complete reproducible training loop with:
  - Mini-batch DataLoader with prefix sub-sessions
  - AdamW optimizer with weight decay
  - ReduceLROnPlateau learning rate scheduler
  - Validation loss and Top-10 validation accuracy tracking
  - Early stopping with patience
  - Best model checkpointing
  - Full training history and hyperparameters logging

Usage:
    python -m training.train
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.build_dataset import build_dataset
from data.dataset import SessionDataset, ItemVocabulary, collate_fn
from models.gru4rec import GRU4Rec


def set_seed(seed=None):
    """Set deterministic seeds for reproducibility."""
    if seed is None:
        seed = config.RANDOM_SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)              # (B, vocab_size)
        loss = criterion(logits, targets)    # targets are item indices
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    """Run validation. Returns average loss and top-10 accuracy."""
    model.eval()
    total_loss = 0.0
    correct_top10 = 0
    correct_top5 = 0
    correct_top1 = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1

            # Top-K accuracy
            top10 = torch.topk(logits, k=min(10, logits.size(-1)), dim=-1).indices
            targets_exp = targets.unsqueeze(-1)
            correct_top10 += (top10 == targets_exp).any(dim=-1).sum().item()
            correct_top5 += (top10[:, :min(5, top10.size(-1))] == targets_exp).any(dim=-1).sum().item()
            correct_top1 += (top10[:, 0] == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / max(num_batches, 1)
    acc_top10 = correct_top10 / max(total_samples, 1)
    acc_top5 = correct_top5 / max(total_samples, 1)
    acc_top1 = correct_top1 / max(total_samples, 1)
    return avg_loss, acc_top10, acc_top5, acc_top1


def run_training(
    train_seqs=None,
    val_seqs=None,
    test_seqs=None,
    vocab=None,
    epochs=None,
    batch_size=None,
    lr=None,
    patience=None,
):
    """
    Full training pipeline.
    Returns the trained model, vocab, and training metadata.
    """
    set_seed()

    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if lr is None:
        lr = config.LEARNING_RATE
    if patience is None:
        patience = config.PATIENCE

    # Build dataset if not provided
    if train_seqs is None or val_seqs is None or vocab is None:
        print("\n🔧 Loading / building dataset...")
        result = build_dataset()
        if result is None:
            raise RuntimeError("Dataset build failed. Cannot train.")
        train_seqs, val_seqs, test_seqs, vocab, report = result

    # Create PyTorch datasets
    print("\n🔧 Creating PyTorch datasets...")
    train_dataset = SessionDataset(train_seqs, vocab)
    val_dataset = SessionDataset(val_seqs, vocab)

    print(f"  Train prefix-target samples: {len(train_dataset):,}")
    print(f"  Val prefix-target samples:   {len(val_dataset):,}")

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples. Check data pipeline.")

    effective_batch = min(batch_size, len(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_batch, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Initialize model
    vocab_size = len(vocab)
    print(f"\n🔧 Initializing Category-Aware GRU4Rec (vocab_size={vocab_size})...")
    model = GRU4Rec(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_GRU_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=getattr(config, "WEIGHT_DECAY", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    # Training loop
    print(f"\n🚀 Training for up to {epochs} epochs (patience={patience})...\n")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_accs_top10 = []
    best_epoch = 0
    start_time = time.time()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc10, val_acc5, val_acc1 = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs_top10.append(val_acc10)

        epoch_time = time.time() - epoch_start

        # Check for improvement on val_loss
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc10
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc_top10": val_acc10,
            }, best_model_path)
            improved = " ✓ saved"
        else:
            epochs_without_improvement += 1

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Top-10: {val_acc10*100:.1f}% | "
            f"Val Top-1: {val_acc1*100:.1f}% | "
            f"LR: {current_lr:.1e} | "
            f"{epoch_time:.1f}s"
            f"{improved}"
        )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.1f}s")
    print(f"  Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}, val_top10={best_val_acc*100:.1f}%)")
    print(f"  Model saved to: {best_model_path}")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save training metadata
    train_meta = {
        "total_epochs": len(train_losses),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc_top10": round(best_val_acc, 4),
        "final_train_loss": round(train_losses[-1], 6),
        "train_losses": [round(l, 6) for l in train_losses],
        "val_losses": [round(l, 6) for l in val_losses],
        "val_accs_top10": [round(a, 4) for a in val_accs_top10],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": effective_batch,
        "learning_rate": lr,
        "model_config": model.get_config(),
        "training_time_seconds": round(total_time, 2),
        "device": str(device),
    }
    meta_path = os.path.join(config.MODEL_DIR, "train_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(train_meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")

    return model, vocab, train_meta, test_seqs


if __name__ == "__main__":
    model, vocab, meta, test_seqs = run_training()
    print(f"\n📊 Training Summary:")
    print(f"  Epochs trained:     {meta['total_epochs']}")
    print(f"  Best val loss:      {meta['best_val_loss']:.4f}")
    print(f"  Best val Top-10:    {meta['best_val_acc_top10']*100:.1f}%")
    print(f"  Parameters:         {sum(p.numel() for p in model.parameters()):,}")
