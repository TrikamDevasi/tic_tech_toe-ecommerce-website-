"""
Unit Tests for GRU4Rec Model Architecture
"""
import pytest
import torch
import torch.nn as nn
from models.gru4rec import GRU4Rec
from models.baselines import PopularityRecommender, RecentlyViewedRecommender


def test_gru4rec_forward_shape():
    vocab_size = 50
    embedding_dim = 32
    hidden_dim = 64
    batch_size = 4
    seq_len = 10

    model = GRU4Rec(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.1,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(dummy_input)

    # Output shape must be (batch_size, vocab_size)
    assert output.shape == (batch_size, vocab_size)


def test_gru4rec_gradient_flow():
    vocab_size = 20
    model = GRU4Rec(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    inputs = torch.randint(1, vocab_size, (2, 5))
    targets = torch.randint(1, vocab_size, (2,))

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    # Check that embedding and GRU weights have gradients
    assert model.embedding.weight.grad is not None
    assert model.output_layer.weight.grad is not None
    assert not torch.isnan(loss)


def test_baselines():
    # Popularity
    pop = PopularityRecommender()
    seqs = [{"product_ids": [1, 2, 2, 3, 2, 1]}]
    pop.train(seqs)
    recs = pop.recommend(session_history=[2], top_k=2)
    # Item 2 is excluded, next most popular is 1, then 3
    rec_pids = [pid for pid, _ in recs]
    assert rec_pids == [1, 3]

    # Recently Viewed
    rec = RecentlyViewedRecommender()
    recent_recs = rec.recommend(session_history=[5, 12, 5, 8], top_k=2)
    # Most recently viewed unique items in reverse: 8, 5
    assert [pid for pid, _ in recent_recs] == [8, 5]


def test_predict_with_confidence():
    vocab_size = 30
    model = GRU4Rec(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32, num_layers=2)
    model.eval()

    dummy_input = torch.tensor([[0, 0, 2, 5, 8]], dtype=torch.long)
    res = model.predict_with_confidence(dummy_input, top_k=5, exclude_indices={2, 5, 8}, confidence_threshold=0.05)

    assert "top_indices" in res
    assert "top_probs" in res
    assert "max_prob" in res
    assert "entropy" in res
    assert "is_low_confidence" in res
    assert len(res["top_indices"]) == 5
    assert res["max_prob"] >= 0.0
    # Excluded indices must not be in top_indices
    assert 2 not in res["top_indices"]
    assert 5 not in res["top_indices"]
    assert 8 not in res["top_indices"]
    # Special tokens (0: PAD, 1: UNK) must not be in top_indices
    assert 0 not in res["top_indices"]
    assert 1 not in res["top_indices"]
