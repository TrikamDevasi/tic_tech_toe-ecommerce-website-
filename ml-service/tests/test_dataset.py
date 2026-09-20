"""
Unit Tests for Data Pipeline & Dataset Utilities
"""
import pytest
from datetime import datetime, timedelta
from data.dataset import ItemVocabulary, SessionDataset, collate_fn
from data.preprocessing import build_sessions, create_sequences, temporal_split
import config


def test_item_vocabulary():
    vocab = ItemVocabulary()
    products = [101, 102, 103, "custom_item"]
    vocab.build(products)

    # Check special tokens
    assert vocab.to_idx("<PAD>") == config.PAD_IDX
    assert vocab.to_idx("<UNK>") == config.UNK_IDX
    assert vocab.to_id(config.PAD_IDX) == "<PAD>"
    assert vocab.to_id(config.UNK_IDX) == "<UNK>"

    # Check valid items
    assert vocab.to_idx(101) >= config.SPECIAL_TOKENS
    assert vocab.to_id(vocab.to_idx(101)) == 101

    # Check unknown item mapping
    assert vocab.to_idx("non_existent_item_xyz") == config.UNK_IDX

    # Check size
    assert len(vocab) == len(products) + config.SPECIAL_TOKENS
    assert vocab.num_items == len(products)


def test_session_preprocessing_and_sequences():
    now = datetime(2026, 1, 1, 10, 0, 0)
    mock_events = [
        {"sessionId": "s1", "productId": 1, "timestamp": now},
        {"sessionId": "s1", "productId": 2, "timestamp": now + timedelta(minutes=1)},
        {"sessionId": "s1", "productId": 3, "timestamp": now + timedelta(minutes=2)},
        {"sessionId": "s2", "productId": 99, "timestamp": now},  # Single-item session (should be filtered)
    ]

    sessions = build_sessions(mock_events)
    assert len(sessions) == 2
    assert "s1" in sessions and "s2" in sessions

    # Filtering happens during sequence generation (min_len=2)
    seqs = create_sequences(sessions, min_len=2)
    assert len(seqs) == 1
    assert seqs[0]["product_ids"] == [1, 2, 3]


def test_temporal_split():
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    sequences = [
        {"session_id": f"s_{i}", "product_ids": [1, 2], "end_time": base_time + timedelta(hours=i)}
        for i in range(100)
    ]

    train, val, test = temporal_split(sequences, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Verify chronological ordering across splits
    assert train[-1]["end_time"] <= val[0]["end_time"]
    assert val[-1]["end_time"] <= test[0]["end_time"]


def test_session_dataset_and_collate():
    vocab = ItemVocabulary()
    vocab.build([10, 20, 30, 40])

    sequences = [
        {"product_ids": [10, 20, 30]},
        {"product_ids": [20, 30, 40, 10]},
    ]

    dataset = SessionDataset(sequences, vocab, max_seq_len=5)
    assert len(dataset) == 5

    # Verify sample structure: dynamic padding is applied by collate_fn
    inputs, target = dataset[1]
    assert len(inputs) == 2  # unpadded prefix [10, 20]
    assert target == vocab.to_idx(30)  # item 30 is target

    # Collate pads to max length within batch (here max(1, 2) = 2)
    batch = [dataset[0], dataset[1]]
    batch_inputs, batch_targets = collate_fn(batch)
    assert batch_inputs.shape == (2, 2)
    assert batch_targets.shape == (2,)
