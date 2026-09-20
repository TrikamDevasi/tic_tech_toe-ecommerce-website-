"""
PriceIQ ML Service — PyTorch Dataset & Vocabulary
Handles product-ID-to-index mapping, sequence padding, and DataLoader collation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from torch.utils.data import Dataset
import config


class ItemVocabulary:
    """
    Bidirectional mapping between canonical product IDs (ints) and model indices.

    Index layout:
        0 = PAD (padding token)
        1 = UNK (unknown / unseen product)
        2 … N+1 = actual product IDs
    """

    def __init__(self):
        self.product_to_idx = {}  # canonical product_id (int) -> model index (int)
        self.idx_to_product = {}  # model index (int) -> canonical product_id (int)
        self.pad_idx = config.PAD_IDX
        self.unk_idx = config.UNK_IDX

    def build(self, product_ids):
        """
        Build vocabulary from a list of canonical product IDs (ints).
        Assigns deterministic indices sorted by product ID.
        """
        self.product_to_idx = {}
        self.idx_to_product = {}
        sorted_ids = sorted(set(product_ids), key=lambda x: str(x))
        for i, pid in enumerate(sorted_ids, start=config.SPECIAL_TOKENS):
            self.product_to_idx[pid] = i
            self.idx_to_product[i] = pid

    def to_idx(self, product_id):
        """Convert product ID (int or str) to model index."""
        if product_id == "<PAD>":
            return self.pad_idx
        if product_id == "<UNK>":
            return self.unk_idx
        try:
            canon = int(product_id)
        except (ValueError, TypeError):
            canon = str(product_id)
        return self.product_to_idx.get(canon, self.unk_idx)

    def to_id(self, idx):
        """Convert model index to product ID."""
        if idx == self.pad_idx:
            return "<PAD>"
        if idx == self.unk_idx:
            return "<UNK>"
        return self.idx_to_product.get(idx, None)

    def encode(self, product_id):
        """Convert canonical product_id (int) to model index."""
        return self.to_idx(product_id)

    def decode(self, idx):
        """Convert model index to canonical product_id (int). Returns None for special tokens."""
        if idx in (self.pad_idx, self.unk_idx):
            return None
        return self.idx_to_product.get(idx, None)

    def encode_sequence(self, product_ids):
        """Encode a list of canonical product IDs to model indices."""
        return [self.to_idx(pid) for pid in product_ids]

    def all_product_ids(self):
        """Return list of all registered product IDs."""
        return list(self.product_to_idx.keys())

    def __len__(self):
        """Total vocabulary size including special tokens."""
        return len(self.product_to_idx) + config.SPECIAL_TOKENS

    @property
    def num_items(self):
        """Number of actual product items (excluding special tokens)."""
        return len(self.product_to_idx)

    def save(self, path=None):
        """Save vocabulary mapping to JSON file."""
        if path is None:
            os.makedirs(config.MAPPINGS_DIR, exist_ok=True)
            path = os.path.join(config.MAPPINGS_DIR, "item_vocab.json")
        data = {
            "product_to_idx": {str(k): v for k, v in self.product_to_idx.items()},
            "idx_to_product": {str(k): v for k, v in self.idx_to_product.items()},
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Vocabulary saved to {path} ({len(self)} total, {self.num_items} items)")

    def load(self, path=None):
        """Load vocabulary mapping from JSON file into self."""
        if path is None:
            path = os.path.join(config.MAPPINGS_DIR, "item_vocab.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.product_to_idx = {}
        for k, v in data["product_to_idx"].items():
            try:
                canon_k = int(k)
            except (ValueError, TypeError):
                canon_k = str(k)
            self.product_to_idx[canon_k] = v
        self.idx_to_product = {}
        for k, v in data["idx_to_product"].items():
            idx_int = int(k)
            try:
                canon_v = int(v)
            except (ValueError, TypeError):
                canon_v = str(v)
            self.idx_to_product[idx_int] = canon_v
        self.pad_idx = data.get("pad_idx", config.PAD_IDX)
        self.unk_idx = data.get("unk_idx", config.UNK_IDX)
        return self

    @classmethod
    def from_file(cls, path=None):
        """Classmethod constructor loading from file."""
        inst = cls()
        return inst.load(path)


class SessionDataset(Dataset):
    """
    PyTorch Dataset for session-based recommendation.

    Each sample is a (input_sequence, target_item) pair.
    From a session [A, B, C, D], we generate:
        ([A], B), ([A,B], C), ([A,B,C], D)

    Sequences are encoded using ItemVocabulary indices.
    """

    def __init__(self, sequences, vocab, max_seq_len=None):
        """
        Args:
            sequences: list of dicts with "product_ids" key (list of canonical ints)
            vocab: ItemVocabulary instance
            max_seq_len: maximum input sequence length (truncates from left)
        """
        self.vocab = vocab
        self.max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        self.samples = []  # list of (input_indices, target_index)

        for seq_data in sequences:
            product_ids = seq_data["product_ids"]
            encoded = vocab.encode_sequence(product_ids)

            # Generate all valid (prefix, target) pairs
            for i in range(1, len(encoded)):
                input_seq = encoded[:i]
                target = encoded[i]

                # Skip if target is UNK
                if target == vocab.unk_idx:
                    continue

                # Truncate input from left if too long
                if len(input_seq) > self.max_seq_len:
                    input_seq = input_seq[-self.max_seq_len:]

                self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function that pads variable-length input sequences.
    Returns (padded_inputs, targets) tensors.
    """
    inputs, targets = zip(*batch)
    max_len = max(inp.size(0) for inp in inputs)

    padded_inputs = torch.full((len(inputs), max_len), config.PAD_IDX, dtype=torch.long)
    for i, inp in enumerate(inputs):
        padded_inputs[i, max_len - inp.size(0):] = inp  # right-align (pad on left)

    targets = torch.stack(targets)
    return padded_inputs, targets
