"""
PriceIQ ML Service — GRU4Rec Model

Session-based recommendation using Gated Recurrent Units.
Architecture: Embedding → GRU → Dropout → Linear → Item logits

Reference: Hidasi et al., "Session-based Recommendations with Recurrent
           Neural Networks" (ICLR 2016)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import config


class GRU4Rec(nn.Module):
    """
    GRU-based session recommendation model.

    Input: batch of padded item-index sequences  (B, T)
    Output: logits over all items in vocabulary   (B, vocab_size)

    Index conventions:
        0 = PAD (padding_idx for embedding)
        1 = UNK
        2 … vocab_size-1 = actual items

    The output logits have dimension vocab_size, and CrossEntropyLoss targets
    should be item indices in [0, vocab_size-1].
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=None,
        hidden_dim=None,
        num_layers=None,
        dropout=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim or config.EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else config.NUM_GRU_LAYERS
        self.dropout_rate = dropout if dropout is not None else config.DROPOUT

        # Layers
        self.embedding = nn.Embedding(
            vocab_size, self.embedding_dim, padding_idx=config.PAD_IDX
        )
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for embedding and linear layers."""
        nn.init.xavier_uniform_(self.embedding.weight.data[1:])  # skip PAD
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, T) tensor of item indices (left-padded)

        Returns:
            logits: (B, vocab_size) raw scores for next-item prediction
        """
        # x shape: (B, T)
        emb = self.embedding(x)        # (B, T, emb_dim)
        emb = self.dropout(emb)
        gru_out, _ = self.gru(emb)     # (B, T, hidden_dim)

        # Use the last time-step output for prediction with LayerNorm
        last_hidden = gru_out[:, -1, :]  # (B, hidden_dim)
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        logits = self.output_layer(last_hidden)  # (B, vocab_size)

        return logits

    def predict_with_confidence(
        self,
        x,
        top_k=5,
        exclude_indices=None,
        temperature=1.0,
        confidence_threshold=None,
    ):
        """
        Run inference and compute calibrated confidence metrics.

        Returns:
            top_indices: List[int] of predicted token indices
            top_probs: List[float] of calibrated probabilities
            max_prob: float, highest predicted probability
            entropy: float, predictive distribution entropy
            is_low_confidence: bool, True if max_prob < confidence_threshold
        """
        threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)[0] / max(temperature, 1e-4)

            # Mask special tokens
            logits[config.PAD_IDX] = -float("inf")
            logits[config.UNK_IDX] = -float("inf")

            if exclude_indices:
                for idx in exclude_indices:
                    if idx >= config.SPECIAL_TOKENS and idx < self.vocab_size:
                        logits[idx] = -float("inf")

            probs = torch.softmax(logits, dim=0)

            # Compute entropy over valid items
            valid_probs = probs[config.SPECIAL_TOKENS:]
            valid_probs = valid_probs[valid_probs > 0]
            entropy = float(-(valid_probs * torch.log(valid_probs + 1e-9)).sum().item())

            k = min(top_k, max(1, self.vocab_size - config.SPECIAL_TOKENS))
            top_res = torch.topk(probs, k=k)
            top_indices = top_res.indices.cpu().tolist()
            top_probs = [float(p) for p in top_res.values.cpu().tolist()]

            max_prob = top_probs[0] if top_probs else 0.0
            is_low_confidence = bool(max_prob < threshold)

        return {
            "top_indices": top_indices,
            "top_probs": top_probs,
            "max_prob": round(max_prob, 4),
            "entropy": round(entropy, 4),
            "is_low_confidence": is_low_confidence,
        }

    def get_config(self):
        """Return model configuration as dict."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }
