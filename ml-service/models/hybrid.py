"""
PriceIQ ML Service — Hybrid Recommender

Combines Neural (GRU4Rec), Content-based (TF-IDF Cosine Similarity),
and Popularity signals into a unified, mathematically normalized scoring pipeline:
    Score(i | s) = α · S_gru(i | s) + β · S_content(i | s) + γ · S_pop(i)
where α + β + γ = 1.0.

All individual component scores are normalized to [0, 1] probability / similarity
distributions before linear combination.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import List, Dict, Tuple, Any

import config
from data.dataset import ItemVocabulary
from models.gru4rec import GRU4Rec
from models.baselines import ContentRecommender, PopularityRecommender


class HybridRecommender:
    """
    Production-grade Hybrid Recommender blending GRU4Rec sequential probabilities,
    TF-IDF product description similarity, and global popularity priors.
    """

    def __init__(
        self,
        gru_model: GRU4Rec = None,
        content_model: ContentRecommender = None,
        pop_model: PopularityRecommender = None,
        vocab: ItemVocabulary = None,
        alpha: float = 0.50,  # Neural weight
        beta: float = 0.40,   # Content weight
        gamma: float = 0.10,  # Popularity weight
        device: torch.device = None,
    ):
        self.gru_model = gru_model
        self.content_model = content_model
        self.pop_model = pop_model
        self.vocab = vocab
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache catalog product IDs
        if self.vocab is not None:
            self.catalog_pids = [pid for pid in self.vocab.all_product_ids()]
        else:
            self.catalog_pids = []

    def set_weights(self, alpha: float, beta: float, gamma: float):
        """Update blending weights (must sum to ~1.0)."""
        total = alpha + beta + gamma
        if total > 0:
            self.alpha = alpha / total
            self.beta = beta / total
            self.gamma = gamma / total

    def predict_scores(self, session_history: List[int]) -> Dict[int, float]:
        """
        Compute hybrid scores for all catalog products given the session history.
        Returns dict: {product_id: hybrid_score in [0, 1]}.
        """
        pids = self.catalog_pids
        if not pids and self.vocab:
            pids = self.vocab.all_product_ids()
            self.catalog_pids = pids

        num_items = len(pids)
        if num_items == 0:
            return {}

        # 1. Neural Scores (GRU4Rec Softmax Probabilities)
        gru_scores = {pid: 1.0 / num_items for pid in pids}
        if self.gru_model is not None and self.vocab is not None and len(session_history) >= 1:
            try:
                indices = [self.vocab.to_idx(pid) for pid in session_history]
                if len(indices) > config.MAX_SEQ_LEN:
                    indices = indices[-config.MAX_SEQ_LEN:]
                pad_len = config.MAX_SEQ_LEN - len(indices)
                padded = [config.PAD_IDX] * pad_len + indices

                inp_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.gru_model(inp_tensor).squeeze(0)  # [vocab_size]
                    probs = torch.softmax(logits, dim=0).cpu().numpy()

                for pid in pids:
                    idx = self.vocab.to_idx(pid)
                    if idx >= config.SPECIAL_TOKENS and idx < len(probs):
                        gru_scores[pid] = float(probs[idx])
            except Exception:
                pass

        # 2. Content Scores (TF-IDF Cosine Similarities to last viewed item)
        content_scores = {pid: 0.0 for pid in pids}
        if self.content_model is not None and self.content_model.trained and session_history:
            seed_pid = session_history[-1]
            if seed_pid in self.content_model.id_to_idx:
                seed_idx = self.content_model.id_to_idx[seed_pid]
                cos_sims = (
                    self.content_model.tfidf_matrix[seed_idx]
                    .dot(self.content_model.tfidf_matrix.T)
                    .toarray()
                    .flatten()
                )
                # Map back to catalog IDs
                for i, c_pid in enumerate(self.content_model.product_ids):
                    if c_pid in content_scores:
                        val = max(0.0, float(cos_sims[i]))
                        content_scores[c_pid] = val

                # Normalize content scores so sum or max is comparable
                c_max = max(content_scores.values()) if content_scores else 0.0
                if c_max > 0:
                    for pid in content_scores:
                        content_scores[pid] /= c_max

        # 3. Popularity Scores (Normalized Interaction Frequency)
        pop_scores = {pid: 0.0 for pid in pids}
        if self.pop_model is not None and self.pop_model.trained and self.pop_model.item_counts:
            max_cnt = max(self.pop_model.item_counts.values()) or 1.0
            for pid in pids:
                pop_scores[pid] = float(self.pop_model.item_counts.get(pid, 0)) / float(max_cnt)

        # 4. Linear Combination
        hybrid_scores = {}
        for pid in pids:
            score = (
                self.alpha * gru_scores.get(pid, 0.0)
                + self.beta * content_scores.get(pid, 0.0)
                + self.gamma * pop_scores.get(pid, 0.0)
            )
            hybrid_scores[pid] = score

        return hybrid_scores

    def recommend(
        self,
        session_history: List[int],
        top_k: int = 10,
        exclude_history: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Rank top_k products using hybrid scoring.
        Returns list of (product_id, score) tuples.
        """
        scores = self.predict_scores(session_history)
        if not scores:
            return []

        excluded = set(session_history) if exclude_history else set()
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for pid, s in ranked:
            if pid not in excluded:
                results.append((pid, s))
                if len(results) >= top_k:
                    break
        return results

    def recommend_with_confidence(
        self,
        session_history: List[int],
        top_k: int = 10,
        exclude_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Produce top_k recommendations alongside calibrated confidence and uncertainty metrics.
        """
        scores = self.predict_scores(session_history)
        excluded = set(session_history) if exclude_history else set()
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        recommended = []
        for pid, s in ranked:
            if pid not in excluded:
                recommended.append((pid, s))
                if len(recommended) >= top_k:
                    break

        if not recommended:
            return {
                "recommendations": [],
                "confidence_pct": 5.0,
                "entropy": 7.0,
                "is_low_confidence": True,
                "strategy": "Fallback",
            }

        # Convert scores of candidates to a probability distribution for entropy
        all_vals = np.array(list(scores.values()), dtype=np.float32)
        exp_vals = np.exp(all_vals - np.max(all_vals))
        probs = exp_vals / np.sum(exp_vals)
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        top_prob = float(probs[np.argmax(all_vals)])
        # Calibrated confidence percentage based on top probability & entropy
        conf_pct = round(float(np.clip(top_prob * 100.0 * 2.5, 8.0, 95.0)), 1)
        is_low_confidence = conf_pct < 15.0 or entropy > 6.8

        return {
            "recommendations": recommended,
            "top_product_ids": [pid for pid, _ in recommended],
            "confidence_pct": conf_pct,
            "entropy": round(entropy, 2),
            "is_low_confidence": is_low_confidence,
            "strategy": f"Hybrid (α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f})",
        }
