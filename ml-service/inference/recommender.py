"""
PriceIQ ML Service — Hybrid Recommender (Inference Engine)

Implements the cold-start fallback ladder:
  - 0 interactions: Popularity (non-personalized)
  - 1-2 interactions: TF-IDF Content Similarity on last viewed product
  - 3+ interactions: GRU4Rec Session-Aware Neural Network

Outputs canonical product IDs (integers/strings matching the Product schema)
enriched with metadata (name, category, price, image) for seamless frontend/backend display.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from typing import List, Dict, Any, Optional
import torch

import config
from database import get_db
from data.dataset import ItemVocabulary
from data.preprocessing import load_product_catalog
from models.gru4rec import GRU4Rec
from models.baselines import PopularityRecommender, ContentRecommender

logger = logging.getLogger("priceiq.inference")


class HybridRecommender:
    """
    Production-ready hybrid recommender combining GRU4Rec with cold-start fallbacks.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[GRU4Rec] = None
        self.vocab: Optional[ItemVocabulary] = None
        self.popularity_model = PopularityRecommender()
        self.content_model = ContentRecommender()
        self.catalog: Dict[Any, Dict[str, Any]] = {}
        self.is_ready = False
        self.train_metadata: Dict[str, Any] = {}
        self.stats = {
            "total_recommendation_requests": 0,
            "gru4rec_served": 0,
            "content_tfidf_served": 0,
            "popularity_served": 0,
            "errors": 0,
        }

    def load(self, force_retrain: bool = False):
        """
        Load models and metadata from artifacts. If artifacts are missing,
        triggers dataset building and training pipeline.
        """
        db = get_db()
        logger.info("Loading product catalog from MongoDB...")
        self.catalog = load_product_catalog(db)
        logger.info(f"Loaded {len(self.catalog)} products into catalog.")

        model_path = os.path.join(config.MODEL_DIR, "gru4rec_best.pt")
        vocab_path = os.path.join(config.MAPPINGS_DIR, "item_vocab.json")
        train_data_path = os.path.join(config.ARTIFACTS_DIR, "data", "train_sequences.json")

        need_train = force_retrain or not (os.path.exists(model_path) and os.path.exists(vocab_path))

        if need_train:
            logger.info("Artifacts missing or retraining requested. Running full training pipeline...")
            from training.train import run_training
            model, vocab, meta, _ = run_training()
            self.model = model
            self.vocab = vocab
            self.train_metadata = meta
        else:
            logger.info(f"Loading vocabulary from {vocab_path}...")
            self.vocab = ItemVocabulary()
            self.vocab.load(vocab_path)

            logger.info(f"Loading GRU4Rec checkpoint from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model = GRU4Rec(vocab_size=len(self.vocab)).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            meta_path = os.path.join(config.MODEL_DIR, "train_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.train_metadata = json.load(f)

        # Train baselines
        logger.info("Training ContentRecommender (TF-IDF) from catalog...")
        if self.catalog:
            self.content_model.train_from_catalog(self.catalog)

        logger.info("Training PopularityRecommender...")
        if os.path.exists(train_data_path):
            with open(train_data_path, "r") as f:
                train_seqs = json.load(f)
            self.popularity_model.train(train_seqs)
        else:
            # Fallback to catalog order
            mock_seqs = [{"product_ids": list(self.catalog.keys())}]
            self.popularity_model.train(mock_seqs)

        self.is_ready = True
        logger.info("HybridRecommender initialization complete. Ready for inference.")

    def _normalize_id(self, raw_id: Any) -> Any:
        """Convert input product ID to canonical format (int if numeric, else str)."""
        if raw_id is None:
            return None
        try:
            return int(raw_id)
        except (ValueError, TypeError):
            return str(raw_id)

    def _predict_gru4rec(
        self,
        clean_history: List[Any],
        top_k: int = 5,
        exclude_history: bool = True,
    ) -> List[tuple]:
        """
        Run GRU4Rec inference. Returns list of (canonical_pid, softmax_score).
        """
        indices = [self.vocab.to_idx(pid) for pid in clean_history]
        if len(indices) > config.MAX_SEQ_LEN:
            indices = indices[-config.MAX_SEQ_LEN:]
        pad_len = config.MAX_SEQ_LEN - len(indices)
        padded = [config.PAD_IDX] * pad_len + indices

        input_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)[0]
            # Mask special tokens
            logits[config.PAD_IDX] = -float("inf")
            logits[config.UNK_IDX] = -float("inf")

            if exclude_history:
                for pid in clean_history:
                    idx = self.vocab.to_idx(pid)
                    if idx >= config.SPECIAL_TOKENS:
                        logits[idx] = -float("inf")

            probs = torch.softmax(logits, dim=0)
            k = min(top_k, self.vocab.num_items)
            top_res = torch.topk(probs, k=k)
            top_indices = top_res.indices.cpu().tolist()
            top_scores = top_res.values.cpu().tolist()

        results = []
        for idx, score in zip(top_indices, top_scores):
            if idx >= config.SPECIAL_TOKENS:
                results.append((self.vocab.to_id(idx), float(score)))
        return results

    def recommend_session(
        self,
        session_history: List[Any],
        top_k: int = 5,
        exclude_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Recommend items for a session using the hybrid cold-start fallback ladder.

        Args:
            session_history: List of viewed product IDs in chronological order.
            top_k: Number of recommendations desired.
            exclude_history: Whether to filter out already viewed products.

        Returns:
            Dict containing recommended items, strategy used, and latency.
        """
        if not self.is_ready:
            self.load()

        start_time = time.perf_counter()
        self.stats["total_recommendation_requests"] += 1

        # Clean and normalize history
        clean_history = [
            self._normalize_id(pid)
            for pid in (session_history or [])
            if pid is not None
        ]
        history_len = len(clean_history)

        raw_recs = []
        strategy = ""

        # Step 1: Cold-start Ladder
        if history_len == 0:
            # 0 interactions → Popularity
            strategy = "popularity_cold_start"
            self.stats["popularity_served"] += 1
            raw_recs = self.popularity_model.recommend([], top_k=top_k)

        elif history_len < config.COLD_START_THRESHOLD:
            # 1-2 interactions → TF-IDF Content similarity
            strategy = "content_tfidf"
            self.stats["content_tfidf_served"] += 1
            raw_recs = self.content_model.recommend(clean_history, top_k=top_k)
            # If content model didn't return enough, backfill with popularity
            if len(raw_recs) < top_k:
                seen_pids = set(clean_history) | {pid for pid, _ in raw_recs}
                pop_recs = self.popularity_model.recommend(list(seen_pids), top_k=top_k - len(raw_recs))
                raw_recs.extend(pop_recs)

        else:
            # 3+ interactions → GRU4Rec
            strategy = "gru4rec_neural"
            self.stats["gru4rec_served"] += 1
            try:
                raw_recs = self._predict_gru4rec(clean_history, top_k=top_k, exclude_history=exclude_history)
            except Exception as e:
                logger.warning(f"GRU4Rec inference error: {e}. Falling back to content recommender.")
                self.stats["errors"] += 1
                strategy = "content_tfidf_fallback"
                raw_recs = self.content_model.recommend(clean_history, top_k=top_k)

            # Backfill with popularity if needed
            if len(raw_recs) < top_k:
                seen_pids = set(clean_history) | {pid for pid, _ in raw_recs}
                pop_recs = self.popularity_model.recommend(list(seen_pids), top_k=top_k - len(raw_recs))
                raw_recs.extend(pop_recs)

        # Step 2: Enrich with catalog product metadata
        items = []
        for pid, score in raw_recs[:top_k]:
            cat_item = self.catalog.get(pid, {})
            items.append({
                "productId": pid,
                "score": round(float(score), 4),
                "name": cat_item.get("name", f"Product #{pid}"),
                "category": cat_item.get("category", "General"),
                "price": cat_item.get("price", 0),
                "image": cat_item.get("image", ""),
                "rating": cat_item.get("rating", 4.0),
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "strategy": strategy,
            "session_length": history_len,
            "latency_ms": round(latency_ms, 2),
            "recommendations": items,
        }

    def recommend_similar_product(self, product_id: Any, top_k: int = 5) -> Dict[str, Any]:
        """
        Recommend items similar to a specific product using Content (TF-IDF).
        Used by the product detail page fallback.
        """
        if not self.is_ready:
            self.load()

        start_time = time.perf_counter()
        canonical_pid = self._normalize_id(product_id)

        raw_recs = self.content_model.recommend_by_product(canonical_pid, top_k=top_k)

        # If TF-IDF returns insufficient results, backfill with popularity
        if len(raw_recs) < top_k:
            exclude = [canonical_pid] + [pid for pid, _ in raw_recs]
            pop_recs = self.popularity_model.recommend(exclude, top_k=top_k - len(raw_recs))
            raw_recs.extend(pop_recs)

        items = []
        for pid, score in raw_recs[:top_k]:
            cat_item = self.catalog.get(pid, {})
            items.append({
                "productId": pid,
                "score": round(float(score), 4),
                "name": cat_item.get("name", f"Product #{pid}"),
                "category": cat_item.get("category", "General"),
                "price": cat_item.get("price", 0),
                "image": cat_item.get("image", ""),
                "rating": cat_item.get("rating", 4.0),
            })

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "strategy": "content_tfidf_item_similarity",
            "seed_product_id": canonical_pid,
            "latency_ms": round(latency_ms, 2),
            "recommendations": items,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return system status and operational statistics."""
        return {
            "status": "ready" if self.is_ready else "initializing",
            "vocab_size": len(self.vocab) if self.vocab else 0,
            "num_catalog_items": len(self.catalog),
            "model_type": "GRU4Rec + Baselines Hybrid",
            "device": str(self.device),
            "stats": self.stats,
            "train_metadata": self.train_metadata,
        }
