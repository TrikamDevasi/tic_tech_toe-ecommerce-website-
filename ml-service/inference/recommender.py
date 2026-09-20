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
from collections import deque
from typing import List, Dict, Any, Optional
import torch

import numpy as np

import config
from database import get_db
from data.dataset import ItemVocabulary
from data.preprocessing import load_product_catalog
from models.gru4rec import GRU4Rec
from models.baselines import PopularityRecommender, ContentRecommender

logger = logging.getLogger("priceiq.inference")


class HybridRecommender:
    """
    Production-ready hybrid recommender combining GRU4Rec with cold-start fallbacks
    and comprehensive operational telemetry.
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
        
        # Telemetry & Production Monitoring metrics
        self.stats = {
            "total_recommendation_requests": 0,
            "fallbacks_served": 0,
            "unknown_products_encountered": 0,
            "clicks_tracked": 0,
            "strategy_distribution": {
                "popularity_cold_start": 0,
                "content_tfidf": 0,
                "gru4rec_neural": 0,
                "content_tfidf_unknown_fallback": 0,
                "gru4rec_low_conf_fallback": 0,
                "content_tfidf_fallback": 0,
            },
            "errors": 0,
        }
        self.recent_latencies_ms = deque(maxlen=2000)
        self.recent_confidences = deque(maxlen=2000)

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
            m_cfg = checkpoint.get("model_config", {})
            self.model = GRU4Rec(
                vocab_size=m_cfg.get("vocab_size", len(self.vocab)),
                embedding_dim=m_cfg.get("embedding_dim", config.EMBEDDING_DIM),
                hidden_dim=m_cfg.get("hidden_dim", config.HIDDEN_DIM),
                num_layers=m_cfg.get("num_layers", config.NUM_GRU_LAYERS),
                dropout=m_cfg.get("dropout", config.DROPOUT),
            ).to(self.device)
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
            self.popularity_total = float(sum(self.popularity_model.item_counts.values()))
        else:
            # Fallback to catalog order
            mock_seqs = [{"product_ids": list(self.catalog.keys())}]
            self.popularity_model.train(mock_seqs)
            self.popularity_total = float(len(self.catalog))

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
    ) -> Dict[str, Any]:
        """
        Run GRU4Rec inference with confidence estimation.
        """
        indices = [self.vocab.to_idx(pid) for pid in clean_history]
        if len(indices) > config.MAX_SEQ_LEN:
            indices = indices[-config.MAX_SEQ_LEN:]
        pad_len = config.MAX_SEQ_LEN - len(indices)
        padded = [config.PAD_IDX] * pad_len + indices

        input_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
        exclude_indices = set(indices) if exclude_history else set()

        conf_res = self.model.predict_with_confidence(
            input_tensor,
            top_k=top_k,
            exclude_indices=exclude_indices,
            temperature=config.INFERENCE_TEMPERATURE,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
        )

        results = []
        for idx, score in zip(conf_res["top_indices"], conf_res["top_probs"]):
            if idx >= config.SPECIAL_TOKENS:
                results.append((self.vocab.to_id(idx), float(score)))

        return {
            "raw_recs": results,
            "max_prob": conf_res["max_prob"],
            "entropy": conf_res["entropy"],
            "is_low_confidence": conf_res["is_low_confidence"],
        }

    def recommend_session(
        self,
        session_history: List[Any],
        top_k: int = 5,
        exclude_history: bool = True,
        category_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recommend items for a session using the explicit hybrid cold-start fallback ladder:
          - 0 interactions: Popularity (optionally category-filtered)
          - 1-2 interactions: Content TF-IDF on viewed item(s)
          - 3+ interactions: GRU4Rec neural session model
          - Unknown products: Detected and routed to content/category fallback
          - Low confidence (< threshold): Low-confidence fallback flag with safe discovery backfill

        Every fallback is explicitly declared via `is_fallback: bool` and `fallback_reason: str`.
        """
        if not self.is_ready:
            self.load()

        start_time = time.perf_counter()

        # Clean and normalize history
        clean_history = [
            self._normalize_id(pid)
            for pid in (session_history or [])
            if pid is not None
        ]
        history_len = len(clean_history)

        # Detect unknown products in history
        known_history = []
        unknown_history = []
        for pid in clean_history:
            if self.vocab and self.vocab.to_idx(pid) != config.UNK_IDX:
                known_history.append(pid)
            else:
                unknown_history.append(pid)

        if unknown_history:
            self.stats["unknown_products_encountered"] += len(unknown_history)

        raw_recs = []
        strategy = ""
        is_fallback = False
        fallback_reason: Optional[str] = None
        max_prob = 0.0
        entropy = 0.0
        is_low_confidence = False

        # ── Cold-Start Decision Ladder ──────────────────────────────────────────
        if history_len == 0:
            # Ladder Step 1: New user / 0 interactions -> Popularity Fallback
            strategy = "popularity_cold_start"
            is_fallback = True
            fallback_reason = "new_user_zero_history"
            
            if category_preference:
                cat_pids = [
                    pid for pid, meta in self.catalog.items()
                    if meta.get("category", "").lower() == category_preference.lower()
                ]
                raw_recs = self.popularity_model.recommend([], top_k=top_k, allowed_items=set(cat_pids))
            else:
                raw_recs = self.popularity_model.recommend([], top_k=top_k)
                
            total = getattr(self, "popularity_total", 0.0)
            top_count = raw_recs[0][1] if raw_recs else 0.0
            max_prob = round(min(1.0, float(top_count) / total) if total > 0 else 0.0, 4)

        elif len(known_history) == 0:
            # Ladder Step 2: Session contains items, but all are unknown to catalog vocabulary
            strategy = "content_tfidf_unknown_fallback"
            is_fallback = True
            fallback_reason = "all_session_products_unknown_in_catalog"
            
            # Content model fallback
            raw_recs = self.content_model.recommend(clean_history, top_k=top_k)
            if len(raw_recs) < top_k:
                seen = set(clean_history) | {pid for pid, _ in raw_recs}
                raw_recs.extend(self.popularity_model.recommend(list(seen), top_k=top_k - len(raw_recs)))
            max_prob = round(min(1.0, float(raw_recs[0][1])) if raw_recs else 0.05, 4)

        elif len(known_history) < config.COLD_START_THRESHOLD:
            # Ladder Step 3: 1-2 interactions -> TF-IDF Content similarity
            strategy = "content_tfidf"
            is_fallback = True
            fallback_reason = "sparse_history_content_fallback"
            
            raw_recs = self.content_model.recommend(known_history, top_k=top_k)
            if len(raw_recs) < top_k:
                seen_pids = set(clean_history) | {pid for pid, _ in raw_recs}
                pop_recs = self.popularity_model.recommend(list(seen_pids), top_k=top_k - len(raw_recs))
                raw_recs.extend(pop_recs)
            max_prob = round(min(1.0, float(raw_recs[0][1])) if raw_recs else 0.0, 4)

        else:
            # Ladder Step 4: 3+ meaningful interactions -> GRU4Rec Neural session model
            strategy = "gru4rec_neural"
            is_fallback = False
            fallback_reason = None
            try:
                gru_output = self._predict_gru4rec(known_history, top_k=top_k, exclude_history=exclude_history)
                raw_recs = gru_output["raw_recs"]
                max_prob = gru_output["max_prob"]
                entropy = gru_output["entropy"]
                is_low_confidence = gru_output["is_low_confidence"]

                # If neural model is not confident, declare explicit fallback
                if is_low_confidence:
                    strategy = "gru4rec_low_conf_fallback"
                    is_fallback = True
                    fallback_reason = "low_model_confidence"
            except Exception as e:
                logger.warning(f"GRU4Rec inference error: {e}. Falling back to content recommender.")
                self.stats["errors"] += 1
                strategy = "content_tfidf_fallback"
                is_fallback = True
                fallback_reason = f"inference_exception: {str(e)[:50]}"
                raw_recs = self.content_model.recommend(known_history, top_k=top_k)
                max_prob = round(raw_recs[0][1], 4) if raw_recs else 0.05
                is_low_confidence = True

            # Backfill with popularity if needed to ensure top_k returned
            if len(raw_recs) < top_k:
                seen_pids = set(clean_history) | {pid for pid, _ in raw_recs}
                pop_recs = self.popularity_model.recommend(list(seen_pids), top_k=top_k - len(raw_recs))
                raw_recs.extend(pop_recs)

        confidence_pct = round(max_prob * 100, 1)
        if strategy == "popularity_cold_start":
            confidence_msg = (
                "New session (no history): serving globally popular products. "
                "Confidence reflects the item's share of all catalog views."
            )
        elif "content_tfidf" in strategy:
            confidence_msg = (
                f"Content (TF-IDF) similarity {confidence_pct}% for this session. "
                "Score is catalog-similarity, not a calibrated next-item probability."
            )
        elif is_low_confidence:
            confidence_msg = "Low-confidence next-item prediction; recommendations are exploratory for this session."
        else:
            confidence_msg = f"Temperature-calibrated next-item confidence: {confidence_pct}%."

        # Step 2: Enrich with catalog product metadata
        items = []
        for pid, score in raw_recs[:top_k]:
            cat_item = self.catalog.get(pid, {})
            items.append({
                "productId": pid,
                "score": round(float(score), 4),
                "confidencePct": round(float(score) * 100, 1),
                "name": cat_item.get("name", f"Product #{pid}"),
                "category": cat_item.get("category", "General"),
                "price": cat_item.get("price", 0),
                "image": cat_item.get("image", ""),
                "rating": cat_item.get("rating", 4.0),
            })

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Update telemetry counters
        self.stats["total_recommendation_requests"] += 1
        if is_fallback:
            self.stats["fallbacks_served"] += 1
        strat_dist = self.stats["strategy_distribution"]
        strat_dist[strategy] = strat_dist.get(strategy, 0) + 1
        self.recent_latencies_ms.append(latency_ms)
        self.recent_confidences.append(max_prob)

        return {
            "strategy": strategy,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "unknown_products_count": len(unknown_history),
            "session_length": history_len,
            "known_session_length": len(known_history),
            "latency_ms": latency_ms,
            "confidence": max_prob,
            "confidence_pct": confidence_pct,
            "is_low_confidence": is_low_confidence,
            "confidence_message": confidence_msg,
            "entropy": entropy,
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

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            "strategy": "content_tfidf_item_similarity",
            "seed_product_id": canonical_pid,
            "latency_ms": latency_ms,
            "recommendations": items,
        }

    def track_click(self, product_id: Any, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Track user click on a recommended product for production CTR monitoring
        without collecting sensitive personal information.
        """
        self.stats["clicks_tracked"] += 1
        return {
            "status": "recorded",
            "productId": self._normalize_id(product_id),
            "strategy": strategy,
            "total_clicks": self.stats["clicks_tracked"],
        }

    def get_telemetry_metrics(self) -> Dict[str, Any]:
        """Compute operational production telemetry metrics."""
        total_reqs = self.stats["total_recommendation_requests"]
        fallbacks = self.stats["fallbacks_served"]
        fallback_rate = round((fallbacks / total_reqs * 100), 2) if total_reqs > 0 else 0.0

        lat_list = list(self.recent_latencies_ms)
        conf_list = list(self.recent_confidences)

        return {
            "total_requests": total_reqs,
            "fallbacks_served": fallbacks,
            "fallback_rate_pct": fallback_rate,
            "unknown_products_encountered": self.stats["unknown_products_encountered"],
            "clicks_tracked": self.stats["clicks_tracked"],
            "strategy_distribution": self.stats["strategy_distribution"],
            "latency_ms": {
                "mean": round(float(np.mean(lat_list)), 2) if lat_list else 0.0,
                "p50": round(float(np.percentile(lat_list, 50)), 2) if lat_list else 0.0,
                "p95": round(float(np.percentile(lat_list, 95)), 2) if lat_list else 0.0,
                "p99": round(float(np.percentile(lat_list, 99)), 2) if lat_list else 0.0,
            },
            "mean_confidence": round(float(np.mean(conf_list)), 4) if conf_list else 0.0,
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
            "telemetry": self.get_telemetry_metrics(),
            "train_metadata": self.train_metadata,
        }


# Backward-compatible alias
SessionRecommender = HybridRecommender

