"""
PriceIQ ML Service — Baseline Recommenders

Three non-neural baselines for honest comparison against GRU4Rec:
  1. PopularityRecommender — ranks by global interaction count
  2. RecentlyViewedRecommender — returns last-N distinct viewed items reversed
  3. ContentRecommender — TF-IDF cosine similarity on product text
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict


class PopularityRecommender:
    """Recommends globally most popular items by interaction count."""

    def __init__(self):
        self.item_counts = Counter()
        self.ranked_items = []
        self.trained = False

    def train(self, sequences):
        """
        Train from sequences (list of dicts with 'product_ids').
        Counts how often each product appears across all sessions.
        """
        self.item_counts = Counter()
        for seq in sequences:
            for pid in seq["product_ids"]:
                self.item_counts[pid] += 1
        self.ranked_items = [pid for pid, _ in self.item_counts.most_common()]
        self.trained = True

    def recommend(self, session_history, top_k=5):
        """
        Return top_k most popular items, excluding those already in session_history.
        Args:
            session_history: list of canonical product IDs (ints)
            top_k: number of recommendations
        Returns:
            list of (product_id, score) tuples
        """
        if not self.trained:
            return []
        excluded = set(session_history) if session_history else set()
        results = []
        for pid in self.ranked_items:
            if pid not in excluded:
                results.append((pid, float(self.item_counts[pid])))
                if len(results) >= top_k:
                    break
        return results


class RecentlyViewedRecommender:
    """Returns the most recently viewed items in reverse order (trivial baseline)."""

    def __init__(self):
        self.trained = True  # no training needed

    def train(self, sequences):
        """No-op: this baseline doesn't learn anything."""
        pass

    def recommend(self, session_history, top_k=5):
        """
        Return last top_k distinct items from session_history, reversed.
        Args:
            session_history: list of canonical product IDs (ints)
            top_k: number of recommendations
        Returns:
            list of (product_id, score) tuples
        """
        if not session_history:
            return []
        # Reverse and deduplicate while maintaining order
        seen = set()
        results = []
        for pid in reversed(session_history):
            if pid not in seen:
                seen.add(pid)
                results.append((pid, float(len(session_history) - len(results))))
                if len(results) >= top_k:
                    break
        return results


class ContentRecommender:
    """
    TF-IDF content-based recommender.
    Computes cosine similarity on product text (name + category + description).
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.product_ids = []   # list of canonical product IDs (ints)
        self.id_to_idx = {}     # canonical product_id -> TF-IDF matrix row index
        self.trained = False

    def train_from_catalog(self, catalog):
        """
        Train from product catalog dict: {canonical_id: {name, category, description}}.
        """
        if len(catalog) < 2:
            return

        self.product_ids = []
        corpus = []
        for pid in sorted(catalog.keys()):
            info = catalog[pid]
            self.product_ids.append(pid)
            text = f"{info.get('name', '')} {info.get('category', '')} {info.get('description', '')}"
            corpus.append(text)

        self.id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.trained = True

    def train(self, sequences):
        """No-op if already trained from catalog. Called for interface compatibility."""
        pass

    def recommend(self, session_history, top_k=5):
        """
        Recommend items similar to the last viewed product.
        Args:
            session_history: list of canonical product IDs (ints)
            top_k: number of recommendations
        Returns:
            list of (product_id, score) tuples
        """
        if not self.trained or not session_history:
            return []

        # Use last viewed product as seed
        seed_pid = session_history[-1]
        if seed_pid not in self.id_to_idx:
            return []

        idx = self.id_to_idx[seed_pid]
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx] = -1  # exclude self

        # Exclude already-viewed
        excluded = set(session_history)
        top_indices = np.argsort(scores)[::-1]
        results = []
        for i in top_indices:
            pid = self.product_ids[i]
            if pid not in excluded and scores[i] > 0:
                results.append((pid, float(scores[i])))
                if len(results) >= top_k:
                    break

        return results

    def recommend_by_product(self, product_id, top_k=5):
        """
        Recommend items similar to a specific product (used by FastAPI endpoint).
        Returns list of (product_id, score) tuples.
        """
        return self.recommend([product_id], top_k)


class ItemTransitionRecommender:
    """
    First-Order Markov Chain recommender.
    Computes empirical transition probabilities P(next_item | last_item) from training sequences.
    """

    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.trained = False

    def train(self, sequences):
        self.transitions = defaultdict(Counter)
        for seq in sequences:
            pids = seq.get("product_ids", [])
            for i in range(len(pids) - 1):
                self.transitions[pids[i]][pids[i + 1]] += 1
        self.trained = True

    def recommend(self, session_history, top_k=5):
        if not self.trained or not session_history:
            return []
        last_pid = session_history[-1]
        next_counts = self.transitions.get(last_pid, Counter())
        if not next_counts:
            return []
        total = sum(next_counts.values())
        excluded = set(session_history)
        results = []
        for pid, count in next_counts.most_common():
            if pid not in excluded:
                results.append((pid, float(count / total)))
                if len(results) >= top_k:
                    break
        return results

    def get_score_dict(self, session_history):
        """Return dict of {pid: normalized_prob} for all candidate items."""
        if not self.trained or not session_history:
            return {}
        last_pid = session_history[-1]
        next_counts = self.transitions.get(last_pid, Counter())
        if not next_counts:
            return {}
        total = sum(next_counts.values())
        return {pid: count / total for pid, count in next_counts.items()}

