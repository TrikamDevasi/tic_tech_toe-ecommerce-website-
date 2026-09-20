# PriceIQ ML Service — Session-Aware Recommendation Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com/)

A production-grade, session-based recommendation service designed for real-time personalization in e-commerce. Built with **GRU4Rec** (Recurrent Neural Networks for Session-Based Recommendation), accompanied by a comprehensive cold-start fallback ladder, rigorous baseline comparisons, and reproducible evaluation benchmarks.

---

## 🏗️ Architecture Overview

```
                         Incoming User Session
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   Session Length Threshold    │
                   └───────────────┬───────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │ 0 items                 │ 1-2 items               │ 3+ items
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│    Popularity    │      │  TF-IDF Content  │      │     GRU4Rec      │
│     Baseline     │      │    Similarity    │      │  (RNN Embedding) │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │ (backfill if needed)    │ (backfill if needed)
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │  Catalog Metadata Enrichment  │
                   │    (id, title, price, img)    │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                       Enriched Recommendations
```

---

## 🚀 Key Features & Defect Resolutions

1. **Native Canonical ID Compatibility**: Normalizes all interactions to native catalog IDs (`id: 1, 2, 3...`) across MongoDB events, Redis session logs, backend Express services, and frontend state.
2. **Mini-Batch Training Pipeline**: Full PyTorch `DataLoader` batching with Adam optimization, gradient clipping, early stopping, and validation loss tracking.
3. **Reproducible Temporal Splits**: 70/15/15 chronological split by session timestamp to prevent future-data leakage.
4. **Transparent Data Provenance**: Explicit distinction between real production MongoDB events and coherent synthetic sessions generated with category personas and realistic user funnels.
5. **Rigorous Benchmarking**: Fair comparative evaluation measuring **Hit@5, Hit@10, Hit@20, NDCG@5, NDCG@10, NDCG@20, MRR, Catalog Coverage, and Latency** against 3 distinct baselines:
   - `PopularityRecommender` (non-personalized global frequency)
   - `RecentlyViewedRecommender` (session-reverse heuristic)
   - `ContentRecommender` (TF-IDF cosine similarity on catalog text)

---

## 📦 Directory Layout

```
ml-service/
├── config.py                 # Central hyperparameters & paths
├── database.py               # MongoDB connection client
├── main.py                   # FastAPI application & lifespan management
├── requirements.txt          # Python dependencies
├── artifacts/                # Persisted artifacts (created on run)
│   ├── model/                # gru4rec_best.pt, train_metadata.json
│   ├── mappings/             # item_vocab.json, split_metadata.json
│   └── data/                 # train/val/test_sequences.json
├── data/
│   ├── build_dataset.py      # Dataset compilation & CLI
│   ├── dataset.py            # PyTorch Dataset, Vocab & CollateFn
│   ├── preprocessing.py      # Event cleaning & session reconstruction
│   └── synthetic.py          # Coherent persona-based event simulator
├── models/
│   ├── gru4rec.py            # PyTorch GRU4Rec implementation
│   └── baselines.py          # Popularity, RecentlyViewed & TF-IDF baselines
├── training/
│   └── train.py              # Full mini-batch training loop
├── evaluation/
│   ├── metrics.py            # Hit@K, NDCG@K, MRR, Coverage
│   └── evaluate.py           # Multi-model benchmark suite
├── inference/
│   └── recommender.py        # HybridRecommender with cold-start fallbacks
└── tests/
    ├── test_dataset.py       # Vocab & preprocessing tests
    ├── test_model.py         # Forward pass & gradient checks
    ├── test_metrics.py       # Pure metric unit tests
    └── test_api.py           # FastAPI integration tests
```

---

## 🛠️ Quick Start

### 1. Setup Environment
```bash
cd ml-service
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Generate Data & Train Model
```bash
# Full dataset build (pulls real data, falls back to coherent synthetic if < 100 events):
python -m data.build_dataset

# Train GRU4Rec with early stopping:
python -m training.train
```

### 4. Evaluate Models on Test Set
```bash
python -m evaluation.evaluate
```

### 5. Launch FastAPI Service
```bash
uvicorn main:app --reload --port 8000
```

---

## 📡 API Reference

### `POST /recommend/session`
Session-based recommendation using GRU4Rec (or cold-start fallbacks).

**Request:**
```json
{
  "session_history": ["1", "4", "12"],
  "top_k": 5,
  "exclude_history": true
}
```

**Response:**
```json
{
  "strategy": "gru4rec_neural",
  "model": "gru4rec",
  "recommendations": [7, 18, 22, 3, 15],
  "items": [
    {
      "productId": 7,
      "score": 0.3812,
      "name": "Wireless Noise-Cancelling Headphones",
      "category": "Electronics",
      "price": 149.99,
      "image": "https://...",
      "rating": 4.5
    }
  ],
  "session_length": 3,
  "latency_ms": 2.41
}
```

---

### `GET /recommend/{product_id}?top_k=5`
Item-to-item similarity using TF-IDF content features.

**Response:**
```json
{
  "product_id": "1",
  "strategy": "content_tfidf_item_similarity",
  "model": "tfidf-content",
  "recommendations": [3, 8, 12, 19, 25],
  "items": [...],
  "latency_ms": 0.82
}
```

---

### `GET /evaluate?force=false`
Returns benchmark metrics comparing GRU4Rec against all baselines on held-out test data.

---

### `GET /health`
Returns service health, device info, vocabulary size, and live serving counters.

---

## 📊 Evaluation Methodology

All models are evaluated on the exact same chronological held-out test set using the leave-one-out next-item prediction protocol:
- **Input sequence**: First $N - 1$ items of the session.
- **Target item**: The true $N$-th item.
- **Metrics**:
  - $\text{Hit}@K = \mathbb{I}(\text{target} \in \text{top-}K)$
  - $\text{NDCG}@K = \frac{1}{\log_2(\text{rank} + 1)}$ if $\text{rank} \le K$, else $0$
  - $\text{MRR} = \frac{1}{\text{rank}}$
  - $\text{Coverage} = \frac{|\bigcup \text{Recommendations}|}{|\text{Catalog}|}$
