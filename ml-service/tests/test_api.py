"""
Integration Tests for FastAPI Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from main import app, recommender
from data.dataset import ItemVocabulary
from models.gru4rec import GRU4Rec


@pytest.fixture(scope="module", autouse=True)
def setup_mock_recommender():
    """Ensure recommender has in-memory mock models so tests don't require MongoDB."""
    catalog = {
        1: {"name": "Smartphone", "category": "Electronics", "price": 499, "image": ""},
        2: {"name": "Laptop", "category": "Electronics", "price": 999, "image": ""},
        3: {"name": "Headphones", "category": "Electronics", "price": 99, "image": ""},
        4: {"name": "Running Shoes", "category": "Footwear", "price": 79, "image": ""},
        5: {"name": "T-Shirt", "category": "Apparel", "price": 25, "image": ""},
    }
    vocab = ItemVocabulary()
    vocab.build(list(catalog.keys()))

    model = GRU4Rec(vocab_size=len(vocab), embedding_dim=16, hidden_dim=32)
    model.eval()

    recommender.catalog = catalog
    recommender.vocab = vocab
    recommender.model = model
    recommender.content_model.train_from_catalog(catalog)
    recommender.popularity_model.train([{"product_ids": [1, 2, 3, 4, 5]}])
    recommender.is_ready = True
    recommender.load = lambda **kwargs: None


def test_root_endpoint():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert "PriceIQ ML" in data["service"]
        assert data["version"] == "2.0.0"


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert data["vocab_size"] > 0


def test_recommend_session_cold_start():
    with TestClient(app) as client:
        # 0 items in history -> popularity
        res = client.post("/recommend/session", json={"session_history": [], "top_k": 3})
        assert res.status_code == 200
        data = res.json()
        assert "popularity" in data["strategy"]
        assert len(data["recommendations"]) <= 3


def test_recommend_session_content_ladder():
    with TestClient(app) as client:
        # 1-2 items in history -> TF-IDF content similarity
        res = client.post("/recommend/session", json={"session_history": [1], "top_k": 3})
        assert res.status_code == 200
        data = res.json()
        assert "content_tfidf" in data["strategy"]
        assert len(data["recommendations"]) <= 3


def test_recommend_session_gru4rec():
    with TestClient(app) as client:
        # 3+ items in history -> GRU4Rec
        res = client.post("/recommend/session", json={"session_history": [1, 2, 3], "top_k": 3})
        assert res.status_code == 200
        data = res.json()
        assert data["strategy"] == "gru4rec_neural"
        assert len(data["recommendations"]) <= 3


def test_recommend_by_product():
    with TestClient(app) as client:
        res = client.get("/recommend/1?top_k=2")
        assert res.status_code == 200
        data = res.json()
        assert data["model"] == "tfidf-content"
        assert len(data["recommendations"]) <= 2
