"""
Tests for PriceIQ Cold-Start Ladder, Fallback Metadata, and Production Monitoring Telemetry.
"""
import pytest
from fastapi.testclient import TestClient
import main
from main import app, recommender


@pytest.fixture(scope="module")
def client():
    # Ensure recommender is loaded
    recommender.load()
    with TestClient(app) as test_client:
        yield test_client


def test_cold_start_zero_history(client):
    """Test 0 interactions returns popularity fallback with explicit metadata."""
    payload = {"session_history": [], "top_k": 3}
    response = client.post("/recommend/session", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is True
    assert data["fallback_reason"] == "new_user_zero_history"
    assert data["strategy"] == "popularity_cold_start"
    assert len(data["recommendations"]) == 3


def test_cold_start_sparse_history(client):
    """Test 1-2 interactions triggers content TF-IDF fallback."""
    payload = {"session_history": [1], "top_k": 3}
    response = client.post("/recommend/session", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is True
    assert data["fallback_reason"] == "sparse_history_content_fallback"
    assert "content_tfidf" in data["strategy"]
    assert len(data["recommendations"]) == 3


def test_cold_start_unknown_product(client):
    """Test session with totally unknown product IDs routes to unknown fallback."""
    payload = {"session_history": [999999, 888888], "top_k": 5}
    response = client.post("/recommend/session", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is True
    assert data["fallback_reason"] == "all_session_products_unknown_in_catalog"
    assert data["unknown_products_count"] == 2


def test_monitoring_telemetry_endpoint(client):
    """Test GET /monitoring/metrics returns telemetry without PII."""
    response = client.get("/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "fallback_rate_pct" in data
    assert "strategy_distribution" in data
    assert "latency_ms" in data
    assert "p50" in data["latency_ms"]
    assert "p95" in data["latency_ms"]


def test_monitoring_click_endpoint(client):
    """Test POST /monitoring/click records user engagement cleanly."""
    initial_resp = client.get("/monitoring/metrics")
    initial_clicks = initial_resp.json().get("clicks_tracked", 0)

    click_payload = {"product_id": 1, "strategy": "gru4rec_neural"}
    response = client.post("/monitoring/click", json=click_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert data["total_clicks"] == initial_clicks + 1
