"""
PriceIQ ML Service — FastAPI Application

Production-ready session recommendation service powered by:
  - GRU4Rec (Neural Session-based RNN)
  - TF-IDF Content Similarity
  - Global Popularity Fallback
  - Real Test Set Evaluation Benchmarks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import config
from inference.recommender import HybridRecommender
from evaluation.evaluate import evaluate_models
from data.build_dataset import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("priceiq.api")

# Singleton recommender instance
recommender = HybridRecommender()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup & shutdown lifespan.
    Loads pre-trained artifacts on startup (does NOT retrain unless missing).
    """
    logger.info("Initializing PriceIQ ML Service...")
    try:
        recommender.load(force_retrain=False)
        logger.info("Recommender loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load recommender on startup: {e}")
        logger.info("Attempting automated recovery/training...")
        try:
            recommender.load(force_retrain=True)
            logger.info("Recovery training completed successfully.")
        except Exception as rec_err:
            logger.critical(f"FATAL: Could not initialize recommender: {rec_err}")

    yield

    logger.info("Shutting down PriceIQ ML Service.")


app = FastAPI(
    title="PriceIQ ML Recommendation Service",
    description="Session-Aware Dynamic Personalization Engine using GRU4Rec and Hybrid Fallbacks",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ───────────────────────────────────────────────

class SessionRequest(BaseModel):
    session_history: List[Any] = Field(
        ...,
        description="Chronological list of product IDs viewed or interacted with in current session",
        json_schema_extra={"example": ["1", "5", "12"]},
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of recommendations requested")
    exclude_history: bool = Field(default=True, description="Exclude items already in session history")


class TrainRequest(BaseModel):
    epochs: Optional[int] = Field(default=None, ge=1, le=200)
    batch_size: Optional[int] = Field(default=None, ge=8, le=512)
    seed_synthetic: bool = Field(default=False, description="Re-generate synthetic events before training")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "PriceIQ ML Recommendation Engine",
        "version": "2.0.0",
        "architecture": "GRU4Rec + TF-IDF Content + Popularity Fallback",
        "docs": "/docs",
        "status": "ready" if recommender.is_ready else "initializing",
    }


@app.get("/health")
def health():
    """Service health, model readiness, vocabulary size, and runtime statistics."""
    return recommender.get_status()


@app.post("/recommend/session")
def recommend_session(req: SessionRequest):
    """
    Primary recommendation endpoint: generates session-aware recommendations
    using GRU4Rec when session length >= 3, with automatic cold-start fallbacks
    (TF-IDF content for 1-2 interactions, popularity for 0 interactions).
    """
    try:
        res = recommender.recommend_session(
            session_history=req.session_history,
            top_k=req.top_k,
            exclude_history=req.exclude_history,
        )

        rec_ids = [item["productId"] for item in res["recommendations"]]

        return {
            "strategy": res["strategy"],
            "model": "gru4rec" if "gru4rec" in res["strategy"] else res["strategy"],
            "recommendations": rec_ids,
            "items": res["recommendations"],
            "session_length": res["session_length"],
            "latency_ms": res["latency_ms"],
            "confidence": res.get("confidence", 0.0),
            "confidence_pct": res.get("confidence_pct", 0.0),
            "is_low_confidence": res.get("is_low_confidence", False),
            "confidence_message": res.get("confidence_message", ""),
            "entropy": res.get("entropy", 0.0),
        }
    except Exception as e:
        logger.error(f"Error in recommend_session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{product_id}")
def recommend_by_product(product_id: str, top_k: int = Query(default=5, ge=1, le=50)):
    """
    Item-to-item recommendation using TF-IDF Content Similarity.
    Used for product detail pages and item-level similarity.
    """
    try:
        res = recommender.recommend_similar_product(product_id=product_id, top_k=top_k)
        rec_ids = [item["productId"] for item in res["recommendations"]]

        return {
            "product_id": res["seed_product_id"],
            "strategy": res["strategy"],
            "model": "tfidf-content",
            "recommendations": rec_ids,
            "items": res["recommendations"],
            "latency_ms": res["latency_ms"],
        }
    except Exception as e:
        logger.error(f"Error in recommend_by_product: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate")
def evaluate(force: bool = Query(default=False, description="Re-run evaluation instead of returning cached results")):
    """
    Returns real benchmark evaluation comparing GRU4Rec against Popularity,
    Recently Viewed, and TF-IDF Content baselines on held-out test data.
    """
    results_path = os.path.join(config.ARTIFACTS_DIR, "experiment_results.json")
    if not force and os.path.exists(results_path):
        try:
            import json
            with open(results_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading cached evaluation: {e}")

    try:
        results = evaluate_models()
        return results
    except Exception as e:
        logger.error(f"Error running evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/data-report")
def data_report():
    """Returns data quality report on MongoDB events and session distributions."""
    try:
        report = build_dataset(report_only=True)
        if report:
            return report.to_dict()
        raise HTTPException(status_code=500, detail="Could not generate data report.")
    except Exception as e:
        logger.error(f"Error generating data report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(req: TrainRequest = TrainRequest()):
    """
    Triggers complete training pipeline:
    data preparation -> temporal split -> model training -> validation -> checkpointing.
    """
    try:
        from training.train import run_training
        logger.info(f"Retraining triggered with epochs={req.epochs}, batch_size={req.batch_size}")
        if req.seed_synthetic:
            build_dataset(seed_synthetic=True)

        model, vocab, meta, test_seqs = run_training(
            epochs=req.epochs,
            batch_size=req.batch_size,
        )

        # Reload recommender with newly trained model
        recommender.load(force_retrain=False)

        # Run fresh evaluation
        eval_summary = evaluate_models(
            test_sequences=test_seqs,
            vocab=vocab,
            model=model,
        )

        return {
            "status": "success",
            "message": "Model trained and evaluated successfully",
            "training_metadata": meta,
            "benchmark_summary": eval_summary,
        }
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
