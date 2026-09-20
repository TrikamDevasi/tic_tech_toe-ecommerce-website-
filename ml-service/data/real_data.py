"""
PriceIQ ML Service — Organic Real-Data Pipeline & Temporal Splitter

Maintains strict separation between:
  - SYNTHETIC DATASET: used for pre-training and simulation
  - REAL DATASET: organic interactions from authentic store visitors

As organic interactions enter MongoDB, this module:
  1. Filters events where metadata.source != 'SYNTHETIC'
  2. Normalizes product IDs and validates session continuity
  3. Prepares a strict chronological split (Train / Val / Test)
  4. Prevents target leakage and session overlap
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict
from datetime import datetime
from database import get_db
import config
from data.preprocessing import load_product_catalog, normalize_product_id, build_sessions, create_sequences, temporal_split
from data.dataset import ItemVocabulary


def load_organic_real_events(db=None, catalog=None):
    """
    Extract strictly organic real-user events from MongoDB.
    Filters out any event tagged with metadata.source == 'SYNTHETIC'.
    """
    if db is None:
        db = get_db()
    if catalog is None:
        catalog = load_product_catalog(db)

    mongo_id_map = {info["mongo_id"]: pid for pid, info in catalog.items() if "mongo_id" in info}

    # Query organic events only
    query = {
        "metadata.source": {"$ne": "SYNTHETIC"},
        "eventType": {"$in": config.PRODUCT_INTERACTION_TYPES},
    }

    raw_events = list(db.events.find(query).sort("timestamp", 1))
    clean_events = []
    unknown_pids = 0

    for e in raw_events:
        sid = e.get("sessionId")
        raw_pid = e.get("productId")
        ts = e.get("timestamp")
        if not sid or not ts or raw_pid is None:
            continue

        canon_pid = normalize_product_id(raw_pid, catalog, mongo_id_map)
        if canon_pid is None:
            unknown_pids += 1
            continue

        clean_events.append({
            "session_id": str(sid),
            "product_id": canon_pid,
            "event_type": e.get("eventType", "page_view"),
            "timestamp": ts,
            "user_id": str(e.get("userId", "")),
        })

    return clean_events, len(raw_events), unknown_pids


def build_real_dataset(db=None):
    """
    Build real-data sequences and temporal split if organic data is sufficient.
    Returns dict with real data statistics and sequences.
    """
    if db is None:
        db = get_db()
    catalog = load_product_catalog(db)
    clean_events, raw_count, unknown_count = load_organic_real_events(db, catalog)

    sessions = build_sessions(clean_events)
    sequences = create_sequences(sessions, min_len=config.MIN_SESSION_LEN)

    is_sufficient = len(sequences) >= 100

    report = {
        "raw_organic_events": raw_count,
        "clean_organic_events": len(clean_events),
        "unknown_product_events": unknown_count,
        "total_organic_sessions": len(sessions),
        "usable_organic_sessions": len(sequences),
        "is_sufficient_for_training": is_sufficient,
        "status": "ready" if is_sufficient else "awaiting_organic_traffic",
    }

    if is_sufficient:
        train_seqs, val_seqs, test_seqs = temporal_split(sequences)
        real_data_dir = os.path.join(config.ARTIFACTS_DIR, "data_real")
        os.makedirs(real_data_dir, exist_ok=True)
        with open(os.path.join(real_data_dir, "train_sequences.json"), "w") as f:
            json.dump(train_seqs, f, default=str)
        with open(os.path.join(real_data_dir, "val_sequences.json"), "w") as f:
            json.dump(val_seqs, f, default=str)
        with open(os.path.join(real_data_dir, "test_sequences.json"), "w") as f:
            json.dump(test_seqs, f, default=str)
        report["split"] = {
            "train": len(train_seqs),
            "val": len(val_seqs),
            "test": len(test_seqs),
        }
        print(f"✅ Real dataset prepared: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test")
    else:
        print(f"ℹ️ Organic real data volume ({len(sequences)} usable sessions) is below minimum threshold (100). Standby.")

    return report


if __name__ == "__main__":
    rep = build_real_dataset()
    print(json.dumps(rep, indent=2))
