"""
PriceIQ ML Service — Data Preprocessing
Loads raw events from MongoDB, normalizes product IDs, builds sessions,
creates training sequences, and generates a data quality report.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from database import get_db
import config


@dataclass
class DataQualityReport:
    """Stores statistics about the loaded and processed data."""
    total_events: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    unknown_product_events: int = 0
    missing_session_events: int = 0
    missing_timestamp_events: int = 0
    duplicate_events: int = 0
    total_sessions: int = 0
    usable_sessions: int = 0
    total_interactions: int = 0
    unique_products: int = 0
    avg_session_length: float = 0.0
    median_session_length: float = 0.0
    max_session_length: int = 0
    event_type_distribution: dict = field(default_factory=dict)
    is_synthetic: bool = False

    def print_report(self):
        print("\n" + "=" * 60)
        print("  DATA QUALITY REPORT")
        if self.is_synthetic:
            print("  ⚠️  DATA SOURCE: SYNTHETIC (not real user data)")
        else:
            print("  📊 DATA SOURCE: REAL MongoDB events")
        print("=" * 60)
        print(f"  Total raw events:          {self.total_events:,}")
        print(f"  Valid events:              {self.valid_events:,}")
        print(f"  Invalid events:            {self.invalid_events:,}")
        print(f"    - Unknown product IDs:   {self.unknown_product_events:,}")
        print(f"    - Missing sessionId:     {self.missing_session_events:,}")
        print(f"    - Missing timestamp:     {self.missing_timestamp_events:,}")
        print(f"    - Duplicates removed:    {self.duplicate_events:,}")
        print(f"  Total sessions:            {self.total_sessions:,}")
        print(f"  Usable sessions (≥{config.MIN_SESSION_LEN} items): {self.usable_sessions:,}")
        print(f"  Total interactions:        {self.total_interactions:,}")
        print(f"  Unique products:           {self.unique_products:,}")
        print(f"  Avg session length:        {self.avg_session_length:.2f}")
        print(f"  Median session length:     {self.median_session_length:.1f}")
        print(f"  Max session length:        {self.max_session_length}")
        print(f"  Event type distribution:")
        for etype, count in sorted(self.event_type_distribution.items(), key=lambda x: -x[1]):
            print(f"    {etype:25s} {count:,}")
        print("=" * 60 + "\n")

    def to_dict(self) -> dict:
        return {
            "total_events": self.total_events,
            "valid_events": self.valid_events,
            "invalid_events": self.invalid_events,
            "unknown_product_events": self.unknown_product_events,
            "missing_session_events": self.missing_session_events,
            "missing_timestamp_events": self.missing_timestamp_events,
            "duplicate_events": self.duplicate_events,
            "total_sessions": self.total_sessions,
            "usable_sessions": self.usable_sessions,
            "total_interactions": self.total_interactions,
            "unique_products": self.unique_products,
            "avg_session_length": round(self.avg_session_length, 2),
            "median_session_length": round(self.median_session_length, 1),
            "max_session_length": self.max_session_length,
            "event_type_distribution": self.event_type_distribution,
            "is_synthetic": self.is_synthetic,
        }


def load_product_catalog(db=None):
    """
    Load the product catalog from MongoDB.
    Returns a dict mapping the canonical product id (int) to product info.
    The canonical ID is the Product.id field (numeric: 1, 2, 3, ...).
    """
    if db is None:
        db = get_db()
    products = list(db.products.find({}, {"_id": 1, "id": 1, "name": 1, "category": 1, "description": 1}))
    catalog = {}
    for p in products:
        pid = p.get("id")
        if pid is not None:
            catalog[int(pid)] = {
                "id": int(pid),
                "mongo_id": str(p["_id"]),
                "name": p.get("name", ""),
                "category": p.get("category", ""),
                "description": p.get("description", ""),
            }
    return catalog


def normalize_product_id(raw_pid, catalog):
    """
    Convert any raw productId format to the canonical integer id.
    Handles: int, float, str of int, ObjectId hex strings.
    Returns the canonical int id, or None if unmatchable.
    """
    if raw_pid is None:
        return None

    # Direct int match
    if isinstance(raw_pid, (int, float)):
        pid = int(raw_pid)
        if pid in catalog:
            return pid
        return None

    # String handling
    raw_str = str(raw_pid).strip()
    if not raw_str:
        return None

    # Try parsing as integer string ("1", "42")
    try:
        pid = int(raw_str)
        if pid in catalog:
            return pid
    except ValueError:
        pass

    # Try matching against MongoDB ObjectId strings
    for pid, info in catalog.items():
        if info["mongo_id"] == raw_str:
            return pid

    return None


def load_events(db=None, catalog=None):
    """
    Load raw events from MongoDB, normalize product IDs, validate fields.
    Returns (list of clean event dicts, DataQualityReport).
    """
    if db is None:
        db = get_db()
    if catalog is None:
        catalog = load_product_catalog(db)

    report = DataQualityReport()
    event_type_counts = Counter()

    raw_events = list(db.events.find(
        {"eventType": {"$in": config.PRODUCT_INTERACTION_TYPES}},
        {"sessionId": 1, "productId": 1, "eventType": 1, "timestamp": 1, "userId": 1}
    ).sort("timestamp", 1))

    report.total_events = len(raw_events)
    clean_events = []
    seen = set()  # for dedup: (sessionId, productId, eventType, timestamp_rounded)

    for e in raw_events:
        session_id = e.get("sessionId")
        raw_pid = e.get("productId")
        event_type = e.get("eventType")
        timestamp = e.get("timestamp")

        # Validate required fields
        if not session_id:
            report.missing_session_events += 1
            report.invalid_events += 1
            continue

        if timestamp is None:
            report.missing_timestamp_events += 1
            report.invalid_events += 1
            continue

        # Normalize product ID to canonical int
        canonical_pid = normalize_product_id(raw_pid, catalog)
        if canonical_pid is None:
            report.unknown_product_events += 1
            report.invalid_events += 1
            continue

        # Deduplication key (session + product + type + second-level timestamp)
        ts_key = timestamp.strftime("%Y%m%d%H%M%S") if isinstance(timestamp, datetime) else str(timestamp)
        dedup_key = (str(session_id), canonical_pid, event_type, ts_key)
        if dedup_key in seen:
            report.duplicate_events += 1
            report.invalid_events += 1
            continue
        seen.add(dedup_key)

        event_type_counts[event_type] += 1
        clean_events.append({
            "session_id": str(session_id),
            "product_id": canonical_pid,
            "event_type": event_type,
            "timestamp": timestamp,
            "user_id": str(e.get("userId", "")),
        })

    report.valid_events = len(clean_events)
    report.event_type_distribution = dict(event_type_counts)
    return clean_events, report, catalog


def build_sessions(events):
    """
    Group events by session_id, sort chronologically within each session.
    Returns dict: session_id -> list of events (sorted by timestamp).
    """
    sessions = defaultdict(list)
    for e in events:
        sid = e.get("session_id") or e.get("sessionId")
        if sid:
            sessions[sid].append(e)

    # Sort each session chronologically
    for sid in sessions:
        sessions[sid].sort(key=lambda x: x["timestamp"])

    return dict(sessions)


def create_sequences(sessions, min_len=None):
    """
    Convert sessions into product interaction sequences for training.
    Each session becomes a list of canonical product IDs (ints) in chronological order.
    Only keeps sessions with >= min_len product interactions.

    Returns list of dicts: {
        "session_id": str,
        "product_ids": [int, ...],
        "end_time": datetime,
    }
    """
    if min_len is None:
        min_len = config.MIN_SESSION_LEN

    sequences = []
    for sid, events in sessions.items():
        # Extract product IDs in order (deduplicate consecutive same-product views)
        product_ids = []
        for e in events:
            pid = e.get("product_id") or e.get("productId")
            if pid is not None and (not product_ids or product_ids[-1] != pid):
                product_ids.append(pid)

        if len(product_ids) >= min_len:
            end_time = events[-1]["timestamp"]
            sequences.append({
                "session_id": sid,
                "product_ids": product_ids,
                "end_time": end_time,
            })

    return sequences


def temporal_split(sequences, train_ratio=None, val_ratio=None, test_ratio=None):
    """
    Split sequences chronologically by session end time.
    Returns (train_seqs, val_seqs, test_seqs).
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = config.VAL_RATIO

    # Sort by end time
    sorted_seqs = sorted(sequences, key=lambda x: x["end_time"])
    n = len(sorted_seqs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_seqs = sorted_seqs[:train_end]
    val_seqs = sorted_seqs[train_end:val_end]
    test_seqs = sorted_seqs[val_end:]

    return train_seqs, val_seqs, test_seqs


def finalize_report(report, sessions, sequences):
    """Fill in session-level statistics in the report."""
    report.total_sessions = len(sessions)
    report.usable_sessions = len(sequences)

    if sequences:
        lengths = [len(s["product_ids"]) for s in sequences]
        report.total_interactions = sum(lengths)
        report.unique_products = len(set(
            pid for s in sequences for pid in s["product_ids"]
        ))
        report.avg_session_length = sum(lengths) / len(lengths)
        sorted_lengths = sorted(lengths)
        mid = len(sorted_lengths) // 2
        report.median_session_length = (
            sorted_lengths[mid] if len(sorted_lengths) % 2 == 1
            else (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
        )
        report.max_session_length = max(lengths)

    return report
