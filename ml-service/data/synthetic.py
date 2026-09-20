"""
PriceIQ ML Service — Coherent Synthetic Data Generator

Generates realistic, sequential e-commerce browsing sessions with true
item-to-item transition affinity, category hierarchies, and behavioral funnels.
Every record is explicitly marked as SYNTHETIC in metadata.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from database import get_db
import config

# Category affinity groups — users exploring one category also transition to related ones
CATEGORY_AFFINITY = {
    "Electronics": ["Gaming", "Appliances"],
    "Gaming": ["Electronics", "Toys"],
    "Fashion": ["Beauty", "Sports"],
    "Beauty": ["Fashion", "Home & Kitchen"],
    "Home & Kitchen": ["Appliances", "Furniture"],
    "Sports": ["Fashion", "Automotive"],
    "Books": ["Toys", "Home & Kitchen"],
    "Toys": ["Gaming", "Books"],
    "Automotive": ["Sports", "Electronics"],
    "Appliances": ["Home & Kitchen", "Electronics"],
    "Furniture": ["Home & Kitchen", "Appliances"],
    "Food & Grocery": ["Home & Kitchen", "Beauty"],
}

# Funnel progression probabilities
FUNNEL_TRANSITIONS = {
    "page_view": {"page_view": 0.65, "add_to_cart": 0.20, "wishlist_add": 0.10, "purchase": 0.05},
    "add_to_cart": {"page_view": 0.25, "add_to_cart": 0.15, "purchase": 0.50, "wishlist_add": 0.10},
    "wishlist_add": {"page_view": 0.50, "add_to_cart": 0.30, "purchase": 0.15, "wishlist_add": 0.05},
    "purchase": {"page_view": 0.60, "add_to_cart": 0.15, "purchase": 0.10, "wishlist_add": 0.15},
}


def build_catalog_transition_graph(products):
    """
    Build a transition probability matrix T(j | i) over catalog products.
    Weights are grounded in real e-commerce behaviors:
      - Same product sub-type (e.g., headphones -> headphones): very high
      - Same brand (e.g., Apple -> Apple): high
      - Same category: moderate
      - Affinity categories: small
      - Cross-category discovery: small exploration epsilon
    """
    pids = [int(p["id"]) for p in products if p.get("id") is not None]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    n = len(pids)

    # Extract tokens and metadata
    metadata = {}
    for p in products:
        pid = int(p["id"])
        name = p.get("name", "").lower()
        cat = p.get("category", "")
        brand = p.get("brand", "").lower()
        price = p.get("livePrice") or p.get("basePrice", 1000)
        metadata[pid] = {
            "name": name,
            "cat": cat,
            "brand": brand,
            "price": price,
            "tokens": set(name.split()),
        }

    # Build affinity matrix
    affinities = np.zeros((n, n), dtype=np.float32)

    for i, pid_i in enumerate(pids):
        meta_i = metadata[pid_i]
        for j, pid_j in enumerate(pids):
            if i == j:
                affinities[i, j] = 0.5  # repeated re-examination
                continue

            meta_j = metadata[pid_j]
            score = 0.05  # baseline exploration

            # Shared category
            if meta_i["cat"] == meta_j["cat"]:
                score += 3.0

            # Affinity category
            elif meta_j["cat"] in CATEGORY_AFFINITY.get(meta_i["cat"], []):
                score += 1.2

            # Shared brand
            if meta_i["brand"] and meta_i["brand"] == meta_j["brand"]:
                score += 2.5

            # Shared keyword tokens (e.g. 'smartwatch', 'shoes', 'jeans')
            shared_tokens = meta_i["tokens"] & meta_j["tokens"]
            # Exclude generic words
            meaningful = {t for t in shared_tokens if len(t) > 3 and t not in ("model", "premium")}
            if meaningful:
                score += 6.0 * len(meaningful)

            # Price similarity (within 40% range)
            pi = meta_i["price"]
            pj = meta_j["price"]
            if max(pi, pj) > 0:
                price_ratio = min(pi, pj) / max(pi, pj)
                if price_ratio > 0.6:
                    score += 1.5 * price_ratio

            affinities[i, j] = score

    # Normalize rows to probabilities
    row_sums = affinities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = affinities / row_sums

    return pids, pid_to_idx, transition_matrix


def generate_synthetic_events(db=None, num_sessions=None, seed=None):
    """
    Generate coherent synthetic browsing events following the catalog transition graph.
    Returns a list of event dicts and the count of sessions generated.
    """
    if db is None:
        db = get_db()
    if num_sessions is None:
        num_sessions = config.SYNTHETIC_NUM_SESSIONS
    if seed is None:
        seed = config.RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)

    # Load real products from DB
    products = list(db.products.find({}, {
        "_id": 1, "id": 1, "category": 1, "name": 1, "brand": 1, "livePrice": 1, "basePrice": 1
    }))
    if not products:
        print("ERROR: No products in MongoDB. Run product seed first.")
        return [], 0

    pids, pid_to_idx, transition_matrix = build_catalog_transition_graph(products)
    prod_map = {int(p["id"]): p for p in products if p.get("id") is not None}

    # Group products by category
    products_by_cat = defaultdict(list)
    for p in products:
        if p.get("id") is not None:
            products_by_cat[p.get("category", "Electronics")].append(int(p["id"]))

    all_categories = list(products_by_cat.keys())
    if not all_categories:
        return [], 0

    # User personas
    num_users = min(config.SYNTHETIC_NUM_USERS, num_sessions)
    users = []
    for _ in range(num_users):
        primary_cat = random.choice(all_categories)
        users.append({
            "user_id": str(uuid.uuid4()),
            "primary_category": primary_cat,
        })

    now = datetime.utcnow()
    events = []

    for session_idx in range(num_sessions):
        user = random.choice(users)
        session_id = str(uuid.uuid4())

        # Session length
        min_events, max_events = config.SYNTHETIC_EVENTS_PER_SESSION_RANGE
        num_events = random.randint(min_events, max_events)

        # Session start timestamp (spaced over 30 days)
        days_ago = random.uniform(0, config.SYNTHETIC_DAYS_SPAN)
        session_start = now - timedelta(days=days_ago)

        # Pick starting product based on user persona
        start_cat = user["primary_category"]
        cat_pids = products_by_cat.get(start_cat) or pids
        current_pid = random.choice(cat_pids)
        current_idx = pid_to_idx[current_pid]

        current_event_type = "page_view"
        current_time = session_start

        for event_idx in range(num_events):
            prod_info = prod_map[current_pid]
            cat = prod_info.get("category", "General")
            price = prod_info.get("livePrice") or prod_info.get("basePrice", 1000)

            # Advance time coherently (10s to 120s between events in same session)
            current_time += timedelta(seconds=random.randint(10, 120))

            events.append({
                "eventType": current_event_type,
                "productId": int(current_pid),
                "sessionId": session_id,
                "userId": user["user_id"],
                "amount": price if current_event_type == "purchase" else 0,
                "timestamp": current_time,
                "createdAt": current_time,
                "metadata": {
                    "source": "SYNTHETIC",
                    "category": cat,
                    "productName": prod_info.get("name", f"Product #{current_pid}"),
                    "price": price,
                    "deviceType": random.choice(["mobile", "desktop", "tablet"]),
                    "referralSource": random.choice(["direct", "google", "instagram", "affiliate"]),
                },
            })

            # Next event type
            transitions = FUNNEL_TRANSITIONS.get(current_event_type, FUNNEL_TRANSITIONS["page_view"])
            next_types = list(transitions.keys())
            next_weights = list(transitions.values())
            current_event_type = random.choices(next_types, weights=next_weights, k=1)[0]

            # Sample next product from the transition matrix
            probs = transition_matrix[current_idx]
            next_idx = np.random.choice(len(pids), p=probs)
            current_pid = pids[next_idx]
            current_idx = next_idx

    return events, num_sessions


def seed_synthetic_events(db=None, num_sessions=None, clear_existing=True):
    """
    Generate and insert coherent synthetic events into MongoDB.
    """
    if db is None:
        db = get_db()

    if clear_existing:
        deleted = db.events.delete_many({})
        print(f"  Cleared {deleted.deleted_count:,} existing events")

    events, num_sess = generate_synthetic_events(db, num_sessions)
    if not events:
        print("  No events generated.")
        return 0

    batch_size = 5000
    for i in range(0, len(events), batch_size):
        db.events.insert_many(events[i:i + batch_size])
        print(f"  Inserted {min(i + batch_size, len(events)):,}/{len(events):,} events")

    print(f"  ✅ Seeded {len(events):,} coherent events across {num_sess:,} sessions")
    return len(events)


if __name__ == "__main__":
    print("=" * 60)
    print("  COHERENT SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    count = seed_synthetic_events()
    print(f"Done. Total events: {count:,}")
