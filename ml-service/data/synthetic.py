"""
PriceIQ ML Service — Coherent Synthetic Data Generator

Generates realistic browsing sessions when real MongoDB event data is
insufficient for training. Every record is explicitly labeled as SYNTHETIC.

Design:
  - Users have category preferences (e.g., an "Electronics" user browses
    electronics 70% of the time, occasionally cross-browses related categories).
  - Sessions follow realistic browsing funnels:
      page_view → page_view → add_to_cart → purchase
  - Timestamps are chronologically coherent within sessions.
  - Product selections within a category are weighted by realistic browsing patterns.

WARNING: Model metrics from synthetic data should NEVER be presented as
real-world performance.  All outputs are labeled "SYNTHETIC".
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
from database import get_db
import config


# Category affinity groups — users who like one category also explore related ones
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
# After each event type, what's the probability of the next type?
FUNNEL_TRANSITIONS = {
    "page_view": {"page_view": 0.60, "add_to_cart": 0.25, "wishlist_add": 0.10, "purchase": 0.05},
    "add_to_cart": {"page_view": 0.25, "add_to_cart": 0.10, "purchase": 0.55, "wishlist_add": 0.10},
    "wishlist_add": {"page_view": 0.50, "add_to_cart": 0.30, "purchase": 0.15, "wishlist_add": 0.05},
    "purchase": {"page_view": 0.70, "add_to_cart": 0.10, "purchase": 0.10, "wishlist_add": 0.10},
}


def generate_synthetic_events(db=None, num_sessions=None, seed=None):
    """
    Generate coherent synthetic browsing events.

    Returns a list of event dicts ready for MongoDB insertion,
    plus the count of sessions generated.
    """
    if db is None:
        db = get_db()
    if num_sessions is None:
        num_sessions = config.SYNTHETIC_NUM_SESSIONS
    if seed is None:
        seed = config.RANDOM_SEED

    random.seed(seed)

    # Load real products from DB
    products = list(db.products.find({}, {"_id": 1, "id": 1, "category": 1, "name": 1}))
    if not products:
        print("ERROR: No products in MongoDB. Run product seed first.")
        return [], 0

    # Group products by category
    products_by_category = defaultdict(list)
    for p in products:
        cat = p.get("category", "Electronics")
        products_by_category[cat].append(p)

    all_categories = list(products_by_category.keys())
    if not all_categories:
        return [], 0

    # Create user personas with category preferences
    num_users = min(config.SYNTHETIC_NUM_USERS, num_sessions)
    users = []
    for _ in range(num_users):
        primary_cat = random.choice(all_categories)
        affinity_cats = CATEGORY_AFFINITY.get(primary_cat, [])
        # 70% primary category, 20% affinity categories, 10% random
        users.append({
            "user_id": str(uuid.uuid4()),
            "primary_category": primary_cat,
            "affinity_categories": affinity_cats,
        })

    now = datetime.utcnow()
    events = []

    for session_idx in range(num_sessions):
        user = random.choice(users)
        session_id = str(uuid.uuid4())

        # Session length follows a realistic distribution
        min_events, max_events = config.SYNTHETIC_EVENTS_PER_SESSION_RANGE
        num_events = random.randint(min_events, max_events)

        # Session start time (spread across SYNTHETIC_DAYS_SPAN days)
        days_ago = random.uniform(0, config.SYNTHETIC_DAYS_SPAN)
        session_start = now - timedelta(days=days_ago)

        current_event_type = "page_view"  # sessions always start with a page view
        current_time = session_start

        for event_idx in range(num_events):
            # Pick category based on user persona
            roll = random.random()
            if roll < 0.70:
                cat = user["primary_category"]
            elif roll < 0.90 and user["affinity_categories"]:
                cat = random.choice(user["affinity_categories"])
            else:
                cat = random.choice(all_categories)

            # Pick a product from the chosen category
            cat_products = products_by_category.get(cat, [])
            if not cat_products:
                cat_products = products_by_category.get(user["primary_category"], products)

            product = random.choice(cat_products)
            canonical_pid = product.get("id")
            if canonical_pid is None:
                continue

            # Advance time within session (5s to 3min between events)
            current_time += timedelta(seconds=random.randint(5, 180))

            events.append({
                "eventType": current_event_type,
                "productId": int(canonical_pid),  # canonical numeric ID
                "sessionId": session_id,
                "userId": user["user_id"],
                "timestamp": current_time,
                "metadata": {
                    "source": "SYNTHETIC",
                    "category": cat,
                    "deviceType": random.choice(["mobile", "desktop", "tablet"]),
                },
            })

            # Determine next event type using funnel transitions
            transitions = FUNNEL_TRANSITIONS.get(current_event_type, FUNNEL_TRANSITIONS["page_view"])
            next_types = list(transitions.keys())
            next_weights = list(transitions.values())
            current_event_type = random.choices(next_types, weights=next_weights, k=1)[0]

    return events, num_sessions


def seed_synthetic_events(db=None, num_sessions=None, clear_existing=True):
    """
    Generate and insert synthetic events into MongoDB.
    Clears existing events first if clear_existing is True.
    """
    if db is None:
        db = get_db()

    if clear_existing:
        deleted = db.events.delete_many({})
        print(f"  Cleared {deleted.deleted_count} existing events")

    events, num_sess = generate_synthetic_events(db, num_sessions)
    if not events:
        print("  No events generated.")
        return 0

    # Batch insert
    batch_size = 5000
    for i in range(0, len(events), batch_size):
        db.events.insert_many(events[i:i + batch_size])
        print(f"  Inserted {min(i + batch_size, len(events)):,}/{len(events):,} SYNTHETIC events")

    print(f"  ✅ Seeded {len(events):,} SYNTHETIC events across {num_sess:,} sessions")
    return len(events)


if __name__ == "__main__":
    print("=" * 60)
    print("  SYNTHETIC DATA GENERATOR")
    print("  ⚠️  This data is for development/training only")
    print("=" * 60)
    count = seed_synthetic_events()
    print(f"Done. Total SYNTHETIC events: {count:,}")
