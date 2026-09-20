"""
PriceIQ ML Service — Coherent E-commerce Behavioral Synthetic Generator

Generates realistic, sequential e-commerce browsing sessions modeling true
human shopping behaviors:
  1. Product Comparison (browsing alternative models in the same subcategory)
  2. Complementary Ecosystem / Cross-Sell (accessories and companion items)
  3. Brand Loyalty (browsing items across subcategories within a single brand)
  4. Category Discovery (exploring within a department)
  5. Price-Tier Shopping (budget or premium constrained exploration)
  6. Re-examination & Backtracking (re-visiting earlier candidates)
  7. 500 Returning User Personas with persistent affinities across 30 days
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
import re
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from database import get_db
import config

# Complementary subcategories for realistic cross-sell bundling
COMPLEMENTARY_SUBCATS = {
    # Electronics
    "Smartphone": ["Smartwatch", "Headphones", "Phone Holder"],
    "Smartwatch": ["Smartphone", "Headphones", "Running Shoes"],
    "Headphones": ["Smartphone", "Smartwatch", "Laptop"],
    "TV": ["Gaming Monitor", "Mechanical Keyboard", "Floor Lamp"],
    # Gaming
    "Gaming Mouse": ["Mechanical Keyboard", "Gaming Headset", "Gaming Chair", "Gaming Monitor"],
    "Mechanical Keyboard": ["Gaming Mouse", "Gaming Headset", "Gaming Monitor"],
    "Gaming Headset": ["Gaming Mouse", "Mechanical Keyboard", "Gaming Chair"],
    "Gaming Chair": ["Study Table", "Gaming Mouse", "Mechanical Keyboard"],
    "Gaming Monitor": ["Mechanical Keyboard", "Gaming Mouse", "Gaming Headset"],
    # Fashion
    "T-Shirt": ["Jeans", "Running Shoes"],
    "Jeans": ["T-Shirt", "Running Shoes"],
    "Running Shoes": ["T-Shirt", "Jeans", "Yoga Mat", "Dumbbell Set"],
    # Sports
    "Yoga Mat": ["Dumbbell Set", "Running Shoes"],
    "Dumbbell Set": ["Yoga Mat", "Protein Powder", "Cricket Bat"],
    "Cricket Bat": ["Dumbbell Set", "Running Shoes"],
    # Beauty
    "Foundation": ["Serum", "Lipstick"],
    "Serum": ["Foundation", "Lipstick"],
    "Lipstick": ["Foundation", "Serum"],
    # Furniture
    "Office Chair": ["Study Table", "Bookshelf", "Floor Lamp"],
    "Study Table": ["Office Chair", "Bookshelf", "Floor Lamp"],
    "Bookshelf": ["Study Table", "Office Chair", "Finance Book", "Self-help Book"],
    "Floor Lamp": ["Study Table", "Office Chair", "Bean Bag"],
    "Bean Bag": ["Floor Lamp", "Study Table"],
    # Appliances & Kitchen
    "Air Fryer": ["Toaster", "Blender", "Mixer Grinder"],
    "Blender": ["Mixer Grinder", "Electric Kettle", "Protein Powder"],
    "Electric Kettle": ["Coffee Beans", "Premium Tea", "Toaster"],
    "Mixer Grinder": ["Pressure Cooker", "Blender", "Air Fryer"],
    "Pressure Cooker": ["Mixer Grinder", "Vacuum Cleaner"],
    "Vacuum Cleaner": ["Air Fryer", "Floor Lamp"],
    # Books
    "Finance Book": ["Self-help Book", "Bookshelf"],
    "Self-help Book": ["Finance Book", "Bookshelf"],
    # Food & Grocery
    "Coffee Beans": ["Electric Kettle", "Snack Pack"],
    "Premium Tea": ["Electric Kettle", "Dry Fruits Mix"],
    "Protein Powder": ["Dumbbell Set", "Blender", "Yoga Mat"],
    # Automotive
    "Phone Holder": ["Smartphone", "Car Air Freshener", "Car Wax"],
    "Car Air Freshener": ["Seat Covers", "Car Wax", "Phone Holder"],
    "Car Wax": ["Car Air Freshener", "Seat Covers"],
    "Seat Covers": ["Car Air Freshener", "Phone Holder"],
}

# Funnel progression probabilities
FUNNEL_TRANSITIONS = {
    "page_view": {"page_view": 0.68, "add_to_cart": 0.18, "wishlist_add": 0.10, "purchase": 0.04},
    "add_to_cart": {"page_view": 0.22, "add_to_cart": 0.18, "purchase": 0.50, "wishlist_add": 0.10},
    "wishlist_add": {"page_view": 0.50, "add_to_cart": 0.30, "purchase": 0.15, "wishlist_add": 0.05},
    "purchase": {"page_view": 0.65, "add_to_cart": 0.15, "purchase": 0.05, "wishlist_add": 0.15},
}


def extract_subcategory(p):
    """Extract fine-grained subcategory from product metadata."""
    name = p.get("name", "")
    brand = p.get("brand", "")
    if brand:
        m = re.search(re.escape(brand) + r"\s+(.*?)\s+Model", name, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    words = name.split()
    return words[1] if len(words) > 1 else "General"


def build_catalog_indices(products):
    """
    Build rich indices over products for intent-driven sampling.
    """
    prod_map = {}
    by_category = defaultdict(list)
    by_subcategory = defaultdict(list)
    by_brand = defaultdict(list)
    by_price_tier = {"budget": [], "mid": [], "premium": []}

    for p in products:
        pid = int(p["id"])
        name = p.get("name", "")
        cat = p.get("category", "General")
        brand = p.get("brand", "")
        subcat = extract_subcategory(p)
        price = float(p.get("livePrice") or p.get("basePrice", 1000))

        item_meta = {
            "id": pid,
            "name": name,
            "category": cat,
            "brand": brand,
            "subcategory": subcat,
            "price": price,
            "rating": float(p.get("rating", 4.0)),
        }
        prod_map[pid] = item_meta
        by_category[cat].append(pid)
        by_subcategory[subcat].append(pid)
        if brand:
            by_brand[brand].append(pid)

        if price < 3000:
            by_price_tier["budget"].append(pid)
        elif price < 15000:
            by_price_tier["mid"].append(pid)
        else:
            by_price_tier["premium"].append(pid)

    return prod_map, by_category, by_subcategory, by_brand, by_price_tier


def sample_intent_sequence(
    intent,
    user_persona,
    prod_map,
    by_category,
    by_subcategory,
    by_brand,
    by_price_tier,
    target_len,
):
    """
    Generate an ordered sequence of canonical product IDs following a shopping intent.
    """
    sequence = []
    pids = list(prod_map.keys())

    if intent == "comparison":
        # User compares alternatives in the same subcategory
        pref_cat = user_persona["primary_category"]
        cat_pids = by_category.get(pref_cat, pids)
        anchor_pid = random.choice(cat_pids)
        anchor_subcat = prod_map[anchor_pid]["subcategory"]
        candidates = by_subcategory.get(anchor_subcat, [anchor_pid])

        sequence.append(anchor_pid)
        for _ in range(target_len - 1):
            # 20% chance of re-examining a previously viewed candidate in this session
            if len(sequence) >= 2 and random.random() < 0.20:
                sequence.append(random.choice(sequence[:-1]))
            else:
                # Pick another candidate in same subcategory (preferring different brand)
                curr_brand = prod_map[sequence[-1]]["brand"]
                diff_brand_cands = [c for c in candidates if prod_map[c]["brand"] != curr_brand]
                pool = diff_brand_cands if diff_brand_cands else candidates
                sequence.append(random.choice(pool))

    elif intent == "complementary":
        # Cross-sell ecosystem bundle
        pref_cat = user_persona["primary_category"]
        anchor_pid = random.choice(by_category.get(pref_cat, pids))
        sequence.append(anchor_pid)

        for _ in range(target_len - 1):
            curr_pid = sequence[-1]
            curr_subcat = prod_map[curr_pid]["subcategory"]
            comp_subcats = COMPLEMENTARY_SUBCATS.get(curr_subcat, [])

            # Check if we should re-examine anchor (15% chance)
            if len(sequence) >= 3 and random.random() < 0.15:
                sequence.append(anchor_pid)
            elif comp_subcats and random.random() < 0.75:
                next_subcat = random.choice(comp_subcats)
                cands = by_subcategory.get(next_subcat, [])
                if cands:
                    sequence.append(random.choice(cands))
                else:
                    sequence.append(random.choice(by_category.get(prod_map[curr_pid]["category"], pids)))
            else:
                # Browse in same category
                same_cat_cands = by_category.get(prod_map[curr_pid]["category"], pids)
                sequence.append(random.choice(same_cat_cands))

    elif intent == "brand_loyalty":
        # Fan of specific brand
        brand = user_persona["brand_affinity"]
        brand_pids = by_brand.get(brand, [])
        if len(brand_pids) >= 2:
            pool = brand_pids
        else:
            pool = by_category.get(user_persona["primary_category"], pids)

        anchor = random.choice(pool)
        sequence.append(anchor)
        for _ in range(target_len - 1):
            if len(sequence) >= 2 and random.random() < 0.18:
                sequence.append(random.choice(sequence[:-1]))
            else:
                sequence.append(random.choice(pool))

    elif intent == "category_discovery":
        # Exploration across a parent department
        cat = user_persona["primary_category"]
        pool = by_category.get(cat, pids)
        sequence.append(random.choice(pool))
        for _ in range(target_len - 1):
            if len(sequence) >= 2 and random.random() < 0.15:
                sequence.append(random.choice(sequence[:-1]))
            else:
                sequence.append(random.choice(pool))

    else:  # price_tier
        tier = user_persona["price_tier"]
        pool = by_price_tier.get(tier, pids)
        sequence.append(random.choice(pool))
        for _ in range(target_len - 1):
            if len(sequence) >= 2 and random.random() < 0.15:
                sequence.append(random.choice(sequence[:-1]))
            else:
                sequence.append(random.choice(pool))

    return sequence


def generate_synthetic_events(db=None, num_sessions=None, seed=None):
    """
    Generate coherent synthetic browsing events following the realistic intent framework.
    Returns (list of event dicts, count of sessions).
    """
    if db is None:
        db = get_db()
    if num_sessions is None:
        num_sessions = config.SYNTHETIC_NUM_SESSIONS
    if seed is None:
        seed = config.RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)

    # 1. Load real catalog from MongoDB
    products = list(db.products.find({}, {
        "_id": 1, "id": 1, "category": 1, "name": 1, "brand": 1, "livePrice": 1, "basePrice": 1, "rating": 1
    }))
    if not products:
        print("ERROR: No products in MongoDB. Run product seed first.")
        return [], 0

    prod_map, by_category, by_subcategory, by_brand, by_price_tier = build_catalog_indices(products)
    all_categories = list(by_category.keys())
    all_brands = [b for b, lst in by_brand.items() if len(lst) >= 2]

    # 2. Build 500 returning user personas
    num_users = min(config.SYNTHETIC_NUM_USERS, num_sessions)
    users = []
    for _ in range(num_users):
        primary_cat = random.choice(all_categories)
        brand = random.choice(all_brands) if all_brands else "Samsung"
        tier = random.choices(["budget", "mid", "premium"], weights=[0.35, 0.45, 0.20])[0]
        users.append({
            "user_id": str(uuid.uuid4()),
            "primary_category": primary_cat,
            "brand_affinity": brand,
            "price_tier": tier,
        })

    # Intent distribution
    intents = ["comparison", "complementary", "brand_loyalty", "category_discovery", "price_tier"]
    intent_weights = [0.40, 0.25, 0.15, 0.15, 0.05]

    now = datetime.utcnow()
    events = []

    for session_idx in range(num_sessions):
        user = random.choice(users)
        session_id = str(uuid.uuid4())
        intent = random.choices(intents, weights=intent_weights)[0]

        # Session length (4 to 16 events)
        min_events, max_events = config.SYNTHETIC_EVENTS_PER_SESSION_RANGE
        target_len = random.randint(min_events, max_events)

        # Generate sequence of product interactions under this intent
        product_seq = sample_intent_sequence(
            intent,
            user,
            prod_map,
            by_category,
            by_subcategory,
            by_brand,
            by_price_tier,
            target_len,
        )

        # Spread sessions chronologically across 30 days
        days_ago = random.uniform(0, config.SYNTHETIC_DAYS_SPAN)
        current_time = now - timedelta(days=days_ago)

        current_event_type = "page_view"

        for pid in product_seq:
            prod_info = prod_map[pid]
            cat = prod_info["category"]
            price = prod_info["price"]

            # Advance time coherently (15s to 120s between clicks)
            current_time += timedelta(seconds=random.randint(15, 120))

            events.append({
                "eventType": current_event_type,
                "productId": int(pid),
                "sessionId": session_id,
                "userId": user["user_id"],
                "amount": price if current_event_type == "purchase" else 0,
                "timestamp": current_time,
                "createdAt": current_time,
                "metadata": {
                    "source": "SYNTHETIC",
                    "intent": intent,
                    "category": cat,
                    "subcategory": prod_info["subcategory"],
                    "brand": prod_info["brand"],
                    "productName": prod_info["name"],
                    "price": price,
                    "deviceType": random.choice(["mobile", "desktop", "tablet"]),
                    "referralSource": random.choice(["direct", "google", "instagram", "affiliate"]),
                },
            })

            # Transition event type through e-commerce funnel
            transitions = FUNNEL_TRANSITIONS.get(current_event_type, FUNNEL_TRANSITIONS["page_view"])
            next_types = list(transitions.keys())
            next_weights = list(transitions.values())
            current_event_type = random.choices(next_types, weights=next_weights, k=1)[0]

    return events, num_sessions


def seed_synthetic_events(db=None, num_sessions=None, clear_existing=True):
    """Generate and insert coherent synthetic events into MongoDB."""
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

    print(f"  ✅ Seeded {len(events):,} intent-driven events across {num_sess:,} sessions")
    return len(events)


if __name__ == "__main__":
    print("=" * 60)
    print("  COHERENT E-COMMERCE INTENT-DRIVEN SYNTHETIC GENERATOR")
    print("=" * 60)
    count = seed_synthetic_events()
    print(f"Done. Total events: {count:,}")
