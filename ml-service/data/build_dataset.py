"""
PriceIQ ML Service — Build Dataset CLI

Loads events from MongoDB (real or synthetic), preprocesses them,
builds sequences, creates temporal train/val/test split, saves vocabulary.

Usage:
    python -m data.build_dataset              # full pipeline
    python -m data.build_dataset --report-only # just print data quality report
    python -m data.build_dataset --seed-synthetic # generate synthetic data first
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from database import get_db
import config
from data.preprocessing import (
    load_product_catalog,
    load_events,
    build_sessions,
    create_sequences,
    temporal_split,
    finalize_report,
)
from data.dataset import ItemVocabulary
from data.synthetic import seed_synthetic_events


def build_dataset(report_only=False, seed_synthetic=False):
    """
    Main dataset building pipeline.
    Returns (train_seqs, val_seqs, test_seqs, vocab, report) or just report if report_only.
    """
    db = get_db()

    # Step 0: Optionally generate synthetic data
    if seed_synthetic:
        print("\n📦 Generating synthetic event data...")
        seed_synthetic_events(db, clear_existing=True)

    # Step 1: Load product catalog
    print("\n📦 Loading product catalog...")
    catalog = load_product_catalog(db)
    print(f"  Found {len(catalog)} products in catalog")

    if not catalog:
        print("  ERROR: No products found. Run backend seed first.")
        return None

    # Step 2: Load and validate events
    print("\n📦 Loading events from MongoDB...")
    events, report, catalog = load_events(db, catalog)

    # Check if we have enough real data
    is_synthetic = False
    if len(events) < 100:
        print(f"\n  ⚠️  Only {len(events)} valid events found.")
        print("  Checking if synthetic data generation is needed...")

        # Check if events collection has any data at all
        total_events_in_db = db.events.count_documents({})
        if total_events_in_db < 100:
            print("  Generating SYNTHETIC data for development...")
            seed_synthetic_events(db, clear_existing=True)
            events, report, catalog = load_events(db, catalog)
            is_synthetic = True
            report.is_synthetic = True
        else:
            print(f"  {total_events_in_db} events in DB but only {len(events)} matched product catalog.")
            print("  This likely means productId format mismatch. Generating SYNTHETIC data...")
            seed_synthetic_events(db, clear_existing=True)
            events, report, catalog = load_events(db, catalog)
            is_synthetic = True
            report.is_synthetic = True

    # Step 3: Build sessions
    print("\n📦 Building sessions...")
    sessions = build_sessions(events)

    # Step 4: Create sequences
    print("📦 Creating sequences...")
    sequences = create_sequences(sessions)

    # Finalize report
    report = finalize_report(report, sessions, sequences)
    report.print_report()

    if report_only:
        return report

    if not sequences:
        print("  ERROR: No usable sequences. Cannot build dataset.")
        return None

    # Step 5: Build vocabulary from ALL product IDs in training data
    print("📦 Building vocabulary...")
    # Use all products from catalog to ensure full coverage
    all_product_ids = list(catalog.keys())
    vocab = ItemVocabulary()
    vocab.build(all_product_ids)
    print(f"  Vocabulary size: {len(vocab)} (PAD=0, UNK=1, items={vocab.num_items})")

    # Step 6: Temporal split
    print("\n📦 Temporal split...")
    train_seqs, val_seqs, test_seqs = temporal_split(sequences)
    print(f"  Train:      {len(train_seqs):,} sequences")
    print(f"  Validation: {len(val_seqs):,} sequences")
    print(f"  Test:       {len(test_seqs):,} sequences")

    if train_seqs and val_seqs and test_seqs:
        print(f"  Train end:  {train_seqs[-1]['end_time']}")
        print(f"  Val end:    {val_seqs[-1]['end_time']}")
        print(f"  Test end:   {test_seqs[-1]['end_time']}")

    # Step 7: Save artifacts
    print("\n📦 Saving artifacts...")
    os.makedirs(config.MAPPINGS_DIR, exist_ok=True)
    vocab.save()

    # Save sequences for evaluation
    sequences_dir = os.path.join(config.ARTIFACTS_DIR, "data")
    os.makedirs(sequences_dir, exist_ok=True)
    with open(os.path.join(sequences_dir, "train_sequences.json"), "w") as f:
        json.dump(train_seqs, f, default=str)
    with open(os.path.join(sequences_dir, "test_sequences.json"), "w") as f:
        json.dump(test_seqs, f, default=str)
    with open(os.path.join(sequences_dir, "val_sequences.json"), "w") as f:
        json.dump(val_seqs, f, default=str)

    # Save split metadata
    split_meta = {
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "test_sequences": len(test_seqs),
        "vocab_size": len(vocab),
        "num_items": vocab.num_items,
        "is_synthetic": is_synthetic,
        "data_quality": report.to_dict(),
    }
    meta_path = os.path.join(config.MAPPINGS_DIR, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(split_meta, f, indent=2, default=str)
    print(f"  Split metadata saved to {meta_path}")
    print(f"  Sequences saved to {sequences_dir}")

    print("\n✅ Dataset build complete!")
    return train_seqs, val_seqs, test_seqs, vocab, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PriceIQ ML dataset")
    parser.add_argument("--report-only", action="store_true", help="Only print data quality report")
    parser.add_argument("--seed-synthetic", action="store_true", help="Generate synthetic data first")
    args = parser.parse_args()

    result = build_dataset(
        report_only=args.report_only,
        seed_synthetic=args.seed_synthetic,
    )

    if result is None:
        print("Dataset build failed.")
        sys.exit(1)
