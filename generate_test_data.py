#!/usr/bin/env python3
"""Generate synthetic Amazon-like .json.gz data for testing the pipeline.

Usage:
    python generate_test_data.py

This creates small synthetic .json.gz files in data/raw/ so you can
test run_pipeline.py without downloading real Amazon data.
"""
import os
import gzip
import json
import random
import time

def generate_synthetic_data(filepath: str, domain_name: str,
                             n_users: int = 500, n_items: int = 300,
                             n_interactions: int = 5000,
                             time_start: int = 1400000000,
                             time_end: int = 1550000000):
    """Generate a synthetic .json.gz file mimicking Amazon Review Data 2018."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    users = [f"USER_{domain_name}_{i}" for i in range(n_users)]
    # Add some shared users across domains
    shared_users = [f"USER_SHARED_{i}" for i in range(n_users // 5)]
    all_users = users + shared_users

    items = [f"ITEM_{domain_name}_{i}" for i in range(n_items)]

    random.seed(42 + hash(domain_name) % 1000)

    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        for _ in range(n_interactions):
            user = random.choice(all_users)
            item = random.choice(items)
            rating = random.choice([1.0, 2.0, 3.0, 4.0, 5.0])
            timestamp = random.randint(time_start, time_end)

            words = random.sample(
                ["great", "good", "bad", "terrible", "amazing", "ok",
                 "love", "hate", "nice", "product", "quality", "price",
                 "fast", "slow", "worth", "buy", "recommend", "enjoy",
                 "disappointed", "excellent", "perfect", "broken"],
                k=random.randint(3, 10))
            review_text = " ".join(words)

            record = {
                "reviewerID": user,
                "asin": item,
                "overall": rating,
                "unixReviewTime": timestamp,
                "reviewText": review_text,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generated {filepath} ({n_interactions} interactions, "
          f"{len(all_users)} users, {n_items} items)")


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")

    # Generate all 4 domains
    configs = [
        ("Digital_Music_test.json.gz", "DM", 400, 200, 4000),
        ("Movies_and_TV_test.json.gz", "MT", 500, 300, 5000),
        ("Video_Games_test.json.gz", "VG", 450, 250, 4500),
        ("Electronics_test.json.gz", "EL", 600, 400, 6000),
    ]

    for filename, short_name, n_u, n_i, n_int in configs:
        generate_synthetic_data(
            os.path.join(data_dir, filename),
            short_name, n_u, n_i, n_int,
            time_start=1400000000,
            time_end=1550000000)

    print(f"\nAll synthetic data generated in {data_dir}/")
    print("Now run: python run_pipeline.py")
