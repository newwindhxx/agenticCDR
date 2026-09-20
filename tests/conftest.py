from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


def _millis(day: int) -> int:
    value = datetime(2021, 10, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return int(value.timestamp() * 1000)


def _write_gzip_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@pytest.fixture
def synthetic_project(tmp_path: Path) -> dict[str, Any]:
    raw = tmp_path / "data/raw/amazon2023"
    book_reviews: list[dict[str, Any]] = []
    movie_reviews: list[dict[str, Any]] = []
    book_meta: list[dict[str, Any]] = []
    movie_meta: list[dict[str, Any]] = []

    for user_number in range(2):
        user_id = f"U{user_number}"
        for index in range(6):
            item_id = f"B{user_number}{index:02d}"
            book_reviews.append({"user_id": user_id, "parent_asin": item_id, "rating": 5.0, "timestamp": _millis(index + 1)})
            book_meta.append({"parent_asin": item_id, "title": f"Book {item_id}", "description": [f"Description {item_id}"], "categories": ["Books", "Fiction"], "price": "9.99"})
        for index, day in enumerate((8, 10, 20, 21)):
            item_id = f"M{user_number}{index:02d}"
            movie_reviews.append({"user_id": user_id, "parent_asin": item_id, "rating": 5.0, "timestamp": _millis(day)})
            movie_meta.append({"parent_asin": item_id, "title": f"Movie {item_id}", "description": [f"Description {item_id}"], "categories": ["Movies", "Drama"], "price": "12.99"})

    # These low-activity users provide enough unseen items for negative sampling.
    for index in range(14):
        movie_id = f"MX{index:02d}"
        movie_reviews.append({"user_id": f"BACKGROUND_M_{index}", "parent_asin": movie_id, "rating": 5.0, "timestamp": _millis(4)})
        movie_meta.append({"parent_asin": movie_id, "title": f"Background Movie {index}", "description": ["A catalog movie"], "categories": ["Movies"]})
        book_id = f"BX{index:02d}"
        book_reviews.append({"user_id": f"BACKGROUND_B_{index}", "parent_asin": book_id, "rating": 5.0, "timestamp": _millis(4)})
        book_meta.append({"parent_asin": book_id, "title": f"Background Book {index}", "description": ["A catalog book"], "categories": ["Books"]})

    files = {
        "Books.jsonl.gz": book_reviews,
        "Movies_and_TV.jsonl.gz": movie_reviews,
        "meta_Books.jsonl.gz": book_meta,
        "meta_Movies_and_TV.jsonl.gz": movie_meta,
    }
    for name, records in files.items():
        _write_gzip_jsonl(raw / name, records)

    return {
        "experiment_name": "synthetic_books_to_movies",
        "source_domain": "Books",
        "target_domain": "Movies_and_TV",
        "seed": 23,
        "candidate_count": 10,
        "_project_root": str(tmp_path),
        "_config_path": str(tmp_path / "config.yaml"),
        "paths": {"raw_dir": "data/raw/amazon2023", "processed_dir": "data/processed/books_to_movies", "runs_dir": "runs"},
        "downloads": [{"name": name, "url": f"synthetic://{name}"} for name in files],
        "data": {
            "start_date": "2021-10-01T00:00:00Z", "end_date": "2022-03-31T23:59:59.999Z", "min_rating": 4.0,
            "min_source_train": 3, "min_target_train": 1, "min_total_interactions": 10, "max_total_interactions": 100,
            "full_user_count": 2, "smoke_user_count": 1, "training_negative_pool_size": 5,
        },
        "llm": {"provider": "deepseek", "model": "deepseek-chat", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "temperature": 0, "max_tokens": 800, "max_retries": 3},
        "embedding": {"model": "fake-embedding", "device": "cpu", "batch_size": 8},
        "group_memory": {"random_seed": 42, "max_clusters": 4, "top_groups": 2, "recent_items_per_domain": 2},
        "profiles": {
            "trial": {
                "user_limit": 1,
                "max_train_interactions_per_user": 1,
                "group_memory": False,
                "users_file": "users_smoke.json",
            },
            "smoke": {"user_limit": 1, "max_train_interactions_per_user": 6, "group_memory": False},
            "full": {"user_limit": 2, "max_train_interactions_per_user": None, "group_memory": True},
        },
    }
