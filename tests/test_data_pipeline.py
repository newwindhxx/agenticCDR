from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from agentic_cdr.data import prepare_pair
from agentic_cdr.io_utils import read_json, read_jsonl


def test_preparation_is_deterministic_and_leak_free(synthetic_project):
    first = prepare_pair(synthetic_project)
    processed = Path(synthetic_project["_project_root"]) / synthetic_project["paths"]["processed_dir"]
    assert not (processed / ".prepare.sqlite").exists()
    train = pd.read_parquet(processed / "train.parquet")
    validation = {row["user_id"]: row for row in read_jsonl(processed / "validation.jsonl")}
    candidates = read_jsonl(processed / "candidates_test.jsonl")
    users = read_json(processed / "users_full.json")

    assert len(users) == 2
    assert set(train["domain"]) == {"Books", "Movies_and_TV"}
    for user_id in users:
        user_train = train[train["user_id"] == user_id]
        assert (user_train["timestamp"] < validation[user_id]["timestamp"]).all()
        assert sum(user_train["domain"] == "Books") >= 3
        assert sum(user_train["domain"] == "Movies_and_TV") >= 1

    observed_movies: dict[str, set[str]] = {user_id: set() for user_id in users}
    raw_path = Path(synthetic_project["_project_root"]) / "data/raw/amazon2023/Movies_and_TV.jsonl.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["user_id"] in observed_movies:
                observed_movies[row["user_id"]].add(row["parent_asin"])
    for row in candidates:
        item_ids = [entry["item_id"] for entry in row["candidates"]]
        negative_ids = set(item_ids) - {row["positive_item_id"]}
        assert len(item_ids) == 10
        assert len(set(item_ids)) == 10
        assert not (negative_ids & observed_movies[row["user_id"]])
        assert row["positive_label"] in {entry["label"] for entry in row["candidates"]}

    second = prepare_pair(synthetic_project, force=True)
    assert first["processed_sha256"] == second["processed_sha256"]
    assert not (processed / ".prepare.sqlite").exists()
