from __future__ import annotations

import gzip
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from .config import configured_path
from .io_utils import sha256_file, stable_int, write_jsonl, atomic_write_json


MAIN_CATEGORIES = {
    "Books": "Books",
    "Movies_and_TV": "Movies & TV",
}


class PreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    rating: float
    timestamp: int
    domain: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "parent_asin": self.item_id,
            "rating": self.rating,
            "timestamp": self.timestamp,
            "domain": self.domain,
        }


def _timestamp_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            user_id TEXT NOT NULL,
            parent_asin TEXT NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            domain TEXT NOT NULL,
            PRIMARY KEY (user_id, domain, parent_asin)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            parent_asin TEXT PRIMARY KEY,
            domain TEXT NOT NULL,
            main_category TEXT NOT NULL,
            title TEXT NOT NULL,
            subtitle TEXT NOT NULL,
            description TEXT NOT NULL,
            categories TEXT NOT NULL,
            price TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_interactions_domain ON interactions(domain)")
    connection.commit()
    return connection


def _domain_files(config: dict[str, Any], metadata: bool = False) -> list[tuple[str, Path]]:
    raw_dir = configured_path(config, "raw_dir")
    domains = [str(config["source_domain"]), str(config["target_domain"])]
    result = []
    for domain in domains:
        prefix = "meta_" if metadata else ""
        result.append((domain, raw_dir / f"{prefix}{domain}.jsonl.gz"))
    return result


def _insert_review_batch(connection: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO interactions(user_id, parent_asin, rating, timestamp, domain)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, domain, parent_asin) DO UPDATE SET
            rating = CASE
                WHEN excluded.timestamp < interactions.timestamp THEN excluded.rating
                ELSE interactions.rating
            END,
            timestamp = MIN(interactions.timestamp, excluded.timestamp)
        """,
        batch,
    )
    connection.commit()


def ingest_reviews(connection: sqlite3.Connection, config: dict[str, Any]) -> Counter[str]:
    start_ms = _timestamp_ms(str(config["data"]["start_date"]))
    end_ms = _timestamp_ms(str(config["data"]["end_date"]))
    min_rating = float(config["data"]["min_rating"])
    counts: Counter[str] = Counter()
    for domain, path in _domain_files(config, metadata=False):
        if not path.exists():
            raise PreparationError(f"Missing review file: {path}; run download_amazon2023.py first")
        batch: list[tuple[Any, ...]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"reviews:{domain}", unit=" rows"):
                try:
                    record = json.loads(line)
                    user_id = str(record.get("user_id") or "").strip()
                    item_id = str(record.get("parent_asin") or "").strip()
                    rating = float(record.get("rating"))
                    timestamp = int(record.get("timestamp", record.get("sort_timestamp")))
                except (TypeError, ValueError, json.JSONDecodeError):
                    counts[f"{domain}_malformed"] += 1
                    continue
                if not user_id or not item_id or rating < min_rating or not (start_ms <= timestamp <= end_ms):
                    continue
                batch.append((user_id, item_id, rating, timestamp, domain))
                counts[f"{domain}_accepted_raw"] += 1
                if len(batch) >= 10_000:
                    _insert_review_batch(connection, batch)
                    batch.clear()
        if batch:
            _insert_review_batch(connection, batch)
    return counts


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_flatten_text(item)}" for key, item in value.items() if item)
    if isinstance(value, (list, tuple, set)):
        return "; ".join(part for item in value if (part := _flatten_text(item)))
    return str(value).strip()


def ingest_metadata(connection: sqlite3.Connection, config: dict[str, Any]) -> Counter[str]:
    wanted = {row[0] for row in connection.execute("SELECT DISTINCT parent_asin FROM interactions")}
    counts: Counter[str] = Counter()
    for domain, path in _domain_files(config, metadata=True):
        if not path.exists():
            raise PreparationError(f"Missing metadata file: {path}; run download_amazon2023.py first")
        batch: list[tuple[str, ...]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"metadata:{domain}", unit=" rows"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counts[f"{domain}_malformed_meta"] += 1
                    continue
                item_id = str(record.get("parent_asin") or "").strip()
                if item_id not in wanted:
                    continue
                title = _flatten_text(record.get("title"))
                if not title:
                    counts[f"{domain}_missing_title"] += 1
                    continue
                description = _flatten_text(record.get("description"))
                if not description:
                    description = _flatten_text(record.get("features"))
                batch.append(
                    (
                        item_id,
                        domain,
                        MAIN_CATEGORIES.get(domain, domain),
                        title,
                        _flatten_text(record.get("subtitle")),
                        description,
                        _flatten_text(record.get("categories")),
                        _flatten_text(record.get("price")),
                    )
                )
                counts[f"{domain}_accepted_meta"] += 1
                if len(batch) >= 10_000:
                    connection.executemany(
                        "INSERT OR REPLACE INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    connection.commit()
                    batch.clear()
        if batch:
            connection.executemany("INSERT OR REPLACE INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
            connection.commit()
    return counts


def _user_records(connection: sqlite3.Connection, user_id: str) -> list[Interaction]:
    rows = connection.execute(
        """
        SELECT i.user_id, i.parent_asin, i.rating, i.timestamp, i.domain
        FROM interactions i JOIN items m USING(parent_asin)
        WHERE i.user_id = ?
        ORDER BY i.timestamp, i.domain, i.parent_asin
        """,
        (user_id,),
    )
    return [Interaction(str(u), str(item), float(rating), int(ts), str(domain)) for u, item, rating, ts, domain in rows]


def _eligible_users(
    connection: sqlite3.Connection, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    source = str(config["source_domain"])
    target = str(config["target_domain"])
    settings = config["data"]
    minimum = int(settings["min_total_interactions"])
    maximum = int(settings["max_total_interactions"])
    candidates = connection.execute(
        """
        SELECT i.user_id
        FROM interactions i JOIN items m USING(parent_asin)
        GROUP BY i.user_id
        HAVING COUNT(*) BETWEEN ? AND ?
        ORDER BY i.user_id
        """,
        (minimum, maximum),
    )
    eligible: dict[str, dict[str, Any]] = {}
    for (user_id,) in candidates:
        records = _user_records(connection, str(user_id))
        target_records = [record for record in records if record.domain == target]
        if len(target_records) < 3:
            continue
        validation = target_records[-2]
        test = target_records[-1]
        train = [record for record in records if record.timestamp < validation.timestamp]
        source_count = sum(record.domain == source for record in train)
        target_count = sum(record.domain == target for record in train)
        if source_count < int(settings["min_source_train"]):
            continue
        if target_count < int(settings["min_target_train"]):
            continue
        eligible[str(user_id)] = {
            "records": records,
            "train": train,
            "validation": validation,
            "test": test,
        }
    return eligible


def _item_pool(connection: sqlite3.Connection, domain: str) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT i.parent_asin
            FROM interactions i JOIN items m USING(parent_asin)
            WHERE i.domain = ?
            ORDER BY i.parent_asin
            """,
            (domain,),
        )
    ]


def _fixed_candidate_record(
    user_id: str,
    positive: Interaction,
    split: str,
    target_pool: list[str],
    observed_target: set[str],
    candidate_count: int,
    seed: int,
) -> dict[str, Any]:
    available = [item for item in target_pool if item not in observed_target and item != positive.item_id]
    negative_count = candidate_count - 1
    if len(available) < negative_count:
        raise PreparationError(
            f"User {user_id} has only {len(available)} target negatives; need {negative_count}"
        )
    rng = random.Random(stable_int(f"candidate:{split}:{user_id}", seed))
    negatives = rng.sample(available, negative_count)
    item_ids = negatives + [positive.item_id]
    rng.shuffle(item_ids)
    candidates = [{"label": f"C{index:02d}", "item_id": item_id} for index, item_id in enumerate(item_ids)]
    positive_label = next(entry["label"] for entry in candidates if entry["item_id"] == positive.item_id)
    return {
        "user_id": user_id,
        "positive_item_id": positive.item_id,
        "positive_label": positive_label,
        "candidates": candidates,
    }


def _fetch_items(connection: sqlite3.Connection, item_ids: set[str]) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    ordered = sorted(item_ids)
    for start in range(0, len(ordered), 500):
        chunk = ordered[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            connection.execute(
                f"SELECT parent_asin, domain, main_category, title, subtitle, description, categories, price "
                f"FROM items WHERE parent_asin IN ({placeholders})",
                chunk,
            ).fetchall()
        )
    columns = [
        "parent_asin",
        "domain",
        "main_category",
        "title",
        "subtitle",
        "description",
        "categories",
        "price",
    ]
    frame = pd.DataFrame(rows, columns=columns).sort_values("parent_asin").reset_index(drop=True)
    if len(frame) != len(item_ids):
        found = set(frame["parent_asin"])
        raise PreparationError(f"Metadata missing for items: {sorted(item_ids - found)[:10]}")
    return frame


def _raw_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_dir = configured_path(config, "raw_dir")
    result = []
    for entry in config.get("downloads", []):
        path = raw_dir / str(entry["name"])
        result.append(
            {
                "name": path.name,
                "url": str(entry["url"]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def prepare_pair(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output_dir = configured_path(config, "processed_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    declared_outputs = [
        "train.parquet",
        "validation.jsonl",
        "test.jsonl",
        "items.parquet",
        "candidates_validation.jsonl",
        "candidates_test.jsonl",
        "users_full.json",
        "users_smoke.json",
        "manifest.json",
    ]
    existing = [name for name in declared_outputs if (output_dir / name).exists()]
    if existing and not force:
        raise PreparationError(
            f"Processed outputs already exist ({', '.join(existing)}); use --force to rebuild"
        )
    database_path = output_dir / ".prepare.sqlite"
    if force:
        for name in declared_outputs:
            (output_dir / name).unlink(missing_ok=True)
        for suffix in ("", "-shm", "-wal"):
            Path(str(database_path) + suffix).unlink(missing_ok=True)

    connection = _connect(database_path)
    review_counts = ingest_reviews(connection, config)
    metadata_counts = ingest_metadata(connection, config)
    eligible = _eligible_users(connection, config)
    required_users = int(config["data"]["full_user_count"])
    if len(eligible) < required_users:
        raise PreparationError(
            f"Only {len(eligible)} eligible users remain after filtering; need {required_users}"
        )

    user_ids = sorted(eligible)
    random.Random(int(config["seed"])).shuffle(user_ids)
    selected = user_ids[:required_users]
    smoke_count = int(config["data"]["smoke_user_count"])
    smoke = selected[:smoke_count]
    source = str(config["source_domain"])
    target = str(config["target_domain"])
    pools = {source: _item_pool(connection, source), target: _item_pool(connection, target)}
    seed = int(config["seed"])
    candidate_count = int(config["candidate_count"])
    pool_limit = int(config["data"]["training_negative_pool_size"])

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    validation_candidates: list[dict[str, Any]] = []
    test_candidates: list[dict[str, Any]] = []
    needed_items: set[str] = set()

    for user_id in selected:
        info = eligible[user_id]
        observed_by_domain = {
            domain: {record.item_id for record in info["records"] if record.domain == domain}
            for domain in (source, target)
        }
        negative_pools: dict[str, list[str]] = {}
        for domain in (source, target):
            available = [item for item in pools[domain] if item not in observed_by_domain[domain]]
            if not available:
                raise PreparationError(f"No training negatives for {user_id} in {domain}")
            rng = random.Random(stable_int(f"train-pool:{user_id}:{domain}", seed))
            sample_size = min(pool_limit, len(available))
            negative_pools[domain] = rng.sample(available, sample_size)

        for record in info["train"]:
            interaction_id = f"{record.user_id}:{record.domain}:{record.item_id}:{record.timestamp}"
            pool = negative_pools[record.domain]
            negative = pool[stable_int(interaction_id, seed) % len(pool)]
            row = record.as_dict()
            row["interaction_id"] = interaction_id
            row["negative_asin"] = negative
            train_rows.append(row)
            needed_items.update((record.item_id, negative))

        for split, positive, split_rows, candidate_rows in (
            ("validation", info["validation"], validation_rows, validation_candidates),
            ("test", info["test"], test_rows, test_candidates),
        ):
            split_rows.append(positive.as_dict())
            candidate = _fixed_candidate_record(
                user_id,
                positive,
                split,
                pools[target],
                observed_by_domain[target],
                candidate_count,
                seed,
            )
            candidate_rows.append(candidate)
            needed_items.update(entry["item_id"] for entry in candidate["candidates"])

    train_frame = pd.DataFrame(train_rows).sort_values(
        ["timestamp", "user_id", "domain", "parent_asin"]
    ).reset_index(drop=True)
    train_frame.to_parquet(output_dir / "train.parquet", index=False)
    _fetch_items(connection, needed_items).to_parquet(output_dir / "items.parquet", index=False)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_jsonl(output_dir / "candidates_validation.jsonl", validation_candidates)
    write_jsonl(output_dir / "candidates_test.jsonl", test_candidates)
    atomic_write_json(output_dir / "users_full.json", selected)
    atomic_write_json(output_dir / "users_smoke.json", smoke)

    processed_hashes = {
        name: sha256_file(output_dir / name)
        for name in declared_outputs
        if name != "manifest.json"
    }
    manifest = {
        "protocol": "books_to_movies_target_leave_two_out_v1",
        "source_domain": source,
        "target_domain": target,
        "seed": seed,
        "filters": dict(config["data"]),
        "raw_files": _raw_manifest(config),
        "review_ingest": dict(review_counts),
        "metadata_ingest": dict(metadata_counts),
        "eligible_users": len(eligible),
        "selected_users": len(selected),
        "smoke_users": len(smoke),
        "train_interactions": len(train_frame),
        "train_by_domain": {
            str(key): int(value) for key, value in train_frame["domain"].value_counts().to_dict().items()
        },
        "validation_samples": len(validation_rows),
        "test_samples": len(test_rows),
        "candidate_count": candidate_count,
        "processed_sha256": processed_hashes,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    connection.close()
    for suffix in ("", "-shm", "-wal"):
        Path(str(database_path) + suffix).unlink(missing_ok=True)
    return manifest


def load_profile_user_ids(config: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    processed = configured_path(config, "processed_dir")
    user_file = processed / str(profile.get("users_file", f"users_{profile['name']}.json"))
    with user_file.open("r", encoding="utf-8") as handle:
        user_ids = [str(value) for value in json.load(handle)]
    return user_ids[: int(profile["user_limit"])]


def load_profile_data(
    config: dict[str, Any], profile: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    processed = configured_path(config, "processed_dir")
    user_ids = load_profile_user_ids(config, profile)
    frame = pd.read_parquet(processed / "train.parquet")
    frame = frame[frame["user_id"].astype(str).isin(set(user_ids))].copy()
    limit = profile.get("max_train_interactions_per_user")
    if limit is not None:
        frame = (
            frame.sort_values(["user_id", "timestamp"])
            .groupby("user_id", group_keys=False)
            .tail(int(limit))
        )
    frame = frame.sort_values(["timestamp", "user_id", "domain", "parent_asin"]).reset_index(drop=True)
    items = pd.read_parquet(processed / "items.parquet")
    return frame, items, user_ids
