from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import atomic_write_json, read_json


def item_initial_memory(row: dict[str, Any]) -> str:
    fields = [
        ("title", row.get("title")),
        ("main category", row.get("main_category")),
        ("subtitle", row.get("subtitle")),
        ("categories", row.get("categories")),
        ("description", row.get("description")),
        ("price", row.get("price")),
    ]
    return "; ".join(f"{name}: {value}" for name, value in fields if str(value or "").strip())


@dataclass
class MemoryStore:
    path: Path
    state: dict[str, Any]

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        user_ids: list[str],
        domains: list[str],
        items: pd.DataFrame,
        profile: str,
        dataset_hash: str,
    ) -> "MemoryStore":
        user_state = {}
        for user_id in user_ids:
            user_state[user_id] = {
                domain: {
                    "private": f"I am an Amazon buyer, and I enjoy {domain} very much.",
                    "fused": f"I am an Amazon buyer, and I enjoy {domain} very much.",
                }
                for domain in domains
            }
        item_state = {
            str(row["parent_asin"]): item_initial_memory(row)
            for row in items.to_dict(orient="records")
        }
        store = cls(
            Path(path),
            {
                "version": 1,
                "profile": profile,
                "dataset_hash": dataset_hash,
                "last_completed": -1,
                "users": user_state,
                "items": item_state,
            },
        )
        store.save()
        return store

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        destination = Path(path)
        return cls(destination, read_json(destination))

    def save(self) -> None:
        atomic_write_json(self.path, self.state)

    def user_memory(self, user_id: str, domain: str) -> dict[str, str]:
        try:
            return self.state["users"][user_id][domain]
        except KeyError as exc:
            raise KeyError(f"Missing user memory for {user_id}/{domain}") from exc

    def item_memory(self, item_id: str) -> str:
        try:
            return str(self.state["items"][item_id])
        except KeyError as exc:
            raise KeyError(f"Missing item memory for {item_id}") from exc
