from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .config import configured_path, get_profile
from .data import load_profile_data
from .io_utils import atomic_write_json, sha256_file
from .llm import LLMClient
from .memory import MemoryStore
from .prompts import group_name_prompt, tag_prompt
from .run_context import RunContext, latest_run


def _tags(value: dict[str, Any]) -> list[str]:
    raw = value["tags"]
    if not isinstance(raw, list):
        raise ValueError("tags must be a list")
    tags = []
    seen = set()
    for entry in raw:
        tag = str(entry).strip()
        normalized = tag.casefold()
        if tag and normalized not in seen:
            tags.append(tag)
            seen.add(normalized)
    if not 3 <= len(tags) <= 10:
        raise ValueError("tags must contain 3 to 10 non-empty unique strings")
    return tags


def _group_name(value: dict[str, Any]) -> str:
    name = str(value["name"]).strip()
    if not name:
        raise ValueError("name must not be empty")
    return name


def build_group_memory(
    config: dict[str, Any],
    profile_name: str,
    run_id: str | None = None,
    llm: Any | None = None,
    encoder: Any | None = None,
) -> RunContext:
    profile = get_profile(config, profile_name)
    if not profile.get("group_memory"):
        raise ValueError(f"Profile {profile_name} has group_memory disabled")
    context = (
        latest_run(config, profile_name)
        if run_id is None
        else RunContext(run_id, configured_path(config, "runs_dir") / run_id, profile_name)
    )
    store = MemoryStore.load(context.state_path)
    train_frame, items, user_ids = load_profile_data(config, profile)
    if int(store.state.get("last_completed", -1)) + 1 != len(train_frame):
        raise RuntimeError("Training is incomplete; finish the full run before building groups")
    llm = llm or LLMClient(config["llm"], context.cache_path)
    domains = [str(config["source_domain"]), str(config["target_domain"])]

    user_tags: dict[str, list[str]] = {}
    for user_id in user_ids:
        private = {domain: store.user_memory(user_id, domain)["private"] for domain in domains}
        fused = {domain: store.user_memory(user_id, domain)["fused"] for domain in domains}
        user_tags[user_id] = llm.call_json(
            tag_prompt(private, fused), f"interest-tags:{user_id}", _tags
        )

    tag_rows = [
        (user_id, tag)
        for user_id in sorted(user_tags)
        for tag in user_tags[user_id]
    ]
    if len(tag_rows) < 2:
        raise RuntimeError("At least two interest tags are required for clustering")
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            str(config["embedding"]["model"]),
            device=str(config["embedding"].get("device", "cpu")),
        )
    embeddings = np.asarray(
        encoder.encode(
            [tag for _, tag in tag_rows],
            batch_size=int(config["embedding"].get("batch_size", 64)),
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )
    group_settings = config["group_memory"]
    cluster_count = min(int(group_settings["max_clusters"]), len(tag_rows))
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=int(group_settings["random_seed"]),
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    cluster_users: dict[int, set[str]] = defaultdict(set)
    cluster_tags: dict[int, list[str]] = defaultdict(list)
    for (user_id, tag), label in zip(tag_rows, labels.tolist()):
        cluster_users[int(label)].add(user_id)
        cluster_tags[int(label)].append(tag)
    ranked_clusters = sorted(
        cluster_users,
        key=lambda label: (-len(cluster_users[label]), label),
    )[: int(group_settings["top_groups"])]

    title_by_item = {
        str(row["parent_asin"]): str(row["title"])
        for row in items.to_dict(orient="records")
    }
    recent_count = int(group_settings["recent_items_per_domain"])
    groups: list[dict[str, Any]] = []
    user_groups: dict[str, list[str]] = {user_id: [] for user_id in user_ids}
    for position, label in enumerate(ranked_clusters):
        group_id = f"G{position:03d}"
        users = sorted(cluster_users[label])
        tags = sorted(set(cluster_tags[label]), key=str.casefold)
        name = llm.call_json(
            group_name_prompt(tags), f"group-name:{group_id}", _group_name
        )
        interactions = train_frame[train_frame["user_id"].astype(str).isin(set(users))]
        recent_items: dict[str, list[str]] = {}
        for domain in domains:
            domain_rows = interactions[interactions["domain"] == domain].sort_values("timestamp")
            recent_ids = domain_rows["parent_asin"].astype(str).tolist()[-recent_count:]
            recent_items[domain] = [title_by_item[item_id] for item_id in recent_ids]
        groups.append(
            {
                "group_id": group_id,
                "name": name,
                "users": users,
                "tags": tags,
                "recent_items": recent_items,
            }
        )
        for user_id in users:
            user_groups[user_id].append(group_id)

    group_payload = {
        "version": 1,
        "profile": profile_name,
        "state_sha256": sha256_file(context.state_path),
        "embedding_model": config["embedding"]["model"],
        "cluster_count": cluster_count,
        "groups": groups,
        "user_groups": user_groups,
        "user_tags": user_tags,
    }
    atomic_write_json(context.run_dir / "memory" / "groups.json", group_payload)
    context.log(f"Built {len(groups)} group memories from {len(tag_rows)} tags")
    return context
