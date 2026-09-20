from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import configured_path, get_profile
from .io_utils import atomic_write_json, read_json, read_jsonl, sha256_file, write_jsonl
from .llm import LLMClient
from .memory import MemoryStore
from .prompts import ranking_prompt
from .run_context import RunContext, latest_run


def _ranking_validator(expected: list[str]):
    expected_set = set(expected)

    def validate(value: dict[str, Any]) -> list[str]:
        ranking = value["ranking"]
        if not isinstance(ranking, list):
            raise ValueError("ranking must be a list")
        labels = [str(label).strip() for label in ranking]
        if len(labels) != len(expected):
            raise ValueError(f"ranking must contain {len(expected)} labels")
        if len(set(labels)) != len(labels):
            raise ValueError("ranking contains duplicate labels")
        if set(labels) != expected_set:
            raise ValueError("ranking labels do not match the candidate labels")
        return labels

    return validate


def _group_text(groups: dict[str, Any] | None, user_id: str, target_domain: str) -> str:
    if not groups:
        return ""
    by_id = {group["group_id"]: group for group in groups["groups"]}
    lines = []
    for group_id in groups["user_groups"].get(user_id, []):
        group = by_id[group_id]
        recent = group["recent_items"].get(target_domain, [])
        if recent:
            lines.append(f"Users in interest group '{group['name']}' recently interacted with: {recent}")
    return "\n".join(lines)


def _metric_at(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def evaluate(
    config: dict[str, Any],
    profile_name: str,
    split: str,
    use_group_memory: bool = False,
    run_id: str | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    profile = get_profile(config, profile_name)
    context = (
        latest_run(config, profile_name)
        if run_id is None
        else RunContext(run_id, configured_path(config, "runs_dir") / run_id, profile_name)
    )
    store = MemoryStore.load(context.state_path)
    processed = configured_path(config, "processed_dir")
    candidates = read_jsonl(processed / f"candidates_{split}.jsonl")
    with (processed / f"users_{profile_name}.json").open("r", encoding="utf-8") as handle:
        user_ids = {str(value) for value in json.load(handle)}
    candidates = [record for record in candidates if str(record["user_id"]) in user_ids]
    if len(candidates) != len(user_ids):
        raise RuntimeError(
            f"Expected one {split} candidate set for each of {len(user_ids)} users, found {len(candidates)}"
        )
    groups = None
    groups_path = context.run_dir / "memory" / "groups.json"
    if use_group_memory:
        if not profile.get("group_memory"):
            raise ValueError(f"Profile {profile_name} has group memory disabled")
        if not groups_path.exists():
            raise FileNotFoundError("Group memory is missing; run build_group_memory.py first")
        groups = read_json(groups_path)
        if groups.get("state_sha256") != sha256_file(context.state_path):
            raise RuntimeError("Group memory was built from a different training state")

    state_hash_before = sha256_file(context.state_path)
    groups_hash_before = sha256_file(groups_path) if use_group_memory else None
    llm = llm or LLMClient(config["llm"], context.cache_path)
    target = str(config["target_domain"])
    evaluation_id = f"{split}_{'group' if use_group_memory else 'nogroup'}"
    prediction_path = context.run_dir / f"predictions_{evaluation_id}.jsonl"
    existing = {
        str(record["user_id"]): record
        for record in read_jsonl(prediction_path)
    } if prediction_path.exists() else {}
    predictions: list[dict[str, Any]] = []

    for candidate_record in candidates:
        user_id = str(candidate_record["user_id"])
        if user_id in existing:
            predictions.append(existing[user_id])
            continue
        user_memory = store.user_memory(user_id, target)
        prompt_candidates = [
            {
                "label": str(entry["label"]),
                "item_id": str(entry["item_id"]),
                "memory": store.item_memory(str(entry["item_id"])),
            }
            for entry in candidate_record["candidates"]
        ]
        expected = [entry["label"] for entry in prompt_candidates]
        ranking = llm.call_json(
            ranking_prompt(
                target,
                user_memory["private"],
                user_memory["fused"],
                _group_text(groups, user_id, target),
                prompt_candidates,
            ),
            f"ranking:{evaluation_id}:{user_id}",
            _ranking_validator(expected),
        )
        positive_label = str(candidate_record["positive_label"])
        rank = ranking.index(positive_label) + 1
        predictions.append(
            {
                "evaluation_id": evaluation_id,
                "user_id": user_id,
                "positive_item_id": str(candidate_record["positive_item_id"]),
                "positive_label": positive_label,
                "ranking": ranking,
                "rank": rank,
            }
        )
        write_jsonl(prediction_path, predictions)

    aggregate_prediction_path = context.run_dir / "predictions.jsonl"
    aggregate_predictions = (
        [
            record
            for record in read_jsonl(aggregate_prediction_path)
            if record.get("evaluation_id") != evaluation_id
        ]
        if aggregate_prediction_path.exists()
        else []
    )
    write_jsonl(aggregate_prediction_path, aggregate_predictions + predictions)

    ranks = [int(record["rank"]) for record in predictions]
    count = len(ranks)
    metrics = {
        "evaluation_id": evaluation_id,
        "profile": profile_name,
        "split": split,
        "use_group_memory": use_group_memory,
        "valid_samples": count,
        "parse_failures": 0,
        "MRR": sum(1.0 / rank for rank in ranks) / count,
        "NDCG@1": sum(_metric_at(rank, 1) for rank in ranks) / count,
        "NDCG@5": sum(_metric_at(rank, 5) for rank in ranks) / count,
        "NDCG@10": sum(_metric_at(rank, 10) for rank in ranks) / count,
        "average_target_rank": sum(ranks) / count,
        "per_user": {record["user_id"]: {"rank": record["rank"]} for record in predictions},
    }
    atomic_write_json(context.run_dir / f"metrics_{evaluation_id}.json", metrics)
    aggregate_path = context.run_dir / "metrics.json"
    aggregate = read_json(aggregate_path) if aggregate_path.exists() else {}
    aggregate[evaluation_id] = metrics
    atomic_write_json(aggregate_path, aggregate)

    if sha256_file(context.state_path) != state_hash_before:
        raise RuntimeError("Evaluation modified the training memory state")
    if use_group_memory and sha256_file(groups_path) != groups_hash_before:
        raise RuntimeError("Evaluation modified group memory")
    context.log(
        f"Evaluation {evaluation_id} complete: MRR={metrics['MRR']:.6f} NDCG@10={metrics['NDCG@10']:.6f}"
    )
    return metrics
