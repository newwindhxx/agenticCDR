from __future__ import annotations

from typing import Any

from .config import configured_path, get_profile
from .data import load_profile_data
from .io_utils import read_jsonl


def estimate_run(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profile = get_profile(config, profile_name)
    train, _, users = load_profile_data(config, profile)
    processed = configured_path(config, "processed_dir")
    validation = [r for r in read_jsonl(processed / "validation.jsonl") if str(r["user_id"]) in set(users)]
    test = [r for r in read_jsonl(processed / "test.jsonl") if str(r["user_id"]) in set(users)]
    training_calls = len(train) * 4
    ranking_calls = len(validation) + len(test)
    tag_calls = len(users) if profile.get("group_memory") else 0
    group_name_calls = int(config["group_memory"]["top_groups"]) if profile.get("group_memory") else 0
    total = training_calls + ranking_calls + tag_calls + group_name_calls
    # Conservative planning estimates. Actual usage is recorded by the API client per request.
    approximate_input_tokens = (
        training_calls * 650
        + ranking_calls * 1_200
        + tag_calls * 600
        + group_name_calls * 250
    )
    return {
        "profile": profile_name,
        "users": len(users),
        "training_interactions": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "requests": {
            "training": training_calls,
            "ranking": ranking_calls,
            "interest_tags": tag_calls,
            "group_names_upper_bound": group_name_calls,
            "total_upper_bound": total,
        },
        "tokens": {
            "approximate_input": approximate_input_tokens,
            "maximum_output": total * int(config["llm"].get("max_tokens", 800)),
            "note": "Input is an approximation; actual usage is saved in llm_cache.jsonl.",
        },
    }
