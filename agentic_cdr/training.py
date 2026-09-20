from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .config import configured_path, get_profile
from .data import load_profile_data
from .io_utils import sha256_file
from .llm import LLMClient
from .memory import MemoryStore
from .prompts import decision_prompt, fused_update_prompt, item_update_prompt, private_update_prompt
from .run_context import RunContext, create_or_resume_run


def _item_payload(item_id: str, memory: str) -> dict[str, str]:
    return {"item_id": item_id, "memory": memory}


def _decision(value: dict[str, Any]) -> dict[str, str]:
    choice = str(value["choice"]).strip().upper()
    reason = str(value["reason"]).strip()
    if choice not in {"NEG", "POS"}:
        raise ValueError("choice must be NEG or POS")
    if not reason:
        raise ValueError("reason must not be empty")
    return {"choice": choice, "reason": reason}


def _memory(value: dict[str, Any]) -> str:
    text = str(value["memory"]).strip()
    if not text:
        raise ValueError("memory must not be empty")
    return text


def _item_memories(value: dict[str, Any]) -> dict[str, str]:
    positive = str(value["positive_memory"]).strip()
    negative = str(value["negative_memory"]).strip()
    if not positive or not negative:
        raise ValueError("both item memories must be non-empty")
    return {"positive": positive, "negative": negative}


def train(
    config: dict[str, Any],
    profile_name: str,
    resume: bool = False,
    run_id: str | None = None,
    llm: Any | None = None,
) -> RunContext:
    profile = get_profile(config, profile_name)
    train_frame, items, user_ids = load_profile_data(config, profile)
    context = create_or_resume_run(config, profile_name, resume=resume, run_id=run_id)
    manifest_hash = sha256_file(configured_path(config, "processed_dir") / "manifest.json")
    domains = [str(config["source_domain"]), str(config["target_domain"])]

    if context.state_path.exists():
        store = MemoryStore.load(context.state_path)
        if store.state.get("profile") != profile_name or store.state.get("dataset_hash") != manifest_hash:
            raise RuntimeError("Checkpoint profile or processed dataset does not match this run")
    else:
        store = MemoryStore.initialize(
            context.state_path, user_ids, domains, items, profile_name, manifest_hash
        )
    llm = llm or LLMClient(config["llm"], context.cache_path)
    start = int(store.state.get("last_completed", -1)) + 1
    context.log(
        f"Training {profile_name}: {len(user_ids)} users, {len(train_frame)} interactions, resume index {start}"
    )

    records = train_frame.to_dict(orient="records")
    for index in tqdm(range(start, len(records)), desc="AgentCF++ train"):
        record = records[index]
        user_id = str(record["user_id"])
        domain = str(record["domain"])
        positive_id = str(record["parent_asin"])
        negative_id = str(record["negative_asin"])
        user_memory = store.user_memory(user_id, domain)
        positive_old = store.item_memory(positive_id)
        negative_old = store.item_memory(negative_id)
        positive_payload = _item_payload(positive_id, positive_old)
        negative_payload = _item_payload(negative_id, negative_old)

        decision = llm.call_json(
            decision_prompt(
                domain,
                user_memory["private"],
                user_memory["fused"],
                negative_payload,
                positive_payload,
            ),
            f"decision:{record['interaction_id']}",
            _decision,
        )
        new_private = llm.call_json(
            private_update_prompt(
                domain,
                user_memory["private"],
                positive_payload,
                negative_payload,
                decision["choice"],
                decision["reason"],
            ),
            f"private-update:{record['interaction_id']}",
            _memory,
        )
        private_memories = {
            current_domain: (
                new_private
                if current_domain == domain
                else store.user_memory(user_id, current_domain)["private"]
            )
            for current_domain in domains
        }
        new_fused = llm.call_json(
            fused_update_prompt(domain, user_memory["fused"], private_memories),
            f"fused-update:{record['interaction_id']}",
            _memory,
        )
        updated_items = llm.call_json(
            item_update_prompt(
                domain,
                new_fused,
                positive_payload,
                negative_payload,
                decision["choice"],
                decision["reason"],
            ),
            f"item-update:{record['interaction_id']}",
            _item_memories,
        )

        # Commit all four logical updates together through one atomic state-file replace.
        store.state["users"][user_id][domain]["private"] = new_private
        store.state["users"][user_id][domain]["fused"] = new_fused
        store.state["items"][positive_id] = updated_items["positive"]
        store.state["items"][negative_id] = updated_items["negative"]
        store.state["last_completed"] = index
        store.save()
        context.log(
            f"completed={index + 1}/{len(records)} user={user_id} domain={domain} item={positive_id}"
        )
    context.log("Training complete")
    return context
