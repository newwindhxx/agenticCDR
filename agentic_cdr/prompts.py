from __future__ import annotations

import json
from typing import Any


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def decision_prompt(
    domain: str,
    private_memory: str,
    fused_memory: str,
    negative_item: dict[str, str],
    positive_item: dict[str, str],
) -> str:
    return f"""You simulate an Amazon user's choice in the {domain} domain.

Domain-separated memory:
{private_memory}

Domain-fused memory:
{fused_memory}

Candidate NEG:
{_dump(negative_item)}

Candidate POS:
{_dump(positive_item)}

Choose the candidate that best matches the memories and explain briefly.
Return exactly one JSON object:
{{"choice":"NEG or POS","reason":"brief evidence-based reason"}}
"""


def private_update_prompt(
    domain: str,
    old_memory: str,
    positive_item: dict[str, str],
    negative_item: dict[str, str],
    choice: str,
    reason: str,
) -> str:
    correctness = "correct" if choice == "POS" else "incorrect"
    return f"""Update an Amazon user's domain-separated memory for {domain}.

Previous memory:
{old_memory}

The observed positive item was:
{_dump(positive_item)}

The sampled negative item was:
{_dump(negative_item)}

The simulated choice was {choice}, which was {correctness}. Its reason was:
{reason}

Infer concrete likes and dislikes from the ground-truth positive-versus-negative feedback. Preserve useful prior preferences, remove contradictions, do not mention item IDs or the update process, and stay under 150 words.
Return exactly one JSON object:
{{"memory":"updated preference text"}}
"""


def fused_update_prompt(
    target_domain: str,
    old_fused_memory: str,
    private_memories: dict[str, str],
) -> str:
    return f"""Update the user's domain-fused memory for the target domain {target_domain}.

Previous fused memory:
{old_fused_memory}

Current domain-separated memories:
{_dump(private_memories)}

Extract only preferences from all domains that are useful for decisions in {target_domain}, then integrate them with the previous fused memory. Exclude source-domain-specific details that do not transfer. Do not mention other domain names or the reasoning process. Stay under 180 words.
Return exactly one JSON object:
{{"memory":"updated target-domain fused preference text"}}
"""


def item_update_prompt(
    domain: str,
    fused_memory: str,
    positive_item: dict[str, str],
    negative_item: dict[str, str],
    choice: str,
    reason: str,
) -> str:
    return f"""Update two item-agent memories after observed feedback in {domain}.

User's fused preference:
{fused_memory}

Ground-truth positive item:
{_dump(positive_item)}

Sampled negative item:
{_dump(negative_item)}

The simulated choice was {choice}; its reason was:
{reason}

Preserve factual item attributes. Add concise evidence about what kinds of preferences each item may or may not satisfy. Do not invent attributes. Keep each memory under 80 words.
Return exactly one JSON object:
{{"positive_memory":"text","negative_memory":"text"}}
"""


def ranking_prompt(
    domain: str,
    private_memory: str,
    fused_memory: str,
    group_memory: str,
    candidates: list[dict[str, str]],
) -> str:
    labels = [entry["label"] for entry in candidates]
    group_section = group_memory or "No group-shared memory is enabled."
    return f"""Rank candidate items for an Amazon user in {domain}.

Domain-separated memory:
{private_memory}

Domain-fused memory:
{fused_memory}

Group-shared memory:
{group_section}

Candidates:
{_dump(candidates)}

Rank all candidates from most to least preferred. Use every label exactly once. Do not add or omit labels.
Required labels: {_dump(labels)}
Return exactly one JSON object:
{{"ranking":{_dump(labels)}}}
"""


def tag_prompt(private_memories: dict[str, str], fused_memories: dict[str, str]) -> str:
    return f"""Extract concise English interest tags from this user's memories.

Domain-separated memories:
{_dump(private_memories)}

Domain-fused memories:
{_dump(fused_memories)}

Return 3 to 10 specific interest tags. Avoid generic tags such as shopping or Amazon.
Return exactly one JSON object:
{{"tags":["tag one","tag two"]}}
"""


def group_name_prompt(tags: list[str]) -> str:
    return f"""Summarize these related user-interest tags as one short English noun phrase:
{_dump(tags)}

Return exactly one JSON object:
{{"name":"short interest group name"}}
"""
