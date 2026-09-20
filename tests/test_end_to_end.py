from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from agentic_cdr.data import prepare_pair
from agentic_cdr.evaluation import evaluate
from agentic_cdr.group_memory import build_group_memory
from agentic_cdr.io_utils import read_json, sha256_file
from agentic_cdr.llm import ScriptedLLM
from agentic_cdr.training import train


def responder(purpose: str, prompt: str):
    if purpose.startswith("decision:"):
        return {"choice": "POS", "reason": "The positive item matches the current preference."}
    if purpose.startswith("private-update:"):
        return {"memory": "Prefers detailed stories and dislikes poorly matched alternatives."}
    if purpose.startswith("fused-update:"):
        return {"memory": "Prefers detailed, character-driven stories."}
    if purpose.startswith("item-update:"):
        return {"positive_memory": "A detailed item suited to story-focused users.", "negative_memory": "An alternative item with a different appeal."}
    if purpose.startswith("ranking:"):
        return {"ranking": sorted(set(re.findall(r"C\d{2}", prompt)))}
    if purpose.startswith("interest-tags:"):
        return {"tags": ["character driven stories", "detailed worlds", "dramatic narratives"]}
    if purpose.startswith("group-name:"):
        return {"name": "narrative enthusiasts"}
    raise AssertionError(f"Unexpected purpose: {purpose}")


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for index, _ in enumerate(texts):
            vector = np.array([index + 1, (index + 1) ** 2, 1.0, 0.5], dtype=float)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors)


def test_smoke_training_resume_and_frozen_evaluation(synthetic_project):
    prepare_pair(synthetic_project)
    llm = ScriptedLLM(responder)
    context = train(synthetic_project, "smoke", run_id="smoke-test", llm=llm)
    assert read_json(context.state_path)["last_completed"] == 5
    calls_after_train = len(llm.calls)
    train(synthetic_project, "smoke", resume=True, run_id="smoke-test", llm=llm)
    assert len(llm.calls) == calls_after_train

    state_hash = sha256_file(context.state_path)
    metrics = evaluate(synthetic_project, "smoke", "test", run_id="smoke-test", llm=llm)
    assert metrics["valid_samples"] == 1
    assert metrics["parse_failures"] == 0
    assert sha256_file(context.state_path) == state_hash
    assert (context.run_dir / "predictions_test_nogroup.jsonl").exists()
    assert (context.run_dir / "predictions.jsonl").exists()
    assert (context.run_dir / "metrics.json").exists()


def test_failed_interaction_is_not_partially_committed(synthetic_project):
    prepare_pair(synthetic_project)

    def failing(purpose: str, prompt: str):
        if purpose.startswith("item-update:"):
            raise TimeoutError("simulated timeout")
        return responder(purpose, prompt)

    with pytest.raises(TimeoutError):
        train(synthetic_project, "smoke", run_id="failed-test", llm=ScriptedLLM(failing))
    state_path = Path(synthetic_project["_project_root"]) / "runs/failed-test/memory/state.json"
    state = read_json(state_path)
    assert state["last_completed"] == -1
    initial = "I am an Amazon buyer, and I enjoy Books very much."
    assert all(user["Books"]["private"] == initial for user in state["users"].values())


def test_full_group_memory_and_group_evaluation(synthetic_project):
    prepare_pair(synthetic_project)
    llm = ScriptedLLM(responder)
    context = train(synthetic_project, "full", run_id="full-test", llm=llm)
    build_group_memory(synthetic_project, "full", run_id="full-test", llm=llm, encoder=FakeEncoder())
    groups = read_json(context.run_dir / "memory/groups.json")
    assert 1 <= len(groups["groups"]) <= 2
    state_hash = sha256_file(context.state_path)
    metrics = evaluate(synthetic_project, "full", "test", use_group_memory=True, run_id="full-test", llm=llm)
    assert metrics["valid_samples"] == 2
    assert sha256_file(context.state_path) == state_hash
